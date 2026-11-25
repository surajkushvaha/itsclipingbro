"""Enforces the import-direction rules of ARCHITECTURE section 1 by parsing every module with ast."""

from __future__ import annotations

import ast
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "clipbro"


@dataclass(frozen=True)
class ImportEdge:
    """One import statement: the file, its line and the absolute module name it reaches."""

    path: str
    line: int
    module: str


def _module_name(path: Path) -> str:
    """Return the dotted module name of a file under the repository root."""
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_parts(path: Path) -> list[str]:
    """Return the package a file belongs to, as a list of name parts (used for relative imports)."""
    name = _module_name(path).split(".")
    if path.name == "__init__.py":
        return name
    return name[:-1]


def _resolve_relative(path: Path, level: int, module: str | None) -> str:
    """Resolve a relative import of the given level inside path to an absolute module name."""
    base = _package_parts(path)
    if level > 1:
        base = base[: len(base) - (level - 1)]
    parts = list(base)
    if module:
        parts.append(module)
    return ".".join(parts)


def _edges_of(path: Path) -> list[ImportEdge]:
    """Return every import edge of one source file, including names imported from packages."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = str(path.relative_to(REPO_ROOT))
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.extend(ImportEdge(rel, node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            edges.extend(_from_edges(path, rel, node))
    return edges


def _from_edges(path: Path, rel: str, node: ast.ImportFrom) -> list[ImportEdge]:
    """Expand one from-import into edges for the module and for each name that may be a submodule."""
    if node.level:
        target = _resolve_relative(path, node.level, node.module)
    else:
        target = node.module or ""
    edges = [ImportEdge(rel, node.lineno, target)]
    edges.extend(ImportEdge(rel, node.lineno, f"{target}.{alias.name}") for alias in node.names)
    return edges


def collect_package_edges() -> list[ImportEdge]:
    """Return the import edges of every .py file under clipbro/."""
    edges: list[ImportEdge] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        edges.extend(_edges_of(path))
    return edges


def _under(module: str, prefix: str) -> bool:
    """Return True when module equals prefix or is a submodule of it."""
    return module == prefix or module.startswith(prefix + ".")


def _is_clipbro(module: str) -> bool:
    """Return True for the clipbro package itself or any of its submodules."""
    return _under(module, "clipbro")


def _violations(edges: list[ImportEdge], source_prefix: str, rule: Callable[[str], str]) -> list[str]:
    """Return failure lines for edges from modules under source_prefix that rule rejects."""
    found: list[str] = []
    for edge in edges:
        if not _under(_module_for_path(edge.path), source_prefix):
            continue
        reason = rule(edge.module)
        if reason:
            found.append(f"{edge.path}:{edge.line} imports {edge.module} ({reason})")
    return found


def _module_for_path(rel: str) -> str:
    """Return the dotted module name for a repository-relative path string."""
    return _module_name(REPO_ROOT / rel)


def _core_rule(module: str) -> str:
    """Core may import only clipbro.core and must not touch subprocess or cv2."""
    if module.split(".")[0] in {"subprocess", "cv2"}:
        return "core must not use subprocess or cv2"
    if _is_clipbro(module) and not _under(module, "clipbro.core"):
        return "core may import only clipbro.core"
    return ""


def _motion_rule(module: str) -> str:
    """Motion may import clipbro.core and clipbro.motion only, and must not spawn processes."""
    if module.split(".")[0] == "subprocess":
        return "motion must not use subprocess"
    if _is_clipbro(module) and not (_under(module, "clipbro.core") or _under(module, "clipbro.motion")):
        return "motion may import only clipbro.core and clipbro.motion"
    return ""


def _adapter_rule(module: str) -> str:
    """Adapters may import clipbro.core and clipbro.adapters only."""
    if _is_clipbro(module) and not (_under(module, "clipbro.core") or _under(module, "clipbro.adapters")):
        return "adapters may import only clipbro.core and clipbro.adapters"
    return ""


def _bootstrap_importers_rule(edges: list[ImportEdge]) -> list[str]:
    """Return failure lines for every module, other than the allowed layers, importing clipbro.bootstrap."""
    allowed = ("clipbro.bootstrap", "clipbro.api", "clipbro.cli")
    found: list[str] = []
    for edge in edges:
        if not _under(edge.module, "clipbro.bootstrap"):
            continue
        importer = _module_for_path(edge.path)
        if not any(_under(importer, prefix) for prefix in allowed):
            found.append(f"{edge.path}:{edge.line} imports {edge.module} (only bootstrap, api and cli may)")
    return found


class ArchitectureTest(unittest.TestCase):
    """Import-direction rules of the layered design."""

    @classmethod
    def setUpClass(cls) -> None:
        """Parse the package once; every rule reads the same edge list."""
        cls.edges = collect_package_edges()

    def _assert_clean(self, found: list[str], rule_name: str) -> None:
        """Fail with every offending file, line and import when a rule is broken."""
        message = f"{rule_name} violated by {len(found)} import(s):\n  " + "\n  ".join(found)
        self.assertEqual(found, [], message)

    def test_parser_sees_known_edges(self) -> None:
        """Guard against a silently empty parse: a known import must be collected."""
        modules = {edge.module for edge in self.edges}
        self.assertIn("clipbro.core.types", modules)
        self.assertIn("clipbro.adapters.ffmpeg_video", modules)

    def test_core_imports_only_core(self) -> None:
        """clipbro.core imports only clipbro.core (plus stdlib and numpy), never subprocess or cv2."""
        self._assert_clean(_violations(self.edges, "clipbro.core", _core_rule), "core rule")

    def test_motion_imports_only_core_and_motion(self) -> None:
        """clipbro.motion imports nothing from adapters, api, cli, bootstrap or hype, and no subprocess."""
        self._assert_clean(_violations(self.edges, "clipbro.motion", _motion_rule), "motion rule")

    def test_adapters_import_only_core_and_adapters(self) -> None:
        """clipbro.adapters imports no motion, api, cli, bootstrap or hype module."""
        self._assert_clean(_violations(self.edges, "clipbro.adapters", _adapter_rule), "adapter rule")

    def test_only_entry_layers_import_bootstrap(self) -> None:
        """Only clipbro.bootstrap, clipbro.api.* and clipbro.cli.* may import clipbro.bootstrap."""
        self._assert_clean(_bootstrap_importers_rule(self.edges), "bootstrap import rule")


if __name__ == "__main__":
    unittest.main()
