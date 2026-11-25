"""Plugin registry: channels, detectors and smoothers self-register here."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Iterable

_NAME_ATTRIBUTE_BY_KIND = {"detector": "kind"}


def _name_attribute(kind: str) -> str:
    """Return the class attribute that holds the registered name for a kind."""
    return _NAME_ATTRIBUTE_BY_KIND.get(kind, "name")


def _topological_order(names: list[str], requires: dict[str, tuple[str, ...]]) -> list[str]:
    """Stable topological sort: repeatedly emit the first name whose deps are placed."""
    placed: set[str] = set()
    pending = list(names)
    ordered: list[str] = []
    while pending:
        ready = next((n for n in pending if all(d in placed for d in requires[n])), None)
        if ready is None:
            raise ValueError(f"dependency cycle among: {', '.join(pending)}")
        ordered.append(ready)
        placed.add(ready)
        pending.remove(ready)
    return ordered


class Registry:
    """Maps (kind, name) to plugin classes."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._entries: dict[str, dict[str, type]] = {}

    def register(self, kind: str, name: str) -> Callable[[type], type]:
        """Return a class decorator that registers the class under (kind, name)."""
        if not name:
            raise ValueError("plugin name must be non-empty")

        def decorator(cls: type) -> type:
            """Store cls, filling its name/kind attribute when the class lacks one."""
            table = self._entries.setdefault(kind, {})
            if name in table:
                raise ValueError(f"{kind} {name!r} is already registered")
            attribute = _name_attribute(kind)
            if not getattr(cls, attribute, None):
                setattr(cls, attribute, name)
            table[name] = cls
            return cls

        return decorator

    def unregister(self, kind: str, name: str) -> None:
        """Remove a registration; unknown entries raise KeyError."""
        self.get(kind, name)
        del self._entries[kind][name]

    def names(self, kind: str) -> list[str]:
        """Return the registered names of a kind, in registration order."""
        return list(self._entries.get(kind, {}))

    def get(self, kind: str, name: str) -> type:
        """Return the class registered under (kind, name)."""
        table = self._entries.get(kind, {})
        if name not in table:
            known = ", ".join(sorted(table)) or "none"
            raise KeyError(f"unknown {kind} {name!r}; registered: {known}")
        return table[name]

    def create(self, kind: str, name: str, **kwargs: Any) -> Any:
        """Instantiate the class registered under (kind, name)."""
        return self.get(kind, name)(**kwargs)

    def ordered(self, kind: str, names: Iterable[str]) -> list[str]:
        """Sort names so every plugin comes after the names listed in its `requires`.

        Dependencies outside `names` raise ValueError, as do cycles.
        """
        wanted = list(dict.fromkeys(names))
        requires = {n: tuple(getattr(self.get(kind, n), "requires", ())) for n in wanted}
        for name, deps in requires.items():
            for dep in deps:
                if dep not in requires:
                    raise ValueError(f"{kind} {name!r} requires {dep!r}, which is not selected")
        return _topological_order(wanted, requires)


def discover(package: str) -> None:
    """Import every direct submodule of a package so its plugins register themselves.

    Importing is idempotent, so calling this repeatedly is harmless.
    """
    module = importlib.import_module(package)
    search_path = getattr(module, "__path__", None)
    if search_path is None:
        return
    for info in pkgutil.iter_modules(search_path):
        importlib.import_module(f"{package}.{info.name}")


REGISTRY = Registry()
register = REGISTRY.register
