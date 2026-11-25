"""Motion analysis configuration and named profiles."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _default_channels() -> list[str]:
    """Default channel names, in the order they are listed in the design."""
    return ["hud", "motion", "flash", "smoke", "face", "ui_change", "centre"]


def _default_detectors() -> list[str]:
    """Default event detector kinds."""
    return ["boom", "smoke", "scream", "hype_word", "hard_cut", "dead_air"]


def _default_weights() -> dict[str, float]:
    """Default combination weights; keys of the form 'a|b' mean element-wise max."""
    return {"motion": 0.35, "flash|smoke": 0.25, "face": 0.25, "ui_change": 0.10, "centre": 0.05}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of base with override merged in, recursing into nested dicts."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass
class MotionConfig:
    """All tunables of the motion pipeline. Plugin thresholds live in `thresholds`."""

    fps: float = 6.0
    width: int = 320
    grid: tuple[int, int] = (8, 8)
    audio_sr: int = 16000
    face_every: int = 2
    top_k: int = 3
    dedupe_window: float = 0.5
    channels: list[str] = field(default_factory=_default_channels)
    detectors: list[str] = field(default_factory=_default_detectors)
    weights: dict[str, float] = field(default_factory=_default_weights)
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise grid to a tuple of two positive ints."""
        if len(self.grid) != 2:
            raise ValueError(f"grid must have two entries (rows, cols), got {self.grid!r}")
        rows, cols = int(self.grid[0]), int(self.grid[1])
        if rows < 1 or cols < 1:
            raise ValueError(f"grid must be positive, got {self.grid!r}")
        self.grid = (rows, cols)

    def threshold(self, plugin: str, key: str, default: float) -> float:
        """Return the configured value for plugin/key, or default when not set."""
        overrides = self.thresholds.get(plugin, {})
        if key in overrides:
            return float(overrides[key])
        return default

    def fingerprint(self) -> str:
        """Return a stable sha256 hex digest of every field."""
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_overrides(self, **changes: Any) -> MotionConfig:
        """Return a new config with changes applied; dict fields are deep-merged."""
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(changes) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        updates: dict[str, Any] = {}
        for key, value in changes.items():
            current = getattr(self, key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                updates[key] = _deep_merge(current, value)
            else:
                updates[key] = copy.deepcopy(value)
        return dataclasses.replace(copy.deepcopy(self), **updates)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionConfig:
        """Build a config from a mapping; absent keys take defaults, unknown keys raise."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        return cls(**copy.deepcopy(dict(data)))

    @classmethod
    def from_toml(cls, path: Path) -> MotionConfig:
        """Load a config from a TOML file (top-level keys). Requires Python 3.11+."""
        try:
            import tomllib
        except ImportError as exc:
            raise ValueError("TOML config needs Python 3.11+") from exc
        with Path(path).open("rb") as handle:
            data = tomllib.load(handle)
        return cls.from_dict(data)

    @classmethod
    def profile(cls, name: str) -> MotionConfig:
        """Return the config for a named profile in PROFILES."""
        if name not in PROFILES:
            valid = ", ".join(sorted(PROFILES))
            raise ValueError(f"unknown profile {name!r}; valid profiles: {valid}")
        return cls.from_dict(PROFILES[name])


PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "talk": {
        "weights": {"face": 0.45, "motion": 0.20, "flash|smoke": 0.10, "ui_change": 0.10, "centre": 0.15},
    },
    "gaming": {
        "weights": {"motion": 0.40, "flash|smoke": 0.30, "face": 0.15, "ui_change": 0.10, "centre": 0.05},
    },
}
