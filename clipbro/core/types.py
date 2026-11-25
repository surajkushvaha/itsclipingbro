"""Core data types for clipbro.

Conventions: coordinates are normalised 0..1 of the frame (x right, y down);
times are absolute seconds in the source media.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import InvalidRange


def _require_finite(name: str, value: float) -> None:
    """Raise ValueError unless value is a finite number."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _check_bounds(name: str, value: float, low: float, high: float) -> None:
    """Raise ValueError unless value is finite and within low..high inclusive."""
    if not (math.isfinite(value) and low <= value <= high):
        raise ValueError(f"{name} must be within [{low}, {high}], got {value!r}")


def _clamp_unit(value: float) -> float:
    """Clamp a finite value into the range 0..1."""
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class TimeRange:
    """Absolute time span in seconds, from start to end."""

    start: float
    end: float

    def __post_init__(self) -> None:
        """Validate that the bounds are finite, non-negative and non-empty."""
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise InvalidRange(f"TimeRange bounds must be finite, got {self.start}, {self.end}")
        if self.start < 0:
            raise InvalidRange(f"TimeRange start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise InvalidRange(f"TimeRange end ({self.end}) must be greater than start ({self.start})")

    @property
    def duration(self) -> float:
        """Length of the range in seconds."""
        return self.end - self.start


@dataclass(frozen=True, eq=False)
class FrameBatch:
    """Sampled RGB frames of a source video, with their absolute times."""

    frames: np.ndarray
    times: np.ndarray
    source_size: tuple[int, int]
    fps: float

    @property
    def n(self) -> int:
        """Number of sampled frames."""
        return int(self.frames.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the sampled frames."""
        return (int(self.frames.shape[2]), int(self.frames.shape[1]))


@dataclass(frozen=True, eq=False)
class AudioTrack:
    """Mono float32 audio samples in -1..1 starting at an absolute time."""

    samples: np.ndarray
    sr: int
    start: float

    @property
    def duration(self) -> float:
        """Length of the track in seconds."""
        return len(self.samples) / self.sr


@dataclass(frozen=True)
class Box:
    """Normalised bounding box with a detection score."""

    x: float
    y: float
    w: float
    h: float
    score: float


@dataclass(frozen=True)
class Word:
    """A transcribed word with its absolute start and end times."""

    text: str
    start: float
    end: float


@dataclass
class Event:
    """A timestamped moment of interest located in normalised frame coordinates."""

    t: float
    kind: str
    strength: float
    x: float
    y: float
    reason: str = ""

    def __post_init__(self) -> None:
        """Reject non-finite values and clamp strength and position into 0..1."""
        _require_finite("Event.t", self.t)
        _require_finite("Event.strength", self.strength)
        _require_finite("Event.x", self.x)
        _require_finite("Event.y", self.y)
        self.strength = _clamp_unit(self.strength)
        self.x = _clamp_unit(self.x)
        self.y = _clamp_unit(self.y)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "t": self.t,
            "kind": self.kind,
            "strength": self.strength,
            "x": self.x,
            "y": self.y,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Rebuild an event from the dict produced by to_dict."""
        return cls(
            t=float(data["t"]),
            kind=str(data["kind"]),
            strength=float(data["strength"]),
            x=float(data["x"]),
            y=float(data["y"]),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class FocusPoint:
    """A weighted focus position at an absolute time, in normalised coordinates."""

    t: float
    cx: float
    cy: float
    weight: float = 1.0


@dataclass(frozen=True)
class Keyframe:
    """A camera keyframe: centre, zoom, shake and playback speed at time t."""

    t: float
    cx: float
    cy: float
    zoom: float = 1.0
    shake: float = 0.0
    speed: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate every field against its allowed range."""
        _check_bounds("Keyframe.t", self.t, 0.0, math.inf)
        _check_bounds("Keyframe.cx", self.cx, 0.0, 1.0)
        _check_bounds("Keyframe.cy", self.cy, 0.0, 1.0)
        _check_bounds("Keyframe.zoom", self.zoom, 1.0, 2.0)
        _check_bounds("Keyframe.speed", self.speed, 0.25, 2.0)
        _check_bounds("Keyframe.shake", self.shake, 0.0, math.inf)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "t": self.t,
            "cx": self.cx,
            "cy": self.cy,
            "zoom": self.zoom,
            "shake": self.shake,
            "speed": self.speed,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keyframe:
        """Rebuild a keyframe from the dict produced by to_dict."""
        return cls(
            t=float(data["t"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            zoom=float(data.get("zoom", 1.0)),
            shake=float(data.get("shake", 0.0)),
            speed=float(data.get("speed", 1.0)),
            reason=str(data.get("reason", "")),
        )


@dataclass(eq=False)
class SaliencyMap:
    """Per-frame, per-cell saliency for one analysed range."""

    times: np.ndarray
    grid: tuple[int, int]
    channels: dict[str, np.ndarray]
    combined: np.ndarray
    hud: np.ndarray

    def channel(self, name: str) -> np.ndarray:
        """Return the named channel, or zeros shaped like combined if it is absent."""
        if name in self.channels:
            return self.channels[name]
        return np.zeros_like(self.combined)

    def nearest(self, t: float) -> int:
        """Return the index of the frame whose time is closest to t."""
        if self.times.size == 0:
            raise ValueError("SaliencyMap has no frames")
        return int(np.argmin(np.abs(self.times - t)))
