"""Ports: Protocol interfaces implemented by adapters.

Nothing in this module performs I/O. Concrete implementations live in
clipbro.adapters, clipbro.hype or clipbro.motion and are wired by clipbro.bootstrap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from .context import AnalysisContext
    from .types import AudioTrack, Box, Event, FocusPoint, FrameBatch, TimeRange, Word


@runtime_checkable
class FrameSource(Protocol):
    """Decodes sampled frames of a time range from a media file."""

    def read(self, media: Path, rng: TimeRange, fps: float, width: int) -> FrameBatch:
        """Return frames sampled at fps and scaled to width."""
        ...


@runtime_checkable
class AudioSource(Protocol):
    """Decodes the audio of a time range from a media file."""

    def read(self, media: Path, rng: TimeRange, sr: int) -> AudioTrack | None:
        """Return mono audio at sr, or None when the media has no audio stream."""
        ...


@runtime_checkable
class CutDetector(Protocol):
    """Finds hard scene cuts inside a time range."""

    def cuts(self, media: Path, rng: TimeRange) -> list[float]:
        """Return absolute cut times inside rng."""
        ...


@runtime_checkable
class FaceDetector(Protocol):
    """Detects faces in a single image."""

    def detect(self, image: np.ndarray) -> list[Box]:
        """Take an RGB uint8 image and return normalised boxes."""
        ...


@runtime_checkable
class WordScorer(Protocol):
    """Scores how exciting a word is."""

    def score(self, word: str) -> float:
        """Return a score in 0..1."""
        ...


@runtime_checkable
class Cache(Protocol):
    """Key-value cache for JSON payloads and named arrays."""

    def get(self, key: str) -> dict | None:
        """Return the stored payload for key, or None."""
        ...

    def put(self, key: str, value: dict) -> None:
        """Store a JSON payload under key."""
        ...

    def put_array(self, key: str, name: str, array: np.ndarray) -> None:
        """Store a named array under key."""
        ...

    def get_array(self, key: str, name: str) -> np.ndarray | None:
        """Return a named array stored under key, or None."""
        ...


@runtime_checkable
class SignalChannel(Protocol):
    """Produces a per-frame, per-cell signal in 0..1."""

    name: str
    requires: tuple[str, ...]

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] values in 0..1."""
        ...


@runtime_checkable
class EventDetector(Protocol):
    """Turns the analysed context into timestamped events."""

    kind: str

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return the events found in the context."""
        ...


@runtime_checkable
class FocusSmoother(Protocol):
    """Smooths a raw focus path."""

    def smooth(self, path: list[FocusPoint]) -> list[FocusPoint]:
        """Return a smoothed copy of the path."""
        ...
