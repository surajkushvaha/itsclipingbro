"""AnalysisContext: shared state for a single analysis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import numpy as np

from .config import MotionConfig
from .ports import FaceDetector, WordScorer
from .types import AudioTrack, FrameBatch, SaliencyMap, TimeRange, Word

T = TypeVar("T")


@dataclass
class AnalysisContext:
    """Everything a channel or detector may read during one analysis run."""

    frames: FrameBatch
    rng: TimeRange
    config: MotionConfig
    audio: AudioTrack | None = None
    cuts: list[float] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    faces: FaceDetector | None = None
    scorer: WordScorer | None = None
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    saliency: SaliencyMap | None = None
    _memo: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def grid(self) -> tuple[int, int]:
        """Grid shape (rows, cols) taken from the config."""
        return self.config.grid

    def memo(self, key: str, compute: Callable[[], T]) -> T:
        """Return the cached value for key, computing it once on first use."""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def threshold(self, plugin: str, key: str, default: float) -> float:
        """Read a tunable threshold, delegating to the config."""
        return self.config.threshold(plugin, key, default)
