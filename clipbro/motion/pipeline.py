"""Motion analysis pipeline: sample media, build saliency, detect events, derive focus.

The analyzer depends only on ports (clipbro.core) and on the motion package. Concrete
adapters are injected by clipbro.bootstrap. Results are cached per media file, range,
config fingerprint and pipeline version.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clipbro.core.config import MotionConfig
from clipbro.core.context import AnalysisContext
from clipbro.core.errors import MediaError
from clipbro.core.ports import AudioSource, Cache, CutDetector, FaceDetector, FocusSmoother, FrameSource, WordScorer
from clipbro.core.registry import REGISTRY, Registry
from clipbro.core.types import AudioTrack, Event, FocusPoint, SaliencyMap, TimeRange, Word
from clipbro.motion.events import run_detectors
from clipbro.motion.focus import focus_path
from clipbro.motion.saliency import run_channels

VERSION = 1
COMBINED_FPS = 2.0
_ROUND_DIGITS = 4


@dataclass
class MotionResult:
    """Output of one analysis: the JSON payload and, on fresh runs only, the saliency map."""

    payload: dict[str, Any]
    saliency: SaliencyMap | None = None


class MotionAnalyzer:
    """Runs the full motion pipeline for a time range of a media file."""

    def __init__(
        self,
        frames: FrameSource,
        config: MotionConfig,
        *,
        audio: AudioSource | None = None,
        cuts: CutDetector | None = None,
        faces: FaceDetector | None = None,
        scorer: WordScorer | None = None,
        cache: Cache | None = None,
        smoother: FocusSmoother | None = None,
        registry: Registry = REGISTRY,
    ) -> None:
        """Store the ports and the configuration used by every analysis."""
        self.frames = frames
        self.config = config
        self.audio = audio
        self.cuts = cuts
        self.faces = faces
        self.scorer = scorer
        self.cache = cache
        self.smoother = smoother
        self.registry = registry

    def analyze(
        self,
        media: Path,
        rng: TimeRange,
        words: list[Word] | None = None,
        use_cache: bool = True,
    ) -> MotionResult:
        """Analyse rng of media and return the payload, reusing the cache when allowed.

        Raises MediaError when media does not exist. A cache hit returns before any adapter runs.
        """
        key = self.cache_key(media, rng)
        if use_cache and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return MotionResult(payload=cached)
        started = time.perf_counter()
        ctx = self._context(Path(media), rng, words or [])
        sampled = time.perf_counter()
        sal = run_channels(ctx, registry=self.registry)
        saliency_done = time.perf_counter()
        events = run_detectors(ctx, registry=self.registry)
        events_done = time.perf_counter()
        focus = focus_path(sal, self.smoother, self.config.top_k)
        timing = {
            "sample_s": sampled - started,
            "saliency_s": saliency_done - sampled,
            "events_s": events_done - saliency_done,
            "total_s": time.perf_counter() - started,
        }
        payload = _payload(ctx, sal, events, focus, timing)
        if self.cache is not None:
            self._store(key, payload, sal, rng)
        return MotionResult(payload=payload, saliency=sal)

    def cache_key(self, media: Path, rng: TimeRange) -> str:
        """Return a sha256 key over the resolved path, mtime, range, version and config fingerprint."""
        resolved = Path(media).resolve()
        try:
            mtime = os.stat(resolved).st_mtime_ns
        except OSError as error:
            raise MediaError(f"media not found: {media}") from error
        text = (
            f"{resolved}|{mtime}|{rng.start:.3f}|{rng.end:.3f}|{VERSION}|{self.config.fingerprint()}"
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _context(self, media: Path, rng: TimeRange, words: list[Word]) -> AnalysisContext:
        """Sample frames, audio and cuts for rng and wrap them in an AnalysisContext."""
        if not media.is_file():
            raise MediaError(f"media not found: {media}")
        frames = self.frames.read(media, rng, self.config.fps, self.config.width)
        audio = self._audio(media, rng)
        cuts = self.cuts.cuts(media, rng) if self.cuts is not None else []
        in_range = [word for word in words if rng.start <= word.start < rng.end]
        return AnalysisContext(
            frames=frames,
            rng=rng,
            config=self.config,
            audio=audio,
            cuts=cuts,
            words=in_range,
            faces=self.faces,
            scorer=self.scorer,
        )

    def _audio(self, media: Path, rng: TimeRange) -> AudioTrack | None:
        """Return the decoded audio of rng, or None when no source is configured or there is no audio."""
        if self.audio is None:
            return None
        return self.audio.read(media, rng, self.config.audio_sr)

    def _store(self, key: str, payload: dict[str, Any], sal: SaliencyMap, rng: TimeRange) -> None:
        """Persist the payload and the 2 fps combined saliency under key."""
        self.cache.put(key, payload)
        self.cache.put_array(key, "combined", _combined_2fps(sal, rng))


def _combined_2fps(sal: SaliencyMap, rng: TimeRange) -> np.ndarray:
    """Pick the nearest analysed frame to each 0.5 s step from rng.start; return float16 [K, rows, cols]."""
    count = int(math.floor(rng.duration * COMBINED_FPS)) + 1
    picks = [sal.nearest(rng.start + step / COMBINED_FPS) for step in range(count)]
    return sal.combined[picks].astype(np.float16)


def _round(value: float) -> float:
    """Round a float to the payload precision."""
    return round(float(value), _ROUND_DIGITS)


def _event_dict(event: Event) -> dict[str, Any]:
    """Serialise an event with its numeric fields rounded to the payload precision."""
    data = event.to_dict()
    for key in ("t", "strength", "x", "y"):
        data[key] = _round(data[key])
    return data


def _payload(
    ctx: AnalysisContext,
    sal: SaliencyMap,
    events: list[Event],
    focus: list[FocusPoint],
    timing: dict[str, float],
) -> dict[str, Any]:
    """Assemble the JSON-safe payload described in ARCHITECTURE section 13."""
    width, height = ctx.frames.source_size
    rows, cols = sal.grid
    return {
        "version": VERSION,
        "source_size": [int(width), int(height)],
        "range": [_round(ctx.rng.start), _round(ctx.rng.end)],
        "grid": [int(rows), int(cols)],
        "events": [_event_dict(event) for event in events],
        "focus": [[_round(p.t), _round(p.cx), _round(p.cy)] for p in focus],
        "hud": [[int(cell) for cell in row] for row in sal.hud],
        "timing": {name: _round(seconds) for name, seconds in timing.items()},
    }
