"""Boom detector: a bright full-frame flash that coincides with a sudden loud audio onset."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.audio_features import loud_jumps
from clipbro.motion.events import hud_free, peak_location, runs_of

DEFAULTS: dict[str, float] = {"flash_min": 0.5, "window": 0.3}

SATURATION_RATIO = 6.0


def _matched_ratios(times: np.ndarray, jumps: list[tuple[float, float]], window: float) -> np.ndarray:
    """Per frame, the largest audio onset ratio within +-window seconds; 0 where none matches."""
    ratios = np.zeros(times.shape, dtype=np.float64)
    for jump_time, ratio in jumps:
        near = np.abs(times - jump_time) <= window
        ratios = np.where(near, np.maximum(ratios, ratio), ratios)
    return ratios


def _boom_event(ctx: AnalysisContext, index: int, flash_level: float, ratio: float) -> Event:
    """Build the boom event for one peak frame."""
    strength = (flash_level + min(1.0, ratio / SATURATION_RATIO)) / 2.0
    x, y = peak_location(hud_free(ctx, "flash")[index], ctx.grid)
    return Event(
        t=float(ctx.frames.times[index]),
        kind="boom",
        strength=strength,
        x=x,
        y=y,
        reason=f"flash {flash_level:.2f} + audio x{ratio:.1f}",
    )


@register("detector", "boom")
class BoomDetector:
    """Finds flash frames that have an audio onset within +-window seconds, one event per burst."""

    kind = "boom"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return one boom per run of qualifying frames, placed on the run's peak flash frame."""
        jumps = loud_jumps(ctx.audio)
        if not jumps:
            return []
        flash_min = ctx.threshold("boom", "flash_min", DEFAULTS["flash_min"])
        window = ctx.threshold("boom", "window", DEFAULTS["window"])
        flash = np.asarray(ctx.saliency.channel("flash"), dtype=np.float64).shape and None
        flash_max = self._flash_levels(ctx)
        ratios = _matched_ratios(ctx.frames.times, jumps, window)
        qualifying = (flash_max >= flash_min) & (ratios > 0.0)
        events: list[Event] = []
        for first, last in runs_of(qualifying):
            peak = first + int(np.argmax(flash_max[first : last + 1]))
            events.append(_boom_event(ctx, peak, float(flash_max[peak]), float(ratios[peak])))
        return events

    def _flash_levels(self, ctx: AnalysisContext) -> np.ndarray:
        """Per-frame flash maximum over non-HUD cells, as float64."""
        return frame_peak_float(ctx)
