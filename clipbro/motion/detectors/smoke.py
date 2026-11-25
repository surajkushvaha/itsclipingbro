"""Smoke detector: sustained runs of smoke-like saliency on non-HUD cells."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.events import frame_peak, hud_free, peak_location, runs_of

DEFAULTS: dict[str, float] = {"min": 0.4, "min_s": 0.5}


def _smoke_event(ctx: AnalysisContext, first: int, last: int, levels: np.ndarray, duration: float) -> Event:
    """Build the smoke event for one run: time of its start, mean level, location of its peak."""
    peak = first + int(np.argmax(levels[first : last + 1]))
    x, y = peak_location(hud_free(ctx, "smoke")[peak], ctx.grid)
    strength = float(np.mean(levels[first : last + 1]))
    return Event(
        t=float(ctx.frames.times[first]),
        kind="smoke",
        strength=strength,
        x=x,
        y=y,
        reason=f"smoke {strength:.2f} for {duration:.1f}s",
    )


@register("detector", "smoke")
class SmokeDetector:
    """Reports each run of frames whose smoke level stays above `min` for at least `min_s` seconds."""

    kind = "smoke"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return one smoke event per qualifying run of frames."""
        level = ctx.threshold("smoke", "min", DEFAULTS["min"])
        min_s = ctx.threshold("smoke", "min_s", DEFAULTS["min_s"])
        levels = frame_peak(ctx, "smoke").astype(np.float64)
        times = ctx.frames.times
        frame_s = 1.0 / ctx.frames.fps
        events: list[Event] = []
        for first, last in runs_of(levels >= level):
            duration = float(times[last] - times[first]) + frame_s
            if duration >= min_s:
                events.append(_smoke_event(ctx, first, last, levels, duration))
        return events
