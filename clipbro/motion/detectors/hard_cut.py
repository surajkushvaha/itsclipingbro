"""Hard-cut detector: a scene cut inside the analysed range is an event, unless it coincides with a flash."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.events import frame_peak, locate

DEFAULTS: dict[str, float] = {"flash_guard": 0.5, "flash_guard_s": 0.25}


def _flash_near(ctx: AnalysisContext, t: float, guard: float, guard_s: float) -> bool:
    """True when a sampled frame within guard_s seconds of t has a non-HUD flash at or above guard.

    A flash is not an edit: a cut inside a flash is the flash's own luminance step.
    """
    flash = frame_peak(ctx, "flash")
    near = np.abs(ctx.frames.times - t) <= guard_s
    return bool(np.any(flash[near] >= guard))


@register("detector", "hard_cut")
class HardCutDetector:
    """One full-strength event per cut in ctx.cuts that lies within ctx.rng and is not on a flash."""

    kind = "hard_cut"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return a hard_cut event at each cut time inside the range that is not flash-guarded."""
        guard = ctx.threshold("hard_cut", "flash_guard", DEFAULTS["flash_guard"])
        guard_s = ctx.threshold("hard_cut", "flash_guard_s", DEFAULTS["flash_guard_s"])
        events: list[Event] = []
        for cut in ctx.cuts:
            if not ctx.rng.start <= cut <= ctx.rng.end:
                continue
            if _flash_near(ctx, cut, guard, guard_s):
                continue
            x, y = locate(ctx, cut)
            events.append(Event(t=float(cut), kind="hard_cut", strength=1.0, x=x, y=y, reason="scene cut"))
        return events
