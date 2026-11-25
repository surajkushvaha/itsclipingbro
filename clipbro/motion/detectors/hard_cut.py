"""Hard-cut detector: every scene cut inside the analysed range is an event."""

from __future__ import annotations

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.events import locate

DEFAULTS: dict[str, float] = {}


@register("detector", "hard_cut")
class HardCutDetector:
    """One full-strength event per cut in ctx.cuts that lies within ctx.rng."""

    kind = "hard_cut"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return a hard_cut event at each cut time inside the range."""
        events: list[Event] = []
        for cut in ctx.cuts:
            if not ctx.rng.start <= cut <= ctx.rng.end:
                continue
            x, y = locate(ctx, cut)
            events.append(Event(t=float(cut), kind="hard_cut", strength=1.0, x=x, y=y, reason="scene cut"))
        return events
