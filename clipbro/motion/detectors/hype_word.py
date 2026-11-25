"""Hype-word detector: transcribed words that a scorer rates as hype."""

from __future__ import annotations

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.events import locate

DEFAULTS: dict[str, float] = {"min": 0.5}


@register("detector", "hype_word")
class HypeWordDetector:
    """One event per word inside ctx.rng whose score reaches `min`, at the word's start time."""

    kind = "hype_word"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return hype-word events; without a scorer or words there are none."""
        if ctx.scorer is None or not ctx.words:
            return []
        level = ctx.threshold("hype_word", "min", DEFAULTS["min"])
        events: list[Event] = []
        for word in ctx.words:
            if not ctx.rng.start <= word.start <= ctx.rng.end:
                continue
            score = float(ctx.scorer.score(word.text))
            if score < level:
                continue
            x, y = locate(ctx, word.start)
            events.append(
                Event(
                    t=float(word.start),
                    kind="hype_word",
                    strength=score,
                    x=x,
                    y=y,
                    reason=f'hype word "{word.text}" {score:.2f}',
                )
            )
        return events
