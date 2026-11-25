"""Boom detector: a bright full-frame flash that coincides with a sudden loud audio onset."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.audio_features import loud_jumps
from clipbro.motion.events import frame_peak, hud_free, hud_mask, locate, peak_location, runs_of

DEFAULTS: dict[str, float] = {"flash_min": 0.5, "window": 0.3, "uniform_spread": 0.15}

SATURATION_RATIO = 6.0


def _matched_ratios(times: np.ndarray, jumps: list[tuple[float, float]], window: float) -> np.ndarray:
    """Per frame, the largest audio onset ratio within +-window seconds; 0 where none matches."""
    ratios = np.zeros(times.shape, dtype=np.float64)
    for jump_time, ratio in jumps:
        near = np.abs(times - jump_time) <= window
        ratios = np.where(near, np.maximum(ratios, ratio), ratios)
    return ratios


def _is_uniform(open_cells: np.ndarray, spread: float) -> bool:
    """True when the non-HUD cells differ by less than spread, so the flash has no single source."""
    if open_cells.size == 0:
        return True
    return float(open_cells.max() - open_cells.min()) < spread


def _boom_location(ctx: AnalysisContext, index: int, spread: float) -> tuple[float, float]:
    """Strongest non-HUD flash cell; when the flash is near-uniform, the combined-saliency peak instead."""
    cells = hud_free(ctx, "flash")[index]
    if _is_uniform(cells[~hud_mask(ctx)], spread):
        return locate(ctx, float(ctx.frames.times[index]))
    return peak_location(cells, ctx.grid)


def _boom_event(ctx: AnalysisContext, index: int, flash_level: float, ratio: float, spread: float) -> Event:
    """Build the boom event for one peak frame, located by _boom_location."""
    strength = (flash_level + min(1.0, ratio / SATURATION_RATIO)) / 2.0
    x, y = _boom_location(ctx, index, spread)
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
        spread = ctx.threshold("boom", "uniform_spread", DEFAULTS["uniform_spread"])
        flash_max = frame_peak(ctx, "flash").astype(np.float64)
        ratios = _matched_ratios(ctx.frames.times, jumps, window)
        qualifying = (flash_max >= flash_min) & (ratios > 0.0)
        events: list[Event] = []
        for first, last in runs_of(qualifying):
            peak = first + int(np.argmax(flash_max[first : last + 1]))
            events.append(_boom_event(ctx, peak, float(flash_max[peak]), float(ratios[peak]), spread))
        return events
