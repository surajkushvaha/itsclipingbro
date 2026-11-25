"""UI-change channel: frame-to-frame content changes inside HUD cells (alerts, kill feeds, pop-ups)."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.channels.hud import hud_mask
from clipbro.motion.grid import cell_mean, stats_gray

DEFAULTS: dict[str, float] = {"scale": 30.0}


@register("channel", "ui_change")
class UiChangeChannel:
    """Mean absolute change per HUD cell between consecutive frames; zero outside HUD cells."""

    name = "ui_change"
    requires: tuple[str, ...] = ("hud",)

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] change strength in 0..1; frame 0 is zero."""
        rows, cols = ctx.grid
        count = ctx.frames.n
        out = np.zeros((count, rows, cols), dtype=np.float32)
        if count < 2:
            return out
        gray = stats_gray(ctx)
        scale = max(ctx.threshold("ui_change", "scale", DEFAULTS["scale"]), 1e-6)
        later, earlier = gray[1:], gray[:-1]
        diffs = np.maximum(later, earlier) - np.minimum(later, earlier)
        out[1:] = cell_mean(diffs, ctx.grid) / scale
        out = np.where(hud_mask(ctx)[None, :, :], out, 0.0)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
