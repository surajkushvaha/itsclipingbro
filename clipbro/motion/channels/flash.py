"""Flash channel: sudden per-cell luminance bursts and orange fire-like colour rises."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.grid import cell_mean, stats_frames, stats_gray

DEFAULTS: dict[str, float] = {"lum_jump": 35.0, "return_frames": 3.0, "orange_jump": 0.12}


def _orange_fraction_cells(frames: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Per-cell fraction of orange pixels (R>200, 90<G<190, B<90) for each frame."""
    red = frames[..., 0]
    green = frames[..., 1]
    blue = frames[..., 2]
    orange = (red > 200) & (green > 90) & (green < 190) & (blue < 90)
    return cell_mean(orange.astype(np.float32), grid)


def _future_min(levels: np.ndarray, horizon: int) -> np.ndarray:
    """For each frame, the lowest level within the next horizon frames (inf when none)."""
    count = levels.shape[0]
    result = np.full(levels.shape, np.inf, dtype=np.float64)
    for step in range(1, horizon + 1):
        if step >= count:
            break
        np.minimum(result[: count - step], levels[step:], out=result[: count - step])
    return result


def _burst_scores(levels: np.ndarray, lum_jump: float, horizon: int) -> np.ndarray:
    """Score one-step luminance rises that fall back by half the rise within horizon frames."""
    rises = np.zeros_like(levels, dtype=np.float64)
    rises[1:] = levels[1:] - levels[:-1]
    floor = _future_min(levels, horizon)
    falls = np.where(np.isfinite(floor), levels - floor, 0.0)
    burst = (rises >= lum_jump) & (falls >= 0.5 * rises)
    return np.where(burst, np.clip(rises / (2.0 * lum_jump), 0.0, 1.0), 0.0)


def _fire_scores(orange: np.ndarray, orange_jump: float) -> np.ndarray:
    """Score per-cell rises in orange fraction, saturating at orange_jump."""
    rises = np.zeros_like(orange, dtype=np.float64)
    rises[1:] = orange[1:] - orange[:-1]
    return np.clip(rises / max(orange_jump, 1e-9), 0.0, 1.0)


@register("channel", "flash")
class FlashChannel:
    """Flash and fire-like bursts per cell; the maximum of the luminance and colour scores."""

    name = "flash"
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] flash strength in 0..1."""
        rows, cols = ctx.grid
        count = ctx.frames.n
        if count < 2:
            return np.zeros((count, rows, cols), dtype=np.float32)
        luminance = cell_mean(stats_gray(ctx).astype(np.float32), ctx.grid)
        orange = _orange_fraction_cells(stats_frames(ctx), ctx.grid)
        lum_jump = ctx.threshold("flash", "lum_jump", DEFAULTS["lum_jump"])
        horizon = int(round(ctx.threshold("flash", "return_frames", DEFAULTS["return_frames"])))
        orange_jump = ctx.threshold("flash", "orange_jump", DEFAULTS["orange_jump"])
        scores = np.maximum(
            _burst_scores(luminance, lum_jump, horizon),
            _fire_scores(orange, orange_jump),
        )
        return np.clip(scores, 0.0, 1.0).astype(np.float32)
