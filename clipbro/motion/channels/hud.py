"""HUD channel: static screen overlays found from low temporal variance plus edge density."""

from __future__ import annotations

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.grid import cell_mean, stats_gray

DEFAULTS: dict[str, float] = {"std_max": 4.0, "edge_min": 0.06, "min_frames": 6}

CANNY_LOW = 80
CANNY_HIGH = 160


def _temporal_std_cells(gray: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Per-cell mean of the per-pixel temporal standard deviation of gray frames."""
    pixel_std = gray.astype(np.float32).std(axis=0)
    return cell_mean(pixel_std, grid)


def _edge_fraction_cells(gray: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Per-cell mean fraction of Canny edge pixels, averaged over all frames."""
    edges = np.stack([cv2.Canny(frame, CANNY_LOW, CANNY_HIGH) for frame in gray])
    return cell_mean((edges > 0).astype(np.float32), grid).mean(axis=0)


def _compute_mask(ctx: AnalysisContext) -> np.ndarray:
    """Build the bool [rows, cols] HUD mask; all False when there are too few frames."""
    gray = stats_gray(ctx)
    rows, cols = ctx.grid
    min_frames = ctx.threshold("hud", "min_frames", DEFAULTS["min_frames"])
    if gray.shape[0] < max(2.0, min_frames):
        return np.zeros((rows, cols), dtype=bool)
    std_max = ctx.threshold("hud", "std_max", DEFAULTS["std_max"])
    edge_min = ctx.threshold("hud", "edge_min", DEFAULTS["edge_min"])
    static = _temporal_std_cells(gray, ctx.grid) < std_max
    detailed = _edge_fraction_cells(gray, ctx.grid) > edge_min
    return static & detailed


def hud_mask(ctx: AnalysisContext) -> np.ndarray:
    """Return the shared bool [rows, cols] HUD mask, computed once per run."""
    return ctx.memo("hud_mask", lambda: _compute_mask(ctx))


@register("channel", "hud")
class HudChannel:
    """Marks static overlay cells and broadcasts the 0/1 mask to every frame."""

    name = "hud"
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] with 1.0 on HUD cells and 0.0 elsewhere."""
        mask = hud_mask(ctx)
        return np.broadcast_to(mask.astype(np.float32), (ctx.frames.n, *mask.shape)).astype(np.float32)
