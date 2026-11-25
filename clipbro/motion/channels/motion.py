"""Motion channel: Farneback optical-flow magnitude per cell, with cut and HUD suppression."""

from __future__ import annotations

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.channels.hud import hud_mask
from clipbro.motion.grid import cell_reduce, to_gray

DEFAULTS: dict[str, float] = {"scale_px": 2.0, "cut_diff": 38.0}

REFERENCE_WIDTH = 160


def _flow_magnitude(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Return the per-pixel Farneback flow magnitude between two gray frames."""
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.hypot(flow[..., 0], flow[..., 1])


def _cell_p90(magnitude: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Per-cell 90th percentile of a magnitude map."""
    return cell_reduce(magnitude, grid, lambda block: float(np.percentile(block, 90)))


def _is_cut(
    index: int,
    gray: np.ndarray,
    times: np.ndarray,
    cut_times: np.ndarray,
    cut_diff: float,
) -> bool:
    """Return True when frame index is a scene cut relative to index - 1."""
    diff = float(np.mean(np.abs(gray[index].astype(np.float32) - gray[index - 1].astype(np.float32))))
    if diff > cut_diff:
        return True
    if cut_times.size == 0:
        return False
    lo, hi = times[index - 1], times[index]
    return bool(np.any((cut_times > lo) & (cut_times <= hi)))


def _flow_cells(
    gray: np.ndarray,
    times: np.ndarray,
    cut_times: np.ndarray,
    grid: tuple[int, int],
    scale: float,
    cut_diff: float,
) -> np.ndarray:
    """Return unclipped per-cell flow for every frame; frame 0 copies frame 1, cuts give 0."""
    count = gray.shape[0]
    out = np.zeros((count, grid[0], grid[1]), dtype=np.float64)
    for index in range(1, count):
        if _is_cut(index, gray, times, cut_times, cut_diff):
            continue
        out[index] = _cell_p90(_flow_magnitude(gray[index - 1], gray[index]), grid) / scale
    out[0] = out[1]
    return out


@register("channel", "motion")
class MotionChannel:
    """Per-cell motion strength from optical flow; HUD cells are forced to zero."""

    name = "motion"
    requires: tuple[str, ...] = ("hud",)

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] motion in 0..1."""
        rows, cols = ctx.grid
        gray = ctx.memo("gray", lambda: to_gray(ctx.frames.frames))
        if gray.shape[0] < 2:
            return np.zeros((gray.shape[0], rows, cols), dtype=np.float32)
        width = ctx.frames.size[0]
        scale = ctx.threshold(
            "motion", "scale_px", DEFAULTS["scale_px"] * width / REFERENCE_WIDTH
        )
        cut_diff = ctx.threshold("motion", "cut_diff", DEFAULTS["cut_diff"])
        cut_times = np.asarray(ctx.cuts, dtype=np.float64)
        flow = _flow_cells(gray, ctx.frames.times, cut_times, ctx.grid, max(scale, 1e-6), cut_diff)
        flow = np.where(hud_mask(ctx)[None, :, :], 0.0, flow)
        return np.clip(flow, 0.0, 1.0).astype(np.float32)
