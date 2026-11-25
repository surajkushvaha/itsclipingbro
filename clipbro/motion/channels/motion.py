"""Motion channel: Farneback optical-flow magnitude per cell, with cut and HUD suppression."""

from __future__ import annotations

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.channels.hud import hud_mask
from clipbro.motion.grid import cell_slices, to_gray

DEFAULTS: dict[str, float] = {"scale_px": 2.0, "cut_diff": 38.0, "flow_scale": 0.5}

REFERENCE_WIDTH = 160


def _flow_magnitude(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return the per-pixel Farneback flow magnitude, defined on the CURRENT frame's pixels.

    Flow is computed backwards (current -> previous) so a moving object lights up where it
    is now, not where it was. Forward flow would be defined on the previous frame's pixels.
    """
    flow = cv2.calcOpticalFlowFarneback(current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.hypot(flow[..., 0], flow[..., 1])


def _p90_plan(shape_hw: tuple[int, int], grid: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return flat pixel indices grouped cell by cell, and the (start, stop) span of each cell."""
    height, width = shape_hw
    index_grid = np.arange(height * width).reshape(height, width)
    parts: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    start = 0
    for row in cell_slices(shape_hw, grid):
        for ys, xs in row:
            block = index_grid[ys, xs].ravel()
            parts.append(block)
            spans.append((start, start + block.size))
            start += block.size
    return np.concatenate(parts), spans


def _segment_p90(values: np.ndarray) -> float:
    """90th percentile with linear interpolation, matching np.percentile, via partition."""
    count = values.size
    position = 0.9 * (count - 1)
    low = int(position)
    high = min(low + 1, count - 1)
    part = np.partition(values, (low, high))
    return float(part[low] + (position - low) * (part[high] - part[low]))


def _cell_p90(magnitude: np.ndarray, plan: tuple[np.ndarray, list[tuple[int, int]]], grid: tuple[int, int]) -> np.ndarray:
    """Per-cell 90th percentile of a magnitude map, using a precomputed cell-major pixel plan."""
    order, spans = plan
    values = magnitude.ravel()[order]
    out = np.array([_segment_p90(values[start:stop]) for start, stop in spans], dtype=np.float64)
    return out.reshape(grid)


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


def _downscale(gray: np.ndarray, flow_scale: float) -> np.ndarray:
    """Resize every gray frame by flow_scale with area averaging, for cheaper optical flow."""
    return np.stack(
        [cv2.resize(frame, None, fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA) for frame in gray]
    )


def _flow_cells(
    gray: np.ndarray,
    small: np.ndarray,
    times: np.ndarray,
    cut_times: np.ndarray,
    grid: tuple[int, int],
    scale: float,
    cut_diff: float,
    flow_scale: float,
) -> np.ndarray:
    """Return unclipped per-cell flow for every frame; frame 0 copies frame 1, cuts give 0.

    Flow is computed on the downscaled frames `small`; magnitudes are divided by flow_scale
    so they stay in sampled-frame pixels. Cut detection uses the full-resolution `gray`.
    """
    count = gray.shape[0]
    plan = _p90_plan(small.shape[1:3], grid)
    out = np.zeros((count, grid[0], grid[1]), dtype=np.float64)
    for index in range(1, count):
        if _is_cut(index, gray, times, cut_times, cut_diff):
            continue
        magnitude = _flow_magnitude(small[index], small[index - 1]) / flow_scale
        out[index] = _cell_p90(magnitude, plan, grid) / scale
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
        flow_scale = min(max(ctx.threshold("motion", "flow_scale", DEFAULTS["flow_scale"]), 1e-3), 1.0)
        small = _downscale(gray, flow_scale)
        flow = _flow_cells(
            gray, small, ctx.frames.times, cut_times, ctx.grid, max(scale, 1e-6), cut_diff, flow_scale
        )
        flow = np.where(hud_mask(ctx)[None, :, :], 0.0, flow)
        return np.clip(flow, 0.0, 1.0).astype(np.float32)
