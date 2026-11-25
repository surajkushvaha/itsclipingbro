"""Pure grid helpers: split frames into rows x cols cells and reduce them per cell.

Imports only numpy and cv2 (for the luma conversion); never clipbro.adapters.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext


def _edges(length: int, parts: int) -> np.ndarray:
    """Return parts + 1 integer edges that split 0..length into near-equal spans."""
    return np.linspace(0, length, parts + 1).round().astype(np.int64)


def cell_slices(shape_hw: tuple[int, int], grid: tuple[int, int]) -> list[list[tuple[slice, slice]]]:
    """Return the (row, column) slices of every cell; the cells tile the frame exactly."""
    height, width = shape_hw
    rows, cols = grid
    row_edges = _edges(height, rows)
    col_edges = _edges(width, cols)
    return [
        [
            (
                slice(int(row_edges[row]), int(row_edges[row + 1])),
                slice(int(col_edges[col]), int(col_edges[col + 1])),
            )
            for col in range(cols)
        ]
        for row in range(rows)
    ]


def cell_reduce(
    img: np.ndarray,
    grid: tuple[int, int],
    fn: Callable[[np.ndarray], float] = np.mean,
) -> np.ndarray:
    """Reduce every cell of a [H, W] image with fn, returning float64 [rows, cols].

    Empty cells (only possible when the frame is smaller than the grid) give 0.
    """
    rows, cols = grid
    out = np.zeros((rows, cols), dtype=np.float64)
    for row, cells in enumerate(cell_slices(img.shape[:2], grid)):
        for col, (ys, xs) in enumerate(cells):
            block = img[ys, xs]
            if block.size:
                out[row, col] = fn(block)
    return out


def cell_mean(stack: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Vectorised per-cell mean over the last two axes: [..., H, W] -> [..., rows, cols].

    Uses the same edges as cell_slices. Raises ValueError when the frame is smaller
    than the grid, because an empty cell has no mean.
    """
    rows, cols = grid
    height, width = stack.shape[-2:]
    if height < rows or width < cols:
        raise ValueError(f"frame {height}x{width} is smaller than grid {rows}x{cols}")
    row_edges = _edges(height, rows)
    col_edges = _edges(width, cols)
    summed = np.add.reduceat(stack, row_edges[:-1], axis=-2, dtype=np.float64)
    summed = np.add.reduceat(summed, col_edges[:-1], axis=-1)
    counts = np.outer(np.diff(row_edges), np.diff(col_edges)).astype(np.float64)
    return summed / counts


def cell_centre(row: int, col: int, grid: tuple[int, int]) -> tuple[float, float]:
    """Return the normalised (x, y) centre of a cell."""
    rows, cols = grid
    return ((col + 0.5) / cols, (row + 0.5) / rows)


def argmax_cell(scores: np.ndarray) -> tuple[int, int]:
    """Return the (row, col) of the largest value in a [rows, cols] array."""
    row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(row), int(col)


def to_gray(frames: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 frames [N, H, W, 3] to luma uint8 [N, H, W]."""
    count, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    gray = np.empty((count, height, width), dtype=np.uint8)
    for index, frame in enumerate(frames):
        gray[index] = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
    return gray


STATS_SCALE = 0.5


def _downscale_rgb(frames: np.ndarray, scale: float) -> np.ndarray:
    """Area-resize RGB uint8 frames [N, H, W, 3] by scale, returning uint8 [N, h, w, 3]."""
    count, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    out = np.empty((count, out_h, out_w, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        out[index] = cv2.resize(np.ascontiguousarray(frame), (out_w, out_h), interpolation=cv2.INTER_AREA)
    return out


def stats_frames(ctx: AnalysisContext) -> np.ndarray:
    """Return RGB uint8 frames at STATS_SCALE for per-cell statistics, computed once per run."""
    return ctx.memo("stats_rgb", lambda: _downscale_rgb(ctx.frames.frames, STATS_SCALE))


def stats_gray(ctx: AnalysisContext) -> np.ndarray:
    """Return luma uint8 frames at STATS_SCALE, computed once per run."""
    return ctx.memo("stats_gray", lambda: to_gray(stats_frames(ctx)))
