"""Face channel: runs an injected face detector every few frames and rasterises the boxes per cell."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Box

DEFAULTS: dict[str, float] = {}


def _overlap(lo: float, hi: float, edges: np.ndarray) -> np.ndarray:
    """Length of the interval [lo, hi] that falls inside each span between consecutive edges."""
    return np.clip(np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]), 0.0, None)


def _box_coverage(box: Box, rows: int, cols: int) -> np.ndarray:
    """Per-cell fraction of the cell area covered by a normalised box, times its score."""
    x_edges = np.linspace(0.0, 1.0, cols + 1)
    y_edges = np.linspace(0.0, 1.0, rows + 1)
    covered_x = _overlap(box.x, box.x + box.w, x_edges) * cols
    covered_y = _overlap(box.y, box.y + box.h, y_edges) * rows
    return np.outer(covered_y, covered_x) * box.score


def _detect_cells(ctx: AnalysisContext, index: int, rows: int, cols: int) -> np.ndarray:
    """Run the detector on one frame and sum the box coverage per cell, clipped to 1."""
    cells = np.zeros((rows, cols), dtype=np.float64)
    for box in ctx.faces.detect(ctx.frames.frames[index]):
        cells += _box_coverage(box, rows, cols)
    return np.clip(cells, 0.0, 1.0)


@register("channel", "face")
class FaceChannel:
    """Face presence per cell; detector runs every config.face_every frames and is held between."""

    name = "face"
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] face coverage in 0..1 (zeros without a detector)."""
        rows, cols = ctx.grid
        count = ctx.frames.n
        out = np.zeros((count, rows, cols), dtype=np.float32)
        if ctx.faces is None or count == 0:
            return out
        every = max(1, int(ctx.config.face_every))
        held = np.zeros((rows, cols), dtype=np.float64)
        for index in range(count):
            if index % every == 0:
                held = _detect_cells(ctx, index, rows, cols)
            out[index] = held
        return out
