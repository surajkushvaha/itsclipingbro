"""Centre channel: a constant Gaussian prior that favours cells near the frame centre."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.grid import cell_centre

DEFAULTS: dict[str, float] = {"sigma": 0.35}


def _centre_weights(rows: int, cols: int, sigma: float, grid: tuple[int, int]) -> np.ndarray:
    """Return the [rows, cols] Gaussian weight of each cell centre's distance from (0.5, 0.5)."""
    weights = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            x, y = cell_centre(row, col, grid)
            dist_sq = (x - 0.5) ** 2 + (y - 0.5) ** 2
            weights[row, col] = np.exp(-dist_sq / (2.0 * sigma * sigma))
    return weights


@register("channel", "centre")
class CentreChannel:
    """Frame-centre prior, identical for every frame."""

    name = "centre"
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] Gaussian weights in 0..1."""
        rows, cols = ctx.grid
        sigma = max(ctx.threshold("centre", "sigma", DEFAULTS["sigma"]), 1e-6)
        weights = _centre_weights(rows, cols, sigma, ctx.grid)
        return np.broadcast_to(weights.astype(np.float32), (ctx.frames.n, rows, cols)).astype(np.float32)
