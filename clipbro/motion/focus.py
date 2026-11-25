"""Raw focus path derived from a saliency map.

Per frame, the top-k saliency cells are reduced to one weighted centroid. The
optional smoother hook in `focus_path` is the Phase 2 extension point.
"""

from __future__ import annotations

import numpy as np

from clipbro.core.ports import FocusSmoother
from clipbro.core.registry import register
from clipbro.core.types import FocusPoint, SaliencyMap

DEFAULT_TOP_K = 3
_CENTRE = (0.5, 0.5)


def _cell_centre(row: np.ndarray, col: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised (x, y) centres of grid cells given their row and column indices."""
    return (col + 0.5) / cols, (row + 0.5) / rows


def _frame_centroid(scores: np.ndarray, top_k: int) -> tuple[float, float, float] | None:
    """Return (cx, cy, total_weight) for one frame, or None when no cell has a positive score."""
    flat = scores.ravel()
    k = min(top_k, flat.size)
    candidates = np.argpartition(-flat, k - 1)[:k]
    chosen = candidates[flat[candidates] > 0]
    if chosen.size == 0:
        return None
    rows, cols = scores.shape
    weights = flat[chosen].astype(np.float64)
    xs, ys = _cell_centre(chosen // cols, chosen % cols, rows, cols)
    cx = float(np.average(xs, weights=weights))
    cy = float(np.average(ys, weights=weights))
    return cx, cy, float(weights.sum())


def raw_focus(sal: SaliencyMap, top_k: int = DEFAULT_TOP_K) -> list[FocusPoint]:
    """Return one focus point per analysed frame from the top_k saliency cells.

    Frames with no positive saliency repeat the previous point with weight 0.
    The first such frame falls back to the frame centre (0.5, 0.5).
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    path: list[FocusPoint] = []
    previous = _CENTRE
    for index, t in enumerate(sal.times):
        found = _frame_centroid(sal.combined[index], top_k)
        if found is None:
            cx, cy, weight = previous[0], previous[1], 0.0
        else:
            cx, cy, weight = found
        path.append(FocusPoint(t=float(t), cx=cx, cy=cy, weight=weight))
        previous = (cx, cy)
    return path


@register("smoother", "identity")
class IdentitySmoother:
    """Smoother that returns the path unchanged."""

    def smooth(self, path: list[FocusPoint]) -> list[FocusPoint]:
        """Return a shallow copy of the path."""
        return list(path)


def focus_path(
    sal: SaliencyMap,
    smoother: FocusSmoother | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[FocusPoint]:
    """Return the focus path for a saliency map, smoothed when a smoother is given.

    This is the Phase 2 extension point: a real smoother plugs in through the
    `smoother` argument without changing the raw centroid maths.
    """
    path = raw_focus(sal, top_k=top_k)
    if smoother is None:
        return path
    return smoother.smooth(path)
