"""HUD channel: static overlays such as watermarks, logos and panels, found per pixel.

A pixel is overlay-like when it is static (low temporal std) and carries a persistent Canny
edge (an edge on most frames). A cell is HUD when enough of its pixels are overlay-like, so a
small logo over moving footage still marks its own cell.
"""

from __future__ import annotations

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.grid import cell_mean, to_gray

DEFAULTS: dict[str, float] = {"std_max": 6.0, "edge_persist": 0.6, "cover": 0.04, "min_frames": 6}

CANNY_LOW = 80
CANNY_HIGH = 160


def _sampled_gray(ctx: AnalysisContext) -> np.ndarray:
    """Luma uint8 [N, H, W] of the sampled frames (not the half-res stats), computed once per run."""
    return ctx.memo("gray", lambda: to_gray(ctx.frames.frames))


def _pixel_stats(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel temporal std (float64 [H, W]) and Canny edge hit count (int32 [H, W]).

    Streams over the frames with running sums, so no [N, H, W] float array is ever built.
    """
    count = gray.shape[0]
    total = np.zeros(gray.shape[1:], dtype=np.float64)
    squares = np.zeros(gray.shape[1:], dtype=np.float64)
    hits = np.zeros(gray.shape[1:], dtype=np.int32)
    for frame in gray:
        values = frame.astype(np.float64)
        total += values
        squares += values * values
        hits += cv2.Canny(frame, CANNY_LOW, CANNY_HIGH) > 0
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
    return std, hits


def _overlay_pixels(gray: np.ndarray, std_max: float, edge_persist: float) -> np.ndarray:
    """Bool [H, W]: static pixels that lie within one pixel of a persistent Canny edge.

    The one-pixel dilation is needed because the edge of a thin glyph over moving footage is
    drawn on the moving side of its outline, while the glyph's own (static) pixels sit inside it.
    """
    std, hits = _pixel_stats(gray)
    persistent = (hits / gray.shape[0] >= edge_persist).astype(np.uint8)
    persistent = cv2.dilate(persistent, np.ones((3, 3), dtype=np.uint8)) > 0
    return (std < std_max) & persistent


def _compute_mask(ctx: AnalysisContext) -> np.ndarray:
    """Build the bool [rows, cols] HUD mask; all False when there are too few frames."""
    gray = _sampled_gray(ctx)
    rows, cols = ctx.grid
    min_frames = ctx.threshold("hud", "min_frames", DEFAULTS["min_frames"])
    if gray.shape[0] < max(2.0, min_frames):
        return np.zeros((rows, cols), dtype=bool)
    std_max = ctx.threshold("hud", "std_max", DEFAULTS["std_max"])
    edge_persist = ctx.threshold("hud", "edge_persist", DEFAULTS["edge_persist"])
    cover = ctx.threshold("hud", "cover", DEFAULTS["cover"])
    overlay = _overlay_pixels(gray, std_max, edge_persist)
    fraction = cell_mean(overlay.astype(np.float32), ctx.grid)
    return fraction >= cover


def hud_mask(ctx: AnalysisContext) -> np.ndarray:
    """Return the shared bool [rows, cols] HUD mask, computed once per run."""
    return ctx.memo("hud_mask", lambda: _compute_mask(ctx))


@register("channel", "hud")
class HudChannel:
    """Marks overlay cells and broadcasts the 0/1 mask to every frame."""

    name = "hud"
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] with 1.0 on HUD cells and 0.0 elsewhere."""
        mask = hud_mask(ctx)
        return np.broadcast_to(mask.astype(np.float32), (ctx.frames.n, *mask.shape)).astype(np.float32)
