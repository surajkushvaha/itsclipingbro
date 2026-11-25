"""Event helpers: locate salient cells, run detectors, merge near-duplicate events.

Shared helpers for detectors (HUD-free channel views, run detection, peak cells) also live
here so that every detector reads the saliency map the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import REGISTRY, Registry, discover
from clipbro.core.types import Event, SaliencyMap
from clipbro.motion.grid import argmax_cell, cell_centre

DETECTOR_PACKAGE = "clipbro.motion.detectors"
CENTRE = (0.5, 0.5)


def _require_saliency(ctx: AnalysisContext) -> SaliencyMap:
    """Return ctx.saliency, or raise ValueError when the saliency stage has not run."""
    if ctx.saliency is None:
        raise ValueError("ctx.saliency is not set; run saliency.run_channels first")
    return ctx.saliency


def peak_location(cells: np.ndarray, grid: tuple[int, int]) -> tuple[float, float]:
    """Normalised (x, y) centre of the largest cell of a [rows, cols] array.

    An array with no positive cell has no salient position, so the frame centre is returned.
    """
    if not np.any(cells > 0):
        return CENTRE
    row, col = argmax_cell(cells)
    return cell_centre(row, col, grid)


def locate(ctx: AnalysisContext, t: float) -> tuple[float, float]:
    """Normalised (x, y) of the argmax cell of the combined saliency at the frame nearest t."""
    saliency = _require_saliency(ctx)
    index = saliency.nearest(t)
    return peak_location(saliency.combined[index], saliency.grid)


def locate_in(ctx: AnalysisContext, channel_name: str, t: float) -> tuple[float, float]:
    """Like locate, but on one channel; falls back to locate when that channel is zero at the frame."""
    saliency = _require_saliency(ctx)
    index = saliency.nearest(t)
    cells = saliency.channel(channel_name)[index]
    if not np.any(cells > 0):
        return locate(ctx, t)
    return peak_location(cells, saliency.grid)


def hud_mask(ctx: AnalysisContext) -> np.ndarray:
    """Bool [rows, cols] mask of the static HUD cells of the analysed range."""
    return _require_saliency(ctx).hud


def hud_free(ctx: AnalysisContext, channel_name: str) -> np.ndarray:
    """A channel as float32 [N, rows, cols] with HUD cells set to zero (computed once per run)."""
    saliency = _require_saliency(ctx)

    def compute() -> np.ndarray:
        """Zero the HUD cells of every frame of the channel."""
        cells = saliency.channel(channel_name)
        return np.where(saliency.hud[None, :, :], 0.0, cells).astype(np.float32)

    return ctx.memo(f"hud_free:{channel_name}", compute)


def frame_peak(ctx: AnalysisContext, channel_name: str) -> np.ndarray:
    """Per-frame maximum over the non-HUD cells of a channel: float32 [N]."""
    cells = hud_free(ctx, channel_name)
    return cells.reshape(cells.shape[0], -1).max(axis=1)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) inclusive indices of each run of consecutive True values in a 1-D mask."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(end) - 1) for start, end in zip(edges[::2], edges[1::2])]


def run_detectors(ctx: AnalysisContext, registry: Registry = REGISTRY) -> list[Event]:
    """Run the configured detectors in config order, dedupe their events and sort them by time.

    ctx.saliency must already be set by saliency.run_channels.
    """
    _require_saliency(ctx)
    discover(DETECTOR_PACKAGE)
    found: list[Event] = []
    for name in ctx.config.detectors:
        detector = registry.create("detector", name)
        found.extend(detector.detect(ctx))
    merged = dedupe(found, ctx.config.dedupe_window)
    return sorted(merged, key=lambda event: event.t)


def _clusters(ordered: list[Event], window: float) -> list[list[Event]]:
    """Split time-ordered events of one kind: an event joins the cluster if it is within window of the last one."""
    clusters: list[list[Event]] = []
    for event in ordered:
        if clusters and event.t - clusters[-1][-1].t <= window:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def dedupe(events: Sequence[Event], window: float) -> list[Event]:
    """Merge events of the same kind that fall within window seconds; keep the strongest of each cluster.

    Events of different kinds never merge. The result is sorted by time.
    """
    kept: list[Event] = []
    for kind in dict.fromkeys(event.kind for event in events):
        ordered = sorted((event for event in events if event.kind == kind), key=lambda event: event.t)
        for cluster in _clusters(ordered, window):
            kept.append(max(cluster, key=lambda event: event.strength))
    return sorted(kept, key=lambda event: event.t)
