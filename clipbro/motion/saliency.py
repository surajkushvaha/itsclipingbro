"""Saliency runner: computes every configured channel and combines them into one map.

Imports only clipbro.core, numpy and the channel package (auto-discovered).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import REGISTRY, Registry, discover
from clipbro.core.types import SaliencyMap

CHANNEL_PACKAGE = "clipbro.motion.channels"
HUD_NAME = "hud"
HUD_THRESHOLD = 0.5


def _validated(name: str, raw: np.ndarray, expected: tuple[int, int, int]) -> np.ndarray:
    """Check a channel's shape against [N, rows, cols] and return it as clipped float32."""
    array = np.asarray(raw)
    if array.shape != expected:
        raise ValueError(f"channel {name!r} returned shape {array.shape}, expected {expected}")
    cleaned = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(cleaned, 0.0, 1.0).astype(np.float32)


def _max_of_parts(channels: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    """Element-wise max of the channels named by an 'a|b' key; None when none of them is present."""
    parts = [part.strip() for part in key.split("|") if part.strip()]
    present = [channels[part] for part in parts if part in channels]
    if not present:
        return None
    term = np.zeros(shape, dtype=np.float64)
    for array in present:
        if array.shape != shape:
            raise ValueError(f"channel in {key!r} has shape {array.shape}, expected {shape}")
        np.maximum(term, array, out=term)
    return term


def combine(
    channels: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Weighted sum of channel terms, clipped to 0..1 and returned as float32 of the given shape.

    A weight key is a channel name, or 'a|b|...' for the element-wise max of several channels.
    Missing channels count as zeros, so a key with no present channel contributes nothing.
    """
    total = np.zeros(shape, dtype=np.float64)
    for key, weight in weights.items():
        term = _max_of_parts(channels, key, shape)
        if term is not None:
            total += float(weight) * term
    return np.clip(total, 0.0, 1.0).astype(np.float32)


def _hud_mask(channels: Mapping[str, np.ndarray], grid: tuple[int, int]) -> np.ndarray:
    """Bool [rows, cols] HUD mask from the first frame of the hud channel; all False when absent."""
    hud = channels.get(HUD_NAME)
    if hud is None or hud.shape[0] == 0:
        return np.zeros(grid, dtype=bool)
    return hud[0] > HUD_THRESHOLD


def run_channels(ctx: AnalysisContext, registry: Registry = REGISTRY) -> SaliencyMap:
    """Compute the configured channels in dependency order, combine them and set ctx.saliency.

    Results are stored in ctx.channels (replacing any from an earlier run on this context).
    """
    discover(CHANNEL_PACKAGE)
    order = registry.ordered("channel", ctx.config.channels)
    rows, cols = ctx.grid
    expected = (ctx.frames.n, rows, cols)
    ctx.channels.clear()
    for name in order:
        channel = registry.create("channel", name)
        ctx.channels[name] = _validated(name, channel.compute(ctx), expected)
    saliency = SaliencyMap(
        times=np.asarray(ctx.frames.times, dtype=np.float64),
        grid=ctx.grid,
        channels=dict(ctx.channels),
        combined=combine(ctx.channels, ctx.config.weights, expected),
        hud=_hud_mask(ctx.channels, ctx.grid),
    )
    ctx.saliency = saliency
    return saliency
