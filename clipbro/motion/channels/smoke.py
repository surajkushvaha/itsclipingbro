"""Smoke channel: desaturated, blurred cells that appear shortly after a flash or audio burst."""

from __future__ import annotations

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.audio_features import loud_jumps
from clipbro.motion.channels.hud import hud_mask
from clipbro.motion.grid import cell_mean, to_gray

DEFAULTS: dict[str, float] = {
    "sat_drop": 0.25,
    "sharp_drop": 0.4,
    "gate_s": 2.0,
    "trigger": 0.5,
    "lum_min": 60.0,
    "lum_max": 190.0,
}


def _cell_stats(
    frames: np.ndarray, gray: np.ndarray, grid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cell saturation (0..255), Laplacian variance and mean luminance per frame."""
    sat = np.stack([cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2HSV)[..., 1] for f in frames])
    laplace = np.stack([cv2.Laplacian(g, cv2.CV_32F) for g in gray])
    mean_lap = cell_mean(laplace, grid)
    sharpness = np.maximum(cell_mean(laplace * laplace, grid) - mean_lap * mean_lap, 0.0)
    saturation = cell_mean(sat.astype(np.float32), grid)
    luminance = cell_mean(gray.astype(np.float32), grid)
    return saturation, sharpness, luminance


def _baseline(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the window frames before each frame; frame 0 gets 0 (no history)."""
    count = values.shape[0]
    cumulative = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)])
    base = np.zeros(values.shape, dtype=np.float64)
    for index in range(1, count):
        start = max(0, index - window)
        base[index] = (cumulative[index] - cumulative[start]) / (index - start)
    return base


def _smoke_scores(
    saturation: np.ndarray,
    sharpness: np.ndarray,
    luminance: np.ndarray,
    window: int,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Per-cell smoke score before gating: min of the saturation and sharpness drops."""
    sat_base = _baseline(saturation, window)
    sharp_base = _baseline(sharpness, window)
    sat_drop = (sat_base - saturation) / np.maximum(sat_base, 1.0)
    sharp_drop = (sharp_base - sharpness) / np.maximum(sharp_base, 1.0)
    score = np.minimum(sat_drop / thresholds["sat_drop"], sharp_drop / thresholds["sharp_drop"])
    in_range = (luminance >= thresholds["lum_min"]) & (luminance <= thresholds["lum_max"])
    score = np.where(in_range, score, 0.0)
    score[0] = 0.0
    return np.clip(score, 0.0, 1.0)


def _gate_open(
    times: np.ndarray,
    flash_max: np.ndarray,
    audio_times: np.ndarray,
    gate_s: float,
    trigger: float,
) -> np.ndarray:
    """True per frame when a flash >= trigger or an audio jump lies in the previous gate_s seconds."""
    events = np.sort(np.concatenate([times[flash_max >= trigger], audio_times]))
    if events.size == 0:
        return np.zeros(times.shape, dtype=bool)
    latest = np.searchsorted(events, times, side="right") - 1
    last_event = events[np.clip(latest, 0, None)]
    return (latest >= 0) & (last_event >= times - gate_s)


def _audio_jump_times(ctx: AnalysisContext) -> np.ndarray:
    """Absolute times of loud audio jumps, empty when there is no audio."""
    return np.asarray([moment for moment, _ in loud_jumps(ctx.audio)], dtype=np.float64)


@register("channel", "smoke")
class SmokeChannel:
    """Smoke-like desaturation and blur per cell, only inside a flash or audio gate."""

    name = "smoke"
    requires: tuple[str, ...] = ("flash",)

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] smoke strength in 0..1."""
        rows, cols = ctx.grid
        count = ctx.frames.n
        if count < 2:
            return np.zeros((count, rows, cols), dtype=np.float32)
        gray = ctx.memo("gray", lambda: to_gray(ctx.frames.frames))
        thresholds = {key: ctx.threshold("smoke", key, value) for key, value in DEFAULTS.items()}
        window = max(1, int(round(ctx.frames.fps)))
        saturation, sharpness, luminance = _cell_stats(ctx.frames.frames, gray, ctx.grid)
        scores = _smoke_scores(saturation, sharpness, luminance, window, thresholds)
        flash = ctx.channels.get("flash", np.zeros((count, rows, cols), dtype=np.float32))
        non_hud = ~hud_mask(ctx)
        flash_max = np.where(non_hud[None, :, :], flash, 0.0).reshape(count, -1).max(axis=1)
        gate = _gate_open(
            ctx.frames.times,
            flash_max,
            _audio_jump_times(ctx),
            thresholds["gate_s"],
            thresholds["trigger"],
        )
        scores = np.where(gate[:, None, None], scores, 0.0)
        return np.clip(scores, 0.0, 1.0).astype(np.float32)
