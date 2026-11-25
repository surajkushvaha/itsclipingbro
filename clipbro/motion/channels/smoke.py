"""Smoke channel: desaturated, blurred cells that appear shortly after a flash or audio burst.

Scene cuts reset the smoke baseline, so frames before a cut never count as the "normal" look
of frames after it. A cut that is part of a returning flash burst (an explosion, brightness back
to where it was) is not a scene change and is ignored, so boom -> smoke still works.
"""

from __future__ import annotations

import cv2
import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.motion.audio_features import loud_jumps
from clipbro.motion.channels.flash import DEFAULTS as FLASH_DEFAULTS
from clipbro.motion.channels.hud import hud_mask
from clipbro.motion.cuts import cut_frames
from clipbro.motion.events import runs_of
from clipbro.motion.grid import cell_mean, stats_frames, stats_gray, to_gray

DEFAULTS: dict[str, float] = {
    "sat_drop": 0.25,
    "sharp_drop": 0.4,
    "gate_s": 2.0,
    "trigger": 0.5,
    "lum_min": 60.0,
    "lum_max": 190.0,
    "cut_reset_s": 1.0,
    "return_frac": 0.2,
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


def _segment_starts(cuts: np.ndarray) -> np.ndarray:
    """For each frame, the index of the last cut frame at or before it (0 when there is none)."""
    marks = np.where(cuts, np.arange(cuts.size), 0)
    return np.maximum.accumulate(marks)


def _reset_mask(cuts: np.ndarray, span: int) -> np.ndarray:
    """Bool [N] true for every frame c .. c + span (inclusive) where c is a cut frame."""
    mask = np.zeros(cuts.shape, dtype=bool)
    for index in np.flatnonzero(cuts):
        mask[index : index + span + 1] = True
    return mask


def _baseline(values: np.ndarray, window: int, starts: np.ndarray) -> np.ndarray:
    """Mean of the window frames before each frame, never reaching back before its segment start.

    A frame with no usable history (the first frame of a segment) gets 0.
    """
    count = values.shape[0]
    cumulative = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)])
    base = np.zeros(values.shape, dtype=np.float64)
    for index in range(count):
        start = max(int(starts[index]), index - window)
        size = index - start
        if size > 0:
            base[index] = (cumulative[index] - cumulative[start]) / size
    return base


def _smoke_scores(
    saturation: np.ndarray,
    sharpness: np.ndarray,
    luminance: np.ndarray,
    window: int,
    starts: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Per-cell smoke score before gating: min of the saturation and sharpness drops."""
    sat_base = _baseline(saturation, window, starts)
    sharp_base = _baseline(sharpness, window, starts)
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


def _luminance(ctx: AnalysisContext) -> np.ndarray:
    """Full-frame mean luminance per sampled frame, float64 [N], from the shared full-res gray."""
    gray = ctx.memo("gray", lambda: to_gray(ctx.frames.frames))
    return gray.reshape(gray.shape[0], -1).mean(axis=1, dtype=np.float64)


def _returning_bursts(
    flash_max: np.ndarray,
    levels: np.ndarray,
    trigger: float,
    horizon: int,
    return_frac: float,
) -> np.ndarray:
    """Bool [N] marking frames of flash bursts whose brightness returns to the pre-burst level.

    A run of flash frames (flash_max >= trigger, first..last) is a returning burst when the level
    `horizon` frames after its last frame is within return_frac of the rise above the level just
    before it. Frames first-1 .. last+horizon are marked; a scene cut there is an explosion, not
    a scene change. A cut to a new scene at another brightness is not marked.
    """
    count = flash_max.shape[0]
    mask = np.zeros(count, dtype=bool)
    for first, last in runs_of(flash_max >= trigger):
        if first < 1:
            continue
        before = float(levels[first - 1])
        rise = float(levels[first : last + 1].max()) - before
        after = min(last + horizon, count - 1)
        if rise > 0 and abs(float(levels[after]) - before) <= return_frac * rise:
            mask[first - 1 : after + 1] = True
    return mask


def _effective_cuts(ctx: AnalysisContext, flash_max: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Shared scene cuts minus the cuts that belong to a returning flash burst."""
    horizon = int(round(ctx.threshold("flash", "return_frames", FLASH_DEFAULTS["return_frames"])))
    returning = _returning_bursts(
        flash_max,
        _luminance(ctx),
        thresholds["trigger"],
        horizon,
        thresholds["return_frac"],
    )
    return cut_frames(ctx) & ~returning


def _non_hud_flash_max(ctx: AnalysisContext) -> np.ndarray:
    """Per-frame maximum of the flash channel over the non-HUD cells, float64 [N]."""
    count, rows, cols = ctx.frames.n, *ctx.grid
    flash = ctx.channels.get("flash", np.zeros((count, rows, cols), dtype=np.float32))
    non_hud = ~hud_mask(ctx)
    return np.where(non_hud[None, :, :], flash, 0.0).reshape(count, -1).max(axis=1).astype(np.float64)


@register("channel", "smoke")
class SmokeChannel:
    """Smoke-like desaturation and blur per cell, only inside a flash or audio gate and not near a cut."""

    name = "smoke"
    requires: tuple[str, ...] = ("flash",)

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] smoke strength in 0..1."""
        rows, cols = ctx.grid
        count = ctx.frames.n
        if count < 2:
            return np.zeros((count, rows, cols), dtype=np.float32)
        thresholds = {key: ctx.threshold("smoke", key, value) for key, value in DEFAULTS.items()}
        window = max(1, int(round(ctx.frames.fps)))
        reset_span = window + int(round(thresholds["cut_reset_s"] * ctx.frames.fps))
        gray = stats_gray(ctx)
        flash_max = _non_hud_flash_max(ctx)
        cuts = _effective_cuts(ctx, flash_max, thresholds)
        saturation, sharpness, luminance = _cell_stats(stats_frames(ctx), gray, ctx.grid)
        scores = _smoke_scores(saturation, sharpness, luminance, window, _segment_starts(cuts), thresholds)
        scores = np.where(_reset_mask(cuts, reset_span)[:, None, None], 0.0, scores)
        flash_max = np.where(cuts, 0.0, flash_max)
        gate = _gate_open(
            ctx.frames.times,
            flash_max,
            _audio_jump_times(ctx),
            thresholds["gate_s"],
            thresholds["trigger"],
        )
        scores = np.where(gate[:, None, None], scores, 0.0)
        return np.clip(scores, 0.0, 1.0).astype(np.float32)
