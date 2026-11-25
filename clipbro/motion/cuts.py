"""Shared scene-cut detection: one bool per sampled frame, computed once per run.

A frame is a cut when its full-resolution mean absolute difference to the previous frame exceeds
`cut_diff`, or when a cut time from ctx.cuts lies in (t[i-1], t[i]].
"""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.motion.grid import to_gray

DEFAULTS: dict[str, float] = {"cut_diff": 30.0}


def _mean_absdiff(gray: np.ndarray) -> np.ndarray:
    """Mean absolute difference of each frame to the one before it, float64 [N]; frame 0 gives 0."""
    count = gray.shape[0]
    diffs = np.zeros(count, dtype=np.float64)
    for index in range(1, count):
        change = np.abs(gray[index].astype(np.int16) - gray[index - 1].astype(np.int16))
        diffs[index] = float(change.mean())
    return diffs


def _time_cut_frames(times: np.ndarray, cut_times: list[float]) -> np.ndarray:
    """Bool [N] marking the frame i for each cut time in (times[i-1], times[i]]; frame 0 is never marked."""
    marks = np.zeros(times.shape, dtype=bool)
    for moment in cut_times:
        index = int(np.searchsorted(times, moment, side="left"))
        if 1 <= index < times.size:
            marks[index] = True
    return marks


def _compute_cuts(ctx: AnalysisContext) -> np.ndarray:
    """Build the bool [N] cut mask from full-resolution gray differences and ctx.cuts."""
    gray = ctx.memo("gray", lambda: to_gray(ctx.frames.frames))
    threshold = ctx.threshold("cuts", "cut_diff", DEFAULTS["cut_diff"])
    cuts = _mean_absdiff(gray) > threshold
    cuts |= _time_cut_frames(ctx.frames.times, list(ctx.cuts))
    if cuts.size:
        cuts[0] = False
    return cuts


def cut_frames(ctx: AnalysisContext) -> np.ndarray:
    """Return the shared bool [N] scene-cut mask of the sampled frames (memoised as "cut_frames")."""
    return ctx.memo("cut_frames", lambda: _compute_cuts(ctx))
