"""Dead-air detector: spans where the audio is quiet and the picture is static."""

from __future__ import annotations

import math

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import AudioTrack, Event
from clipbro.motion.audio_features import rms_track
from clipbro.motion.events import frame_peak, locate, runs_of

DEFAULTS: dict[str, float] = {"rms_ratio": 0.25, "rms_floor": 0.01, "motion_max": 0.15, "min_s": 1.5}

AUDIO_HOP_S = 0.04
FULL_STRENGTH_S = 5.0
_EPS = 1e-9


def _interval_rms(track: AudioTrack, hop_rms: np.ndarray, start: float, end: float) -> float:
    """RMS of the audio hops that overlap the interval [start, end); 0 when none do."""
    first = max(0, math.floor((start - track.start) / AUDIO_HOP_S + _EPS))
    last = min(hop_rms.size, math.ceil((end - track.start) / AUDIO_HOP_S - _EPS))
    if last <= first:
        return 0.0
    return float(np.sqrt(np.mean(hop_rms[first:last] ** 2)))


def _frame_rms(ctx: AnalysisContext) -> np.ndarray:
    """RMS of the audio overlapping each sampled frame interval [t, t + 1/fps); silent without audio."""
    times = ctx.frames.times
    if ctx.audio is None:
        return np.zeros(times.shape, dtype=np.float64)
    _, hop_rms = rms_track(ctx.audio, AUDIO_HOP_S)
    frame_s = 1.0 / ctx.frames.fps
    return np.array([_interval_rms(ctx.audio, hop_rms, t, t + frame_s) for t in times], dtype=np.float64)


@register("detector", "dead_air")
class DeadAirDetector:
    """Reports spans of consecutive frames that are both silent and static, at least min_s long."""

    kind = "dead_air"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return one dead_air event per qualifying span, starting at its first frame."""
        frame_rms = _frame_rms(ctx)
        motion = frame_peak(ctx, "motion")
        quiet = self._quiet_frames(ctx, frame_rms, motion)
        times = ctx.frames.times
        frame_s = 1.0 / ctx.frames.fps
        min_s = ctx.threshold("dead_air", "min_s", DEFAULTS["min_s"])
        events: list[Event] = []
        for first, last in runs_of(quiet):
            duration = float(times[last] - times[first]) + frame_s
            if duration < min_s:
                continue
            start = float(times[first])
            x, y = locate(ctx, start)
            events.append(
                Event(
                    t=start,
                    kind="dead_air",
                    strength=min(1.0, duration / FULL_STRENGTH_S),
                    x=x,
                    y=y,
                    reason=f"silent+static {duration:.1f}s",
                )
            )
        return events

    def _quiet_frames(self, ctx: AnalysisContext, frame_rms: np.ndarray, motion: np.ndarray) -> np.ndarray:
        """Bool per frame: RMS below max(rms_ratio * median RMS, rms_floor) and motion below motion_max."""
        rms_ratio = ctx.threshold("dead_air", "rms_ratio", DEFAULTS["rms_ratio"])
        rms_floor = ctx.threshold("dead_air", "rms_floor", DEFAULTS["rms_floor"])
        motion_max = ctx.threshold("dead_air", "motion_max", DEFAULTS["motion_max"])
        silent_level = max(rms_ratio * float(np.median(frame_rms)), rms_floor)
        return (frame_rms < silent_level) & (motion < motion_max)
