"""Scream detector: a sustained, loud 2-4 kHz band spike while a face is on screen."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.audio_features import band_ratio, rms_track
from clipbro.motion.events import locate_in, runs_of

DEFAULTS: dict[str, float] = {"k": 6.0, "face_min": 0.2, "loud_ratio": 2.0, "min_hops": 5.0}

BAND_LO_HZ = 2000.0
BAND_HI_HZ = 4000.0
MAD_FLOOR = 1e-3
MAD_TO_SIGMA = 1.4826
_EPS = 1e-12


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score over the whole window: (value - median) / (1.4826 * MAD + MAD_FLOOR)."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / (MAD_TO_SIGMA * mad + MAD_FLOOR)


def _qualifying(z: np.ndarray, rms: np.ndarray, k: float, loud_ratio: float) -> np.ndarray:
    """Hops whose band z-score reaches k and whose RMS is at least loud_ratio x the median RMS."""
    loud = rms >= loud_ratio * max(float(np.median(rms)), _EPS)
    return (z >= k) & loud


def _strength(z_peak: float, rms_ratio: float, k: float) -> float:
    """Mean of the band over-shoot term and the loudness term, plus 0.25, capped at 1."""
    overshoot = float(np.clip((z_peak - k) / max(k, _EPS), 0.0, 1.0))
    louder = float(np.clip(rms_ratio / 4.0, 0.0, 1.0))
    return min(1.0, 0.5 * (overshoot + louder) + 0.25)


def _face_level(ctx: AnalysisContext, t: float) -> float:
    """Largest face coverage at the frame nearest t (the face channel is not HUD-masked)."""
    saliency = ctx.saliency
    return float(saliency.channel("face")[saliency.nearest(t)].max())


@register("detector", "scream")
class ScreamDetector:
    """Reports one scream per sustained loud 2-4 kHz run, at its peak hop, located on the face."""

    kind = "scream"
    MIN_STRENGTH = 0.35

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return one scream per qualifying run (>= min_hops hops) inside ctx.rng with a face on screen."""
        if ctx.audio is None:
            return []
        k = ctx.threshold("scream", "k", DEFAULTS["k"])
        face_min = ctx.threshold("scream", "face_min", DEFAULTS["face_min"])
        loud_ratio = ctx.threshold("scream", "loud_ratio", DEFAULTS["loud_ratio"])
        min_hops = ctx.threshold("scream", "min_hops", DEFAULTS["min_hops"])
        times, ratio = band_ratio(ctx.audio, BAND_LO_HZ, BAND_HI_HZ)
        _, rms = rms_track(ctx.audio)
        z = robust_z(ratio)
        mask = _qualifying(z, rms, k, loud_ratio)
        median_rms = max(float(np.median(rms)), _EPS)
        events: list[Event] = []
        for first, last in runs_of(mask):
            if last - first + 1 < min_hops:
                continue
            peak = first + int(np.argmax(z[first : last + 1]))
            moment = float(times[peak])
            if not ctx.rng.start <= moment <= ctx.rng.end:
                continue
            face = _face_level(ctx, moment)
            if face < face_min:
                continue
            rms_ratio = float(rms[peak]) / median_rms
            x, y = locate_in(ctx, "face", moment)
            events.append(
                Event(
                    t=moment,
                    kind="scream",
                    strength=_strength(float(z[peak]), rms_ratio, k),
                    x=x,
                    y=y,
                    reason=f"2-4 kHz z {z[peak]:.1f} loud x{rms_ratio:.1f} + face {face:.2f}",
                )
            )
        return events
