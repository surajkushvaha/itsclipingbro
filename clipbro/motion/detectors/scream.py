"""Scream detector: a 2-4 kHz energy spike in loud audio while a face is on screen."""

from __future__ import annotations

import numpy as np

from clipbro.core.context import AnalysisContext
from clipbro.core.registry import register
from clipbro.core.types import Event
from clipbro.motion.audio_features import band_ratio, rms_track
from clipbro.motion.events import locate_in, runs_of

DEFAULTS: dict[str, float] = {"k": 6.0, "face_min": 0.2}

BAND_LO_HZ = 2000.0
BAND_HI_HZ = 4000.0
MAD_FLOOR = 1e-3


def _spike_mask(ratio: np.ndarray, rms: np.ndarray, k: float) -> np.ndarray:
    """Hops whose band ratio exceeds median + k * MAD, while their RMS is above the median RMS."""
    median = float(np.median(ratio))
    spread = float(np.median(np.abs(ratio - median)))
    threshold = median + k * max(spread, MAD_FLOOR)
    return (ratio > threshold) & (rms > float(np.median(rms)))


def _face_level(ctx: AnalysisContext, t: float) -> float:
    """Largest face coverage at the frame nearest t (the face channel is not HUD-masked)."""
    saliency = ctx.saliency
    return float(saliency.channel("face")[saliency.nearest(t)].max())


@register("detector", "scream")
class ScreamDetector:
    """Reports 2-4 kHz spikes in loud hops where a face is visible, located on the face."""

    kind = "scream"

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return one scream per run of spike hops inside ctx.rng that has a face on screen."""
        if ctx.audio is None:
            return []
        k = ctx.threshold("scream", "k", DEFAULTS["k"])
        face_min = ctx.threshold("scream", "face_min", DEFAULTS["face_min"])
        times, ratio = band_ratio(ctx.audio, BAND_LO_HZ, BAND_HI_HZ)
        _, rms = rms_track(ctx.audio)
        mask = _spike_mask(ratio, rms, k)
        events: list[Event] = []
        for first, last in runs_of(mask):
            start = float(times[first])
            if not ctx.rng.start <= start <= ctx.rng.end:
                continue
            face = _face_level(ctx, start)
            if face < face_min:
                continue
            x, y = locate_in(ctx, "face", start)
            peak = float(np.max(ratio[first : last + 1]))
            events.append(
                Event(
                    t=start,
                    kind="scream",
                    strength=min(1.0, peak),
                    x=x,
                    y=y,
                    reason=f"2-4 kHz {peak:.2f} + face {face:.2f}",
                )
            )
        return events
