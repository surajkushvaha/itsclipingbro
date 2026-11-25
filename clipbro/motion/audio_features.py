"""Pure numpy features on AudioTrack: short-hop RMS, band energy ratio and loud jumps."""

from __future__ import annotations

import bisect

import numpy as np

from clipbro.core.types import AudioTrack

MERGE_WINDOW_S = 0.2
_EPS = 1e-12


def _hop_blocks(track: AudioTrack, hop: float) -> np.ndarray:
    """Split the samples into non-overlapping hops, returned as [count, hop_len]."""
    hop_len = max(1, int(round(hop * track.sr)))
    count = len(track.samples) // hop_len
    data = np.asarray(track.samples, dtype=np.float64)[: count * hop_len]
    return data.reshape(count, hop_len)


def _hop_times(track: AudioTrack, hop: float, count: int) -> np.ndarray:
    """Return the absolute centre time of each of count hops."""
    return track.start + (np.arange(count) + 0.5) * hop


def rms_track(track: AudioTrack, hop: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Return (absolute times, RMS) of non-overlapping hops of hop seconds."""
    blocks = _hop_blocks(track, hop)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    return _hop_times(track, hop, rms.size), rms


def band_ratio(track: AudioTrack, lo: float, hi: float, hop: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Return (absolute times, ratio) where ratio is the share of energy in [lo, hi] Hz per hop.

    Each hop is Hann-windowed before the FFT. Silent hops give 0.
    """
    blocks = _hop_blocks(track, hop)
    hop_len = blocks.shape[1]
    spectrum = np.abs(np.fft.rfft(blocks * np.hanning(hop_len), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop_len, d=1.0 / track.sr)
    in_band = (freqs >= lo) & (freqs <= hi)
    total = spectrum.sum(axis=1)
    band_energy = spectrum[:, in_band].sum(axis=1)
    ratio = np.divide(band_energy, total, out=np.zeros_like(total), where=total > _EPS)
    return _hop_times(track, hop, blocks.shape[0]), ratio


def _keep_strongest(times: np.ndarray, strengths: np.ndarray) -> list[tuple[float, float]]:
    """Greedily keep the strongest hits so that kept hits are >= MERGE_WINDOW_S apart."""
    taken: list[float] = []
    kept: list[int] = []
    for index in np.argsort(-strengths, kind="stable"):
        moment = float(times[index])
        pos = bisect.bisect_left(taken, moment)
        left_ok = pos == 0 or moment - taken[pos - 1] >= MERGE_WINDOW_S
        right_ok = pos == len(taken) or taken[pos] - moment >= MERGE_WINDOW_S
        if left_ok and right_ok:
            bisect.insort(taken, moment)
            kept.append(int(index))
    kept.sort(key=lambda idx: float(times[idx]))
    return [(float(times[idx]), float(strengths[idx])) for idx in kept]


def loud_jumps(track: AudioTrack | None, ratio: float = 3.0, floor: float = 0.02) -> list[tuple[float, float]]:
    """Return (time, rms / median) for sudden loudness rises.

    A hop is a jump when its RMS exceeds max(ratio * median, floor) and is greater
    than the previous hop's RMS (the first hop counts as rising from silence).
    Hits closer than 0.2 s are merged, keeping the strongest. None gives [].
    """
    if track is None:
        return []
    times, rms = rms_track(track)
    if rms.size == 0:
        return []
    median = float(np.median(rms))
    threshold = max(ratio * median, floor)
    previous = np.concatenate(([0.0], rms[:-1]))
    hits = np.flatnonzero((rms > threshold) & (rms > previous))
    if hits.size == 0:
        return []
    strengths = rms[hits] / max(median, _EPS)
    return _keep_strongest(times[hits], strengths)
