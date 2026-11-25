"""Pure numpy features on AudioTrack: short-hop RMS, band energy ratio and loud onsets."""

from __future__ import annotations

import numpy as np

from clipbro.core.types import AudioTrack

HOP_S = 0.04
GROUP_WINDOW_S = 0.25
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


def rms_track(track: AudioTrack, hop: float = HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """Return (absolute times, RMS) of non-overlapping hops of hop seconds."""
    blocks = _hop_blocks(track, hop)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    return _hop_times(track, hop, rms.size), rms


def band_ratio(track: AudioTrack, lo: float, hi: float, hop: float = HOP_S) -> tuple[np.ndarray, np.ndarray]:
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


def _lookback_mean(rms: np.ndarray, hop: float, lookback: float) -> np.ndarray:
    """Mean RMS of the lookback seconds of hops before each hop; hops before the start count as 0."""
    span = max(1, int(round(lookback / hop)))
    padded = np.concatenate((np.zeros(span), rms))
    cumulative = np.concatenate(([0.0], np.cumsum(padded)))
    index = np.arange(rms.size)
    return (cumulative[index + span] - cumulative[index]) / span


def _group_onsets(times: np.ndarray, ratios: np.ndarray) -> list[tuple[float, float]]:
    """Group qualifying hops that follow each other within GROUP_WINDOW_S.

    Each group returns (time of its earliest hop, maximum ratio in the group).
    """
    groups: list[tuple[float, float]] = []
    last_time: float | None = None
    for moment, strength in zip(times, ratios):
        moment_f = float(moment)
        strength_f = float(strength)
        if last_time is None or moment_f - last_time > GROUP_WINDOW_S:
            groups.append((moment_f, strength_f))
        else:
            start, best = groups[-1]
            groups[-1] = (start, max(best, strength_f))
        last_time = moment_f
    return groups


def loud_jumps(
    track: AudioTrack | None,
    ratio: float = 3.0,
    floor: float = 0.02,
    rise: float = 1.8,
    lookback: float = 0.12,
) -> list[tuple[float, float]]:
    """Return onset hits as (time, rms / median) for sudden loudness rises.

    A hop qualifies when rms > max(ratio * median, floor) and rms > rise * mean RMS of the
    previous `lookback` seconds of hops (hops before the start count as 0). Qualifying hops
    within 0.25 s form one group, reported at its earliest hop with the group's maximum ratio.
    A sustained loud tone therefore yields a single hit at its start. None gives [].
    """
    if track is None:
        return []
    times, rms = rms_track(track, HOP_S)
    if rms.size == 0:
        return []
    median = float(np.median(rms))
    threshold = max(ratio * median, floor)
    history = _lookback_mean(rms, HOP_S, lookback)
    hits = np.flatnonzero((rms > threshold) & (rms > rise * history))
    if hits.size == 0:
        return []
    return _group_onsets(times[hits], rms[hits] / max(median, _EPS))
