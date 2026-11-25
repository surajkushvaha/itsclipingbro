"""FFmpeg-backed audio source: decodes mono float32 samples of a time range."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clipbro.core.types import AudioTrack, TimeRange

from ._ffmpeg import probe_streams, run_binary


class FfmpegAudioSource:
    """Implements AudioSource by piping f32le mono samples out of ffmpeg."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", timeout: float = 120.0) -> None:
        """Store the binary names and the per-call timeout in seconds."""
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.timeout = timeout

    def has_audio(self, media: Path) -> bool:
        """Return True when media contains at least one audio stream."""
        return bool(probe_streams(self.ffprobe, media, "a", self.timeout))

    def read(self, media: Path, rng: TimeRange, sr: int) -> AudioTrack | None:
        """Return mono audio of rng resampled to sr, or None if there is no audio stream."""
        if sr <= 0:
            raise ValueError(f"sample rate must be positive, got {sr}")
        if not self.has_audio(media):
            return None
        args = [
            self.ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{rng.start:.3f}",
            "-t",
            f"{rng.duration:.3f}",
            "-i",
            str(media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "pipe:1",
        ]
        completed = run_binary(args, self.timeout)
        usable = len(completed.stdout) // 4 * 4
        samples = np.frombuffer(completed.stdout[:usable], dtype="<f4").astype(np.float32, copy=True)
        if samples.size == 0:
            return None
        return AudioTrack(samples=samples, sr=sr, start=rng.start)
