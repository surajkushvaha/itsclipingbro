"""FFmpeg-backed frame source: decodes a sampled, scaled RGB frame stack."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from clipbro.core.errors import MediaError
from clipbro.core.types import FrameBatch, TimeRange

from ._ffmpeg import probe_streams, run_binary


def probe_duration(media: Path, ffprobe: str = "ffprobe", timeout: float = 30.0) -> float:
    """Return the container duration of media in seconds.

    Raises MediaError when ffprobe fails or reports no usable positive duration.
    """
    args = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media)]
    completed = run_binary(args, timeout)
    try:
        payload = json.loads(completed.stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as error:
        raise MediaError(f"ffprobe returned invalid JSON for {media}") from error
    try:
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise MediaError(f"no duration reported for {media}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError(f"invalid duration {duration!r} for {media}")
    return duration


def sampled_height(source_w: int, source_h: int, width: int) -> int:
    """Return the even height closest to width * source_h / source_w (minimum 2)."""
    raw = width * source_h / source_w
    return max(2, 2 * round(raw / 2))


class FfmpegFrameSource:
    """Implements FrameSource by piping rawvideo RGB frames out of ffmpeg."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", timeout: float = 120.0) -> None:
        """Store the binary names and the per-call timeout in seconds."""
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.timeout = timeout

    def probe_size(self, media: Path) -> tuple[int, int]:
        """Return (width, height) of the first video stream of media."""
        streams = probe_streams(self.ffprobe, media, "v:0", self.timeout, "stream=width,height")
        if not streams:
            raise MediaError(f"no video stream in {media}")
        try:
            width = int(streams[0]["width"])
            height = int(streams[0]["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise MediaError(f"cannot read video size of {media}") from error
        if width <= 0 or height <= 0:
            raise MediaError(f"invalid video size {width}x{height} in {media}")
        return width, height

    def read(self, media: Path, rng: TimeRange, fps: float, width: int) -> FrameBatch:
        """Decode frames of rng sampled at fps and scaled to width, as RGB uint8."""
        if fps <= 0 or width < 2:
            raise ValueError(f"fps must be > 0 and width >= 2, got fps={fps}, width={width}")
        source_w, source_h = self.probe_size(media)
        height = sampled_height(source_w, source_h, width)
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
            "-an",
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        completed = run_binary(args, self.timeout)
        frames = _decode_frames(completed.stdout, width, height)
        if frames.shape[0] == 0:
            raise MediaError(f"no frames decoded from {media} in {rng.start:.3f}..{rng.end:.3f}")
        times = rng.start + np.arange(frames.shape[0], dtype=np.float64) / fps
        return FrameBatch(frames=frames, times=times, source_size=(source_w, source_h), fps=fps)


def _decode_frames(raw: bytes, width: int, height: int) -> np.ndarray:
    """Reshape a rawvideo rgb24 byte stream into [N, H, W, 3] uint8, dropping any partial frame."""
    frame_bytes = width * height * 3
    count = len(raw) // frame_bytes
    buffer = np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8)
    return buffer.reshape(count, height, width, 3).copy()
