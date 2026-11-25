"""FFmpeg-backed hard-cut detector based on the scene-change score."""

from __future__ import annotations

import re
from pathlib import Path

from clipbro.core.errors import MediaError
from clipbro.core.types import TimeRange

from ._ffmpeg import run_binary

_PTS_PATTERN = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


class FfmpegCutDetector:
    """Implements CutDetector by parsing showinfo output of a scene-select filter.

    Cut detection is an optional signal, so any failure yields an empty list.
    """

    def __init__(self, threshold: float = 0.4, timeout: float = 30.0) -> None:
        """Store the scene-change threshold (0..1) and the timeout in seconds."""
        self.threshold = threshold
        self.timeout = timeout

    def cuts(self, media: Path, rng: TimeRange) -> list[float]:
        """Return absolute times of scene cuts inside rng; [] when detection fails."""
        video_filter = f"scale=160:-2,select='gt(scene,{self.threshold})',showinfo"
        args = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "info",
            "-ss",
            f"{rng.start:.3f}",
            "-t",
            f"{rng.duration:.3f}",
            "-i",
            str(media),
            "-an",
            "-vf",
            video_filter,
            "-f",
            "null",
            "-",
        ]
        try:
            completed = run_binary(args, self.timeout)
        except MediaError:
            return []
        return _absolute_cuts(completed.stderr.decode("utf-8", errors="replace"), rng)


def _absolute_cuts(log: str, rng: TimeRange) -> list[float]:
    """Convert showinfo pts_time values in a log into absolute times inside rng."""
    found: list[float] = []
    for match in _PTS_PATTERN.finditer(log):
        absolute = rng.start + float(match.group(1))
        if rng.start <= absolute <= rng.end:
            found.append(absolute)
    return found
