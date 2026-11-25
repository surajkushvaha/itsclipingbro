"""Shared test helpers: lavfi clip builders and in-memory fakes for every port.

Clips are generated with FFmpeg's lavfi sources, so no real media is needed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

from clipbro.core.types import AudioTrack, Box, FrameBatch, TimeRange


def ffmpeg_available() -> bool:
    """Return True when both ffmpeg and ffprobe are on PATH and runnable."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def make_clip(
    path: Path,
    *,
    duration: float,
    size: tuple[int, int] = (640, 360),
    fps: int = 30,
    video_filter: str,
    audio: str | None,
) -> Path:
    """Render a lavfi test clip to path and return it.

    video_filter is a -vf chain applied to a dark grey colour source ('' for none).
    audio is an lavfi audio source expression, or None for a video-only file.
    """
    width, height = size
    args = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    args += ["-f", "lavfi", "-i", f"color=c=0x202020:s={width}x{height}:r={fps}:d={duration}"]
    if audio is not None:
        args += ["-f", "lavfi", "-i", audio]
    if video_filter:
        args += ["-vf", video_filter]
    args += ["-map", "0:v"]
    if audio is not None:
        args += ["-map", "1:a", "-c:a", "aac"]
    else:
        args += ["-an"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    args += ["-t", f"{duration}", str(path)]
    subprocess.run(args, capture_output=True, check=True, timeout=120)
    return path


def moving_box_clip(directory: Path) -> Path:
    """6 s clip: a white 80x80 box sliding right, plus a static high-contrast HUD in the top-left."""
    hud = (
        "drawbox=x=0:y=0:w=120:h=60:color=white:t=2,"
        "drawbox=x=8:y=8:w=40:h=20:color=white:t=fill,"
        "drawbox=x=64:y=32:w=48:h=20:color=white:t=fill"
    )
    box = "drawbox=x='40+t*80':y=140:w=80:h=80:color=white:t=fill"
    return make_clip(
        Path(directory) / "moving_box.mp4",
        duration=6.0,
        video_filter=f"{hud},{box}",
        audio=None,
    )


def flash_boom_clip(directory: Path) -> Path:
    """6 s dim scene with noise; a full-frame flash at 2.0 s and a loud tone burst at 2.0 s."""
    video = (
        "noise=alls=8:allf=t,"
        "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,2.0,2.07)'"
    )
    audio = "aevalsrc=exprs='if(between(t,2,2.1),0.8,0.02)*sin(2*PI*220*t)':s=16000:d=6"
    return make_clip(
        Path(directory) / "flash_boom.mp4",
        duration=6.0,
        video_filter=video,
        audio=audio,
    )


def dead_air_clip(directory: Path) -> Path:
    """7 s clip: moving box and tone 0-2 s, frozen frame and silence 2-5 s, moving again 5-7 s."""
    box_x = "if(lt(t,2),40+t*80,if(lt(t,5),200,40+(t-3)*80))"
    video = f"drawbox=x='{box_x}':y=140:w=80:h=80:color=white:t=fill"
    audio = "aevalsrc=exprs='if(between(t,0,2)+between(t,5,7),0.3,0)*sin(2*PI*330*t)':s=16000:d=7"
    return make_clip(
        Path(directory) / "dead_air.mp4",
        duration=7.0,
        video_filter=video,
        audio=audio,
    )


class FakeFrameSource:
    """In-memory FrameSource returning a fixed frame stack; records how often it was called."""

    def __init__(self, frames: np.ndarray, fps: float, source_size: tuple[int, int]) -> None:
        """Store the frames [N, H, W, 3], sample rate and the reported source size."""
        self.frames = frames
        self.fps = fps
        self.source_size = source_size
        self.calls = 0

    def read(self, media: Path, rng: TimeRange, fps: float, width: int) -> FrameBatch:
        """Return the stored frames with times starting at rng.start."""
        self.calls += 1
        times = rng.start + np.arange(self.frames.shape[0], dtype=np.float64) / self.fps
        return FrameBatch(frames=self.frames, times=times, source_size=self.source_size, fps=self.fps)


class FakeAudioSource:
    """In-memory AudioSource returning a fixed track (or None); records how often it was called."""

    def __init__(self, track: AudioTrack | None) -> None:
        """Store the track to return; None simulates a file without audio."""
        self.track = track
        self.calls = 0

    def read(self, media: Path, rng: TimeRange, sr: int) -> AudioTrack | None:
        """Return the stored track unchanged."""
        self.calls += 1
        return self.track


class FakeCutDetector:
    """In-memory CutDetector returning the stored cut times that fall inside the range."""

    def __init__(self, cuts: list[float]) -> None:
        """Store the absolute cut times."""
        self.known_cuts = list(cuts)

    def cuts(self, media: Path, rng: TimeRange) -> list[float]:
        """Return the stored cuts that lie within rng."""
        return [time for time in self.known_cuts if rng.start <= time <= rng.end]


class FakeFaceDetector:
    """In-memory FaceDetector returning fixed boxes, or the result of a callable applied to the image."""

    def __init__(self, boxes: list[Box] | Callable[[np.ndarray], list[Box]]) -> None:
        """Store either a fixed list of boxes or a function of the image."""
        self.boxes = boxes
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Box]:
        """Return the fixed boxes, or compute them from the image."""
        self.calls += 1
        if callable(self.boxes):
            return self.boxes(image)
        return list(self.boxes)
