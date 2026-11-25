"""Composition root: the only module that wires concrete adapters into the motion analyzer.

Plugins (channels, detectors, smoothers) are discovered here so that importing the
motion package alone never has side effects on the registry.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clipbro.adapters.ffmpeg_audio import FfmpegAudioSource
from clipbro.adapters.ffmpeg_cuts import FfmpegCutDetector
from clipbro.adapters.ffmpeg_video import FfmpegFrameSource, probe_duration
from clipbro.adapters.json_cache import JsonFileCache
from clipbro.adapters.yunet_faces import YuNetFaceDetector, default_model_path
from clipbro.core.config import MotionConfig
from clipbro.core.ports import FaceDetector
from clipbro.core.registry import discover
from clipbro.hype.lexicon import LexiconWordScorer
from clipbro.motion.pipeline import MotionAnalyzer

logger = logging.getLogger(__name__)

_PLUGIN_PACKAGES = ("clipbro.motion", "clipbro.motion.channels", "clipbro.motion.detectors")


def _resolve_config(config: MotionConfig | None, profile: str | None) -> MotionConfig:
    """Return the explicit config, else the named profile, else the defaults."""
    if config is not None:
        return config
    if profile:
        return MotionConfig.profile(profile)
    return MotionConfig()


def _discover_plugins() -> None:
    """Import every plugin package so channels, detectors and smoothers register themselves."""
    for package in _PLUGIN_PACKAGES:
        discover(package)


def _build_faces(enabled: bool) -> FaceDetector | None:
    """Return a YuNet detector when enabled and the model file exists; otherwise None."""
    if not enabled:
        return None
    model = default_model_path()
    if not model.is_file():
        logger.warning("YuNet model not found at %s, face channel will be empty", model)
        return None
    return YuNetFaceDetector(model)


def build_analyzer(
    config: MotionConfig | None = None,
    data_dir: Path = Path("data"),
    faces: bool = True,
    profile: str | None = None,
) -> MotionAnalyzer:
    """Wire FFmpeg, YuNet, lexicon and JSON cache adapters into a MotionAnalyzer.

    An explicit config wins over the profile, and with neither the MotionConfig() defaults apply.
    Cache files live in data_dir / "motion".
    """
    resolved = _resolve_config(config, profile)
    _discover_plugins()
    return MotionAnalyzer(
        frames=FfmpegFrameSource(),
        config=resolved,
        audio=FfmpegAudioSource(),
        cuts=FfmpegCutDetector(),
        faces=_build_faces(faces),
        scorer=LexiconWordScorer(),
        cache=JsonFileCache(Path(data_dir) / "motion"),
    )


def default_media_duration(path: Path) -> float:
    """Return the duration of a media file in seconds, probed with ffprobe."""
    return probe_duration(Path(path))
