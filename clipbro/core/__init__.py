"""Core layer: data types, ports, context, registry and configuration.

Imports only the standard library and numpy.
"""

from .config import PROFILES, MotionConfig
from .context import AnalysisContext
from .errors import ClipbroError, InvalidRange, MediaError
from .ports import (
    AudioSource,
    Cache,
    CutDetector,
    EventDetector,
    FaceDetector,
    FocusSmoother,
    FrameSource,
    SignalChannel,
    WordScorer,
)
from .registry import REGISTRY, Registry, discover, register
from .types import (
    AudioTrack,
    Box,
    Event,
    FocusPoint,
    FrameBatch,
    Keyframe,
    SaliencyMap,
    TimeRange,
    Word,
)

__all__ = [
    "AnalysisContext",
    "AudioSource",
    "AudioTrack",
    "Box",
    "Cache",
    "ClipbroError",
    "CutDetector",
    "Event",
    "EventDetector",
    "FaceDetector",
    "FocusPoint",
    "FocusSmoother",
    "FrameBatch",
    "FrameSource",
    "InvalidRange",
    "Keyframe",
    "MediaError",
    "MotionConfig",
    "PROFILES",
    "REGISTRY",
    "Registry",
    "SaliencyMap",
    "SignalChannel",
    "TimeRange",
    "Word",
    "WordScorer",
    "discover",
    "register",
]
