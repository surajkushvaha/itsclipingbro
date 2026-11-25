"""Plug-and-play proof: a new channel or detector changes the output without touching the pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from clipbro.core import AnalysisContext, Event, MotionConfig, TimeRange
from clipbro.core.registry import REGISTRY, discover
from clipbro.motion.pipeline import MotionAnalyzer
from tests.fixtures import FakeFrameSource

HEIGHT = 180
WIDTH = 320
FPS = 6.0
FRAME_COUNT = 18
CHANNEL = "test_const"
DETECTOR = "test_ping"


class ConstantCornerChannel:
    """Test channel: 1.0 in the top-left cell on every frame, 0.0 elsewhere."""

    name = CHANNEL
    requires: tuple[str, ...] = ()

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return float32 [N, rows, cols] with a single bright corner cell."""
        out = np.zeros((ctx.frames.n, *ctx.grid), dtype=np.float32)
        out[:, 0, 0] = 1.0
        return out


class PingDetector:
    """Test detector: emits exactly one event one second into the range."""

    kind = DETECTOR

    def detect(self, ctx: AnalysisContext) -> list[Event]:
        """Return a single event at rng.start + 1.0 seconds, centred in the frame."""
        return [Event(t=ctx.rng.start + 1.0, kind=DETECTOR, strength=0.5, x=0.5, y=0.5, reason="ping")]


def setUpModule() -> None:
    """Discover the built-in plugins once for the module."""
    discover("clipbro.motion.channels")
    discover("clipbro.motion.detectors")


def _static_frames() -> np.ndarray:
    """Return a flat grey frame stack: no motion, no edges, no flashes, no audio."""
    return np.full((FRAME_COUNT, HEIGHT, WIDTH, 3), 40, dtype=np.uint8)


class PluginPipelineTest(unittest.TestCase):
    """Register temporary plugins, run the pipeline and check the payload."""

    def setUp(self) -> None:
        """Create a dummy media file and register the temporary channel and detector."""
        self._temp = tempfile.TemporaryDirectory()
        self.media = Path(self._temp.name) / "static.mp4"
        self.media.write_bytes(b"not decoded: the fake frame source ignores the path")
        REGISTRY.register("channel", CHANNEL)(ConstantCornerChannel)
        REGISTRY.register("detector", DETECTOR)(PingDetector)

    def tearDown(self) -> None:
        """Remove the temporary plugins and the media directory."""
        REGISTRY.unregister("channel", CHANNEL)
        REGISTRY.unregister("detector", DETECTOR)
        self._temp.cleanup()

    def _analyze(self, config: MotionConfig) -> dict:
        """Run the analyzer with fake frames and no cache; return the payload."""
        frames = FakeFrameSource(_static_frames(), fps=FPS, source_size=(640, 360))
        analyzer = MotionAnalyzer(frames=frames, config=config)
        return analyzer.analyze(self.media, TimeRange(0.0, FRAME_COUNT / FPS), use_cache=False).payload

    def test_plugin_channel_moves_focus_to_its_corner(self) -> None:
        """A weighted test channel in the top-left corner pulls the focus path into that corner."""
        baseline = self._analyze(MotionConfig())
        config = MotionConfig().with_overrides(
            channels=[*MotionConfig().channels, CHANNEL],
            weights={CHANNEL: 1.0},
        )
        payload = self._analyze(config)
        baseline_cx = np.mean([point[1] for point in baseline["focus"]])
        corner_points = np.array(payload["focus"])
        self.assertGreater(baseline_cx, 0.4, "without the plugin the focus should sit near the centre")
        self.assertLess(float(np.max(corner_points[:, 1])), 0.25, "focus x should stay in the left corner")
        self.assertLess(float(np.max(corner_points[:, 2])), 0.25, "focus y should stay in the top corner")

    def test_plugin_detector_event_appears_in_payload(self) -> None:
        """A registered detector listed in config.detectors contributes its event to the payload."""
        config = MotionConfig().with_overrides(detectors=[*MotionConfig().detectors, DETECTOR])
        payload = self._analyze(config)
        pings = [event for event in payload["events"] if event["kind"] == DETECTOR]
        self.assertEqual(len(pings), 1)
        self.assertAlmostEqual(pings[0]["t"], 1.0, places=4)
        self.assertEqual(pings[0]["reason"], "ping")

    def test_plugin_detector_is_absent_when_not_configured(self) -> None:
        """Registering a detector alone does not run it: it must be listed in the config."""
        payload = self._analyze(MotionConfig())
        self.assertEqual([event for event in payload["events"] if event["kind"] == DETECTOR], [])


if __name__ == "__main__":
    unittest.main()
