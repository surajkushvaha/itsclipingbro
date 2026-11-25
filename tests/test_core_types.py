"""Tests for clipbro.core.types and AnalysisContext."""

from __future__ import annotations

import unittest

import numpy as np

from clipbro.core.config import MotionConfig
from clipbro.core.context import AnalysisContext
from clipbro.core.errors import InvalidRange
from clipbro.core.types import Event, FrameBatch, Keyframe, SaliencyMap, TimeRange


class TimeRangeTests(unittest.TestCase):
    """TimeRange validation and duration."""

    def test_valid_range_has_duration(self) -> None:
        """A well-formed range reports its length."""
        self.assertAlmostEqual(TimeRange(2.0, 5.5).duration, 3.5)

    def test_negative_start_rejected(self) -> None:
        """A negative start raises InvalidRange."""
        with self.assertRaises(InvalidRange):
            TimeRange(-1.0, 2.0)

    def test_end_not_after_start_rejected(self) -> None:
        """Equal or reversed bounds raise InvalidRange."""
        with self.assertRaises(InvalidRange):
            TimeRange(3.0, 3.0)
        with self.assertRaises(InvalidRange):
            TimeRange(3.0, 2.0)

    def test_nan_rejected(self) -> None:
        """NaN in either bound raises InvalidRange."""
        with self.assertRaises(InvalidRange):
            TimeRange(float("nan"), 2.0)
        with self.assertRaises(InvalidRange):
            TimeRange(0.0, float("nan"))

    def test_invalid_range_is_value_error(self) -> None:
        """InvalidRange can be caught as ValueError."""
        with self.assertRaises(ValueError):
            TimeRange(5.0, 1.0)


class KeyframeTests(unittest.TestCase):
    """Keyframe range validation and serialisation."""

    def test_valid_keyframe_accepted(self) -> None:
        """An in-range keyframe is built without error."""
        keyframe = Keyframe(t=1.0, cx=0.5, cy=0.4, zoom=1.5, shake=0.2, speed=1.0, reason="boom")
        self.assertEqual(keyframe.zoom, 1.5)

    def test_zoom_above_two_rejected(self) -> None:
        """Zoom 3.0 is out of range."""
        with self.assertRaises(ValueError):
            Keyframe(t=0.0, cx=0.5, cy=0.5, zoom=3.0)

    def test_cx_above_one_rejected(self) -> None:
        """cx 1.2 is out of range."""
        with self.assertRaises(ValueError):
            Keyframe(t=0.0, cx=1.2, cy=0.5)

    def test_zero_speed_rejected(self) -> None:
        """Speed 0 is below the 0.25 minimum."""
        with self.assertRaises(ValueError):
            Keyframe(t=0.0, cx=0.5, cy=0.5, speed=0.0)

    def test_negative_shake_rejected(self) -> None:
        """Negative shake is rejected."""
        with self.assertRaises(ValueError):
            Keyframe(t=0.0, cx=0.5, cy=0.5, shake=-1.0)

    def test_keyframe_roundtrip(self) -> None:
        """to_dict followed by from_dict restores the same keyframe."""
        keyframe = Keyframe(t=2.5, cx=0.1, cy=0.9, zoom=1.2, shake=0.3, speed=0.5, reason="x")
        self.assertEqual(Keyframe.from_dict(keyframe.to_dict()), keyframe)


class EventTests(unittest.TestCase):
    """Event clamping, validation and serialisation."""

    def test_strength_and_position_clamped(self) -> None:
        """Out-of-range strength, x and y are clamped into 0..1."""
        event = Event(t=1.0, kind="boom", strength=1.7, x=-0.2, y=2.0)
        self.assertEqual(event.strength, 1.0)
        self.assertEqual(event.x, 0.0)
        self.assertEqual(event.y, 1.0)

    def test_non_finite_time_rejected(self) -> None:
        """A NaN time raises ValueError."""
        with self.assertRaises(ValueError):
            Event(t=float("nan"), kind="boom", strength=0.5, x=0.5, y=0.5)

    def test_event_roundtrip(self) -> None:
        """to_dict followed by from_dict restores the same event."""
        event = Event(t=3.25, kind="smoke", strength=0.6, x=0.25, y=0.75, reason="run")
        restored = Event.from_dict(event.to_dict())
        self.assertEqual(restored.to_dict(), event.to_dict())

    def test_from_dict_defaults_reason(self) -> None:
        """A dict without a reason yields an empty reason."""
        event = Event.from_dict({"t": 1, "kind": "hard_cut", "strength": 1, "x": 0.5, "y": 0.5})
        self.assertEqual(event.reason, "")


class SaliencyMapTests(unittest.TestCase):
    """Nearest-frame lookup and channel fallback."""

    def setUp(self) -> None:
        """Build a small saliency map with three frames at 0, 1 and 2 seconds."""
        self.combined = np.ones((3, 2, 2), dtype=np.float32)
        self.saliency = SaliencyMap(
            times=np.array([0.0, 1.0, 2.0]),
            grid=(2, 2),
            channels={"motion": np.full((3, 2, 2), 0.5, dtype=np.float32)},
            combined=self.combined,
            hud=np.zeros((2, 2), dtype=bool),
        )

    def test_nearest_picks_closest_frame(self) -> None:
        """nearest returns the index of the closest timestamp."""
        self.assertEqual(self.saliency.nearest(0.4), 0)
        self.assertEqual(self.saliency.nearest(0.6), 1)
        self.assertEqual(self.saliency.nearest(9.0), 2)

    def test_present_channel_returned(self) -> None:
        """An existing channel is returned unchanged."""
        self.assertTrue(np.all(self.saliency.channel("motion") == 0.5))

    def test_absent_channel_is_zeros(self) -> None:
        """A missing channel yields zeros shaped like combined."""
        absent = self.saliency.channel("flash")
        self.assertEqual(absent.shape, self.combined.shape)
        self.assertFalse(np.any(absent))


class AnalysisContextTests(unittest.TestCase):
    """memo caching and threshold delegation on AnalysisContext."""

    def setUp(self) -> None:
        """Create a minimal context around a one-frame batch."""
        frames = FrameBatch(
            frames=np.zeros((1, 4, 4, 3), dtype=np.uint8),
            times=np.array([0.0]),
            source_size=(16, 16),
            fps=6.0,
        )
        self.config = MotionConfig(thresholds={"smoke": {"min": 0.6}})
        self.ctx = AnalysisContext(frames=frames, rng=TimeRange(0.0, 1.0), config=self.config)

    def test_memo_computes_once(self) -> None:
        """The compute callable runs only on the first memo call for a key."""
        calls: list[int] = []

        def compute() -> int:
            """Record a call and return a value."""
            calls.append(1)
            return 42

        first = self.ctx.memo("gray", compute)
        second = self.ctx.memo("gray", compute)
        self.assertEqual((first, second), (42, 42))
        self.assertEqual(len(calls), 1)

    def test_threshold_delegates_to_config(self) -> None:
        """ctx.threshold reads the config override, falling back to the default."""
        self.assertEqual(self.ctx.threshold("smoke", "min", 0.4), 0.6)
        self.assertEqual(self.ctx.threshold("smoke", "other", 0.4), 0.4)

    def test_grid_comes_from_config(self) -> None:
        """ctx.grid mirrors config.grid."""
        self.assertEqual(self.ctx.grid, (8, 8))


if __name__ == "__main__":
    unittest.main()
