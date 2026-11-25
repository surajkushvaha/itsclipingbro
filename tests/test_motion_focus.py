"""Tests for clipbro.motion.focus: raw centroid, identity smoother and the smoother hook."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from clipbro.core.registry import REGISTRY
from clipbro.core.types import FocusPoint, SaliencyMap
from clipbro.motion.focus import IdentitySmoother, focus_path, raw_focus

ROWS = 4
COLS = 8


def _saliency(frames: list[dict[tuple[int, int], float]]) -> SaliencyMap:
    """Build a SaliencyMap whose frame i has the given (row, col) -> score cells set."""
    combined = np.zeros((len(frames), ROWS, COLS), dtype=np.float32)
    for index, cells in enumerate(frames):
        for (row, col), score in cells.items():
            combined[index, row, col] = score
    return SaliencyMap(
        times=np.arange(len(frames), dtype=np.float64) * 0.5,
        grid=(ROWS, COLS),
        channels={},
        combined=combined,
        hud=np.zeros((ROWS, COLS), dtype=bool),
    )


def _centre(row: int, col: int) -> tuple[float, float]:
    """Expected normalised centre of a grid cell."""
    return (col + 0.5) / COLS, (row + 0.5) / ROWS


class RawFocusTests(unittest.TestCase):
    """Centroid maths on hand-built saliency maps."""

    def test_single_hot_cell_follows_it_left_to_right(self) -> None:
        """A hot cell sliding rightwards drags the focus with it."""
        frames = [{(2, col): 1.0} for col in range(COLS)]
        path = raw_focus(_saliency(frames), top_k=3)
        self.assertEqual(len(path), COLS)
        for col, point in enumerate(path):
            exp_cx, exp_cy = _centre(2, col)
            self.assertLess(abs(point.cx - exp_cx), 0.07)
            self.assertLess(abs(point.cy - exp_cy), 0.07)
        xs = [point.cx for point in path]
        self.assertEqual(xs, sorted(xs))
        self.assertGreater(xs[-1], xs[0])

    def test_top_k_weighted_centroid_between_two_cells(self) -> None:
        """Two hot cells with top_k=2 give the score-weighted centroid of their centres."""
        sal = _saliency([{(0, 0): 1.0, (0, 4): 0.5}])
        point = raw_focus(sal, top_k=2)[0]
        x_a, _ = _centre(0, 0)
        x_b, _ = _centre(0, 4)
        expected_cx = (x_a * 1.0 + x_b * 0.5) / 1.5
        self.assertAlmostEqual(point.cx, expected_cx, places=6)
        self.assertAlmostEqual(point.cy, _centre(0, 0)[1], places=6)
        self.assertAlmostEqual(point.weight, 1.5, places=6)

    def test_top_k_one_picks_only_the_hottest_cell(self) -> None:
        """With top_k=1 only the strongest cell contributes."""
        sal = _saliency([{(0, 0): 1.0, (0, 4): 0.5}])
        point = raw_focus(sal, top_k=1)[0]
        self.assertAlmostEqual(point.cx, _centre(0, 0)[0], places=6)
        self.assertAlmostEqual(point.weight, 1.0, places=6)

    def test_all_zero_frame_holds_previous_point(self) -> None:
        """A frame without saliency repeats the previous point with zero weight."""
        sal = _saliency([{(1, 2): 0.8}, {}, {(3, 6): 0.6}])
        path = raw_focus(sal)
        self.assertEqual((path[1].cx, path[1].cy), (path[0].cx, path[0].cy))
        self.assertEqual(path[1].weight, 0.0)
        self.assertGreater(path[0].weight, 0.0)

    def test_leading_zero_frame_uses_centre(self) -> None:
        """When the first frame is empty the focus starts at (0.5, 0.5)."""
        path = raw_focus(_saliency([{}, {(0, 0): 1.0}]))
        self.assertEqual((path[0].cx, path[0].cy), (0.5, 0.5))
        self.assertEqual(path[0].weight, 0.0)

    def test_all_coordinates_are_normalised(self) -> None:
        """Random saliency never produces coordinates outside 0..1."""
        rng = np.random.default_rng(7)
        frames = [
            {(int(rng.integers(ROWS)), int(rng.integers(COLS))): float(rng.random()) for _ in range(5)}
            for _ in range(20)
        ]
        for point in raw_focus(_saliency(frames), top_k=3):
            self.assertIsInstance(point, FocusPoint)
            self.assertTrue(0.0 <= point.cx <= 1.0)
            self.assertTrue(0.0 <= point.cy <= 1.0)

    def test_invalid_top_k_rejected(self) -> None:
        """top_k below 1 is a configuration error."""
        with self.assertRaises(ValueError):
            raw_focus(_saliency([{(0, 0): 1.0}]), top_k=0)


class SmootherTests(unittest.TestCase):
    """Identity smoother registration and the smoother hook in focus_path."""

    def test_identity_smoother_is_registered(self) -> None:
        """The identity smoother is discoverable through the registry."""
        self.assertIs(REGISTRY.get("smoother", "identity"), IdentitySmoother)
        self.assertEqual(IdentitySmoother.name, "identity")

    def test_identity_smoother_returns_equal_path(self) -> None:
        """The identity smoother leaves the path unchanged."""
        path = [FocusPoint(t=0.0, cx=0.2, cy=0.3), FocusPoint(t=0.5, cx=0.4, cy=0.6)]
        self.assertEqual(IdentitySmoother().smooth(path), path)

    def test_focus_path_without_smoother_equals_raw(self) -> None:
        """focus_path with no smoother returns the raw path."""
        sal = _saliency([{(0, 1): 1.0}, {(2, 5): 1.0}])
        self.assertEqual(focus_path(sal), raw_focus(sal))

    def test_focus_path_applies_custom_smoother(self) -> None:
        """A smoother passed to focus_path transforms the raw path."""

        class PinnedYSmoother:
            """Test smoother that pins every cy to 0.5."""

            def smooth(self, path: list[FocusPoint]) -> list[FocusPoint]:
                """Return copies of the path with cy set to 0.5."""
                return [dataclasses.replace(point, cy=0.5) for point in path]

        sal = _saliency([{(0, 1): 1.0}, {(3, 5): 1.0}])
        path = focus_path(sal, smoother=PinnedYSmoother())
        self.assertEqual([point.cy for point in path], [0.5, 0.5])
        self.assertEqual([point.cx for point in path], [point.cx for point in raw_focus(sal)])

    def test_focus_path_with_registered_identity_smoother(self) -> None:
        """Passing the registered identity smoother gives the raw path."""
        sal = _saliency([{(1, 1): 1.0}])
        smoother = REGISTRY.create("smoother", "identity")
        self.assertEqual(focus_path(sal, smoother=smoother), raw_focus(sal))


if __name__ == "__main__":
    unittest.main()
