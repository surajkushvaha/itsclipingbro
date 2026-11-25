"""End-to-end motion pipeline tests on FFmpeg-rendered clips. Skipped when ffmpeg is unavailable."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from clipbro.bootstrap import build_analyzer
from clipbro.core import TimeRange
from clipbro.motion.pipeline import VERSION, MotionAnalyzer
from tests.fixtures import dead_air_clip, ffmpeg_available, flash_boom_clip, make_clip, moving_box_clip

SOURCE_SIZE = [640, 360]
BOOM_TIME = 2.0
DEAD_AIR_START = 2.0
EVENT_TOLERANCE = 0.2
LONG_CLIP_SECONDS = 60.0
LONG_CLIP_LIMIT_SECONDS = 60.0


def box_centre_x(t: float) -> float:
    """Normalised x of the moving box centre at time t in the 640 px wide moving_box_clip: (40 + 80t + 40) / 640."""
    return (40.0 + t * 80.0 + 40.0) / 640.0


def events_of(payload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Return the payload events of one kind, in payload order."""
    return [event for event in payload["events"] if event["kind"] == kind]


def long_moving_clip(directory: Path) -> Path:
    """60 s 640x360 clip: a sliding box that wraps horizontally, plus a steady tone."""
    box = "drawbox=x='mod(40+t*80,560)':y=140:w=80:h=80:color=white:t=fill"
    audio = "aevalsrc=exprs='0.3*sin(2*PI*330*t)':s=16000:d=60"
    return make_clip(
        Path(directory) / "long_moving.mp4",
        duration=LONG_CLIP_SECONDS,
        video_filter=box,
        audio=audio,
    )


@unittest.skipUnless(ffmpeg_available(), "ffmpeg and ffprobe are required")
class MotionPipelineTest(unittest.TestCase):
    """Render the reference clips once and check the analysis payload against their known content."""

    @classmethod
    def setUpClass(cls) -> None:
        """Render the reference clips into a temporary directory and build an analyzer without faces."""
        cls._temp = tempfile.TemporaryDirectory()
        directory = Path(cls._temp.name)
        cls.moving = moving_box_clip(directory)
        cls.flash = flash_boom_clip(directory)
        cls.dead_air = dead_air_clip(directory)
        cls.long_clip = long_moving_clip(directory)
        cls.analyzer: MotionAnalyzer = build_analyzer(data_dir=directory / "cache", faces=False)

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete the rendered clips and cache."""
        cls._temp.cleanup()

    def _analyze(self, media: Path, end: float) -> dict[str, Any]:
        """Run a fresh analysis (no cache) of 0..end seconds and return the payload."""
        return self.analyzer.analyze(media, TimeRange(0.0, end), use_cache=False).payload

    def test_cache_hit_skips_adapters_and_matches_payload(self) -> None:
        """A second analysis of the same range is served from cache: no subprocess, identical payload, fast."""
        with tempfile.TemporaryDirectory() as cache_dir:
            analyzer = build_analyzer(data_dir=Path(cache_dir), faces=False)
            rng = TimeRange(0.0, 6.0)
            first = analyzer.analyze(self.moving, rng)
            self.assertIsNotNone(first.saliency, "first run must be a fresh analysis")
            with mock.patch("subprocess.run") as run:
                started = time.perf_counter()
                second = analyzer.analyze(self.moving, rng)
                elapsed = time.perf_counter() - started
            run.assert_not_called()
            self.assertLess(elapsed, 0.1)
            self.assertIsNone(second.saliency)
            self.assertEqual(
                json.dumps(first.payload, sort_keys=True),
                json.dumps(second.payload, sort_keys=True),
            )
            key = analyzer.cache_key(self.moving, rng)
            combined = analyzer.cache.get_array(key, "combined")
            self.assertIsNotNone(combined)
            self.assertEqual(combined.dtype, np.float16)
            self.assertEqual(combined.shape[0], 13)

    def test_moving_box_focus_follows_box(self) -> None:
        """Focus x rises with time and tracks the box centre; the static HUD does not attract focus."""
        payload = self._analyze(self.moving, 6.0)
        focus = np.array(payload["focus"])
        window = focus[(focus[:, 0] >= 0.5) & (focus[:, 0] <= 5.5)]
        times, cx = window[:, 0], window[:, 1]
        correlation = float(np.corrcoef(times, cx)[0, 1])
        mean_error = float(np.mean(np.abs(cx - np.array([box_centre_x(t) for t in times]))))
        self.assertGreater(correlation, 0.8, f"corr(t, cx) was {correlation:.3f}")
        self.assertLess(mean_error, 0.1, f"mean |cx - box centre| was {mean_error:.3f}")

    def test_hud_top_left_cell_is_flagged(self) -> None:
        """The static overlay in the top-left corner is marked in the HUD mask."""
        payload = self._analyze(self.moving, 6.0)
        hud = np.array(payload["hud"])
        self.assertEqual(hud.shape, (8, 8))
        self.assertEqual(int(hud[0][0]), 1, f"hud grid was {hud.tolist()}")

    def test_flash_boom_yields_one_boom_at_two_seconds(self) -> None:
        """A flash with a loud tone at 2.0 s produces exactly one boom event within 0.2 s of it."""
        payload = self._analyze(self.flash, 6.0)
        booms = events_of(payload, "boom")
        self.assertEqual(len(booms), 1, f"boom events: {booms}")
        self.assertLessEqual(abs(booms[0]["t"] - BOOM_TIME), EVENT_TOLERANCE, f"boom at {booms[0]['t']}")

    def test_dead_air_yields_one_event_near_the_freeze(self) -> None:
        """A silent, static 3 s span yields exactly one dead_air event near where it starts."""
        payload = self._analyze(self.dead_air, 7.0)
        gaps = events_of(payload, "dead_air")
        self.assertEqual(len(gaps), 1, f"dead_air events: {gaps}")
        self.assertLessEqual(abs(gaps[0]["t"] - DEAD_AIR_START), 0.5, f"dead_air at {gaps[0]['t']}")

    def test_payload_contract_and_bounds(self) -> None:
        """Every clip: version 1, source size 640x360, and all coordinates and strengths within 0..1."""
        for name, media, end in (
            ("moving", self.moving, 6.0),
            ("flash", self.flash, 6.0),
            ("dead_air", self.dead_air, 7.0),
        ):
            with self.subTest(clip=name):
                payload = self._analyze(media, end)
                self.assertEqual(payload["version"], VERSION)
                self.assertEqual(payload["source_size"], SOURCE_SIZE)
                for event in payload["events"]:
                    for key in ("strength", "x", "y"):
                        self.assertGreaterEqual(event[key], 0.0)
                        self.assertLessEqual(event[key], 1.0)
                for _, cx, cy in payload["focus"]:
                    self.assertTrue(0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)

    def test_sixty_second_analysis_time(self) -> None:
        """Time a fresh 60 s analysis of a 640x360 clip with a moving box and a tone; print the result."""
        started = time.perf_counter()
        payload = self._analyze(self.long_clip, LONG_CLIP_SECONDS)
        elapsed = time.perf_counter() - started
        print(f"\n60 s clip analysis: {elapsed:.2f} s wall, timing={payload['timing']}")
        self.assertLess(elapsed, LONG_CLIP_LIMIT_SECONDS)


if __name__ == "__main__":
    unittest.main()
