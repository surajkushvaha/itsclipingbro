"""Tests for the FFmpeg, cache and YuNet adapters, plus the in-memory fakes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from clipbro.adapters.ffmpeg_audio import FfmpegAudioSource
from clipbro.adapters.ffmpeg_cuts import FfmpegCutDetector
from clipbro.adapters.ffmpeg_video import FfmpegFrameSource, sampled_height
from clipbro.adapters.json_cache import JsonFileCache
from clipbro.adapters.yunet_faces import YuNetFaceDetector, default_model_path
from clipbro.core.errors import MediaError
from clipbro.core.ports import (
    AudioSource,
    Cache,
    CutDetector,
    FaceDetector,
    FrameSource,
)
from clipbro.core.types import AudioTrack, Box, TimeRange
from tests.fixtures import (
    FakeAudioSource,
    FakeCutDetector,
    FakeFaceDetector,
    FakeFrameSource,
    ffmpeg_available,
    make_clip,
    moving_box_clip,
)

FFMPEG_OK = ffmpeg_available()
SKIP_NO_FFMPEG = "ffmpeg/ffprobe not available"


class SampledHeightTests(unittest.TestCase):
    """Height rounding used by the frame source."""

    def test_keeps_aspect_and_is_even(self) -> None:
        """320 wide from a 640x360 source gives 180 high."""
        self.assertEqual(sampled_height(640, 360, 320), 180)

    def test_rounds_to_even(self) -> None:
        """An odd raw height is rounded to the nearest even number."""
        self.assertEqual(sampled_height(100, 101, 33), 34)


@unittest.skipUnless(FFMPEG_OK, SKIP_NO_FFMPEG)
class FfmpegFrameSourceTests(unittest.TestCase):
    """Frame decoding from real lavfi clips."""

    @classmethod
    def setUpClass(cls) -> None:
        """Render the shared moving-box clip once."""
        cls._dir = tempfile.TemporaryDirectory()
        cls.clip = moving_box_clip(Path(cls._dir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary clip directory."""
        cls._dir.cleanup()

    def test_reads_expected_shape_dtype_and_times(self) -> None:
        """Frames are RGB uint8 at the requested width, with evenly spaced absolute times."""
        batch = FfmpegFrameSource().read(self.clip, TimeRange(0.0, 6.0), fps=6.0, width=320)
        self.assertEqual(batch.frames.dtype, np.uint8)
        self.assertEqual(batch.frames.shape[1:], (180, 320, 3))
        self.assertEqual(batch.size, (320, 180))
        self.assertAlmostEqual(float(batch.n), 36, delta=2)
        self.assertEqual(batch.source_size, (640, 360))
        self.assertAlmostEqual(float(batch.times[0]), 0.0)
        self.assertTrue(np.allclose(np.diff(batch.times), 1.0 / 6.0))

    def test_range_offset_shifts_times(self) -> None:
        """Times are absolute: a range starting at 2 s begins at 2 s."""
        batch = FfmpegFrameSource().read(self.clip, TimeRange(2.0, 4.0), fps=6.0, width=320)
        self.assertAlmostEqual(float(batch.times[0]), 2.0)

    def test_missing_media_raises_media_error(self) -> None:
        """A missing file is reported as MediaError."""
        with self.assertRaises(MediaError):
            FfmpegFrameSource().read(Path("/nonexistent/clip.mp4"), TimeRange(0.0, 1.0), 6.0, 320)

    def test_probe_size_reports_source_dimensions(self) -> None:
        """probe_size returns the stored video dimensions."""
        self.assertEqual(FfmpegFrameSource().probe_size(self.clip), (640, 360))


@unittest.skipUnless(FFMPEG_OK, SKIP_NO_FFMPEG)
class FfmpegAudioSourceTests(unittest.TestCase):
    """Audio decoding and the no-audio case."""

    @classmethod
    def setUpClass(cls) -> None:
        """Render a tone clip with audio and a silent-of-audio clip."""
        cls._dir = tempfile.TemporaryDirectory()
        directory = Path(cls._dir.name)
        cls.tone = make_clip(
            directory / "tone.mp4", duration=6.0, video_filter="", audio="sine=f=440:d=6"
        )
        cls.video_only = moving_box_clip(directory)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary clip directory."""
        cls._dir.cleanup()

    def test_returns_mono_track_at_requested_rate(self) -> None:
        """A 6 s tone yields about 6 * sr float32 samples at the requested rate."""
        track = FfmpegAudioSource().read(self.tone, TimeRange(0.0, 6.0), sr=16000)
        self.assertIsNotNone(track)
        assert track is not None
        self.assertEqual(track.sr, 16000)
        self.assertEqual(track.samples.dtype, np.float32)
        self.assertAlmostEqual(float(len(track.samples)), 6.0 * 16000, delta=0.1 * 16000)
        self.assertAlmostEqual(track.start, 0.0)

    def test_returns_none_without_audio_stream(self) -> None:
        """A clip with no audio stream yields None."""
        self.assertIsNone(FfmpegAudioSource().read(self.video_only, TimeRange(0.0, 6.0), sr=16000))


@unittest.skipUnless(FFMPEG_OK, SKIP_NO_FFMPEG)
class FfmpegCutDetectorTests(unittest.TestCase):
    """Scene-cut detection on clips with a known cut."""

    @classmethod
    def setUpClass(cls) -> None:
        """Render a clip that turns from dark grey to white at t=3 s."""
        cls._dir = tempfile.TemporaryDirectory()
        cls.clip = make_clip(
            Path(cls._dir.name) / "cut.mp4",
            duration=6.0,
            video_filter="drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='gte(t,3)'",
            audio=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary clip directory."""
        cls._dir.cleanup()

    def test_finds_colour_switch_near_three_seconds(self) -> None:
        """The cut is reported within 0.2 s of t=3."""
        cuts = FfmpegCutDetector(threshold=0.4).cuts(self.clip, TimeRange(0.0, 6.0))
        self.assertTrue(cuts, "expected at least one cut")
        self.assertTrue(any(abs(time - 3.0) <= 0.2 for time in cuts), cuts)

    def test_cuts_outside_range_are_excluded(self) -> None:
        """A range that ends before the switch reports no cut."""
        cuts = FfmpegCutDetector(threshold=0.4).cuts(self.clip, TimeRange(0.0, 2.0))
        self.assertEqual(cuts, [])

    def test_missing_file_returns_empty_list(self) -> None:
        """Detection failure is not an error: it yields []."""
        self.assertEqual(FfmpegCutDetector().cuts(Path("/nonexistent/x.mp4"), TimeRange(0.0, 1.0)), [])


class JsonFileCacheTests(unittest.TestCase):
    """JSON and array persistence, key validation and corruption handling."""

    KEY = "0123456789abcdef"

    def setUp(self) -> None:
        """Create a cache in a fresh temporary directory."""
        self._dir = tempfile.TemporaryDirectory()
        self.cache = JsonFileCache(Path(self._dir.name) / "nested")

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._dir.cleanup()

    def test_json_roundtrip(self) -> None:
        """A stored dict comes back equal."""
        payload = {"version": 1, "events": [{"t": 1.5}], "name": "x"}
        self.cache.put(self.KEY, payload)
        self.assertEqual(self.cache.get(self.KEY), payload)

    def test_missing_key_returns_none(self) -> None:
        """Reading an absent entry returns None."""
        self.assertIsNone(self.cache.get(self.KEY))

    def test_invalid_keys_raise_value_error(self) -> None:
        """Keys outside ^[0-9a-f]{16,64}$ are rejected everywhere."""
        for bad in ["short", "ABCDEF0123456789", "../../etc/passwd0", "g" * 16, "a" * 65]:
            with self.subTest(key=bad):
                with self.assertRaises(ValueError):
                    self.cache.get(bad)
                with self.assertRaises(ValueError):
                    self.cache.put(bad, {})

    def test_array_roundtrip(self) -> None:
        """A float16 array is stored and loaded back unchanged."""
        array = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        self.cache.put_array(self.KEY, "combined", array)
        loaded = self.cache.get_array(self.KEY, "combined")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, array)
        self.assertEqual(loaded.dtype, np.float16)

    def test_missing_array_returns_none(self) -> None:
        """Absent arrays read back as None."""
        self.assertIsNone(self.cache.get_array(self.KEY, "combined"))

    def test_corrupt_json_returns_none(self) -> None:
        """A truncated JSON file is treated as a miss."""
        self.cache.put(self.KEY, {"a": 1})
        (self.cache.directory / f"{self.KEY}.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.cache.get(self.KEY))

    def test_corrupt_array_returns_none(self) -> None:
        """A garbage .npy file is treated as a miss."""
        self.cache.put_array(self.KEY, "combined", np.zeros(3))
        (self.cache.directory / f"{self.KEY}.combined.npy").write_bytes(b"garbage")
        self.assertIsNone(self.cache.get_array(self.KEY, "combined"))


class YuNetFaceDetectorTests(unittest.TestCase):
    """YuNet adapter against the real model, when it is present."""

    def test_missing_model_raises_file_not_found(self) -> None:
        """An explicit path that does not exist fails fast."""
        with self.assertRaises(FileNotFoundError):
            YuNetFaceDetector(model_path=Path("/nonexistent/model.onnx"))

    @unittest.skipUnless(default_model_path().is_file(), "YuNet model not downloaded")
    def test_blank_image_gives_normalised_boxes(self) -> None:
        """A blank frame yields a list (usually empty) whose coordinates lie in 0..1."""
        detector = YuNetFaceDetector()
        boxes = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertIsInstance(boxes, list)
        for box in boxes:
            self.assertIsInstance(box, Box)
            for value in (box.x, box.y, box.w, box.h):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class FakePortConformanceTests(unittest.TestCase):
    """The in-memory fakes satisfy the runtime-checkable ports and behave simply."""

    def test_fakes_satisfy_ports(self) -> None:
        """Each fake passes isinstance against its Protocol."""
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        self.assertIsInstance(FakeFrameSource(frames, 6.0, (8, 8)), FrameSource)
        self.assertIsInstance(FakeAudioSource(None), AudioSource)
        self.assertIsInstance(FakeCutDetector([]), CutDetector)
        self.assertIsInstance(FakeFaceDetector([]), FaceDetector)
        self.assertIsInstance(JsonFileCache(Path(".")), Cache)

    def test_fake_frame_source_counts_calls(self) -> None:
        """The fake frame source returns its frames and counts reads."""
        source = FakeFrameSource(np.zeros((3, 2, 2, 3), dtype=np.uint8), 2.0, (4, 4))
        batch = source.read(Path("x"), TimeRange(1.0, 2.5), 2.0, 2)
        self.assertEqual(batch.n, 3)
        self.assertAlmostEqual(float(batch.times[0]), 1.0)
        self.assertEqual(source.calls, 1)

    def test_fake_audio_returns_track_or_none(self) -> None:
        """The fake audio source returns the configured track unchanged."""
        track = AudioTrack(samples=np.zeros(10, dtype=np.float32), sr=10, start=0.0)
        self.assertIs(FakeAudioSource(track).read(Path("x"), TimeRange(0.0, 1.0), 10), track)
        self.assertIsNone(FakeAudioSource(None).read(Path("x"), TimeRange(0.0, 1.0), 10))

    def test_fake_cut_detector_filters_to_range(self) -> None:
        """Only cuts inside the requested range are returned."""
        detector = FakeCutDetector([0.5, 3.0, 9.0])
        self.assertEqual(detector.cuts(Path("x"), TimeRange(1.0, 5.0)), [3.0])

    def test_fake_face_detector_accepts_callable(self) -> None:
        """A callable is applied to the image on every call."""
        box = Box(x=0.1, y=0.1, w=0.2, h=0.2, score=0.9)
        detector = FakeFaceDetector(lambda image: [box] if image.any() else [])
        self.assertEqual(detector.detect(np.ones((2, 2, 3), dtype=np.uint8)), [box])
        self.assertEqual(detector.detect(np.zeros((2, 2, 3), dtype=np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
