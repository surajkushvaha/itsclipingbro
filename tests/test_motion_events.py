"""Tests for saliency combination, event location, dedupe and the event detectors on synthetic data."""

from __future__ import annotations

import unittest

import numpy as np

from clipbro.core import AnalysisContext, AudioTrack, Box, Event, FrameBatch, MotionConfig, TimeRange, Word
from clipbro.core.registry import REGISTRY
from clipbro.motion import events as events_module
from clipbro.motion import saliency

FPS = 6.0
SR = 16000
SIZE = 160
GRID = (8, 8)


class FakeFaceDetector:
    """Returns one fixed box for every frame."""

    def __init__(self, box: Box) -> None:
        """Store the box to report."""
        self.box = box

    def detect(self, image: np.ndarray) -> list[Box]:
        """Return the stored box regardless of the image."""
        return [self.box]


class FakeScorer:
    """Scores words from a fixed dictionary; unknown words score 0."""

    def __init__(self, table: dict[str, float]) -> None:
        """Store the word table."""
        self.table = table

    def score(self, word: str) -> float:
        """Return the table score for word, or 0.0."""
        return self.table.get(word, 0.0)


def make_context(
    frames: np.ndarray,
    *,
    audio: AudioTrack | None = None,
    cuts: list[float] | None = None,
    words: list[Word] | None = None,
    faces: FakeFaceDetector | None = None,
    scorer: FakeScorer | None = None,
    config: MotionConfig | None = None,
) -> AnalysisContext:
    """Build an AnalysisContext over a stack of RGB frames sampled at FPS from t=0."""
    count, height, width = frames.shape[:3]
    batch = FrameBatch(
        frames=frames,
        times=np.arange(count, dtype=np.float64) / FPS,
        source_size=(width, height),
        fps=FPS,
    )
    rng = TimeRange(0.0, count / FPS)
    cfg = config or MotionConfig(fps=FPS, width=width, grid=GRID, face_every=1)
    return AnalysisContext(
        frames=batch,
        rng=rng,
        config=cfg,
        audio=audio,
        cuts=list(cuts or []),
        words=list(words or []),
        faces=faces,
        scorer=scorer,
    )


def run_pipeline(ctx: AnalysisContext, detectors: list[str] | None = None) -> list[Event]:
    """Run saliency then the detectors (optionally restricted to a list of kinds)."""
    if detectors is not None:
        ctx.config = ctx.config.with_overrides(detectors=detectors)
    saliency.run_channels(ctx)
    return events_module.run_detectors(ctx)


def of_kind(found: list[Event], kind: str) -> list[Event]:
    """Filter events by kind."""
    return [event for event in found if event.kind == kind]


def assert_valid_events(test: unittest.TestCase, found: list[Event]) -> None:
    """Every event has coordinates and strength within 0..1 and a finite time."""
    for event in found:
        test.assertTrue(0.0 <= event.x <= 1.0, event)
        test.assertTrue(0.0 <= event.y <= 1.0, event)
        test.assertTrue(0.0 <= event.strength <= 1.0, event)
        test.assertTrue(np.isfinite(event.t), event)


def noisy_dark_frames(count: int, seed: int = 0) -> np.ndarray:
    """Dim grey frames with per-pixel noise, RGB uint8 [N, H, W, 3]."""
    rng = np.random.default_rng(seed)
    gray = np.clip(rng.normal(40.0, 8.0, (count, SIZE, SIZE)), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def boom_scene() -> tuple[np.ndarray, AudioTrack]:
    """6 s dark noisy video with a full-frame white flash on frames 12-13 (t=2.0 s), plus audio.

    The audio is a quiet 220 Hz bed with a loud 220 Hz burst from 2.0 s to 2.1 s.
    """
    frames = noisy_dark_frames(36)
    frames[12:14] = 255
    samples = 0.01 * np.sin(2 * np.pi * 220 * np.arange(6 * SR) / SR)
    samples[2 * SR : int(2.1 * SR)] = 0.8 * np.sin(2 * np.pi * 220 * np.arange(2 * SR, int(2.1 * SR)) / SR)
    return frames, AudioTrack(samples=samples.astype(np.float32), sr=SR, start=0.0)


def dead_air_scene() -> tuple[np.ndarray, AudioTrack]:
    """7 s at 6 fps: moving square and tone 0-2 s, static square and silence 2-5 s, moving with tone 5-7 s."""
    count = 42
    frames = np.full((count, SIZE, SIZE, 3), 32, dtype=np.uint8)
    for index in range(count):
        moment = index / FPS
        offset = 24.0 * moment if moment < 2.0 else (48.0 + 24.0 * (moment - 5.0) if moment >= 5.0 else 48.0)
        x0 = int(round(40 + offset))
        frames[index, 60:100, x0 : x0 + 40] = 255
    samples = np.zeros(7 * SR, dtype=np.float64)
    times = np.arange(7 * SR) / SR
    tone_on = ((times < 2.0) | (times >= 5.0)) & (times < 7.0)
    samples[tone_on] = 0.3 * np.sin(2 * np.pi * 330 * times[tone_on])
    return frames, AudioTrack(samples=samples.astype(np.float32), sr=SR, start=0.0)


def scream_scene() -> tuple[np.ndarray, AudioTrack]:
    """4 s of frames over a 300 Hz bed; a 3 kHz burst from 2.0 s to 2.3 s."""
    frames = noisy_dark_frames(24)
    times = np.arange(4 * SR) / SR
    samples = 0.02 * np.sin(2 * np.pi * 300 * times)
    burst = (times >= 2.0) & (times < 2.3)
    samples[burst] = 0.5 * np.sin(2 * np.pi * 3000 * times[burst])
    return frames, AudioTrack(samples=samples.astype(np.float32), sr=SR, start=0.0)


class TestCombine(unittest.TestCase):
    """Weighted combination of channel maps."""

    shape = (2, 2, 2)

    def test_pipe_key_uses_element_wise_max(self) -> None:
        """A 'a|b' key takes the larger of the two channels per cell."""
        channels = {"flash": np.full(self.shape, 0.2, np.float32), "smoke": np.full(self.shape, 0.6, np.float32)}
        result = saliency.combine(channels, {"flash|smoke": 1.0}, self.shape)
        np.testing.assert_allclose(result, 0.6, atol=1e-6)

    def test_missing_channel_is_ignored(self) -> None:
        """Absent channels contribute nothing, and a key with only absent parts is skipped."""
        channels = {"motion": np.full(self.shape, 0.4, np.float32)}
        weights = {"face": 0.5, "motion": 0.3, "flash|smoke": 0.9}
        result = saliency.combine(channels, weights, self.shape)
        np.testing.assert_allclose(result, 0.12, atol=1e-6)

    def test_weights_are_respected_and_result_is_clipped(self) -> None:
        """Terms are scaled by their weight and the sum is clipped to 1."""
        channels = {"motion": np.full(self.shape, 0.5, np.float32), "centre": np.full(self.shape, 1.0, np.float32)}
        result = saliency.combine(channels, {"motion": 0.5, "centre": 0.25}, self.shape)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)
        clipped = saliency.combine(channels, {"motion": 2.0, "centre": 2.0}, self.shape)
        self.assertEqual(clipped.dtype, np.float32)
        self.assertLessEqual(float(clipped.max()), 1.0)


class TestRunChannels(unittest.TestCase):
    """The saliency runner stores channels, combined map and HUD mask."""

    def test_outputs_are_valid_and_shaped(self) -> None:
        """Every channel and the combined map are float32 [N, rows, cols] within 0..1."""
        ctx = make_context(noisy_dark_frames(12))
        result = saliency.run_channels(ctx)
        self.assertIs(ctx.saliency, result)
        self.assertEqual(result.combined.shape, (12, *GRID))
        self.assertEqual(result.combined.dtype, np.float32)
        self.assertTrue(0.0 <= float(result.combined.min()) and float(result.combined.max()) <= 1.0)
        self.assertEqual(result.hud.shape, GRID)
        self.assertIn("flash", result.channels)

    def test_run_detectors_requires_saliency(self) -> None:
        """Detectors refuse to run before the saliency stage."""
        ctx = make_context(noisy_dark_frames(4))
        with self.assertRaises(ValueError):
            events_module.run_detectors(ctx)


class TestBoom(unittest.TestCase):
    """Flash frames coinciding with audio onsets."""

    def test_flash_with_audio_burst_gives_one_boom(self) -> None:
        """A flash on two frames with a loud onset at the same time gives exactly one boom near 2.0 s."""
        frames, audio = boom_scene()
        found = run_pipeline(make_context(frames, audio=audio), ["boom"])
        booms = of_kind(found, "boom")
        self.assertEqual(len(booms), 1, booms)
        self.assertLessEqual(abs(booms[0].t - 2.0), 0.2)
        self.assertIn("flash", booms[0].reason)
        assert_valid_events(self, found)

    def test_flash_without_audio_burst_is_not_a_boom(self) -> None:
        """The same flash over a quiet bed with no onset produces no boom."""
        frames, audio = boom_scene()
        quiet = AudioTrack(samples=np.full(audio.samples.shape, 0.01, np.float32), sr=SR, start=0.0)
        self.assertEqual(of_kind(run_pipeline(make_context(frames, audio=quiet), ["boom"]), "boom"), [])
        self.assertEqual(of_kind(run_pipeline(make_context(frames, audio=None), ["boom"]), "boom"), [])


class TestDeadAir(unittest.TestCase):
    """Silent and static spans on the common frame timeline."""

    def test_single_span_of_silence_and_stillness(self) -> None:
        """Exactly one dead_air starts within 0.4 s of 2.0 s and lasts about 3 s."""
        frames, audio = dead_air_scene()
        found = run_pipeline(make_context(frames, audio=audio), ["dead_air"])
        spans = of_kind(found, "dead_air")
        self.assertEqual(len(spans), 1, spans)
        self.assertLessEqual(abs(spans[0].t - 2.0), 0.4)
        duration = float(spans[0].reason.split()[-1].rstrip("s"))
        self.assertLessEqual(abs(duration - 3.0), 0.5, spans[0].reason)
        assert_valid_events(self, found)


class TestHypeWord(unittest.TestCase):
    """Hype words from a scorer, restricted to the analysed range."""

    def test_scored_word_gives_event_and_unknown_is_ignored(self) -> None:
        """'omg' produces an event at its start time; words the scorer does not know are dropped."""
        words = [Word("omg", 1.0, 1.3), Word("hello", 2.0, 2.4), Word("zzz", 3.0, 3.2)]
        ctx = make_context(noisy_dark_frames(12), words=words, scorer=FakeScorer({"omg": 1.0}))
        found = run_pipeline(ctx, ["hype_word"])
        self.assertEqual([event.t for event in found], [1.0])
        self.assertEqual(found[0].strength, 1.0)
        assert_valid_events(self, found)

    def test_no_scorer_gives_nothing(self) -> None:
        """Without a scorer there are no hype-word events."""
        ctx = make_context(noisy_dark_frames(12), words=[Word("omg", 1.0, 1.3)])
        self.assertEqual(run_pipeline(ctx, ["hype_word"]), [])


class TestHardCut(unittest.TestCase):
    """Scene cuts become one event each."""

    def test_one_event_per_cut(self) -> None:
        """Each cut in the range gives exactly one hard_cut with strength 1."""
        ctx = make_context(noisy_dark_frames(18), cuts=[1.0, 2.5])
        found = run_pipeline(ctx, ["hard_cut"])
        self.assertEqual([event.t for event in found], [1.0, 2.5])
        self.assertTrue(all(event.strength == 1.0 for event in found))
        assert_valid_events(self, found)


class TestScream(unittest.TestCase):
    """Band-energy spikes that need a visible face."""

    def test_scream_with_face_is_located_on_face(self) -> None:
        """A 3 kHz burst with a centred face gives a scream close to the face centre."""
        frames, audio = scream_scene()
        face = FakeFaceDetector(Box(x=0.4, y=0.4, w=0.2, h=0.2, score=0.9))
        found = run_pipeline(make_context(frames, audio=audio, faces=face), ["scream"])
        screams = of_kind(found, "scream")
        self.assertEqual(len(screams), 1, screams)
        self.assertLessEqual(abs(screams[0].t - 2.0), 0.2)
        self.assertLessEqual(abs(screams[0].x - 0.5), 0.15)
        self.assertLessEqual(abs(screams[0].y - 0.5), 0.15)
        assert_valid_events(self, found)

    def test_scream_without_face_is_ignored(self) -> None:
        """The same burst without a face detector gives no scream."""
        frames, audio = scream_scene()
        found = run_pipeline(make_context(frames, audio=audio, faces=None), ["scream"])
        self.assertEqual(of_kind(found, "scream"), [])


class TestDedupe(unittest.TestCase):
    """Window-based merging of events."""

    def test_same_kind_within_window_keeps_strongest(self) -> None:
        """Two same-kind events 0.4 s apart merge into the stronger one."""
        weak = Event(t=1.0, kind="boom", strength=0.3, x=0.1, y=0.1)
        strong = Event(t=1.4, kind="boom", strength=0.9, x=0.2, y=0.2)
        self.assertEqual(events_module.dedupe([weak, strong], 0.5), [strong])

    def test_different_kinds_are_kept(self) -> None:
        """Events of different kinds never merge, even at the same time."""
        boom = Event(t=1.0, kind="boom", strength=0.3, x=0.1, y=0.1)
        cut = Event(t=1.0, kind="hard_cut", strength=1.0, x=0.1, y=0.1)
        self.assertEqual(len(events_module.dedupe([boom, cut], 0.5)), 2)

    def test_events_beyond_window_are_kept(self) -> None:
        """Same-kind events 0.6 s apart with a 0.5 s window both survive."""
        first = Event(t=1.0, kind="boom", strength=0.3, x=0.1, y=0.1)
        second = Event(t=1.6, kind="boom", strength=0.9, x=0.2, y=0.2)
        self.assertEqual([event.t for event in events_module.dedupe([second, first], 0.5)], [1.0, 1.6])


class TestLocate(unittest.TestCase):
    """Location of the salient cell at a time."""

    def test_locate_returns_centre_of_argmax_cell(self) -> None:
        """A single hot cell in the combined map is located at its centre."""
        ctx = make_context(noisy_dark_frames(6))
        saliency.run_channels(ctx)
        ctx.saliency.combined[:] = 0.0
        ctx.saliency.combined[:, 1, 6] = 0.9
        x, y = events_module.locate(ctx, 0.5)
        self.assertAlmostEqual(x, 6.5 / 8)
        self.assertAlmostEqual(y, 1.5 / 8)
        self.assertTrue(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)


if __name__ == "__main__":
    unittest.main()
