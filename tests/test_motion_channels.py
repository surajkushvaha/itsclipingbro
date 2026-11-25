"""Tests for the motion channels, grid helpers and audio features on synthetic numpy data."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from clipbro.core import AnalysisContext, AudioTrack, Box, FrameBatch, MotionConfig, TimeRange
from clipbro.core.registry import REGISTRY, discover
from clipbro.motion.audio_features import band_ratio, loud_jumps, rms_track
from clipbro.motion.channels.hud import hud_mask
from clipbro.motion.grid import argmax_cell, cell_mean, cell_reduce, cell_slices, to_gray

ALL_CHANNELS = ["hud", "motion", "flash", "smoke", "face", "ui_change", "centre"]
HEIGHT = 180
WIDTH = 320
FPS = 6.0
SQUARE = 40
SQUARE_Y = 100


def setUpModule() -> None:
    """Discover the channel plugins once for the whole module."""
    discover("clipbro.motion.channels")


def make_context(
    frames: np.ndarray,
    *,
    audio: AudioTrack | None = None,
    faces: object | None = None,
    cuts: list[float] | None = None,
    config: MotionConfig | None = None,
) -> AnalysisContext:
    """Wrap an RGB uint8 [N, H, W, 3] array in an AnalysisContext at FPS."""
    count, height, width, _ = frames.shape
    batch = FrameBatch(
        frames=frames,
        times=np.arange(count) / FPS,
        source_size=(width, height),
        fps=FPS,
    )
    return AnalysisContext(
        frames=batch,
        rng=TimeRange(0.0, count / FPS),
        config=config or MotionConfig(),
        audio=audio,
        cuts=list(cuts or []),
        faces=faces,
    )


def run_channels(ctx: AnalysisContext, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Run the named channels in dependency order and store the outputs in ctx.channels."""
    selected = names if names is not None else list(ctx.config.channels)
    for name in REGISTRY.ordered("channel", selected):
        ctx.channels[name] = REGISTRY.create("channel", name).compute(ctx)
    return ctx.channels


def dark_noise(count: int, seed: int = 0) -> np.ndarray:
    """A static dark-noise background repeated for every frame."""
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 40, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    return np.repeat(background[None], count, axis=0)


def draw_square(frames: np.ndarray, index: int, x: int, y: int, value: int = 220) -> None:
    """Paint a bright square onto one frame in place."""
    frames[index, y : y + SQUARE, x : x + SQUARE] = value


def moving_square_frames(count: int) -> np.ndarray:
    """Dark noise with a bright square moving 10 px per frame along a fixed row."""
    frames = dark_noise(count)
    for index in range(count):
        draw_square(frames, index, 20 + 10 * index, SQUARE_Y)
    return frames


def checkerboard(height: int, width: int, cell: int = 8) -> np.ndarray:
    """A high-contrast black and white checkerboard with cell-pixel squares."""
    yy, xx = np.mgrid[0:height, 0:width]
    board = ((yy // cell + xx // cell) % 2) * 255
    return np.repeat(board[:, :, None], 3, axis=2).astype(np.uint8)


class FakeDetector:
    """Face detector stub that always reports one centred face and counts its calls."""

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Box]:
        """Return one box in the middle of the frame."""
        self.calls += 1
        return [Box(0.4, 0.4, 0.2, 0.2, 0.9)]


class RegistryTests(unittest.TestCase):
    """All seven channels are registered and ordered by their requirements."""

    def test_all_seven_channels_register(self) -> None:
        """Every default channel name is present in the global registry."""
        names = REGISTRY.names("channel")
        for name in ALL_CHANNELS:
            self.assertIn(name, names)

    def test_ordered_puts_dependencies_first(self) -> None:
        """hud precedes motion and ui_change; flash precedes smoke."""
        order = REGISTRY.ordered("channel", ALL_CHANNELS)
        self.assertLess(order.index("hud"), order.index("motion"))
        self.assertLess(order.index("hud"), order.index("ui_change"))
        self.assertLess(order.index("flash"), order.index("smoke"))


class GridTests(unittest.TestCase):
    """Cell tiling, reductions and the argmax helper."""

    def test_cell_slices_cover_every_pixel_exactly_once(self) -> None:
        """Cells tile an uneven frame with no gaps and no overlaps."""
        covered = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        for row in cell_slices((HEIGHT, WIDTH), (8, 8)):
            for ys, xs in row:
                covered[ys, xs] += 1
        self.assertTrue(np.all(covered == 1))

    def test_cell_mean_matches_cell_reduce(self) -> None:
        """The vectorised cell mean agrees with the per-cell reference."""
        rng = np.random.default_rng(3)
        image = rng.random((HEIGHT, WIDTH))
        expected = cell_reduce(image, (8, 8), np.mean)
        np.testing.assert_allclose(cell_mean(image, (8, 8)), expected, rtol=1e-9)

    def test_argmax_cell_returns_row_and_column(self) -> None:
        """argmax_cell finds the largest entry as (row, col)."""
        scores = np.zeros((4, 5))
        scores[2, 3] = 1.0
        self.assertEqual(argmax_cell(scores), (2, 3))

    def test_to_gray_gives_uint8_luma(self) -> None:
        """to_gray keeps the shape and gives a uint8 luma image."""
        frames = np.full((2, 4, 6, 3), 255, dtype=np.uint8)
        gray = to_gray(frames)
        self.assertEqual(gray.shape, (2, 4, 6))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertTrue(np.all(gray == 255))


class MotionChannelTests(unittest.TestCase):
    """Optical-flow motion follows the moving square and respects the HUD mask."""

    def test_motion_peaks_where_square_passes_and_column_increases(self) -> None:
        """The argmax column follows the square's horizontal progress."""
        frames = moving_square_frames(24)
        ctx = make_context(frames)
        motion = run_channels(ctx, ["hud", "motion"])["motion"]
        columns = [argmax_cell(motion[index])[1] for index in range(24)]
        self.assertEqual(motion.shape, (24, 8, 8))
        self.assertGreater(columns[-1], columns[0])
        self.assertTrue(all(b >= a for a, b in zip(columns, columns[1:])))

    def test_motion_sits_on_current_position_not_previous(self) -> None:
        """Backward flow lights where the square IS now, not where it WAS.

        The square is 40 px and jumps 20 px per frame, so its current and previous centres lie
        in different cells. Flow saturates across several cells, so the argmax cell is a tie
        artefact; the column-profile centroid of the motion map is the stable measure.
        """
        size, top, jump, count = 40, 80, 20, 12
        frames = dark_noise(count)
        starts = [20 + jump * index for index in range(count)]
        for index, x in enumerate(starts):
            frames[index, top : top + size, x : x + size] = 220
        channels = run_channels(make_context(frames), ["hud", "motion"])
        column_centres = (np.arange(8) + 0.5) / 8
        for index in range(2, count):
            profile = channels["motion"][index].sum(axis=0)
            centroid = float((profile * column_centres).sum() / profile.sum())
            current = (starts[index] + size / 2) / WIDTH
            previous = (starts[index - 1] + size / 2) / WIDTH
            with self.subTest(frame=index):
                self.assertLess(abs(centroid - current), 0.05, f"centroid {centroid:.3f} vs current {current:.3f}")
                self.assertLess(abs(centroid - current), abs(centroid - previous))

    def test_hud_cells_are_masked_and_moving_area_is_not_hud(self) -> None:
        """Static checkerboard cells are HUD with zero motion; the moving area is not HUD."""
        frames = moving_square_frames(24)
        frames[:, 0:45, 0:80] = checkerboard(45, 80)
        ctx = make_context(frames)
        channels = run_channels(ctx)
        hud = hud_mask(ctx)
        self.assertTrue(np.all(channels["hud"][0][0:2, 0:2] == 1.0))
        self.assertTrue(hud[0, 0] and hud[0, 1] and hud[1, 0] and hud[1, 1])
        self.assertFalse(hud[5, 4])
        self.assertTrue(np.all(channels["motion"][:, 0:2, 0:2] == 0.0))
        moving_row = channels["motion"][:, 4:7, :]
        self.assertGreater(float(moving_row.max()), 0.3)


class FlashChannelTests(unittest.TestCase):
    """Full-frame flashes produce a strong flash score on the flash frame only."""

    def test_flash_fires_on_burst_frames_only(self) -> None:
        """A white flash on frames 10 and 11 scores high at 10 and low at 3 and 20."""
        frames = dark_noise(24)
        frames[10:12] = 255
        channels = run_channels(make_context(frames), ["flash"])
        flash = channels["flash"]
        self.assertGreaterEqual(float(flash[10].max()), 0.5)
        self.assertLess(float(flash[3].max()), 0.2)
        self.assertLess(float(flash[20].max()), 0.2)


def _blur_cell(frames: np.ndarray, start: int, ys: slice, xs: slice) -> None:
    """Desaturate and blur one cell in every frame from start onwards, in place."""
    for index in range(start, frames.shape[0]):
        region = np.ascontiguousarray(frames[index, ys, xs])
        grey = cv2.cvtColor(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        frames[index, ys, xs] = cv2.GaussianBlur(grey, (15, 15), 0)


def _colour_texture(count: int, seed: int = 5) -> np.ndarray:
    """A static saturated random-colour texture repeated over frames."""
    rng = np.random.default_rng(seed)
    base = rng.integers(80, 200, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    return np.repeat(base[None], count, axis=0)


class SmokeChannelTests(unittest.TestCase):
    """Smoke appears after a flash and is gated off without a trigger."""

    CELL = (2, 3)

    def _cell_slices(self) -> tuple[slice, slice]:
        """Return the pixel slices of the test cell."""
        return cell_slices((HEIGHT, WIDTH), (8, 8))[self.CELL[0]][self.CELL[1]]

    def test_blur_after_flash_scores_in_gate(self) -> None:
        """A desaturated, blurred cell after a flash gets smoke > 0."""
        frames = _colour_texture(24)
        frames[3] = 255
        ys, xs = self._cell_slices()
        _blur_cell(frames, 8, ys, xs)
        channels = run_channels(make_context(frames), ["flash", "smoke"])
        smoke = channels["smoke"]
        self.assertGreater(float(smoke[8:16, self.CELL[0], self.CELL[1]].max()), 0.5)

    def test_identical_blur_without_flash_or_audio_is_gated_off(self) -> None:
        """The same blur with no flash and no audio scores exactly zero."""
        frames = _colour_texture(24)
        ys, xs = self._cell_slices()
        _blur_cell(frames, 8, ys, xs)
        channels = run_channels(make_context(frames), ["flash", "smoke"])
        self.assertTrue(np.all(channels["smoke"] == 0.0))


class FaceChannelTests(unittest.TestCase):
    """The face channel rasterises a detector's boxes and runs it every few frames."""

    def test_centre_cells_light_up_and_corners_stay_dark(self) -> None:
        """The centred fake face covers central cells and not the corners."""
        detector = FakeDetector()
        ctx = make_context(dark_noise(24), faces=detector)
        face = run_channels(ctx, ["face"])["face"]
        self.assertGreater(float(face[:, 3:5, 3:5].min()), 0.4)
        for row, col in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            self.assertLess(float(face[:, row, col].max()), 1e-6)

    def test_detector_runs_about_every_face_every_frames(self) -> None:
        """With face_every=2 over 24 frames the detector runs 12 times."""
        detector = FakeDetector()
        ctx = make_context(dark_noise(24), faces=detector)
        run_channels(ctx, ["face"])
        self.assertEqual(detector.calls, 24 // ctx.config.face_every)

    def test_no_detector_gives_zeros(self) -> None:
        """Without a detector the face channel is all zeros."""
        ctx = make_context(dark_noise(12), faces=None)
        face = run_channels(ctx, ["face"])["face"]
        self.assertEqual(face.shape, (12, 8, 8))
        self.assertTrue(np.all(face == 0.0))


class UiChangeChannelTests(unittest.TestCase):
    """Overlay changes inside HUD cells show up as ui_change on that frame."""

    def test_single_frame_text_swap_in_hud_cell(self) -> None:
        """A text box that appears once in a HUD cell scores above zero on that frame."""
        frames = moving_square_frames(24)
        ys, xs = cell_slices((HEIGHT, WIDTH), (8, 8))[0][0]
        frames[:, ys, xs] = checkerboard(ys.stop - ys.start, xs.stop - xs.start)
        frames[12, ys.start + 4 : ys.start + 10, xs.start + 4 : xs.start + 14] = 120
        channels = run_channels(make_context(frames))
        ui = channels["ui_change"]
        self.assertGreater(float(ui[12, 0, 0]), 0.0)
        self.assertEqual(float(ui[5, 0, 0]), 0.0)
        self.assertTrue(np.all(ui[:, 4:7, :] == 0.0))


class CentreChannelTests(unittest.TestCase):
    """The centre prior favours the middle and never changes over time."""

    def test_centre_beats_corners_and_is_constant(self) -> None:
        """Centre cells outweigh corner cells, and every frame is identical."""
        centre = run_channels(make_context(dark_noise(10)), ["centre"])["centre"]
        self.assertGreater(float(centre[0, 3, 3]), float(centre[0, 0, 0]))
        np.testing.assert_array_equal(centre[0], centre[-1])


class AudioFeatureTests(unittest.TestCase):
    """Audio helpers on synthetic tones and bursts."""

    SR = 16000

    def _track(self, samples: np.ndarray, start: float = 0.0) -> AudioTrack:
        """Wrap float samples in an AudioTrack at the test sample rate."""
        return AudioTrack(samples=samples.astype(np.float32), sr=self.SR, start=start)

    def _sine(self, freq: float, seconds: float) -> np.ndarray:
        """A full-scale-half sine at freq Hz lasting seconds."""
        time_axis = np.arange(int(seconds * self.SR)) / self.SR
        return 0.5 * np.sin(2 * np.pi * freq * time_axis)

    def test_rms_track_length_and_times(self) -> None:
        """Three seconds at a 40 ms hop gives 75 hops with centred times."""
        times, rms = rms_track(self._track(self._sine(300.0, 3.0)))
        self.assertEqual(len(rms), 75)
        self.assertEqual(len(times), 75)
        self.assertAlmostEqual(float(times[0]), 0.02, places=6)

    def test_band_ratio_separates_3khz_from_300hz(self) -> None:
        """A 3 kHz tone has most of its energy in 2-4 kHz; a 300 Hz tone has almost none."""
        _, high = band_ratio(self._track(self._sine(3000.0, 1.0)), 2000.0, 4000.0)
        _, low = band_ratio(self._track(self._sine(300.0, 1.0)), 2000.0, 4000.0)
        self.assertGreater(float(np.median(high)), 0.9)
        self.assertLess(float(np.median(low)), 0.1)

    def test_loud_jumps_finds_burst_at_two_seconds(self) -> None:
        """A percussive burst starting at 2.0 s is reported within 0.05 s of that onset.

        The burst decays, as a real hit does, so its onset is the loudest hop and the
        only rising one; a sustained tone would also register micro-rises in its RMS.
        """
        rng = np.random.default_rng(11)
        samples = 0.001 * rng.standard_normal(3 * self.SR)
        onset = int(2.0 * self.SR)
        tail = np.arange(int(0.5 * self.SR)) / self.SR
        samples[onset : onset + tail.size] += 0.5 * np.exp(-tail / 0.08) * np.sin(2 * np.pi * 440.0 * tail)
        jumps = loud_jumps(self._track(samples))
        self.assertEqual(len(jumps), 1)
        self.assertLessEqual(abs(jumps[0][0] - 2.0), 0.05)

    def test_sustained_tone_gives_single_onset_hit(self) -> None:
        """A 0.5 s loud 440 Hz tone at 2.0 s on a quiet bed yields exactly one hit at its start."""
        rng = np.random.default_rng(12)
        samples = 0.001 * rng.standard_normal(3 * self.SR)
        onset = int(2.0 * self.SR)
        samples[onset : onset + int(0.5 * self.SR)] += self._sine(440.0, 0.5)
        jumps = loud_jumps(self._track(samples))
        self.assertEqual(len(jumps), 1)
        self.assertLessEqual(abs(jumps[0][0] - 2.0), 0.05)

    def test_two_bursts_one_second_apart_give_two_hits(self) -> None:
        """Two decaying bursts 1 s apart are reported as two separate onsets."""
        rng = np.random.default_rng(13)
        samples = 0.001 * rng.standard_normal(4 * self.SR)
        tail = np.arange(int(0.4 * self.SR)) / self.SR
        burst = 0.5 * np.exp(-tail / 0.08) * np.sin(2 * np.pi * 440.0 * tail)
        for start in (1.0, 2.0):
            index = int(start * self.SR)
            samples[index : index + tail.size] += burst
        jumps = loud_jumps(self._track(samples))
        self.assertEqual(len(jumps), 2)
        self.assertLessEqual(abs(jumps[0][0] - 1.0), 0.05)
        self.assertLessEqual(abs(jumps[1][0] - 2.0), 0.05)

    def test_loud_jumps_none_track_is_empty(self) -> None:
        """No audio gives no jumps."""
        self.assertEqual(loud_jumps(None), [])


class AllChannelsContractTests(unittest.TestCase):
    """Every channel returns float32 [N, 8, 8] values inside 0..1."""

    def test_outputs_are_bounded_float32_grids(self) -> None:
        """All seven channels run together and satisfy the output contract."""
        frames = moving_square_frames(24)
        frames[:, 0:45, 0:80] = checkerboard(45, 80)
        frames[10:12, 100:, 200:] = 255
        ctx = make_context(frames, faces=FakeDetector(), cuts=[3.0])
        channels = run_channels(ctx)
        self.assertEqual(set(channels), set(ALL_CHANNELS))
        for name, values in channels.items():
            with self.subTest(channel=name):
                self.assertEqual(values.dtype, np.float32)
                self.assertEqual(values.shape, (24, 8, 8))
                self.assertGreaterEqual(float(values.min()), 0.0)
                self.assertLessEqual(float(values.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
