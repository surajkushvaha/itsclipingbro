"""Tests for the debug contact sheet CLI: rendered JPEG, printed table and exit codes."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import cv2

from clipbro.cli.debug import format_table, main
from clipbro.core import Event
from tests.fixtures import ffmpeg_available, flash_boom_clip

TILE_WIDTH = 480
COLS = 5


def hand_made_events() -> list[Event]:
    """Three events of different kinds, one with a long reason, given out of time order."""
    return [
        Event(t=2.0, kind="boom", strength=0.82, x=0.4375, y=0.4375, reason="flash 1.00 + audio x80.0"),
        Event(t=0.5, kind="hard_cut", strength=1.0, x=0.1, y=0.9, reason="scene cut"),
        Event(t=12.25, kind="dead_air", strength=0.6, x=0.5, y=0.5, reason="silent+static 3.0s"),
    ]


class TestFormatTable(unittest.TestCase):
    """The event table is aligned and sorted by time."""

    def test_header_and_rows_are_aligned_and_sorted(self) -> None:
        """Columns line up on the same character positions and rows follow time order."""
        lines = format_table(hand_made_events()).splitlines()
        self.assertTrue(lines[0].startswith("t"))
        self.assertIn("kind", lines[0])
        self.assertIn("reason", lines[0])
        body = lines[2:]
        self.assertEqual([line.split("|")[0].strip() for line in body], ["0.50", "2.00", "12.25"])
        kind_column = [line.index("|") for line in [lines[0]] + body]
        self.assertEqual(len(set(kind_column)), 1, lines)
        self.assertIn("boom", body[1])
        self.assertIn("0.82", body[1])
        self.assertIn("flash 1.00 + audio x80.0", body[1])

    def test_empty_table_says_so(self) -> None:
        """With no events the table shows the header and a placeholder row."""
        self.assertTrue(format_table([]).splitlines()[-1].startswith("(no events)"))


class TestArgumentErrors(unittest.TestCase):
    """Bad arguments and missing media exit with status 2 and a message on stderr."""

    def test_missing_media_exits_two(self) -> None:
        """A media path that does not exist is a usage error, not a crash."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["/nonexistent/clip.mp4", "0", "6"]), 2)
        self.assertIn("media not found", stderr.getvalue())

    def test_inverted_range_exits_two(self) -> None:
        """An end before the start is rejected with status 2."""
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "clip.mp4"
            media.write_bytes(b"not really a video")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(media), "6", "2"]), 2)

    def test_non_numeric_time_exits_two(self) -> None:
        """argparse rejects a non-numeric start with status 2."""
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["clip.mp4", "soon", "6"]), 2)


@unittest.skipUnless(ffmpeg_available(), "ffmpeg and ffprobe are required")
class TestContactSheet(unittest.TestCase):
    """End-to-end: analyse the flash_boom clip and render its contact sheet."""

    def test_sheet_is_written_and_table_is_printed(self) -> None:
        """Exit 0, a JPEG at least five tiles wide, and stdout with the header, the boom and the timing line."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            media = flash_boom_clip(tmp_dir)
            out = tmp_dir / "out" / "s.jpg"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(media), "0", "6", "--out", str(out), "--no-faces", "--data-dir", str(tmp_dir)])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            image = cv2.imread(str(out))
            self.assertIsNotNone(image, "sheet is not a readable image")
            self.assertGreaterEqual(image.shape[1], COLS * TILE_WIDTH)
            text = stdout.getvalue()
            self.assertIn("clipbro debug:", text)
            self.assertIn("boom", text)
            self.assertIn("timing:", text)


if __name__ == "__main__":
    unittest.main()
