"""Tests for clipbro.motion.keyframes: versioned save/load and keyframe validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clipbro.core.types import Event, FocusPoint, Keyframe
from clipbro.motion import keyframes
from clipbro.motion.keyframes import keyframes_from, load, save


class SaveLoadTests(unittest.TestCase):
    """Round-tripping payloads through the JSON file."""

    def setUp(self) -> None:
        """Create a temporary directory for the keyframe file."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_roundtrip_keeps_version_and_keyframes(self) -> None:
        """Saved keyframes load back equal, with version 1 at the top level."""
        original = [
            Keyframe(t=0.0, cx=0.25, cy=0.75, zoom=1.5, shake=0.1, speed=1.25, reason="start"),
            Keyframe(t=2.0, cx=0.5, cy=0.5),
        ]
        target = self.dir / "shot.json"
        save(target, {"range": [0.0, 10.0], "keyframes": original})
        data = load(target)
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["range"], [0.0, 10.0])
        self.assertEqual(keyframes_from(data), original)

    def test_file_is_compact_utf8_json(self) -> None:
        """The file uses compact separators and no temp files remain."""
        target = self.dir / "shot.json"
        save(target, {"keyframes": [Keyframe(t=1.0, cx=0.1, cy=0.2, reason="café")]})
        text = target.read_text(encoding="utf-8")
        self.assertNotIn(": ", text)
        self.assertIn("café", text)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["shot.json"])

    def test_events_and_focus_points_serialise(self) -> None:
        """Event and FocusPoint objects are written in their JSON forms."""
        target = self.dir / "motion.json"
        save(
            target,
            {
                "events": [Event(t=1.5, kind="cut", strength=0.9, x=0.1, y=0.2, reason="r")],
                "focus": [FocusPoint(t=0.5, cx=0.3, cy=0.4, weight=2.0)],
            },
        )
        data = load(target)
        self.assertEqual(data["events"][0]["kind"], "cut")
        self.assertEqual(data["events"][0]["t"], 1.5)
        self.assertEqual(data["focus"], [[0.5, 0.3, 0.4]])

    def test_numpy_values_serialise(self) -> None:
        """Numpy arrays, scalars and tuples are written as plain JSON values."""
        target = self.dir / "numpy.json"
        save(
            target,
            {
                "array": np.array([1.0, 2.5], dtype=np.float32),
                "scalar": np.float32(0.5),
                "count": np.int64(7),
                "flag": np.bool_(True),
                "pair": (1, 2),
            },
        )
        data = load(target)
        self.assertEqual(data["array"], [1.0, 2.5])
        self.assertEqual(data["scalar"], 0.5)
        self.assertEqual(data["count"], 7)
        self.assertIs(data["flag"], True)
        self.assertEqual(data["pair"], [1, 2])

    def test_save_rejects_conflicting_version(self) -> None:
        """A payload that declares another schema version is refused."""
        with self.assertRaises(ValueError):
            save(self.dir / "bad.json", {"version": 2})
        self.assertFalse((self.dir / "bad.json").exists())

    def test_load_rejects_version_two(self) -> None:
        """A file with a future schema version is refused."""
        target = self.dir / "v2.json"
        target.write_text(json.dumps({"version": 2, "keyframes": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load(target)

    def test_load_rejects_missing_version(self) -> None:
        """A file without a version field is refused."""
        target = self.dir / "nover.json"
        target.write_text(json.dumps({"keyframes": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load(target)

    def test_load_rejects_non_object(self) -> None:
        """A top-level JSON list is refused."""
        target = self.dir / "list.json"
        target.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(ValueError):
            load(target)

    def test_schema_version_constant(self) -> None:
        """The module declares schema version 1."""
        self.assertEqual(keyframes.SCHEMA_VERSION, 1)


class KeyframesFromTests(unittest.TestCase):
    """Validation of keyframe dicts through Keyframe.from_dict."""

    def test_valid_entries_are_rebuilt(self) -> None:
        """Well-formed dicts become Keyframe objects."""
        payload = {"keyframes": [{"t": 1.0, "cx": 0.5, "cy": 0.5, "zoom": 2.0}]}
        self.assertEqual(keyframes_from(payload), [Keyframe(t=1.0, cx=0.5, cy=0.5, zoom=2.0)])

    def test_zoom_three_rejected(self) -> None:
        """zoom above 2.0 raises ValueError."""
        with self.assertRaises(ValueError):
            keyframes_from({"keyframes": [{"t": 0.0, "cx": 0.5, "cy": 0.5, "zoom": 3.0}]})

    def test_cx_out_of_range_rejected(self) -> None:
        """cx above 1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            keyframes_from({"keyframes": [{"t": 0.0, "cx": 1.2, "cy": 0.5}]})

    def test_zero_speed_rejected(self) -> None:
        """speed below 0.25 raises ValueError."""
        with self.assertRaises(ValueError):
            keyframes_from({"keyframes": [{"t": 0.0, "cx": 0.5, "cy": 0.5, "speed": 0}]})

    def test_missing_key_rejected(self) -> None:
        """A payload without the keyframes list raises ValueError."""
        with self.assertRaises(ValueError):
            keyframes_from({"other": []})

    def test_missing_field_rejected(self) -> None:
        """An entry without a required field raises ValueError, not KeyError."""
        with self.assertRaises(ValueError):
            keyframes_from({"keyframes": [{"t": 0.0, "cx": 0.5}]})


if __name__ == "__main__":
    unittest.main()
