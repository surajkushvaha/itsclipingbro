"""Tests for clipbro.core.config."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from clipbro.core.config import PROFILES, MotionConfig


class DefaultsTests(unittest.TestCase):
    """Default values and grid normalisation."""

    def test_default_values(self) -> None:
        """Defaults match the architecture contract."""
        config = MotionConfig()
        self.assertEqual(config.fps, 6.0)
        self.assertEqual(config.grid, (8, 8))
        self.assertEqual(config.channels, ["hud", "motion", "flash", "smoke", "face", "ui_change", "centre"])
        self.assertEqual(config.detectors, ["boom", "smoke", "scream", "hype_word", "hard_cut", "dead_air"])
        self.assertAlmostEqual(sum(config.weights.values()), 1.0)

    def test_list_grid_becomes_tuple(self) -> None:
        """A list grid is stored as a tuple."""
        config = MotionConfig(grid=[4, 6])  # type: ignore[arg-type]
        self.assertEqual(config.grid, (4, 6))

    def test_default_containers_not_shared(self) -> None:
        """Mutating one instance's weights does not change another's."""
        first = MotionConfig()
        first.weights["extra"] = 0.1
        self.assertNotIn("extra", MotionConfig().weights)


class ProfileTests(unittest.TestCase):
    """Named profiles."""

    def test_every_profile_weights_sum_to_one(self) -> None:
        """All profiles produce weights summing to 1.0."""
        for name in PROFILES:
            with self.subTest(profile=name):
                weights = MotionConfig.profile(name).weights
                self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_profile_replaces_weights_entirely(self) -> None:
        """A profile's weights replace the default dict rather than merging into it."""
        talk = MotionConfig.profile("talk")
        self.assertEqual(set(talk.weights), set(PROFILES["talk"]["weights"]))
        self.assertNotIn("extra", talk.weights)
        self.assertEqual(talk.weights["flash|smoke"], 0.10)

    def test_unknown_profile_lists_valid_names(self) -> None:
        """An unknown profile raises ValueError naming the valid profiles."""
        with self.assertRaises(ValueError) as ctx:
            MotionConfig.profile("nope")
        message = str(ctx.exception)
        self.assertIn("gaming", message)
        self.assertIn("talk", message)


class ThresholdTests(unittest.TestCase):
    """Threshold lookup and fingerprinting."""

    def test_threshold_override_and_default(self) -> None:
        """A configured threshold wins; a missing one falls back to the default."""
        config = MotionConfig(thresholds={"smoke": {"min": 0.6}})
        self.assertEqual(config.threshold("smoke", "min", 0.4), 0.6)
        self.assertEqual(config.threshold("smoke", "min_s", 0.5), 0.5)
        self.assertEqual(config.threshold("boom", "flash_min", 0.5), 0.5)

    def test_fingerprint_stable_and_sensitive(self) -> None:
        """Equal configs share a fingerprint; changing a threshold changes it."""
        base = MotionConfig()
        self.assertEqual(base.fingerprint(), MotionConfig().fingerprint())
        changed = base.with_overrides(thresholds={"smoke": {"min": 0.9}})
        self.assertNotEqual(base.fingerprint(), changed.fingerprint())
        self.assertEqual(len(base.fingerprint()), 64)


class SerialisationTests(unittest.TestCase):
    """from_dict, with_overrides and from_toml."""

    def test_from_dict_rejects_unknown_key(self) -> None:
        """An unknown key raises ValueError naming it."""
        with self.assertRaises(ValueError) as ctx:
            MotionConfig.from_dict({"fps": 4.0, "bogus": 1})
        self.assertIn("bogus", str(ctx.exception))

    def test_from_dict_applies_values(self) -> None:
        """Known keys are applied and absent keys keep defaults."""
        config = MotionConfig.from_dict({"fps": 4.0})
        self.assertEqual(config.fps, 4.0)
        self.assertEqual(config.width, 320)

    def test_with_overrides_deep_merges_dicts(self) -> None:
        """Nested threshold dicts merge; the original instance is untouched."""
        base = MotionConfig(thresholds={"smoke": {"min": 0.4}})
        merged = base.with_overrides(thresholds={"smoke": {"min_s": 0.5}}, fps=3.0)
        self.assertEqual(merged.thresholds, {"smoke": {"min": 0.4, "min_s": 0.5}})
        self.assertEqual(merged.fps, 3.0)
        self.assertEqual(base.thresholds, {"smoke": {"min": 0.4}})
        self.assertEqual(base.fps, 6.0)

    def test_with_overrides_merges_weights_per_key(self) -> None:
        """Changing one weight keeps the others."""
        merged = MotionConfig().with_overrides(weights={"motion": 0.5})
        self.assertEqual(merged.weights["motion"], 0.5)
        self.assertEqual(merged.weights["centre"], 0.05)

    def test_with_overrides_rejects_unknown_field(self) -> None:
        """An unknown field name raises ValueError."""
        with self.assertRaises(ValueError):
            MotionConfig().with_overrides(nonsense=1)

    @unittest.skipUnless(sys.version_info >= (3, 11), "tomllib needs Python 3.11+")
    def test_from_toml_loads_file(self) -> None:
        """A TOML file with top-level keys loads into a config."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.toml"
            path.write_text('fps = 2.5\ngrid = [4, 4]\n\n[thresholds.smoke]\nmin = 0.7\n', encoding="utf-8")
            config = MotionConfig.from_toml(path)
        self.assertEqual(config.fps, 2.5)
        self.assertEqual(config.grid, (4, 4))
        self.assertEqual(config.threshold("smoke", "min", 0.4), 0.7)


if __name__ == "__main__":
    unittest.main()
