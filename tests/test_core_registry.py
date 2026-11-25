"""Tests for clipbro.core.registry using fresh Registry instances."""

from __future__ import annotations

import unittest

from clipbro.core.registry import Registry, discover


class RegisterTests(unittest.TestCase):
    """Decorator behaviour, lookup and removal."""

    def setUp(self) -> None:
        """Give each test its own registry."""
        self.registry = Registry()

    def test_channel_gets_name_attribute(self) -> None:
        """A channel class without a name receives the registered name."""

        @self.registry.register("channel", "motion")
        class Motion:
            """Test channel."""

        self.assertEqual(Motion.name, "motion")

    def test_detector_gets_kind_attribute(self) -> None:
        """A detector class without a kind receives the registered name as kind."""

        @self.registry.register("detector", "boom")
        class Boom:
            """Test detector."""

        self.assertEqual(Boom.kind, "boom")
        self.assertFalse(hasattr(Boom, "name"))

    def test_existing_name_is_kept(self) -> None:
        """A class that already declares its name keeps it."""

        @self.registry.register("channel", "alias")
        class Named:
            """Test channel with its own name."""

            name = "original"

        self.assertEqual(Named.name, "original")

    def test_duplicate_name_rejected(self) -> None:
        """Registering the same kind and name twice raises ValueError."""

        @self.registry.register("channel", "hud")
        class First:
            """First registration."""

        with self.assertRaises(ValueError):

            @self.registry.register("channel", "hud")
            class Second:
                """Duplicate registration."""

    def test_get_unknown_raises_key_error_with_hint(self) -> None:
        """Looking up an unknown name raises KeyError listing what exists."""

        @self.registry.register("channel", "hud")
        class Hud:
            """Test channel."""

        with self.assertRaises(KeyError) as ctx:
            self.registry.get("channel", "missing")
        self.assertIn("hud", str(ctx.exception))

    def test_names_and_create(self) -> None:
        """names lists registrations and create instantiates with kwargs."""

        @self.registry.register("smoother", "identity")
        class Identity:
            """Test smoother."""

            def __init__(self, gain: int = 1) -> None:
                """Store the gain."""
                self.gain = gain

        self.assertEqual(self.registry.names("smoother"), ["identity"])
        self.assertEqual(self.registry.create("smoother", "identity", gain=3).gain, 3)

    def test_unregister_removes_entry(self) -> None:
        """After unregister the name is gone; unknown unregister raises KeyError."""

        @self.registry.register("channel", "tmp")
        class Tmp:
            """Temporary channel."""

        self.registry.unregister("channel", "tmp")
        self.assertEqual(self.registry.names("channel"), [])
        with self.assertRaises(KeyError):
            self.registry.get("channel", "tmp")
        with self.assertRaises(KeyError):
            self.registry.unregister("channel", "tmp")


class OrderedTests(unittest.TestCase):
    """Dependency ordering over the requires attribute."""

    def setUp(self) -> None:
        """Register a small channel graph: motion requires hud, flash is independent."""
        self.registry = Registry()

        @self.registry.register("channel", "hud")
        class Hud:
            """No dependencies."""

            requires: tuple[str, ...] = ()

        @self.registry.register("channel", "motion")
        class Motion:
            """Depends on hud."""

            requires = ("hud",)

        @self.registry.register("channel", "flash")
        class Flash:
            """Independent, with no requires attribute at all."""

        self.Hud, self.Motion, self.Flash = Hud, Motion, Flash

    def test_dependencies_come_first(self) -> None:
        """A plugin is placed after the names it requires, keeping input order otherwise."""
        order = self.registry.ordered("channel", ["motion", "flash", "hud"])
        self.assertEqual(order, ["flash", "hud", "motion"])

    def test_already_sorted_input_unchanged(self) -> None:
        """Input that already respects dependencies comes back unchanged."""
        self.assertEqual(self.registry.ordered("channel", ["hud", "motion"]), ["hud", "motion"])

    def test_missing_dependency_names_it(self) -> None:
        """A dependency absent from the selection raises ValueError naming it."""
        with self.assertRaises(ValueError) as ctx:
            self.registry.ordered("channel", ["motion"])
        self.assertIn("hud", str(ctx.exception))

    def test_cycle_rejected(self) -> None:
        """Mutually dependent plugins raise ValueError."""

        @self.registry.register("channel", "a")
        class A:
            """Depends on b."""

            requires = ("b",)

        @self.registry.register("channel", "b")
        class B:
            """Depends on a."""

            requires = ("a",)

        with self.assertRaises(ValueError) as ctx:
            self.registry.ordered("channel", ["a", "b"])
        self.assertIn("cycle", str(ctx.exception))


class DiscoverTests(unittest.TestCase):
    """discover is safe to call repeatedly."""

    def test_discover_is_idempotent(self) -> None:
        """Calling discover twice on an existing package raises nothing."""
        discover("clipbro.core")
        discover("clipbro.core")


if __name__ == "__main__":
    unittest.main()
