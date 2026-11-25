"""Tests for the motion HTTP routes with a fake analyzer and a temporary media file. Skipped without fastapi and httpx."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any

from clipbro.core.errors import MediaError
from clipbro.core.types import TimeRange

HAS_HTTP_STACK = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx"))
DURATION = 100.0
PAYLOAD: dict[str, Any] = {"version": 1, "source_size": [640, 360], "range": [1.5, 4.0], "events": []}


class FakeResult:
    """Stands in for MotionResult: only the payload attribute is used by the routes."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the payload to return."""
        self.payload = payload


class FakeAnalyzer:
    """Records every analyze call and returns a fixed payload, or raises a fixed error."""

    def __init__(self, error: Exception | None = None) -> None:
        """Store the optional error to raise on every call."""
        self.error = error
        self.calls: list[tuple[Path, TimeRange]] = []

    def analyze(self, media: Path, rng: TimeRange) -> FakeResult:
        """Record the call, then return the payload or raise the stored error."""
        self.calls.append((Path(media), rng))
        if self.error is not None:
            raise self.error
        return FakeResult(PAYLOAD)


def fixed_duration(path: Path) -> float:
    """Pretend every media file lasts DURATION seconds."""
    return DURATION


if HAS_HTTP_STACK:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from clipbro.api.motion_routes import create_app, make_router, media_dir_resolver


@unittest.skipUnless(HAS_HTTP_STACK, "fastapi and httpx are required")
class MotionRoutesTest(unittest.TestCase):
    """Exercise GET /motion/{source_id} through a TestClient with a fake analyzer."""

    def setUp(self) -> None:
        """Create a temporary media file and an app whose analyzer and media lookup are fakes."""
        self._temp = tempfile.TemporaryDirectory()
        self.media = Path(self._temp.name) / "src1.mp4"
        self.media.write_bytes(b"not really a video")
        self.analyzer = FakeAnalyzer()
        self.client = self._client(self.analyzer)

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._temp.cleanup()

    def _client(self, analyzer: FakeAnalyzer) -> TestClient:
        """Return a TestClient over an app mounting make_router with the given analyzer."""
        resolver = media_dir_resolver(Path(self._temp.name))
        app = FastAPI()
        app.include_router(make_router(analyzer, resolver, fixed_duration))
        return TestClient(app)

    def test_returns_payload_and_passes_time_range(self) -> None:
        """A valid request returns the analyzer payload and analyses the requested range."""
        response = self.client.get("/motion/src1", params={"start": "1.5", "end": "4"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), PAYLOAD)
        self.assertEqual(len(self.analyzer.calls), 1)
        media, rng = self.analyzer.calls[0]
        self.assertEqual(media, self.media)
        self.assertEqual((rng.start, rng.end), (1.5, 4.0))

    def test_invalid_requests_return_400_without_analysis(self) -> None:
        """Every validation failure answers 400 and never reaches the analyzer."""
        cases = {
            "non-alphanumeric id": ("/motion/src.1", {"start": "0", "end": "1"}),
            "overlong id": ("/motion/" + "a" * 65, {"start": "0", "end": "1"}),
            "missing media": ("/motion/ghost", {"start": "0", "end": "1"}),
            "start equals end": ("/motion/src1", {"start": "3", "end": "3"}),
            "end before start": ("/motion/src1", {"start": "5", "end": "2"}),
            "negative start": ("/motion/src1", {"start": "-1", "end": "2"}),
            "range over 600 s": ("/motion/src1", {"start": "0", "end": "601"}),
            "end beyond duration": ("/motion/src1", {"start": "50", "end": "100.5"}),
            "nan start": ("/motion/src1", {"start": "nan", "end": "2"}),
            "infinite end": ("/motion/src1", {"start": "0", "end": "inf"}),
            "missing start": ("/motion/src1", {"end": "2"}),
            "non-numeric end": ("/motion/src1", {"start": "0", "end": "soon"}),
        }
        for name, (url, params) in cases.items():
            with self.subTest(case=name):
                response = self.client.get(url, params=params)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIsInstance(response.json()["detail"], str)
                self.assertEqual(self.analyzer.calls, [])

    def test_end_within_slack_of_duration_is_accepted(self) -> None:
        """An end up to 0.1 s past the probed duration is still accepted."""
        response = self.client.get("/motion/src1", params={"start": "99", "end": "100.05"})
        self.assertEqual(response.status_code, 200)

    def test_media_error_from_analyzer_returns_422(self) -> None:
        """A MediaError raised during analysis is reported as 422 with its message."""
        client = self._client(FakeAnalyzer(error=MediaError("no frames decoded")))
        response = client.get("/motion/src1", params={"start": "0", "end": "2"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "no frames decoded")

    def test_media_error_from_duration_probe_returns_422(self) -> None:
        """A MediaError while probing the duration is reported as 422, not as a server error."""

        def broken_probe(path: Path) -> float:
            """Simulate ffprobe failing."""
            raise MediaError("ffprobe failed")

        app = FastAPI()
        resolver = media_dir_resolver(Path(self._temp.name))
        app.include_router(make_router(self.analyzer, resolver, broken_probe))
        response = TestClient(app).get("/motion/src1", params={"start": "0", "end": "2"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.analyzer.calls, [])

    def test_create_app_mounts_motion_route(self) -> None:
        """create_app builds an app that exposes the motion route."""
        with tempfile.TemporaryDirectory() as data_dir:
            app = create_app(Path(data_dir))
        self.assertIn("/motion/{source_id}", app.openapi()["paths"])


@unittest.skipUnless(HAS_HTTP_STACK, "fastapi and httpx are required")
class MediaDirResolverTest(unittest.TestCase):
    """media_dir_resolver maps ids to the first existing file with a known suffix."""

    def setUp(self) -> None:
        """Create a temporary media directory with a few candidate files."""
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name)
        (self.directory / "a.mkv").write_bytes(b"x")
        (self.directory / "a.mp4").write_bytes(b"x")
        (self.directory / "b.webm").write_bytes(b"x")
        (self.directory / "c.txt").write_bytes(b"x")

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        self._temp.cleanup()

    def test_returns_first_existing_candidate_in_suffix_order(self) -> None:
        """With both a.mkv and a.mp4 present, the default suffix order picks .mp4 first."""
        resolve = media_dir_resolver(self.directory)
        self.assertEqual(resolve("a"), self.directory / "a.mp4")
        self.assertEqual(resolve("b"), self.directory / "b.webm")

    def test_missing_or_unsupported_ids_resolve_to_none(self) -> None:
        """Unknown ids, unsupported suffixes and path-like ids all resolve to None."""
        resolve = media_dir_resolver(self.directory)
        self.assertIsNone(resolve("c"))
        self.assertIsNone(resolve("missing"))
        self.assertIsNone(resolve("../a"))

    def test_custom_suffixes_are_honoured(self) -> None:
        """Passing suffixes restricts the search to them."""
        resolve = media_dir_resolver(self.directory, suffixes=(".mkv",))
        self.assertEqual(resolve("a"), self.directory / "a.mkv")
        self.assertIsNone(resolve("b"))


if __name__ == "__main__":
    unittest.main()
