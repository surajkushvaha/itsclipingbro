"""HTTP routes for the motion analyzer. FastAPI is optional: this module is imported only when the API is used.

Route:
    GET /motion/{source_id}?start=<float>&end=<float>  -> the motion payload of that time range

Validation failures return 400 with a human-readable message. Media and analysis errors
(clipbro.core.errors.MediaError) return 422. The analyzer is typed structurally so this layer
imports only clipbro.core and clipbro.bootstrap, as ARCHITECTURE section 1 requires.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from fastapi import APIRouter, FastAPI, HTTPException

from clipbro import __version__
from clipbro.bootstrap import build_analyzer, default_media_duration
from clipbro.core.errors import MediaError
from clipbro.core.types import TimeRange

MAX_ID_LENGTH = 64
MAX_RANGE_SECONDS = 600.0
DURATION_SLACK_SECONDS = 0.1
MEDIA_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm")


class MotionRunner(Protocol):
    """The part of MotionAnalyzer the routes use: analyze(media, rng) returning an object with .payload."""

    def analyze(self, media: Path, rng: TimeRange) -> Any:
        """Analyse rng of media and return a result carrying the payload."""
        ...


def _bad_request(message: str) -> HTTPException:
    """Build a 400 error carrying a human-readable message."""
    return HTTPException(status_code=400, detail=message)


def _media_error(error: MediaError) -> HTTPException:
    """Build a 422 error from a media or analysis failure."""
    return HTTPException(status_code=422, detail=str(error))


def _check_source_id(source_id: str) -> None:
    """Reject ids that are not alphanumeric or are too long before they reach the resolver."""
    if not source_id.isalnum() or len(source_id) > MAX_ID_LENGTH:
        raise _bad_request(f"source id must be alphanumeric and at most {MAX_ID_LENGTH} characters")


def _parse_seconds(name: str, raw: str | None) -> float:
    """Parse a start or end query value into a finite float, or raise 400."""
    if raw is None:
        raise _bad_request(f"{name} is required, in seconds")
    try:
        value = float(raw)
    except ValueError as error:
        raise _bad_request(f"{name} must be a number of seconds") from error
    if not math.isfinite(value):
        raise _bad_request(f"{name} must be a finite number of seconds")
    return value


def _check_bounds(start: float, end: float) -> None:
    """Check order, sign and length of the range without touching the media."""
    if start < 0:
        raise _bad_request("start must be 0 or greater")
    if end <= start:
        raise _bad_request("end must be greater than start")
    if end - start > MAX_RANGE_SECONDS:
        raise _bad_request(f"range must be at most {MAX_RANGE_SECONDS:g} seconds")


def _resolve_source(source_id: str, resolve_media: Callable[[str], Path | None]) -> Path:
    """Return the media file for source_id, or raise 400 when the id is invalid or no file exists."""
    _check_source_id(source_id)
    found = resolve_media(source_id)
    if found is None or not Path(found).is_file():
        raise _bad_request(f"no media found for source {source_id}")
    return Path(found)


def _check_duration(end: float, media: Path, duration_of: Callable[[Path], float]) -> None:
    """Raise 400 when end lies beyond the media length plus a small slack. Probe errors become 422."""
    try:
        duration = duration_of(media)
    except MediaError as error:
        raise _media_error(error) from error
    if end > duration + DURATION_SLACK_SECONDS:
        raise _bad_request(f"end {end:g} s is beyond the media length of {duration:.1f} s")


def make_router(
    analyzer: MotionRunner,
    resolve_media: Callable[[str], Path | None],
    duration_of: Callable[[Path], float],
) -> APIRouter:
    """Build the router that serves GET /motion/{source_id} for the given analyzer and media lookup."""
    router = APIRouter()

    @router.get("/motion/{source_id}")
    def motion(source_id: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Return the motion payload of the time range [start, end] seconds of the source.

        Declared as a plain def so FastAPI runs the analysis in its threadpool.
        """
        path = _resolve_source(source_id, resolve_media)
        begin = _parse_seconds("start", start)
        finish = _parse_seconds("end", end)
        _check_bounds(begin, finish)
        _check_duration(finish, path, duration_of)
        try:
            return analyzer.analyze(path, TimeRange(begin, finish)).payload
        except MediaError as error:
            raise _media_error(error) from error

    return router


def media_dir_resolver(
    directory: Path,
    suffixes: tuple[str, ...] = MEDIA_SUFFIXES,
) -> Callable[[str], Path | None]:
    """Return a resolver that maps a source id to the first existing directory/(id + suffix).

    Ids that are not alphanumeric resolve to None, so they can never escape the directory.
    """
    base = Path(directory)

    def resolve(source_id: str) -> Path | None:
        """Return the media file of source_id, or None when no candidate exists."""
        if not source_id.isalnum():
            return None
        for suffix in suffixes:
            candidate = base / f"{source_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    return resolve


def create_app(data_dir: Path = Path("data")) -> FastAPI:
    """Build the motion API: analyzer from bootstrap, media read from data_dir / "media".

    Run with: uvicorn clipbro.api.motion_routes:create_app --factory
    """
    data = Path(data_dir)
    app = FastAPI(title="clipbro motion API", version=__version__)
    analyzer = build_analyzer(data_dir=data)
    resolver = media_dir_resolver(data / "media")
    app.include_router(make_router(analyzer, resolver, default_media_duration))
    return app
