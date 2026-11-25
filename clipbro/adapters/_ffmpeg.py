"""Private helpers that run the ffmpeg and ffprobe binaries for the adapters.

Every failure (missing binary, timeout, non-zero exit) is converted into
clipbro.core.errors.MediaError so callers only handle one exception type.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from clipbro.core.errors import MediaError


def run_binary(args: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    """Run a command, capturing stdout and stderr, and return the completed process.

    Raises MediaError when the binary cannot start, times out or exits non-zero.
    """
    try:
        completed = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except OSError as error:
        raise MediaError(f"cannot run {args[0]}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise MediaError(f"{args[0]} timed out after {timeout:.0f}s") from error
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError(f"{args[0]} failed ({completed.returncode}): {message[-500:]}")
    return completed


def probe_streams(
    ffprobe: str, media: Path, select: str, timeout: float, entries: str = "stream=index"
) -> list[dict[str, Any]]:
    """Return the ffprobe stream dicts matching select (for example 'v:0' or 'a')."""
    args = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        select,
        "-show_entries",
        entries,
        "-of",
        "json",
        str(media),
    ]
    completed = run_binary(args, timeout)
    try:
        payload = json.loads(completed.stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as error:
        raise MediaError(f"ffprobe returned invalid JSON for {media}") from error
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    return [stream for stream in streams if isinstance(stream, dict)]
