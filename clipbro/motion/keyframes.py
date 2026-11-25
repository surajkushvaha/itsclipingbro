"""Versioned JSON persistence for camera keyframes and motion payloads.

Files are written atomically: a temp file in the target directory is renamed
over the destination, so readers never see a half-written document.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from clipbro.core.types import Event, FocusPoint, Keyframe

SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    """Recursively convert a payload value into JSON-safe built-in types."""
    if isinstance(value, (Keyframe, Event)):
        return _jsonable(value.to_dict())
    if isinstance(value, FocusPoint):
        return [float(value.t), float(value.cx), float(value.cy)]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write document as compact UTF-8 JSON to a temp file, then rename it onto path."""
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    replaced = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=None, separators=(",", ":"), ensure_ascii=False)
        os.replace(temp_name, path)
        replaced = True
    finally:
        if not replaced:
            with suppress(OSError):
                os.unlink(temp_name)


def save(path: Path, payload: dict) -> None:
    """Write payload to path as JSON with a leading `"version": 1` field.

    Keyframe, Event and FocusPoint objects anywhere in the payload are serialised.
    A payload that already carries a different version is rejected.
    """
    if "version" in payload and payload["version"] != SCHEMA_VERSION:
        raise ValueError(f"payload version must be {SCHEMA_VERSION}, got {payload['version']!r}")
    document = {"version": SCHEMA_VERSION, **_jsonable(payload)}
    _write_atomic(path, document)


def load(path: Path) -> dict:
    """Read a keyframe file and return its payload dict.

    Raises ValueError when the file is not a JSON object or its version is not 1.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    version = data.get("version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ValueError(f"unsupported keyframe schema version {version!r}, expected {SCHEMA_VERSION}")
    return data


def keyframes_from(payload: dict, key: str = "keyframes") -> list[Keyframe]:
    """Validate and rebuild the list of keyframes stored under key.

    Each entry goes through Keyframe.from_dict, so out-of-range values raise ValueError.
    """
    items = payload.get(key)
    if not isinstance(items, list):
        raise ValueError(f"payload[{key!r}] must be a list of keyframes")
    try:
        return [Keyframe.from_dict(item) for item in items]
    except (KeyError, TypeError) as error:
        raise ValueError(f"malformed keyframe in payload[{key!r}]: {error}") from error
