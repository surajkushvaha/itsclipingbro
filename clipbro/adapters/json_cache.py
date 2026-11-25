"""File-system cache: JSON payloads and NumPy arrays stored under one directory (Cache port)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

_KEY_PATTERN = re.compile(r"^[0-9a-f]{16,64}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,32}$")


def _check_key(key: str) -> None:
    """Raise ValueError unless key is a lowercase hex string of 16 to 64 characters."""
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError(f"cache key must match {_KEY_PATTERN.pattern}, got {key!r}")


def _check_name(name: str) -> None:
    """Raise ValueError unless name is a short alphanumeric array name."""
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"array name must match {_NAME_PATTERN.pattern}, got {name!r}")


def _atomic_write(target: Path, write: Callable[[Path], None]) -> None:
    """Write to a temporary sibling file via write(), then atomically move it onto target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        write(temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


class JsonFileCache:
    """Stores '<key>.json' payloads and '<key>.<name>.npy' arrays in one directory."""

    def __init__(self, directory: Path) -> None:
        """Remember the cache directory; it is created on first write."""
        self.directory = Path(directory)

    def _json_path(self, key: str) -> Path:
        """Return the JSON file path for a validated key."""
        _check_key(key)
        return self.directory / f"{key}.json"

    def _array_path(self, key: str, name: str) -> Path:
        """Return the .npy file path for a validated key and array name."""
        _check_key(key)
        _check_name(name)
        return self.directory / f"{key}.{name}.npy"

    def get(self, key: str) -> dict | None:
        """Return the stored JSON object for key, or None if missing or corrupt."""
        path = self._json_path(key)
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict) -> None:
        """Atomically store a JSON-serialisable dict under key."""
        path = self._json_path(key)
        text = json.dumps(value, separators=(",", ":"), allow_nan=False)

        def write(temp: Path) -> None:
            """Write the serialised payload to the temporary file."""
            temp.write_text(text, encoding="utf-8")

        _atomic_write(path, write)

    def put_array(self, key: str, name: str, array: np.ndarray) -> None:
        """Atomically store a NumPy array under key and name."""
        path = self._array_path(key, name)

        def write(temp: Path) -> None:
            """Save the array through an open handle so numpy does not append '.npy'."""
            with temp.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)

        _atomic_write(path, write)

    def get_array(self, key: str, name: str) -> np.ndarray | None:
        """Return the stored array for key and name, or None if missing or corrupt."""
        path = self._array_path(key, name)
        if not path.is_file():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
