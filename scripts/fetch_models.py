"""Download the YuNet face detection model into models/, verifying its SHA-256.

Usage: python scripts/fetch_models.py
An existing file with the expected checksum is left untouched.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1 << 16
TIMEOUT_S = 60.0


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path) -> bool:
    """Return True when path exists and matches the expected checksum."""
    return path.is_file() and sha256_of(path) == MODEL_SHA256


def download(url: str, destination: Path) -> None:
    """Stream url into destination through a temporary file, then move it into place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response, temp.open("wb") as handle:
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                handle.write(chunk)
        if sha256_of(temp) != MODEL_SHA256:
            raise ValueError("downloaded model failed SHA-256 verification")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    """Fetch the model if needed and return a process exit code."""
    target = REPO_ROOT / "models" / MODEL_NAME
    if is_valid(target):
        print(f"already present and valid: {target}")
        return 0
    print(f"downloading {MODEL_URL}")
    try:
        download(MODEL_URL, target)
    except (OSError, ValueError) as error:
        print(f"failed: {error}", file=sys.stderr)
        return 1
    print(f"saved {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
