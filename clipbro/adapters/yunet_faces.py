"""OpenCV YuNet face detector adapter (FaceDetector port)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from clipbro.core.types import Box

_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
_ENV_VAR = "CLIPBRO_YUNET"
_SCORE_COLUMN = 14
_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    """Return $CLIPBRO_YUNET if set, otherwise the model inside the repository's models/ folder."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override)
    return _REPO_ROOT / "models" / _MODEL_NAME


class YuNetFaceDetector:
    """Implements FaceDetector with cv2.FaceDetectorYN, returning normalised boxes."""

    def __init__(self, model_path: Path | None = None, score_threshold: float = 0.6, nms: float = 0.3) -> None:
        """Load the ONNX model; raises FileNotFoundError when the model file is missing."""
        self.model_path = Path(model_path) if model_path is not None else default_model_path()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YuNet model not found: {self.model_path}")
        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320), score_threshold, nms
        )

    def detect(self, image: np.ndarray) -> list[Box]:
        """Detect faces in an RGB uint8 image and return boxes normalised to 0..1."""
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return []
        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((int(width), int(height)))
        _, faces = self._detector.detect(bgr)
        if faces is None:
            return []
        return _to_boxes(faces, width, height)


def _to_boxes(faces: np.ndarray, width: int, height: int) -> list[Box]:
    """Convert raw YuNet rows (x, y, w, h, landmarks..., score) into clamped normalised boxes."""
    boxes: list[Box] = []
    for row in faces:
        left = min(1.0, max(0.0, float(row[0]) / width))
        top = min(1.0, max(0.0, float(row[1]) / height))
        right = min(1.0, max(0.0, (float(row[0]) + float(row[2])) / width))
        bottom = min(1.0, max(0.0, (float(row[1]) + float(row[3])) / height))
        if right <= left or bottom <= top:
            continue
        boxes.append(
            Box(x=left, y=top, w=right - left, h=bottom - top, score=float(row[_SCORE_COLUMN]))
        )
    return boxes
