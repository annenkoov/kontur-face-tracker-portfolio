"""Small, deterministic face crop variants for fair offline comparisons."""

from __future__ import annotations

import numpy as np
import cv2


PAD_RATIO = 0.15
PADDED_CROP_ID = "mesh-bbox+15pct-context-v1"


def padded_face_crop(frame: np.ndarray, face_box: tuple[int, int, int, int],
                     padding: float = PAD_RATIO) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return an expanded crop plus the face bbox in crop-local coordinates."""
    if not 0 <= padding <= 0.5:
        raise ValueError("padding must be between 0 and 0.5")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in face_box)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError("Face box is outside the frame or empty")
    pad_x = round((x2 - x1) * padding)
    pad_y = round((y2 - y1) * padding)
    px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    px2, py2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    crop = np.ascontiguousarray(frame[py1:py2, px1:px2])
    local_box = (x1 - px1, y1 - py1, x2 - px1, y2 - py1)
    return crop, local_box


def crop_variants(image: np.ndarray, preprocessing: str,
                  face_box: tuple[int, int, int, int] | None = None) -> dict[str, np.ndarray]:
    """Produce stored-context and tight-face crops when context is available."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty three-channel image")
    variants = {"stored": image}
    if "+15pct-context" in preprocessing:
        height, width = image.shape[:2]
        # A manifest box handles padding clipped by the edge of the source frame.
        # Reconstruct symmetric margins only for samples from the first build.
        x1, y1, x2, y2 = face_box or _expected_face_box(image.shape)
        height, width = image.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Stored face box is outside its crop")
        if x2 > x1 and y2 > y1:
            key = "tight" if face_box is not None else "tight_approx"
            variants[key] = np.ascontiguousarray(image[y1:y2, x1:x2])
    return variants


def _expected_face_box(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Approximate symmetric crop for legacy samples without saved geometry."""
    height, width = shape[:2]
    x_margin = max(0, round(width * PAD_RATIO / (1 + 2 * PAD_RATIO)))
    y_margin = max(0, round(height * PAD_RATIO / (1 + 2 * PAD_RATIO)))
    return x_margin, y_margin, width - x_margin, height - y_margin


def to_model_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a decoded OpenCV BGR image to the RGB order used by live inference."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Expected a non-empty three-channel BGR image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
