"""Offline OpenCV Zoo expression candidate for locally labeled face crops."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
import onnx


MODEL_FILENAME = "facial_expression_recognition_mobilefacenet_2022july.onnx"
MODEL_SIZE_BYTES = 4_791_892
MODEL_SHA256 = "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"
MODEL_URL = (
    "https://huggingface.co/opencv/facial_expression_recognition/resolve/"
    f"b473270576eabddaacffd08adb4bed27b3fc54e6/{MODEL_FILENAME}"
)

# Keep the names and output order from OpenCV Zoo's facial_fer_model.py.
NATIVE_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
ALIGNMENT = "MediaPipe five points / OpenCV Zoo 112x112 affine template"
# OpenCV Zoo's reference locations: subject's right eye, left eye, nose tip,
# right mouth corner, left mouth corner (see demo.cpp and the research note).
_TEMPLATE = np.array([
    (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
    (41.5493, 92.3655), (70.7299, 92.2041),
], dtype=np.float32)
_SIZE = (112, 112)


def _verified(weights: bytes) -> bool:
    return len(weights) == MODEL_SIZE_BYTES and hashlib.sha256(weights).hexdigest() == MODEL_SHA256


def load_candidate_weights() -> bytes:
    """Reuse a verified cache or fetch the pinned artifact into it atomically."""
    base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    path = base / "Kontur" / "model-cache" / MODEL_FILENAME
    try:
        with path.open("rb") as cached:
            weights = cached.read(MODEL_SIZE_BYTES + 1)
        if _verified(weights):
            return weights
    except FileNotFoundError:
        pass

    # The only network payload is the weight file. Bound the response even when
    # Content-Length is missing; never send a crop or a label to the host.
    with urlopen(MODEL_URL, timeout=30) as response:
        weights = response.read(MODEL_SIZE_BYTES + 1)
    if not _verified(weights):
        raise RuntimeError(f"OpenCV Zoo download failed size or SHA-256 verification: {MODEL_FILENAME}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="opencv-fer-", suffix=".part",
                                         delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(weights)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return weights


class AlignmentAbstention(ValueError):
    """The saved crop could not provide a single usable set of five points."""


class OpenCVZooFER:
    """CPU classifier; the caller supplies a preexisting Face Landmarker asset."""

    def __init__(self, weights: bytes, landmarker_path: Path):
        # Load the same verified bytes, including on Windows paths with non-ASCII
        # usernames. The ONNX graph tells us whether the scores already use Softmax.
        self.scores_are_probabilities = _has_softmax_output(weights)
        self._net = cv2.dnn.readNetFromONNX(np.frombuffer(weights, dtype=np.uint8))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._forward(np.zeros((112, 112, 3), dtype=np.uint8))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=landmarker_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        try:
            # Exclude first-call initialization from per-crop latency.
            self._detect(np.zeros((112, 112, 3), dtype=np.uint8))
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._landmarker.close()

    def _detect(self, rgb: np.ndarray):
        return self._landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
        ).face_landmarks

    def _forward(self, rgb: np.ndarray) -> np.ndarray:
        # The crop is already RGB. OpenCV packs the normalized HWC image into NCHW.
        normalized = rgb.astype(np.float32) * (2.0 / 255.0) - 1.0
        self._net.setInput(cv2.dnn.blobFromImage(normalized), "data")
        scores = np.asarray(self._net.forward("label")).reshape(-1)
        if scores.size != len(NATIVE_LABELS) or not np.all(np.isfinite(scores)):
            raise ValueError("OpenCV Zoo FER produced invalid class scores")
        return scores

    def predict_scores(self, rgb: np.ndarray) -> np.ndarray:
        faces = self._detect(rgb)
        if not faces:
            raise AlignmentAbstention("no_face_landmarks")
        if len(faces) != 1:
            raise AlignmentAbstention("multiple_faces")
        points = _five_points(faces[0], rgb.shape[1], rgb.shape[0])
        transform, inliers = cv2.estimateAffine2D(points, _TEMPLATE)
        if (transform is None or inliers is None or np.count_nonzero(inliers) < 3
                or not np.all(np.isfinite(transform)) or np.linalg.det(transform[:, :2]) <= 0):
            raise AlignmentAbstention("invalid_alignment")
        aligned = cv2.warpAffine(rgb, transform, _SIZE)
        return self._forward(aligned)


def _five_points(landmarks, width: int, height: int) -> np.ndarray:
    if len(landmarks) <= 362:
        raise AlignmentAbstention("invalid_face_landmarks")

    def xy(index: int) -> np.ndarray:
        point = landmarks[index]
        return np.array((point.x, point.y), dtype=np.float32)

    # Midpoints of the inner/outer eye corners approximate the detector's eye centers.
    points = np.array([
        (xy(33) + xy(133)) / 2, (xy(362) + xy(263)) / 2, xy(4), xy(61), xy(291),
    ], dtype=np.float32)
    if (not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1)
            or points[0, 0] >= points[1, 0] or points[3, 0] >= points[4, 0]):
        raise AlignmentAbstention("invalid_face_landmarks")
    return points * np.array((width, height), dtype=np.float32)


def _has_softmax_output(weights: bytes) -> bool:
    graph = onnx.load_model_from_string(weights).graph
    producers = {output: node for node in graph.node for output in node.output}
    node = producers.get("label")
    while node is not None and node.op_type in {"Identity", "Cast"}:
        node = producers.get(node.input[0])
    return node is not None and node.op_type == "Softmax"
