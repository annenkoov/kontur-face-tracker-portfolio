"""Face landmarks and expression probabilities from one in-memory frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tempfile
import threading
import time
from urllib.request import urlretrieve
import zipfile

import cv2
import mediapipe as mp
import numpy as np
import onnx
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from .action_units import PyFeatActionUnitAnalyzer


LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
EMOTION_MODEL_NAME = "enet_b0_8_best_vgaf"
EMOTION_MODEL_VERSION = f"HSEmotion/{EMOTION_MODEL_NAME} · hsemotion-onnx 0.3.1"
EMOTION_MODEL_URL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/"
    "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx?raw=true"
)
FERPLUS_MODEL_NAME = "emotion-ferplus-7.onnx"
FERPLUS_MODEL_URL = (
    "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-7.onnx"
)
FERPLUS_LABELS = (
    "Neutral", "Happiness", "Surprise", "Sadness",
    "Anger", "Disgust", "Fear", "Contempt",
)
BLENDSHAPE_KEYS = {
    "brow_down": ("browDownLeft", "browDownRight"),
    "brow_inner_up": ("browInnerUp",),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "jaw_open": ("jawOpen",),
    "mouth_frown": ("mouthFrownLeft", "mouthFrownRight"),
    "mouth_press": ("mouthPressLeft", "mouthPressRight"),
}
PATTERN_BLENDSHAPE_KEYS = {
    "brow_down": ("browDownLeft", "browDownRight"),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "mouth_press": ("mouthPressLeft", "mouthPressRight"),
    "mouth_frown": ("mouthFrownLeft", "mouthFrownRight"),
    "jaw_open": ("jawOpen",),
    "eye_wide": ("eyeWideLeft", "eyeWideRight"),
    "upper_lip_raise": ("mouthUpperUpLeft", "mouthUpperUpRight"),
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
}
PATTERN_ASYMMETRY_KEYS = {
    "brow_asymmetry": ("browDownLeft", "browDownRight"),
    "eye_squint_asymmetry": ("eyeSquintLeft", "eyeSquintRight"),
    "mouth_press_asymmetry": ("mouthPressLeft", "mouthPressRight"),
}
REQUIRED_BLENDSHAPE_KEYS = frozenset(
    key
    for groups in (BLENDSHAPE_KEYS.values(), PATTERN_BLENDSHAPE_KEYS.values(),
                   PATTERN_ASYMMETRY_KEYS.values())
    for group in groups
    for key in group
)
EXPRESSION_SIGNAL_WEIGHTS = {
    "Anger": 1.0,
    "Fear": 0.9,
    "Sadness": 0.7,
    "Disgust": 0.55,
    "Contempt": 0.5,
    "Neutral": 0.15,
    "Happiness": 0.1,
    "Surprise": 0.25,
}
FACIAL_CONTOUR_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
)


def has_complete_blendshapes(scores: dict[str, float]) -> bool:
    """Return whether every blendshape used by features/patterns has a valid score."""
    if not REQUIRED_BLENDSHAPE_KEYS.issubset(scores):
        return False
    try:
        return all(np.isfinite(float(scores[key])) and 0 <= float(scores[key]) <= 1
                   for key in REQUIRED_BLENDSHAPE_KEYS)
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class FaceObservation:
    timestamp: float
    present: bool
    quality: float
    quality_reason: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    expressions: dict[str, float] = field(default_factory=dict)
    face_box: tuple[int, int, int, int] | None = None
    patterns: dict[str, float] = field(default_factory=dict)
    monotonic_timestamp: float | None = None
    sharpness: float | None = None
    saturated_pixel_fraction: float | None = None
    mean_face_brightness: float | None = None
    # Интенсивность предсказания AU из Py-Feat (0–1), отдельная от blendshape.
    action_units: dict[str, float] = field(default_factory=dict)
    action_units_status: str = "Ожидание пригодного лица"
    # Нормализованные MediaPipe x/y для живой визуализации; не являются стресс-метрикой.
    landmarks: tuple[tuple[float, float], ...] = ()
    # Второй, готовый классификатор FER+; не участвует в индексе состояния.
    secondary_expressions: dict[str, float] = field(default_factory=dict)
    # Средний Eye Aspect Ratio по MediaPipe landmark-ам глаз; сырая геометрия,
    # не выражение эмоции и не часть индекса состояния.
    eye_aspect_ratio: float | None = None


class _LatestActionUnitWorker:
    """Run optional AU inference off the camera-analysis thread, keeping one latest crop."""

    def __init__(self, analyzer_factory=PyFeatActionUnitAnalyzer) -> None:
        self._analyzer_factory = analyzer_factory
        self._condition = threading.Condition()
        self._pending_crop: np.ndarray | None = None
        self._latest: dict[str, float] = {}
        self._device = "CPU"
        self._error: str | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="action-unit-inference", daemon=True)
        self._thread.start()

    def submit(self, face_rgb: np.ndarray) -> None:
        # Копируем только один crop; новый кадр заменяет ожидающий старый.
        crop = np.ascontiguousarray(face_rgb).copy()
        with self._condition:
            if self._stopping or self._error is not None:
                return
            self._pending_crop = crop
            self._condition.notify()

    def snapshot(self) -> tuple[dict[str, float], str | None, str]:
        with self._condition:
            return dict(self._latest), self._error, self._device

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending_crop = None
            self._condition.notify_all()
        # Не задерживаем остановку камеры, если сторонний инференс ещё завершается.
        if self._thread.is_alive():
            self._thread.join(timeout=0.25)

    def _run(self) -> None:
        analyzer = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or self._pending_crop is not None)
                if self._stopping:
                    return
                crop = self._pending_crop
                self._pending_crop = None
            try:
                if analyzer is None:
                    # Пакет, GPU и веса Py-Feat инициализируются только здесь,
                    # отдельно от потока захвата/основного классификатора.
                    analyzer = self._analyzer_factory()
                    self._device = "GPU CUDA" if analyzer.device == "cuda" else "CPU"
                result = analyzer.analyze(crop) or {}
                with self._condition:
                    self._latest = dict(result)
            except Exception as exc:
                with self._condition:
                    self._error = str(exc)
                    self._pending_crop = None
                return


class FaceAnalyzer:
    def __init__(self, model_dir: str | Path = "models") -> None:
        model_path = Path(model_dir) / "face_landmarker.task"
        _download_atomically(LANDMARKER_URL, model_path, 1024 * 1024, _validate_task)
        ensure_emotion_model(EMOTION_MODEL_NAME)
        self._emotion_model = HSEmotionRecognizer(model_name=EMOTION_MODEL_NAME)
        self._ferplus_session = None
        try:
            import onnxruntime as ort

            ferplus_path = ensure_ferplus_model()
            session_options = ort.SessionOptions()
            # Один поток инференса, чтобы не отнимать все CPU-ядра у камеры и UI.
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            self._ferplus_session = ort.InferenceSession(
                str(ferplus_path), sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._ferplus_input = self._ferplus_session.get_inputs()[0].name
        except Exception as exc:
            # Если загрузка второго движка не удалась, остаётся HSEmotion.
            self._ferplus_session = None
            self._ferplus_error = str(exc)
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve()))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        # Ошибка загрузки AU-стека не отключает базовые эмоции/мимику.
        self._au_worker: _LatestActionUnitWorker | None = None
        self._au_unavailable_reason: str | None = None

    @property
    def emotion_models_status(self) -> str:
        if self._ferplus_session is not None:
            return "HSEmotion + FER+ · 2 локальные модели"
        return "HSEmotion · FER+ недоступна, работает запасная модель"

    def _predict_ferplus(self, face_rgb: np.ndarray) -> dict[str, float]:
        """Оценить выражение готовой FER+ CNN на сером crop 64×64."""
        if self._ferplus_session is None:
            return {}
        try:
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            tensor = resized.astype(np.float32, copy=False)[None, None, :, :]
            logits = self._ferplus_session.run(None, {self._ferplus_input: tensor})[0]
            probabilities = softmax_probabilities(np.asarray(logits).reshape(-1), FERPLUS_LABELS)
            return probabilities or {}
        except Exception:
            # FER+ дополнительная модель: её сбой не должен останавливать поток камеры.
            return {}

    def _analyze_action_units(self, face_rgb: np.ndarray) -> dict[str, float]:
        """Queue the newest crop and return the last available AU result immediately."""
        if self._au_unavailable_reason is not None:
            return {}
        if self._au_worker is None:
            self._au_worker = _LatestActionUnitWorker()
        self._au_worker.submit(face_rgb)
        values, error, _device = self._au_worker.snapshot()
        if error:
            self._au_unavailable_reason = error
            self._au_worker.close()
            self._au_worker = None
            return {}
        return values

    def analyze(self, frame: np.ndarray, timestamp: float | None = None,
                monotonic_timestamp: float | None = None) -> FaceObservation:
        now = float(timestamp if timestamp is not None else time.time())
        monotonic_now = float(monotonic_timestamp if monotonic_timestamp is not None else time.monotonic())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(monotonic_now * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        faces = result.face_landmarks or []
        if len(faces) != 1:
            reason = "Лицо не найдено" if not faces else "В кадре должно быть только одно лицо"
            return FaceObservation(now, bool(faces), 0.0, reason, monotonic_timestamp=monotonic_now)

        landmarks = faces[0]
        landmark_points = tuple((float(point.x), float(point.y)) for point in landmarks)
        if (not landmark_points or
                any(not np.isfinite(x) or not np.isfinite(y) for x, y in landmark_points)):
            return FaceObservation(now, True, 0.0, "Некорректные ключевые точки",
                                   monotonic_timestamp=monotonic_now)
        height, width = frame.shape[:2]
        xs = np.array([p.x for p in landmarks])
        ys = np.array([p.y for p in landmarks])
        x1, x2 = np.clip([xs.min(), xs.max()], 0, 1)
        y1, y2 = np.clip([ys.min(), ys.max()], 0, 1)
        box = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
        face_fraction = max(0.0, (x2 - x1) * (y2 - y1))
        if face_fraction < 0.045:
            return FaceObservation(now, True, 0.1, "Подойди ближе к камере", face_box=box,
                                   monotonic_timestamp=monotonic_now, landmarks=landmark_points)

        matrices = result.facial_transformation_matrixes or []
        if not matrices:
            return FaceObservation(now, True, 0.0, "Не удалось оценить ракурс", face_box=box,
                                   monotonic_timestamp=monotonic_now, landmarks=landmark_points)
        try:
            yaw, pitch, roll = face_pose_degrees(np.asarray(matrices[0]))
        except (TypeError, ValueError, IndexError, OverflowError, np.linalg.LinAlgError):
            return FaceObservation(now, True, 0.0, "Не удалось надёжно оценить ракурс", face_box=box,
                                   monotonic_timestamp=monotonic_now, landmarks=landmark_points)
        pose_score, pose_reason = pose_quality(yaw, pitch, roll)
        if pose_reason:
            return FaceObservation(now, True, 0.0, pose_reason, face_box=box,
                                   monotonic_timestamp=monotonic_now, landmarks=landmark_points)

        blendshape_items = result.face_blendshapes[0] if result.face_blendshapes else []
        scores = {item.category_name: float(item.score) for item in blendshape_items}
        if not has_complete_blendshapes(scores):
            return FaceObservation(now, True, 0.0, "Не удалось получить полные мимические признаки",
                                   face_box=box, monotonic_timestamp=monotonic_now,
                                   landmarks=landmark_points)
        features = {
            name: float(np.mean([scores[key] for key in keys]))
            for name, keys in BLENDSHAPE_KEYS.items()
        }
        patterns = {
            name: float(np.mean([scores[key] for key in keys]))
            for name, keys in PATTERN_BLENDSHAPE_KEYS.items()
        }
        patterns.update({
            name: abs(scores[left] - scores[right])
            for name, (left, right) in PATTERN_ASYMMETRY_KEYS.items()
        })
        patterns["brow_eye_coactivation"] = min(patterns["brow_down"], patterns["eye_squint"])
        patterns["mouth_coactivation"] = min(patterns["mouth_press"], patterns["mouth_frown"])
        crop = rgb[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            return FaceObservation(now, True, 0.0, "Не удалось выделить лицо", face_box=box,
                                   monotonic_timestamp=monotonic_now, landmarks=landmark_points)
        gray = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
        sharpness = face_sharpness(gray)
        face_crop_bgr = frame[box[1]:box[3], box[0]:box[2]]
        saturated_fraction = saturated_pixel_fraction(face_crop_bgr)
        labels = list(self._emotion_model.idx_to_class.values())
        _, logits = self._emotion_model.predict_emotions(crop, logits=True)
        expressions = softmax_probabilities(logits, labels)
        if expressions is None:
            return FaceObservation(now, True, 0.0, "Модель вернула некорректный ответ",
                                   face_box=box, patterns=patterns, monotonic_timestamp=monotonic_now,
                                   landmarks=landmark_points)
        secondary_expressions = self._predict_ferplus(crop)

        brightness = float(np.mean(gray))
        quality = pose_score * min(1.0, face_fraction / 0.12) * min(1.0, brightness / 65.0)
        reason = "Недостаточно света" if brightness < 35 else None
        if reason:
            quality = 0.0
        action_units = self._analyze_action_units(crop)
        eye_aspect_ratio = calculate_eye_aspect_ratio(landmark_points)
        _au_values, au_error, au_device = (
            self._au_worker.snapshot() if self._au_worker is not None else ({}, None, "CPU")
        )
        au_status = (f"Готово · {au_device} · модельная активность 0–1" if action_units else
                     (f"AU недоступны: {self._au_unavailable_reason or au_error}"
                      if self._au_unavailable_reason or au_error else
                      "AU-модель загружается в фоне"))
        return FaceObservation(now, True, quality, reason, features, expressions, box, patterns,
                               monotonic_now, sharpness, saturated_fraction, brightness,
                               action_units, au_status, landmark_points, secondary_expressions,
                               eye_aspect_ratio=eye_aspect_ratio)

    def close(self) -> None:
        self._landmarker.close()
        if self._au_worker is not None:
            self._au_worker.close()
            self._au_worker = None
        self._ferplus_session = None


def face_pose_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return approximate yaw, pitch, roll from a MediaPipe 4x4 transform."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Expected a finite 4x4 facial transformation matrix")
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(determinant) or abs(determinant) < 1e-6:
        raise ValueError("Facial rotation matrix is singular")
    yaw = np.degrees(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return tuple(round(float(value), 2) for value in (yaw, pitch, roll))


def calculate_eye_aspect_ratio(landmarks: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> float | None:
    """Calculate the average EAR from standard MediaPipe Face Mesh eye points.

    Returns a unitless geometric measurement. Consumers may derive a closed-eye
    state with temporal hysteresis; the measurement itself is not an emotion.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] <= 387:
        return None
    if not np.all(np.isfinite(points[:, :2])):
        return None
    # (outer corner, upper outer, upper inner, inner corner, lower inner, lower outer)
    eye_indices = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
    ratios = []
    for corner_a, upper_a, upper_b, corner_b, lower_b, lower_a in eye_indices:
        horizontal = float(np.linalg.norm(points[corner_a, :2] - points[corner_b, :2]))
        if horizontal <= 1e-8:
            continue
        vertical_a = float(np.linalg.norm(points[upper_a, :2] - points[lower_a, :2]))
        vertical_b = float(np.linalg.norm(points[upper_b, :2] - points[lower_b, :2]))
        ratios.append((vertical_a + vertical_b) / (2.0 * horizontal))
    if not ratios:
        return None
    value = float(np.mean(ratios))
    return value if np.isfinite(value) else None


def pose_quality(yaw: float, pitch: float, roll: float) -> tuple[float, str | None]:
    if not all(np.isfinite(value) for value in (yaw, pitch, roll)):
        return 0.0, "Не удалось надёжно оценить ракурс"
    if abs(yaw) > 25 or abs(pitch) > 20 or abs(roll) > 25:
        return 0.0, "Повернись лицом к камере"
    return 1.0, None


def face_sharpness(gray_face: np.ndarray, size: int = 128) -> float | None:
    """Variance of Laplacian on a normalized face crop; pixels are never retained."""
    image = np.asarray(gray_face)
    if image.ndim != 2 or image.size == 0 or size < 16:
        return None
    interpolation = cv2.INTER_AREA if max(image.shape) > size else cv2.INTER_LINEAR
    normalized = cv2.resize(image, (size, size), interpolation=interpolation)
    value = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    return value if np.isfinite(value) and value >= 0 else None


def saturated_pixel_fraction(face_bgr: np.ndarray) -> float | None:
    """Report the fraction of crop pixels with any channel at uint8 maximum.

    This is a display-only exposure diagnostic; it is not a quality gate or stress signal.
    """
    image = np.asarray(face_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8:
        return None
    return float(np.any(image == np.iinfo(np.uint8).max, axis=2).mean())


def softmax_probabilities(logits: np.ndarray, labels: list[str] | tuple[str, ...]) -> dict[str, float] | None:
    """Convert one finite logit per unique class to a stable normalized mapping."""
    values = np.asarray(logits, dtype=np.float64)
    if (values.ndim != 1 or len(values) != len(labels) or not labels or
            len(set(labels)) != len(labels) or not np.all(np.isfinite(values))):
        return None
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    total = float(exponentials.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return {label: float(exponentials[index] / total) for index, label in enumerate(labels)}


def consensus_expression_probabilities(observation: FaceObservation) -> dict[str, float]:
    """Average the ready-made classifiers for display; leave scoring input unchanged."""
    primary = observation.expressions or {}
    secondary = observation.secondary_expressions or {}
    if not secondary:
        return dict(primary)
    common = set(primary).intersection(secondary)
    if not common:
        return dict(primary)
    # Обе модели возвращают нормированные оценки одних и тех же восьми классов.
    return {label: (float(primary[label]) + float(secondary[label])) / 2.0
            for label in common}


def _validate_onnx(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)


def ensure_emotion_model(model_name: str) -> Path:
    """Atomically download and validate one supported HSEmotion ONNX model."""
    from .model_registry import get_model_spec

    get_model_spec(model_name)
    path = Path.home() / ".hsemotion" / f"{model_name}.onnx"
    url = ("https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/"
           f"models/affectnet_emotions/onnx/{model_name}.onnx?raw=true")
    _download_atomically(url, path, 10 * 1024 * 1024, _validate_onnx)
    return path


def ensure_ferplus_model() -> Path:
    """Ensure the ready-made FER+ weights exist in Kontur's local model cache."""
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    path = local_app_data / "Kontur" / "models" / FERPLUS_MODEL_NAME
    _download_atomically(FERPLUS_MODEL_URL, path, 30 * 1024 * 1024, _validate_onnx)
    return path


def _validate_task(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError("Файл Face Landmarker не является корректным архивом модели")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Архив Face Landmarker повреждён")


def _download_atomically(url: str, destination: Path, minimum_bytes: int, validator) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if destination.stat().st_size >= minimum_bytes:
                validator(destination)
                return
        except Exception:
            pass
        destination.unlink(missing_ok=True)

    fd, temporary_name = tempfile.mkstemp(prefix=f"{destination.stem}-", suffix=".part", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        urlretrieve(url, temporary_path)
        if temporary_path.stat().st_size < minimum_bytes:
            raise RuntimeError(f"Загруженный файл модели слишком мал: {destination.name}")
        validator(temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
