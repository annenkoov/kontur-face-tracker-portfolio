"""Evaluate labeled local frames using the production MediaPipe face crop path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EmotionEvaluation:
    labels: tuple[str, ...]
    per_class: dict[str, ClassMetrics]
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]


def evaluate_predictions(rows: Iterable[tuple[str, str]], labels: Iterable[str]) -> EmotionEvaluation:
    """Calculate metrics with explicit truth/prediction labels (never numeric class indices)."""
    names = tuple(labels)
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(x, str) for x in names):
        raise ValueError("labels must contain at least two unique class names")
    positions = {name: index for index, name in enumerate(names)}
    matrix = [[0 for _ in names] for _ in names]
    count = 0
    for truth, prediction in rows:
        if truth not in positions or prediction not in positions:
            raise ValueError("prediction row contains a missing or unsupported class")
        matrix[positions[truth]][positions[prediction]] += 1
        count += 1
    if not count:
        raise ValueError("at least one labeled prediction is required")
    metrics = {}
    for index, label in enumerate(names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = ClassMetrics(precision, recall, f1, support)
    return EmotionEvaluation(names, metrics, sum(x.f1 for x in metrics.values()) / len(names),
                             tuple(tuple(row) for row in matrix))


def top_two_summary(probabilities: dict[str, float], margin: float = 0.08) -> tuple[str, str, bool]:
    """Return top class, runner-up and whether their probabilities are too close to call."""
    if (len(probabilities) < 2 or not 0 <= margin <= 1 or
            any(not isinstance(k, str) or not isinstance(v, (int, float)) or not 0 <= v <= 1
                for k, v in probabilities.items())):
        raise ValueError("at least two named probabilities in [0, 1] are required")
    # Stable sort preserves model class ordering on ties; callers use the
    # model's label map, never an assumed numeric Sadness index.
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return ranked[0][0], ranked[1][0], ranked[0][1] - ranked[1][1] < margin


def _evaluate_manifest(manifest: Path) -> int:
    import cv2
    import mediapipe as mp
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    from .model_registry import MODEL_SPECS
    from .preprocessing import mediapipe_face_crop
    from .vision import create_face_landmarker, softmax_probabilities

    task_path = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
    if not task_path.is_file():
        raise FileNotFoundError(f"Local MediaPipe asset not found: {task_path}")
    landmarker = create_face_landmarker(task_path, mp.tasks.vision.RunningMode.IMAGE)
    try:
        rows = []
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                path = Path(record["path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame_bgr = cv2.imread(str(path))
                if frame_bgr is None:
                    raise ValueError(f"Cannot read local frame: {path}")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detection = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
                faces = detection.face_landmarks or []
                if len(faces) != 1:
                    raise ValueError(f"Expected exactly one face in local frame: {path}")
                face_crop, _ = mediapipe_face_crop(frame_rgb, faces[0])
                rows.append((face_crop, record["label"].strip()))
        if not rows:
            raise ValueError("manifest contains no labeled frames")
    finally:
        landmarker.close()

    classes = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
    print(f"Local labeled frames: {len(rows)}. Faces were cropped with the shared production MediaPipe "
          "landmark bbox helper. Use a person/session-separated holdout.")
    evaluated = 0
    for spec in MODEL_SPECS:
        try:
            model = HSEmotionRecognizer(model_name=spec.name)
        except Exception as exc:
            print(f"{spec.name}: skipped ({type(exc).__name__}: {exc})")
            continue
        predictions = []
        for face_rgb, truth in rows:
            _, logits = model.predict_emotions(face_rgb, logits=True)
            probabilities = softmax_probabilities(logits, list(model.idx_to_class.values()))
            if probabilities is None:
                raise ValueError(f"{spec.name} returned invalid class probabilities")
            predicted, _, _ = top_two_summary(probabilities)
            predictions.append((truth, predicted))
        result = evaluate_predictions(predictions, classes)
        sadness = result.per_class["Sadness"]
        print(f"{spec.name}: macro-F1={result.macro_f1:.3f}; Sadness P/R/F1="
              f"{sadness.precision:.3f}/{sadness.recall:.3f}/{sadness.f1:.3f}")
        evaluated += 1
    return evaluated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="local CSV path,label for full frames containing exactly one visible face")
    args = parser.parse_args()
    if _evaluate_manifest(args.manifest) == 0:
        print("No model was evaluated: every registered model was unavailable or failed to load.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
