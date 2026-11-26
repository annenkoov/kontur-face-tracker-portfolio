"""Local classifier for explicit personal self-reports based on visible movement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import math
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .labels import LabeledSample


PERSONAL_LABELS = ("устал", "сосредоточен", "спокоен")
FEATURE_VERSION = "mediapipe-face-landmarker-v1-patterns-v1"
FEATURE_NAMES = (
    "brow_down", "eye_squint", "eye_wide", "jaw_open",
    "mouth_frown", "mouth_press", "smile", "upper_lip_raise",
)
MIN_FRAMES_PER_SESSION_LABEL = 2
MIN_TRAIN_SESSIONS = 3
MIN_VALIDATION_SESSIONS = 2
MIN_VALIDATION_RECALL = 0.75
MAX_DISTANCE = 0.18
MAX_DISTANCE_RATIO = 0.75

FeatureVector = tuple[float, ...]


def feature_vector(values: Mapping[str, float]) -> FeatureVector:
    """Validate the exact persisted schema and return values in stable model order."""
    if set(values) != set(FEATURE_NAMES):
        raise ValueError("Unexpected personal feature schema")
    vector = tuple(values[key] for key in FEATURE_NAMES)
    if not all(type(value) in (float, int) and 0.0 <= value <= 1.0 for value in vector):
        raise ValueError("Invalid personal feature value")
    return tuple(float(value) for value in vector)


def _predict(centers: dict[str, FeatureVector], vector: FeatureVector) -> str | None:
    distances = sorted(
        (math.sqrt(mean((value - target) ** 2 for value, target in zip(vector, center))), label)
        for label, center in centers.items()
    )
    (nearest, label), (runner_up, _) = distances[:2]
    if nearest > MAX_DISTANCE or runner_up == 0 or nearest >= MAX_DISTANCE_RATIO * runner_up:
        return None
    return label


@dataclass(frozen=True)
class PersonalFit:
    status: str
    reason: str
    sessions: int
    coverage: dict[str, tuple[int, int]]
    centers: dict[str, FeatureVector] | None = None

    def predict(self, vector: FeatureVector) -> str | None:
        return _predict(self.centers, vector) if self.centers is not None else None


def fit_personal_model(samples: Iterable[LabeledSample], *, active_session: str | None = None) -> PersonalFit:
    """Fit only completed capture groups, with a fixed, label-independent ~1/3 holdout.

    Each eligible session contributes equally within each label. Held-out
    abstentions count as misses and every label must meet the recall floor.
    The deployed centers remain the ones checked on holdout data.
    """
    grouped: dict[tuple[str, str], list[FeatureVector]] = defaultdict(list)
    for sample in samples:
        if (sample.label_kind != "personal" or sample.label not in PERSONAL_LABELS or
                sample.features is None or not sample.session_id or
                sample.session_id in ("unknown-session", active_session)):
            continue
        grouped[sample.session_id, sample.label].append(sample.features)

    train: dict[str, list[list[FeatureVector]]] = {label: [] for label in PERSONAL_LABELS}
    validation: dict[str, list[list[FeatureVector]]] = {label: [] for label in PERSONAL_LABELS}
    sessions = set()
    for (session, label), vectors in grouped.items():
        if len(vectors) < MIN_FRAMES_PER_SESSION_LABEL:
            continue
        sessions.add(session)
        # This assignment stays fixed as examples are added, edited or deleted.
        held_out = sha256(session.encode()).digest()[0] < 85
        (validation if held_out else train)[label].append(vectors)

    coverage = {label: (len(train[label]), len(validation[label])) for label in PERSONAL_LABELS}
    if any(a < MIN_TRAIN_SESSIONS or b < MIN_VALIDATION_SESSIONS for a, b in coverage.values()):
        return PersonalFit("Недостаточно примеров", "Собери больше примеров в разных сессиях.",
                           len(sessions), coverage)

    centers = {
        label: tuple(mean(mean(vector[i] for vector in vectors) for vectors in groups)
                     for i in range(len(FEATURE_NAMES)))
        for label, groups in train.items()
    }
    for label, groups in validation.items():
        recall = mean(mean(_predict(centers, vector) == label for vector in vectors) for vectors in groups)
        if recall < MIN_VALIDATION_RECALL:
            return PersonalFit("Не уверен", "На отдельных сессиях личные метки часто не совпадают.",
                               len(sessions), coverage)
    return PersonalFit("Проверено на отдельных сессиях", "", len(sessions), coverage, centers)
