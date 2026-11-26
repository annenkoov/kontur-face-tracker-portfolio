"""Personal calibration and interpretable two-branch facial score."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
from statistics import median

from .vision import EXPRESSION_SIGNAL_WEIGHTS, FaceObservation


@dataclass(frozen=True)
class CalibrationProfile:
    medians: dict[str, float]
    scales: dict[str, float]
    pattern_medians: dict[str, float] = field(default_factory=dict)
    pattern_scales: dict[str, float] = field(default_factory=dict)
    sharpness_median: float | None = None
    sharpness_scale: float | None = None
    mean_brightness_median: float | None = None
    expression_baseline: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_layer(self.medians, self.scales, "признаков")
        self._validate_layer(self.pattern_medians, self.pattern_scales, "паттернов")
        if (self.sharpness_median is None) != (self.sharpness_scale is None):
            raise ValueError("Некорректная калибровка резкости: нужны медиана и масштаб")
        if self.sharpness_median is not None:
            center, scale = float(self.sharpness_median), float(self.sharpness_scale)
            if not math.isfinite(center) or center <= 0 or not math.isfinite(scale) or scale <= 0:
                raise ValueError("Некорректная калибровка резкости")
            object.__setattr__(self, "sharpness_median", center)
            object.__setattr__(self, "sharpness_scale", scale)
        if self.mean_brightness_median is not None:
            brightness = float(self.mean_brightness_median)
            if not math.isfinite(brightness) or not 0 <= brightness <= 255:
                raise ValueError("Некорректная медианная яркость калибровки")
            object.__setattr__(self, "mean_brightness_median", brightness)
        expression_baseline = {key: float(value) for key, value in self.expression_baseline.items()}
        if expression_baseline:
            if set(expression_baseline) != set(EXPRESSION_SIGNAL_WEIGHTS):
                raise ValueError("Некорректная калибровка распределения выражений")
            if any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in expression_baseline.values()):
                raise ValueError("Некорректные вероятности выражений в калибровке")
            total = math.fsum(expression_baseline.values())
            if total <= 0:
                raise ValueError("Пустое распределение выражений в калибровке")
            expression_baseline = {key: value / total for key, value in expression_baseline.items()}
        object.__setattr__(self, "expression_baseline", expression_baseline)
        object.__setattr__(self, "medians", {key: float(value) for key, value in self.medians.items()})
        object.__setattr__(self, "scales", {key: float(value) for key, value in self.scales.items()})
        object.__setattr__(self, "pattern_medians",
                           {key: float(value) for key, value in self.pattern_medians.items()})
        object.__setattr__(self, "pattern_scales",
                           {key: float(value) for key, value in self.pattern_scales.items()})

    @staticmethod
    def _validate_layer(medians: dict[str, float], scales: dict[str, float], label: str) -> None:
        if set(medians) != set(scales):
            raise ValueError(f"Некорректная калибровка {label}: медианы и масштабы не совпадают")
        try:
            centers = [float(value) for value in medians.values()]
            widths = [float(value) for value in scales.values()]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Некорректная калибровка {label}: требуются числа") from exc
        if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in centers) or
                any(not math.isfinite(value) or value <= 0 for value in widths)):
            raise ValueError(f"Некорректная калибровка {label}: значения вне допустимого диапазона")

    @classmethod
    def from_observations(cls, observations: list[FaceObservation]) -> "CalibrationProfile":
        usable = []
        for item in observations:
            try:
                quality = float(item.quality)
                feature_values = [float(value) for value in item.features.values()]
                pattern_values = [float(value) for value in item.patterns.values()]
            except (TypeError, ValueError, AttributeError):
                continue
            if (item.present and math.isfinite(quality) and quality >= 0.45 and quality <= 1 and item.features and
                    all(math.isfinite(value) and 0 <= value <= 1 for value in feature_values) and
                    all(math.isfinite(value) and 0 <= value <= 1 for value in pattern_values)):
                usable.append(item)
        if len(usable) < 5:
            raise ValueError("Для калибровки нужно не менее пяти надёжных наблюдений")
        common = set.intersection(*(set(item.features) for item in usable))
        if not common:
            raise ValueError("В наблюдениях нет общих мимических признаков")
        medians: dict[str, float] = {}
        scales: dict[str, float] = {}
        for key in sorted(common):
            values = [float(item.features[key]) for item in usable]
            center = median(values)
            mad = median([abs(value - center) for value in values])
            medians[key] = center
            scales[key] = max(1.4826 * mad, 0.04)
        pattern_common = set.intersection(*(set(item.patterns) for item in usable)) if all(item.patterns for item in usable) else set()
        pattern_medians: dict[str, float] = {}
        pattern_scales: dict[str, float] = {}
        for key in sorted(pattern_common):
            values = [float(item.patterns[key]) for item in usable]
            center = median(values)
            mad = median([abs(value - center) for value in values])
            pattern_medians[key] = center
            pattern_scales[key] = max(1.4826 * mad, 0.04)
        sharpness_values = []
        for item in usable:
            try:
                value = float(item.sharpness)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                sharpness_values.append(value)
        sharpness_median = sharpness_scale = None
        if len(sharpness_values) >= 5:
            sharpness_median = median(sharpness_values)
            sharpness_mad = median([abs(value - sharpness_median) for value in sharpness_values])
            sharpness_scale = max(1.4826 * sharpness_mad, sharpness_median * 0.05, 1e-3)
        brightness_values = []
        for item in usable:
            try:
                value = float(item.mean_face_brightness)
            except (TypeError, ValueError, AttributeError):
                continue
            if math.isfinite(value) and 0 <= value <= 255:
                brightness_values.append(value)
        mean_brightness_median = median(brightness_values) if len(brightness_values) >= 5 else None
        expected_labels = set(EXPRESSION_SIGNAL_WEIGHTS)
        expression_rows = []
        for item in usable:
            try:
                probabilities = {key: float(value) for key, value in item.expressions.items()}
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if (set(probabilities) != expected_labels or
                    any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values())):
                continue
            total = math.fsum(probabilities.values())
            if total > 0:
                expression_rows.append({key: value / total for key, value in probabilities.items()})
        expression_baseline: dict[str, float] = {}
        if len(expression_rows) >= 5:
            expression_baseline = {
                key: math.fsum(row[key] for row in expression_rows) / len(expression_rows)
                for key in sorted(expected_labels)
            }
        return cls(medians, scales, pattern_medians, pattern_scales,
                   sharpness_median, sharpness_scale, mean_brightness_median, expression_baseline)

    @property
    def sharpness_threshold(self) -> float | None:
        if self.sharpness_median is None or self.sharpness_scale is None:
            return None
        allowance = max(3.0 * self.sharpness_scale, self.sharpness_median * 0.2)
        return max(0.0, self.sharpness_median - allowance)


def apply_personal_sharpness_gate(observation: FaceObservation,
                                 profile: CalibrationProfile) -> FaceObservation:
    """Mark blur outliers unusable relative to the user's calibration distribution."""
    threshold = profile.sharpness_threshold
    if threshold is None:
        return observation
    try:
        sharpness = float(observation.sharpness)
    except (TypeError, ValueError):
        sharpness = float("nan")
    if not math.isfinite(sharpness) or sharpness < threshold:
        return replace(observation, quality=0.0, quality_reason="Кадр смазан относительно личной калибровки")
    return observation


class CalibrationSampler:
    """Time-spaced samples prevent a fast camera from filling baseline with adjacent frames."""

    def __init__(self, min_interval_seconds: float = 0.1, minimum_duration_seconds: float = 5.0,
                 max_gap_seconds: float = 1.8):
        settings = (min_interval_seconds, minimum_duration_seconds, max_gap_seconds)
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in settings):
            raise ValueError("Некорректный временной режим калибровки")
        self.min_interval_seconds = float(min_interval_seconds)
        self.minimum_duration_seconds = float(minimum_duration_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.started_at: float | None = None
        self.first_sample_at: float | None = None
        self.last_sample_at: float | None = None
        self.sample_count = 0
        self.continuity_broken = False

    def start(self, now: float) -> None:
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("Время калибровки должно быть конечным")
        self.started_at = timestamp
        self.first_sample_at = None
        self.last_sample_at = None
        self.sample_count = 0
        self.continuity_broken = False

    def accept(self, now: float) -> bool:
        timestamp = float(now)
        self.continuity_broken = False
        if not math.isfinite(timestamp) or self.started_at is None or timestamp < self.started_at:
            return False
        if self.last_sample_at is not None:
            elapsed = timestamp - self.last_sample_at
            if elapsed > self.max_gap_seconds:
                self.started_at = timestamp
                self.first_sample_at = None
                self.last_sample_at = None
                self.sample_count = 0
                self.continuity_broken = True
            elif elapsed + 1e-9 < self.min_interval_seconds:
                return False
        if self.first_sample_at is None:
            self.first_sample_at = timestamp
        self.last_sample_at = timestamp
        self.sample_count += 1
        return True

    def ready(self, now: float, required_samples: int = 50) -> bool:
        try:
            timestamp = float(now)
        except (TypeError, ValueError):
            return False
        return (self.first_sample_at is not None and math.isfinite(timestamp) and
                self.sample_count >= required_samples and
                timestamp - self.first_sample_at >= self.minimum_duration_seconds)


@dataclass(frozen=True)
class ScoreSnapshot:
    expression_score: float | None
    deviation_score: float | None
    index: float | None
    quality: float
    usable: bool
    reason: str | None = None


class Scorer:
    def __init__(self, profile: CalibrationProfile, smoothing_window: int | None = None,
                 ema_alpha: float = 0.35, max_gap_seconds: float = 1.8,
                 nominal_interval_seconds: float = 0.1, smoothing_window_seconds: float = 0.5):
        settings = (ema_alpha, max_gap_seconds, nominal_interval_seconds)
        if (any(not math.isfinite(float(value)) for value in settings) or
                (smoothing_window is not None and smoothing_window < 1) or
                not math.isfinite(float(smoothing_window_seconds)) or smoothing_window_seconds <= 0 or
                not 0 < ema_alpha <= 1):
            raise ValueError("Некорректные настройки сглаживания")
        if max_gap_seconds <= 0 or nominal_interval_seconds <= 0:
            raise ValueError("Некорректные временные настройки сглаживания")
        if not profile.medians:
            raise ValueError("Некорректная калибровка: нет признаков")
        self.profile = profile
        # The default filter spans wall-clock time so camera/processor FPS does not
        # change the effective duration. Explicit legacy frame counts retain their
        # original sample-count semantics.
        self._window: deque[tuple[float, float]] = deque()
        self._legacy_window_size = smoothing_window
        self._window_seconds = float(smoothing_window_seconds)
        self._ema: float | None = None
        self._alpha = ema_alpha
        self._max_gap = max_gap_seconds
        self._nominal_interval = nominal_interval_seconds
        self._last_valid_at: float | None = None

    def reset(self) -> None:
        self._window.clear()
        self._ema = None
        self._last_valid_at = None

    def _robust_window_value(self, now: float) -> float:
        if self._legacy_window_size is not None:
            while len(self._window) > self._legacy_window_size:
                self._window.popleft()
            return median(value for _, value in self._window)

        while self._window and now - self._window[0][0] > self._window_seconds:
            self._window.popleft()
        if len(self._window) < 2:
            return self._window[-1][1]

        # Midpoint/Voronoi cells allocate elapsed support to each score. Clip the
        # complete partition to the live window so no sample is weighted outside it.
        window_start = max(now - self._window_seconds, self._window[0][0])
        total_weight = 0.0
        cumulative: list[tuple[float, float]] = []
        for index, (sample_at, value) in enumerate(self._window):
            left = (window_start if index == 0 else
                    (self._window[index - 1][0] + sample_at) / 2.0)
            right = (now if index == len(self._window) - 1 else
                     (sample_at + self._window[index + 1][0]) / 2.0)
            weight = max(0.0, right - left)
            if weight > 0:
                cumulative.append((value, weight))
                total_weight += weight
        if total_weight <= 0:
            return median(value for _, value in self._window)
        cumulative.sort(key=lambda row: row[0])
        halfway = total_weight / 2.0
        elapsed_weight = 0.0
        for value, weight in cumulative:
            elapsed_weight += weight
            if elapsed_weight >= halfway:
                return value
        return cumulative[-1][0]

    def update(self, observation: FaceObservation, now: float | None = None) -> ScoreSnapshot:
        observation = apply_personal_sharpness_gate(observation, self.profile)
        try:
            source_timestamp = (now if now is not None else
                                observation.monotonic_timestamp if observation.monotonic_timestamp is not None
                                else observation.timestamp)
            timestamp = float(source_timestamp)
            quality = float(observation.quality)
        except (TypeError, ValueError):
            self.reset()
            return ScoreSnapshot(None, None, None, 0.0, False, "Некорректное наблюдение")
        if not math.isfinite(timestamp):
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Некорректное время наблюдения")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            self.reset()
            return ScoreSnapshot(None, None, None, 0.0, False, "Некорректное качество наблюдения")
        if not observation.present or quality < 0.45:
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 observation.quality_reason or "Низкое качество наблюдения")
        if not observation.expressions or not observation.features:
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Недостаточно признаков лица")
        if self._last_valid_at is not None:
            gap = timestamp - self._last_valid_at
            if gap <= 0 or gap > self._max_gap:
                self.reset()
        try:
            raw_probabilities = {key: float(value) for key, value in observation.expressions.items()}
            feature_values = [float(value) for value in observation.features.values()]
        except (TypeError, ValueError):
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Некорректные данные классификатора")
        expected_labels = set(EXPRESSION_SIGNAL_WEIGHTS)
        if set(raw_probabilities) != expected_labels:
            self.reset()
            if set(raw_probabilities) - expected_labels:
                reason = "Классификатор вернул неизвестный класс"
            else:
                reason = "Классификатор вернул неполный набор классов"
            return ScoreSnapshot(None, None, None, quality, False, reason)
        if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in raw_probabilities.values())
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in feature_values)):
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Некорректные данные классификатора")
        probabilities = raw_probabilities
        total = sum(probabilities.values())
        if total <= 0:
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Классификатор не вернул вероятности")
        normalized = [probability / total for probability in probabilities.values()]
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in normalized)
        normalized_entropy = entropy / math.log(max(2, len(normalized)))
        if max(normalized) < 0.30 or normalized_entropy > 0.92:
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Классификатор неуверен")
        if self.profile.expression_baseline:
            current_distribution = dict(zip(probabilities, normalized))
            expression = 100.0 * _jensen_shannon_distance(
                current_distribution, self.profile.expression_baseline)
        else:
            expression = 100.0 * sum(
                normalized[i] * EXPRESSION_SIGNAL_WEIGHTS.get(label, 0.0)
                for i, label in enumerate(probabilities)
            )
        profile_keys = set(self.profile.medians)
        if not profile_keys.issubset(observation.features):
            self.reset()
            return ScoreSnapshot(None, None, None, quality, False,
                                 "Не хватает мимических признаков из калибровки")
        keys = profile_keys
        deviations = [
            min(1.0, abs(float(observation.features[key]) - self.profile.medians[key]) /
                (4.0 * self.profile.scales[key]))
            for key in keys
        ]
        deviation = 100.0 * sum(deviations) / len(deviations)
        raw = min(100.0, max(0.0, 0.5 * expression + 0.5 * deviation))
        self._window.append((timestamp, raw))
        robust = self._robust_window_value(timestamp)
        if self._ema is None or self._last_valid_at is None:
            self._ema = robust
        else:
            elapsed = timestamp - self._last_valid_at
            alpha = 1 - (1 - self._alpha) ** (elapsed / self._nominal_interval)
            self._ema = alpha * robust + (1 - alpha) * self._ema
        self._last_valid_at = timestamp
        return ScoreSnapshot(round(expression, 1), round(deviation, 1), round(self._ema, 1),
                             round(quality, 2), True)


def _jensen_shannon_distance(left: dict[str, float], right: dict[str, float]) -> float:
    """Return a bounded [0, 1] distance between two normalized distributions."""
    if set(left) != set(right) or not left:
        raise ValueError("Распределения выражений должны иметь одинаковые категории")
    midpoint = {key: (left[key] + right[key]) / 2.0 for key in left}
    divergence = 0.0
    for distribution in (left, right):
        for key, probability in distribution.items():
            if probability > 0:
                divergence += 0.5 * probability * math.log(probability / midpoint[key])
    return math.sqrt(min(1.0, max(0.0, divergence / math.log(2.0))))
