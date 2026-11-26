"""Observable eyelid state from the existing MediaPipe face landmarks and scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import dist, isfinite
from typing import Literal


EyeLabel = Literal["open", "partial", "closed", "blink", "unknown"]

# Контуры Face Landmarker: два уголка и две пары точек верхнего/нижнего века.
# Стороны относятся к человеку, а не к зеркальному предпросмотру.
_LEFT_EYE = (362, 263, 385, 380, 387, 373)
_RIGHT_EYE = (33, 133, 158, 153, 160, 144)
_MIN_QUALITY = 0.45
_CLOSED_EAR = 0.10
_OPEN_EAR = 0.29
_MAX_DISAGREEMENT = 0.55
_CLOSE_THRESHOLD = 0.68
_RELEASE_THRESHOLD = 0.53
_OPEN_THRESHOLD = 0.32
_OPEN_HOLD_THRESHOLD = 0.42
_MIN_CLOSED_SECONDS = 0.05
_MAX_BLINK_SECONDS = 0.40
_MAX_BLINK_REOPEN_SKEW = 0.12
_MAX_FRAME_GAP_SECONDS = 0.30


@dataclass(frozen=True)
class EyeState:
    left: EyeLabel
    right: EyeLabel
    overall: EyeLabel
    closure: float | None
    confidence: float
    blink_count: int
    closed_duration_seconds: float
    reason: str | None


def _unit_value(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) and 0.0 <= number <= 1.0 else None


def _eye_aspect_ratio(landmarks: Sequence[tuple[float, float]], indices: tuple[int, ...],
                      y_to_x_scale: float) -> float | None:
    try:
        if len(landmarks) <= max(indices):
            return None
        points = []
        for index in indices:
            x, y = landmarks[index]
            x, y = _unit_value(x), _unit_value(y)
            if x is None or y is None:
                return None
            # MediaPipe normalizes x and y against different frame dimensions.
            # Convert y into the same normalized pixel scale as x before distances.
            points.append((x, y * y_to_x_scale))
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    corner_a, corner_b, upper_a, lower_a, upper_b, lower_b = points
    width = dist(corner_a, corner_b)
    if width <= 1e-6:
        return None
    # EAR: средняя высота глазной щели относительно расстояния между уголками.
    return (dist(upper_a, lower_a) + dist(upper_b, lower_b)) / (2.0 * width)


def _measure_eye(landmarks: Sequence[tuple[float, float]], indices: tuple[int, ...],
                 blendshapes: Mapping[str, float], key: str,
                 y_to_x_scale: float) -> tuple[float | None, float, str | None]:
    ear = _eye_aspect_ratio(landmarks, indices, y_to_x_scale)
    if ear is None:
        return None, 0.0, "Некорректные точки век"
    blink_score = _unit_value(blendshapes.get(key))
    if blink_score is None:
        return None, 0.0, "Не удалось получить оценку моргания"
    geometry = max(0.0, min(1.0, (_OPEN_EAR - ear) / (_OPEN_EAR - _CLOSED_EAR)))
    disagreement = abs(geometry - blink_score)
    if disagreement >= _MAX_DISAGREEMENT:
        return None, 0.0, "Оценки по точкам век и мимике расходятся"
    return (geometry + blink_score) / 2.0, 1.0 - disagreement, None


@dataclass
class _LidTrack:
    state: EyeLabel = "partial"
    closed_since: float | None = None
    armed: bool = False
    blink_candidate: tuple[float, float] | None = None
    recent_blink: tuple[float, float, float] | None = None

    def update(self, timestamp: float, closure: float) -> EyeLabel:
        # Гистерезис: удерживаем состояние в узкой зоне около порога,
        # но подтверждаем закрытие только после нескольких наблюдений.
        if closure >= _CLOSE_THRESHOLD or (self.closed_since is not None and closure >= _RELEASE_THRESHOLD):
            if self.closed_since is None:
                self.closed_since = timestamp
                self.blink_candidate = (timestamp, timestamp) if self.armed else None
                self.armed = False
            self.recent_blink = None
            if self.blink_candidate is not None:
                self.blink_candidate = (self.blink_candidate[0], timestamp)
            if timestamp - self.closed_since >= _MIN_CLOSED_SECONDS:
                self.state = "closed"
            return self.state

        self.closed_since = None
        if closure <= _OPEN_THRESHOLD or (self.state == "open" and closure <= _OPEN_HOLD_THRESHOLD):
            self.state = "open"
            self.armed = True
            candidate, self.blink_candidate = self.blink_candidate, None
            if candidate is not None:
                start, last_closed = candidate
                if last_closed - start >= _MIN_CLOSED_SECONDS and timestamp - start <= _MAX_BLINK_SECONDS:
                    self.recent_blink = (start, last_closed, timestamp)
                    return "blink"
            return "open"

        self.state = "partial"
        self.recent_blink = None
        return "partial"


class EyeStateEstimator:
    """Track eyelids with capture time; report closure and confidence on a 0–1 scale."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start a new observation interval, clearing the count and temporal state."""
        self._blink_count = 0
        self._break_continuity()

    def _break_continuity(self) -> None:
        self._last_timestamp: float | None = None
        self._last_missing_eye_at: float | None = None
        self._left = _LidTrack()
        self._right = _LidTrack()

    def _unknown(self, reason: str) -> EyeState:
        self._break_continuity()
        return EyeState("unknown", "unknown", "unknown", None, 0.0, self._blink_count, 0.0, reason)

    def update(self, timestamp: float, quality: float, landmarks: Sequence[tuple[float, float]],
               blendshapes: Mapping[str, float], y_to_x_scale: float = 1.0) -> EyeState:
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return self._unknown("Некорректное время кадра")
        if not isfinite(timestamp):
            return self._unknown("Некорректное время кадра")
        try:
            y_to_x_scale = float(y_to_x_scale)
        except (TypeError, ValueError, OverflowError):
            return self._unknown("Некорректный формат кадра")
        if not isfinite(y_to_x_scale) or y_to_x_scale <= 0:
            return self._unknown("Некорректный формат кадра")
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                return self._unknown("Нарушен порядок кадров")
            if timestamp - self._last_timestamp > _MAX_FRAME_GAP_SECONDS:
                self._break_continuity()
        frame_quality = _unit_value(quality)
        if frame_quality is None or frame_quality < _MIN_QUALITY:
            return self._unknown("Недостаточное качество кадра для оценки глаз")
        if not isinstance(blendshapes, Mapping):
            return self._unknown("Не удалось получить оценку моргания")
        self._last_timestamp = timestamp

        left_value, left_confidence, left_reason = _measure_eye(
            landmarks, _LEFT_EYE, blendshapes, "eyeBlinkLeft", y_to_x_scale)
        right_value, right_confidence, right_reason = _measure_eye(
            landmarks, _RIGHT_EYE, blendshapes, "eyeBlinkRight", y_to_x_scale)
        if left_value is None:
            self._left = _LidTrack()
        if right_value is None:
            self._right = _LidTrack()
        left = self._left.update(timestamp, left_value) if left_value is not None else "unknown"
        right = self._right.update(timestamp, right_value) if right_value is not None else "unknown"

        if left_value is None or right_value is None:
            self._last_missing_eye_at = timestamp
            return EyeState(left, right, "unknown", None, 0.0, self._blink_count, 0.0, left_reason or right_reason)

        duration = 0.0
        reason = None
        if self._pair_blinks():
            overall = "blink"
        elif left == right == "closed":
            overall = "closed"
            left_since, right_since = self._left.closed_since, self._right.closed_since
            if left_since is not None and right_since is not None:
                duration = timestamp - max(left_since, right_since)
        elif left == right == "open":
            overall = "open"
        else:
            overall = "partial"
            if left != right:
                reason = "Состояния левого и правого глаза различаются"
        return EyeState(left, right, overall, (left_value + right_value) / 2.0,
                        frame_quality * min(left_confidence, right_confidence), self._blink_count, duration, reason)

    def _pair_blinks(self) -> bool:
        left, right = self._left.recent_blink, self._right.recent_blink
        if left is None or right is None:
            return False
        self._left.recent_blink = self._right.recent_blink = None
        left_start, left_last_closed, left_reopened = left
        right_start, right_last_closed, right_reopened = right
        # Общее событие требует наблюдения обоих глаз с начала закрытия каждого века.
        if self._last_missing_eye_at is not None and min(left_start, right_start) <= self._last_missing_eye_at:
            return False
        overlap = min(left_last_closed, right_last_closed) - max(left_start, right_start)
        if overlap >= _MIN_CLOSED_SECONDS and abs(left_reopened - right_reopened) <= _MAX_BLINK_REOPEN_SKEW:
            self._blink_count += 1
            return True
        return False
