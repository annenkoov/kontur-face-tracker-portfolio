"""Quality-weighted temporal consensus for per-frame expression probabilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ExpressionState:
    label: str | None
    probabilities: dict[str, float]
    ready: bool
    reason: str | None = None


class ExpressionSmoother:
    """Smooth short-term noise without carrying predictions over camera gaps."""

    def __init__(self, window_seconds: float = 1.0, max_gap_seconds: float = 1.8,
                 min_frames: int = 4, min_duration_seconds: float = 0.35,
                 min_confidence: float = 0.32, min_margin: float = 0.06,
                 switch_duration_seconds: float = 0.35, switch_margin: float = 0.08):
        settings = (window_seconds, max_gap_seconds, min_duration_seconds, min_confidence,
                    min_margin, switch_duration_seconds, switch_margin)
        if (any(not math.isfinite(float(value)) for value in settings) or window_seconds <= 0 or
                max_gap_seconds <= 0 or min_frames < 1 or min_duration_seconds <= 0 or
                not 0 <= min_confidence <= 1 or not 0 <= min_margin <= 1 or
                switch_duration_seconds <= 0 or not 0 <= switch_margin <= 1):
            raise ValueError("Invalid temporal smoother configuration")
        self.window_seconds = window_seconds
        self.max_gap_seconds = max_gap_seconds
        self.min_frames = min_frames
        self.min_duration_seconds = min_duration_seconds
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.switch_duration_seconds = switch_duration_seconds
        self.switch_margin = switch_margin
        self._history: deque[tuple[float, dict[str, float], float]] = deque()
        self._labels: tuple[str, ...] | None = None
        self._stable_label: str | None = None
        self._candidate_label: str | None = None
        self._candidate_since: float | None = None
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._history.clear()
        self._labels = None
        self._stable_label = None
        self._candidate_label = None
        self._candidate_since = None
        self._last_timestamp = None

    def update(self, timestamp: float, quality: float,
               expressions: dict[str, float] | None) -> ExpressionState:
        try:
            now = float(timestamp)
            q = float(quality)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return ExpressionState(None, {}, False, "Некорректное наблюдение")
        if not math.isfinite(now) or not math.isfinite(q) or not 0 <= q <= 1:
            self.reset()
            return ExpressionState(None, {}, False, "Некорректное наблюдение")
        if not expressions or q < 0.45:
            self.reset()
            return ExpressionState(None, {}, False, "Недостаточное качество кадра")
        try:
            labels = tuple(sorted(expressions))
            values = [float(expressions[label]) for label in labels]
        except (TypeError, ValueError, OverflowError, KeyError):
            self.reset()
            return ExpressionState(None, {}, False, "Некорректные вероятности модели")
        if (len(labels) < 2 or any(not isinstance(label, str) for label in labels) or
                any(not math.isfinite(value) or value < 0 for value in values)):
            self.reset()
            return ExpressionState(None, {}, False, "Недостаточный или некорректный вывод модели")
        maximum = max(values)
        if maximum <= 0:
            self.reset()
            return ExpressionState(None, {}, False, "Модель не вернула вероятности")
        scaled_values = [value / maximum for value in values]
        total = sum(scaled_values)
        if self._last_timestamp is not None:
            gap = now - self._last_timestamp
            if gap <= 0 or gap > self.max_gap_seconds or labels != self._labels:
                self.reset()
        self._labels = labels
        self._last_timestamp = now
        normalized = {label: value / total for label, value in zip(labels, scaled_values)}
        self._history.append((now, normalized, q))
        while self._history and now - self._history[0][0] > self.window_seconds:
            self._history.popleft()

        history = list(self._history)
        window_start = max(now - self.window_seconds, history[0][0])
        weights = []
        for index, (sample_at, _, item_quality) in enumerate(history):
            left = (window_start if index == 0 else
                    (history[index - 1][0] + sample_at) / 2.0)
            right = (now if index == len(history) - 1 else
                     (sample_at + history[index + 1][0]) / 2.0)
            represented_seconds = max(0.0, right - left)
            weights.append(max(0.05, item_quality) * represented_seconds)
        weight_total = sum(weights)
        if weight_total <= 0:
            weights = [max(0.05, item_quality) for _, _, item_quality in history]
            weight_total = sum(weights)
        smoothed = {
            label: sum(probs[label] * weight for (_, probs, _), weight in zip(history, weights)) / weight_total
            for label in labels
        }
        probability_total = sum(smoothed.values())
        smoothed = {label: value / probability_total for label, value in smoothed.items()}
        ranked = sorted(smoothed.items(), key=lambda item: item[1], reverse=True)
        top_label, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        duration = now - self._history[0][0]
        enough_data = len(self._history) >= self.min_frames and duration >= self.min_duration_seconds
        confident = top_probability >= self.min_confidence and top_probability - second_probability >= self.min_margin
        if not enough_data:
            return ExpressionState(None, smoothed, False, "Собираем устойчивые кадры…")
        if not confident:
            self._candidate_label = None
            self._candidate_since = None
            return ExpressionState(None, smoothed, False, "Модель не уверена в выражении")

        if self._stable_label is None:
            self._stable_label = top_label
        elif top_label != self._stable_label:
            if self._candidate_label != top_label:
                self._candidate_label = top_label
                self._candidate_since = now
            elif (self._candidate_since is not None and now - self._candidate_since >= self.switch_duration_seconds
                  and smoothed[top_label] >= smoothed[self._stable_label] + self.switch_margin):
                self._stable_label = top_label
                self._candidate_label = None
                self._candidate_since = None
        else:
            self._candidate_label = None
            self._candidate_since = None
        return ExpressionState(self._stable_label, smoothed, True)
