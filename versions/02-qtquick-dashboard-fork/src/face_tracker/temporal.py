"""Quality-weighted temporal consensus for per-frame expression probabilities."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExpressionState:
    """Consensus result and diagnostics from the smoothed class distribution."""

    label: str | None
    probabilities: dict[str, float]
    ready: bool
    reason: str | None = None
    top_probability: float | None = None
    margin: float | None = None
    ready_reason: str | None = None


class ExpressionSmoother:
    """Smooth short-term noise, optionally enforcing a fixed class set."""

    def __init__(self, window_seconds: float = 1.0, max_gap_seconds: float = 1.8,
                 min_frames: int = 4, min_duration_seconds: float = 0.35,
                 min_confidence: float = 0.32, min_margin: float = 0.06,
                 switch_duration_seconds: float = 0.35, switch_margin: float = 0.08,
                 *, expected_labels: Collection[str] | None = None):
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
        self._expected_labels = frozenset(expected_labels) if expected_labels is not None else None
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

    def _unavailable(self, code: str, reason: str) -> ExpressionState:
        self.reset()
        return ExpressionState(None, {}, False, reason, ready_reason=code)

    def update(self, timestamp: float, quality: float,
               expressions: Mapping[str, float] | None) -> ExpressionState:
        try:
            now = float(timestamp)
            q = float(quality)
        except (TypeError, ValueError, OverflowError):
            return self._unavailable("invalid_observation", "Некорректное наблюдение")
        if not math.isfinite(now) or not math.isfinite(q) or not 0 <= q <= 1:
            return self._unavailable("invalid_observation", "Некорректное наблюдение")
        if expressions is not None and not isinstance(expressions, Mapping):
            return self._unavailable("invalid_probabilities", "Некорректные данные модели")
        if not expressions:
            if q < 0.45:
                return self._unavailable("low_quality", "Низкое качество кадра")
            return self._unavailable("missing_probabilities", "Нет данных модели")
        if (len(expressions) < 2 or any(not isinstance(label, str) or not label.strip()
                                      for label in expressions) or
                (self._expected_labels is not None and frozenset(expressions) != self._expected_labels)):
            return self._unavailable("invalid_labels", "Неожиданные классы модели")
        try:
            labels = tuple(sorted(expressions))
            values = [float(expressions[label]) for label in labels]
        except (TypeError, ValueError, OverflowError, KeyError):
            return self._unavailable("invalid_probabilities", "Некорректные данные модели")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            return self._unavailable("invalid_probabilities", "Некорректные данные модели")
        total = math.fsum(values)
        # Allow rounding drift in softmax output before renormalizing it.
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-3):
            return self._unavailable("invalid_probability_sum", "Некорректные данные модели")
        if q < 0.45:
            return self._unavailable("low_quality", "Низкое качество кадра")
        if self._last_timestamp is not None:
            gap = now - self._last_timestamp
            if gap <= 0 or gap > self.max_gap_seconds or labels != self._labels:
                self.reset()
        self._labels = labels
        self._last_timestamp = now
        normalized = {label: value / total for label, value in zip(labels, values)}
        self._history.append((now, normalized, q))
        while self._history and now - self._history[0][0] > self.window_seconds:
            self._history.popleft()

        history = list(self._history)
        left = max(now - self.window_seconds, history[0][0])
        weights = []
        for index, (sample_at, _, item_quality) in enumerate(history):
            right = (now if index == len(history) - 1 else
                     sample_at + (history[index + 1][0] - sample_at) / 2.0)
            represented_seconds = max(0.0, right - left)
            weights.append(max(0.05, item_quality) * represented_seconds)
            left = right
        weight_total = sum(weights)
        if weight_total <= 0:
            weights = [max(0.05, item_quality) for _, _, item_quality in history]
            weight_total = sum(weights)
        normalized_weights = [weight / weight_total for weight in weights]
        smoothed = {
            label: math.fsum(probs[label] * weight
                             for (_, probs, _), weight in zip(history, normalized_weights))
            for label in labels
        }
        probability_total = math.fsum(smoothed.values())
        smoothed = {label: value / probability_total for label, value in smoothed.items()}
        ranked = sorted(smoothed.items(), key=lambda item: item[1], reverse=True)
        top_label, top_probability = ranked[0]
        margin = top_probability - ranked[1][1]
        duration = now - self._history[0][0]
        enough_data = len(self._history) >= self.min_frames and duration >= self.min_duration_seconds
        if not enough_data:
            self._stable_label = None
            self._candidate_label = None
            self._candidate_since = None
            return ExpressionState(None, smoothed, False, "Собираем данные",
                                   top_probability, margin, "collecting")
        if top_probability < self.min_confidence or margin == 0 or margin < self.min_margin:
            self._candidate_label = None
            self._candidate_since = None
            return ExpressionState(None, smoothed, False, "Не уверен",
                                   top_probability, margin, "ambiguous")

        if self._stable_label is None:
            self._stable_label = top_label
        elif top_label != self._stable_label:
            if self._candidate_label != top_label:
                self._candidate_label = top_label
                self._candidate_since = now
            elif (self._candidate_since is not None and now - self._candidate_since >= self.switch_duration_seconds
                  and top_probability >= smoothed[self._stable_label] + self.switch_margin):
                self._stable_label = top_label
                self._candidate_label = None
                self._candidate_since = None
        else:
            self._candidate_label = None
            self._candidate_since = None
        # Keep the previous class for switch timing only while a challenger leads.
        if self._stable_label != top_label:
            return ExpressionState(None, smoothed, False, "Уточняем выражение",
                                   top_probability, margin, "switching")
        return ExpressionState(top_label, smoothed, True, top_probability=top_probability,
                               margin=margin, ready_reason="stable")
