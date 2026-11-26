"""Personal-baseline diagnostics for sustained facial movement patterns.

These observations are deliberately kept separate from the stress-like index:
facial action units are context-sensitive and are not direct measurements of
internal emotion or stress.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median

from .scoring import CalibrationProfile


PATTERN_LABELS = {
    "brow_down": "Сведение бровей",
    "eye_squint": "Сужение глаз",
    "mouth_press": "Сжатие губ",
    "mouth_frown": "Опускание уголков губ",
    "jaw_open": "Открывание рта · речь/зевок",
    "eye_wide": "Расширение глаз",
    "upper_lip_raise": "Подъём верхней губы",
    "smile": "Активность улыбки",
    "brow_eye_coactivation": "Брови + сужение глаз",
    "mouth_coactivation": "Сжатие губ + опущение уголков",
    "brow_asymmetry": "Разница между бровями",
    "eye_squint_asymmetry": "Разница между глазами",
    "mouth_press_asymmetry": "Разница сжатия губ по сторонам",
}


@dataclass(frozen=True)
class PatternSignal:
    value: float | None
    baseline: float | None
    delta: float | None
    robust_z: float | None
    above_baseline: bool = False


@dataclass(frozen=True)
class PatternSummary:
    ready: bool
    reason: str | None
    signals: dict[str, PatternSignal]


class FacialPatternTracker:
    """Median-filter blendshape patterns and require a sustained personal deviation."""

    def __init__(self, profile: CalibrationProfile, *, window_seconds: float = 1.0,
                 enter_seconds: float = 0.8, exit_seconds: float = 1.0,
                 max_gap_seconds: float = 1.8, enter_z: float = 2.5,
                 exit_z: float = 1.2, min_delta: float = 0.12,
                 exit_delta: float = 0.06):
        settings = (window_seconds, enter_seconds, exit_seconds, max_gap_seconds,
                    enter_z, exit_z, min_delta, exit_delta)
        if (any(not math.isfinite(float(value)) for value in settings) or window_seconds <= 0 or
                enter_seconds <= 0 or exit_seconds <= 0 or max_gap_seconds <= 0 or
                enter_z <= exit_z or exit_z < 0 or min_delta <= exit_delta or exit_delta < 0):
            raise ValueError("Временные настройки паттернов должны быть положительными")
        self.profile = profile
        self.window_seconds = window_seconds
        self.enter_seconds = enter_seconds
        self.exit_seconds = exit_seconds
        self.max_gap_seconds = max_gap_seconds
        self.enter_z = enter_z
        self.exit_z = exit_z
        self.min_delta = min_delta
        self.exit_delta = exit_delta
        self._history: deque[tuple[float, dict[str, float]]] = deque()
        self._last_timestamp: float | None = None
        self._active: set[str] = set()
        self._candidate_since: dict[str, float] = {}
        self._release_since: dict[str, float] = {}

    def reset(self) -> None:
        self._history.clear()
        self._last_timestamp = None
        self._active.clear()
        self._candidate_since.clear()
        self._release_since.clear()

    def _empty(self, reason: str) -> PatternSummary:
        return PatternSummary(False, reason, {
            key: PatternSignal(None, self.profile.pattern_medians.get(key), None, None, False)
            for key in self.profile.pattern_medians
        })

    def update(self, timestamp: float, quality: float, patterns: dict[str, float] | None) -> PatternSummary:
        try:
            now = float(timestamp)
            q = float(quality)
        except (TypeError, ValueError):
            self.reset()
            return self._empty("Некорректные данные наблюдения")
        if not math.isfinite(now) or not math.isfinite(q) or not 0 <= q <= 1:
            self.reset()
            return self._empty("Некорректные данные наблюдения")
        if q < 0.45 or not patterns:
            self.reset()
            return self._empty("Нужно устойчивое качественное изображение лица")
        if self._last_timestamp is not None:
            gap = now - self._last_timestamp
            if gap <= 0 or gap > self.max_gap_seconds:
                self.reset()
        common = set(self.profile.pattern_medians) & set(patterns)
        values: dict[str, float] = {}
        try:
            values = {key: float(patterns[key]) for key in common}
        except (TypeError, ValueError):
            self.reset()
            return self._empty("Некорректное значение паттерна")
        if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            self.reset()
            return self._empty("Некорректные значения паттернов")
        self._history.append((now, values))
        self._last_timestamp = now
        while self._history and now - self._history[0][0] > self.window_seconds:
            self._history.popleft()

        signals: dict[str, PatternSignal] = {}
        for key in self.profile.pattern_medians:
            available = [row[key] for _, row in self._history if key in row]
            if not available:
                signals[key] = PatternSignal(None, self.profile.pattern_medians[key], None, None, False)
                continue
            value = median(available)
            baseline = self.profile.pattern_medians[key]
            scale = max(self.profile.pattern_scales.get(key, 0.04), 0.04)
            delta = value - baseline
            z_score = delta / max(scale, 0.07)
            elevated = delta >= self.min_delta and z_score >= self.enter_z
            released = delta <= self.exit_delta or z_score < self.exit_z
            if key not in self._active:
                self._release_since.pop(key, None)
                if elevated:
                    self._candidate_since.setdefault(key, now)
                    if now - self._candidate_since[key] >= self.enter_seconds:
                        self._active.add(key)
                        self._candidate_since.pop(key, None)
                else:
                    self._candidate_since.pop(key, None)
            else:
                self._candidate_since.pop(key, None)
                if released:
                    self._release_since.setdefault(key, now)
                    if now - self._release_since[key] >= self.exit_seconds:
                        self._active.discard(key)
                        self._release_since.pop(key, None)
                else:
                    self._release_since.pop(key, None)
            signals[key] = PatternSignal(round(value, 3), round(baseline, 3), round(delta, 3),
                                         round(z_score, 2), key in self._active)
        return PatternSummary(True, None, signals)
