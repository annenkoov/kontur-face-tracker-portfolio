"""Debounced normal/tension state transitions with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .scoring import ScoreSnapshot


class TrackerState(str, Enum):
    NORMAL = "Близко к личной базе"
    TENSION = "Сигнал выше личного порога"


@dataclass(frozen=True)
class StateTransition:
    state: TrackerState
    timestamp: float
    index: float
    quality: float
    qualifying_observations: int = 0
    qualifying_span_seconds: float = 0.0
    mean_inter_observation_seconds: float | None = None
    max_inter_observation_seconds: float | None = None


class EventTracker:
    def __init__(self, enter_threshold: float = 62.0, exit_threshold: float = 48.0,
                 enter_seconds: float = 6.0, exit_seconds: float = 5.0,
                 max_gap_seconds: float = 1.8):
        settings = (enter_threshold, exit_threshold, enter_seconds, exit_seconds, max_gap_seconds)
        if any(not math.isfinite(float(value)) for value in settings):
            raise ValueError("Пороговые настройки должны быть конечными числами")
        if not 0 <= exit_threshold < enter_threshold <= 100:
            raise ValueError("Порог возврата должен быть ниже порога напряжения")
        if enter_seconds <= 0 or exit_seconds <= 0 or max_gap_seconds <= 0:
            raise ValueError("Длительности порогов должны быть положительными")
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.enter_seconds = enter_seconds
        self.exit_seconds = exit_seconds
        self.max_gap_seconds = max_gap_seconds
        self.state = TrackerState.NORMAL
        self._candidate_since: float | None = None
        self._candidate_observations = 0
        self._candidate_last_qualifying_at: float | None = None
        self._candidate_max_interval: float | None = None
        self._last_timestamp: float | None = None

    def _clear_candidate(self) -> None:
        self._candidate_since = None
        self._candidate_observations = 0
        self._candidate_last_qualifying_at = None
        self._candidate_max_interval = None

    def reset(self) -> None:
        self.state = TrackerState.NORMAL
        self._clear_candidate()
        self._last_timestamp = None

    def update(self, snapshot: ScoreSnapshot, now: float, wall_now: float | None = None) -> list[StateTransition]:
        try:
            now = float(now)
            event_timestamp = float(wall_now if wall_now is not None else now)
            index = float(snapshot.index) if snapshot.index is not None else None
            quality = float(snapshot.quality)
        except (TypeError, ValueError):
            self._clear_candidate()
            self._last_timestamp = None
            return []
        if not math.isfinite(now) or not math.isfinite(event_timestamp):
            self._clear_candidate()
            self._last_timestamp = None
            return []
        if self._last_timestamp is not None and (
            now <= self._last_timestamp or now - self._last_timestamp > self.max_gap_seconds
        ):
            self._clear_candidate()
        self._last_timestamp = now
        if (not snapshot.usable or index is None or not math.isfinite(index) or not 0 <= index <= 100
                or not math.isfinite(quality) or not 0 <= quality <= 1):
            self._clear_candidate()
            return []
        if self.state == TrackerState.NORMAL:
            qualifies = index >= self.enter_threshold
            required = self.enter_seconds
            next_state = TrackerState.TENSION
        else:
            qualifies = index <= self.exit_threshold
            required = self.exit_seconds
            next_state = TrackerState.NORMAL
        if not qualifies:
            self._clear_candidate()
            return []
        if self._candidate_since is None:
            self._candidate_since = now
            self._candidate_observations = 1
            self._candidate_last_qualifying_at = now
            return []
        if self._candidate_last_qualifying_at is not None:
            interval = now - self._candidate_last_qualifying_at
            self._candidate_max_interval = (interval if self._candidate_max_interval is None
                                            else max(self._candidate_max_interval, interval))
        self._candidate_last_qualifying_at = now
        self._candidate_observations += 1
        if now - self._candidate_since < required:
            return []
        observed_span = max(0.0, now - self._candidate_since)
        mean_interval = (observed_span / (self._candidate_observations - 1)
                         if self._candidate_observations > 1 else None)
        qualifying_observations = self._candidate_observations
        max_interval = self._candidate_max_interval
        self.state = next_state
        self._clear_candidate()
        return [StateTransition(next_state, event_timestamp, index, quality,
                                qualifying_observations=qualifying_observations,
                                qualifying_span_seconds=observed_span,
                                mean_inter_observation_seconds=mean_interval,
                                max_inter_observation_seconds=max_interval)]
