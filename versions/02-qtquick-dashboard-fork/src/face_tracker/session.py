"""Coordinates a session's score transitions and journal writes."""

from __future__ import annotations

import math
from uuid import uuid4

from .events import EventTracker
from .scoring import ScoreSnapshot
from .storage import JournalStore


class TrackingSession:
    def __init__(self, store: JournalStore, event_tracker: EventTracker) -> None:
        self.store = store
        self.event_tracker = event_tracker
        self.session_id: int | None = None
        self.capture_id: str | None = None
        self._last_trusted_wall_time: float | None = None
        self._last_trusted_monotonic_time: float | None = None
        self._session_started_wall_time: float | None = None
        self._usable_seconds = 0.0
        self._usable_observations = 0

    def start(self, now: float) -> int:
        if self.session_id is not None:
            raise RuntimeError("Сессия уже запущена")
        self.event_tracker.reset()
        self._last_trusted_wall_time = None
        self._last_trusted_monotonic_time = None
        self.capture_id = uuid4().hex
        self._session_started_wall_time = float(now)
        self._usable_seconds = 0.0
        self._usable_observations = 0
        self.session_id = self.store.start_session(now)
        return self.session_id

    def update(self, snapshot: ScoreSnapshot, now: float, wall_now: float | None = None) -> None:
        if self.session_id is None:
            return
        try:
            monotonic_now = float(now)
            persisted_at = float(wall_now if wall_now is not None else now)
        except (TypeError, ValueError):
            self.interrupt_observation_gap()
            return
        if not math.isfinite(monotonic_now) or not math.isfinite(persisted_at):
            self.interrupt_observation_gap()
            return
        previous_monotonic = self._last_trusted_monotonic_time
        if previous_monotonic is not None and (
            monotonic_now <= previous_monotonic
            or monotonic_now - previous_monotonic > self.event_tracker.max_gap_seconds
        ):
            # A fresh frame can arrive after a UI stall without the stale-frame poll seeing
            # the old frame. Detect that discontinuity here before scoring its event.
            self.interrupt_observation_gap()
            previous_monotonic = None
        if not snapshot.usable:
            self.interrupt_observation_gap()
            return
        if previous_monotonic is not None:
            self._usable_seconds += monotonic_now - previous_monotonic
        self._usable_observations += 1
        if self._last_trusted_wall_time is not None and persisted_at < self._last_trusted_wall_time:
            elapsed = max(0.0, monotonic_now - previous_monotonic) if previous_monotonic is not None else 0.0
            persisted_at = self._last_trusted_wall_time + elapsed
        self._last_trusted_wall_time = persisted_at
        self._last_trusted_monotonic_time = monotonic_now
        self.store.update_open_episode(self.session_id, snapshot, persisted_at)
        for transition in self.event_tracker.update(snapshot, monotonic_now, persisted_at):
            self.store.record_transition(self.session_id, transition)

    def interrupt_observation_gap(self, last_trusted_wall_time: float | None = None) -> None:
        """End an open episode at the last observed frame and reset dwell state."""
        if self.session_id is None:
            return
        boundary = self._last_trusted_wall_time
        if boundary is None:
            boundary = last_trusted_wall_time
        if boundary is not None:
            self.store.interrupt_open_episode(self.session_id, boundary)
        self.event_tracker.reset()
        self._last_trusted_monotonic_time = None

    def finish(self, now: float, interrupted: bool = False) -> None:
        if self.session_id is not None:
            requested_end = float(now)
            start_time = (self._session_started_wall_time if self._session_started_wall_time is not None
                          else requested_end)
            trusted_time = (self._last_trusted_wall_time if self._last_trusted_wall_time is not None
                            else requested_end)
            end_time = max(requested_end, start_time, trusted_time)
            self.store.finish_session(self.session_id, end_time, interrupted,
                                      usable_seconds=self._usable_seconds,
                                      usable_observations=self._usable_observations)
            self.session_id = None
            self._last_trusted_wall_time = None
            self._last_trusted_monotonic_time = None
            self._session_started_wall_time = None
            self._usable_seconds = 0.0
            self._usable_observations = 0
