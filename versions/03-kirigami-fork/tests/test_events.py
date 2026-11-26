import unittest

from face_tracker.events import EventTracker, TrackerState
from face_tracker.scoring import ScoreSnapshot


def snap(index, usable=True, quality=1.0, reason=None):
    return ScoreSnapshot(expression_score=index, deviation_score=index, index=index,
                         quality=quality, usable=usable, reason=reason)


class EventTrackerTests(unittest.TestCase):
    def test_non_finite_threshold_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            EventTracker(enter_threshold=float("nan"))

    def test_display_states_describe_mimic_range_not_psychological_health(self):
        self.assertEqual(TrackerState.NORMAL.value, "Близко к личной базе")
        self.assertEqual(TrackerState.TENSION.value, "Сигнал выше личного порога")

    def test_short_spike_does_not_trigger_but_sustained_change_does(self):
        tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                               enter_seconds=3, exit_seconds=2, max_gap_seconds=5)
        self.assertEqual(tracker.update(snap(80), 0), [])
        self.assertEqual(tracker.update(snap(20), 1), [])
        transitions = tracker.update(snap(80), 2)
        self.assertEqual(transitions, [])
        transitions = tracker.update(snap(80), 5)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].state, TrackerState.TENSION)

    def test_recovery_uses_lower_exit_threshold_and_poor_quality_pauses(self):
        tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                               enter_seconds=1, exit_seconds=2, max_gap_seconds=5)
        tracker.update(snap(80), 0)
        tracker.update(snap(80), 1)
        self.assertEqual(tracker.update(snap(0, usable=False, reason="лицо потеряно"), 2), [])
        self.assertEqual(tracker.state, TrackerState.TENSION)
        self.assertEqual(tracker.update(snap(40), 4), [])
        transitions = tracker.update(snap(40), 6)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].state, TrackerState.NORMAL)

    def test_long_observation_gap_does_not_count_as_sustained_tension(self):
        tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                               enter_seconds=3, exit_seconds=2, max_gap_seconds=2)
        self.assertEqual(tracker.update(snap(80), 0), [])
        self.assertEqual(tracker.update(snap(80), 3600), [])
        self.assertEqual(tracker.update(snap(80), 3602), [])
        transitions = tracker.update(snap(80), 3603)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].state, TrackerState.TENSION)

    def test_gap_over_live_freshness_window_restarts_event_dwell(self):
        tracker = EventTracker(enter_threshold=60, exit_threshold=45, enter_seconds=1)
        self.assertEqual(tracker.update(snap(80), 0.0), [])
        self.assertEqual(tracker.update(snap(80), 1.9), [])
        self.assertEqual(tracker.update(snap(80), 2.8), [])
        transitions = tracker.update(snap(80), 2.9)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].state, TrackerState.TENSION)

    def test_non_finite_timestamp_resets_candidate_without_transition(self):
        tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                               enter_seconds=1, exit_seconds=1)
        self.assertEqual(tracker.update(snap(80), 0.0), [])
        self.assertEqual(tracker.update(snap(80), float("nan")), [])
        self.assertEqual(tracker.update(snap(80), 2.0), [])
        self.assertEqual(tracker.state, TrackerState.NORMAL)

    def test_monotonic_dwell_keeps_wall_clock_timestamp_for_journal(self):
        tracker = EventTracker(enter_threshold=60, exit_threshold=45, enter_seconds=1)
        self.assertEqual(tracker.update(snap(80), 10.0, wall_now=1000.0), [])
        transitions = tracker.update(snap(80), 11.0, wall_now=500.0)
        self.assertEqual(transitions[0].timestamp, 500.0)


if __name__ == "__main__":
    unittest.main()
