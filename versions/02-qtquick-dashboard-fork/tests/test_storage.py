import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from face_tracker.events import StateTransition, TrackerState
from face_tracker.scoring import ScoreSnapshot
from face_tracker.events import EventTracker
from face_tracker.session import TrackingSession
from face_tracker.storage import JournalStore


class JournalStoreTests(unittest.TestCase):
    def test_episode_tracks_peak_quality_and_closes_on_normal_transition(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            session_id = store.start_session(100.0)
            store.record_transition(session_id, StateTransition(TrackerState.TENSION, 105.0, 67.0, 0.8))
            store.update_open_episode(session_id, ScoreSnapshot(70, 80, 75, 0.9, True), 110.0)
            store.record_transition(session_id, StateTransition(TrackerState.NORMAL, 120.0, 42.0, 0.95))
            store.finish_session(session_id, 130.0)
            episodes = store.list_episodes(session_id)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0]["started_at"], 105.0)
            self.assertEqual(episodes[0]["ended_at"], 120.0)
            self.assertEqual(episodes[0]["peak_index"], 75.0)
            self.assertAlmostEqual(episodes[0]["average_quality"], 0.85)
            self.assertFalse(episodes[0]["interrupted"])
            store.close()

    def test_open_episode_stays_without_end_time_when_session_is_interrupted(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            session_id = store.start_session(10.0)
            store.record_transition(session_id, StateTransition(TrackerState.TENSION, 20.0, 70.0, 0.8))
            store.finish_session(session_id, 25.0, interrupted=True)
            episode = store.list_episodes(session_id)[0]
            self.assertIsNone(episode["ended_at"])
            self.assertTrue(episode["interrupted"])
            store.close()

    def test_delete_all_clears_history(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            session_id = store.start_session(1.0)
            store.record_self_rating(session_id, 2.0, 2.0, 5, 50.0, 0.9)
            store.delete_all()
            self.assertEqual(store.list_sessions(10), [])
            self.assertEqual(store.list_all_self_ratings(), [])
            store.close()

    def test_self_rating_is_local_session_data_and_validated(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            session_id = store.start_session(1.0)
            store.record_self_rating(session_id, 2.0, 1.9, 7, 64.0, 0.8)
            rows = store.list_self_ratings(session_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rating"], 7)
            self.assertEqual(rows[0]["face_index"], 64.0)
            self.assertEqual(rows[0]["observed_at"], 1.9)
            summary = store.list_sessions()[0]
            self.assertEqual(summary.rating_count, 1)
            with self.assertRaisesRegex(ValueError, "уже сохранена"):
                store.record_self_rating(session_id, 2.1, 1.9, 8, 66.0, 0.8)
            store.record_self_rating(session_id, 2.2, 2.0, 8, 66.0, 0.8)
            self.assertEqual(len(store.list_self_ratings(session_id)), 2)
            with self.assertRaises(ValueError):
                store.record_self_rating(session_id, 3.0, 3.0, 11, 50.0, 0.8)
            with self.assertRaises(ValueError):
                store.record_self_rating(session_id, 3.0, 3.0, 5.5, 50.0, 0.8)
            store.close()

    def test_restarting_tracker_session_resets_state_and_records_new_episode(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                                   enter_seconds=1, exit_seconds=1)
            session = TrackingSession(store, tracker)
            first_id = session.start(0)
            high = ScoreSnapshot(80, 80, 80, 1.0, True)
            session.update(high, 0)
            session.update(high, 1)
            session.finish(2, interrupted=True)
            second_id = session.start(10)
            session.update(high, 10)
            session.update(high, 11)
            self.assertEqual(len(store.list_episodes(first_id)), 1)
            self.assertEqual(len(store.list_episodes(second_id)), 1)
            self.assertEqual(tracker.state.value, "Сигнал выше личного порога")
            session.finish(12, interrupted=True)
            store.close()

    def test_poor_quality_during_tension_closes_episode_at_last_trusted_frame(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                                   enter_seconds=1, exit_seconds=1)
            session = TrackingSession(store, tracker)
            session_id = session.start(100.0)
            high = ScoreSnapshot(80, 80, 80, 1.0, True)
            session.update(high, 10.0, 100.0)
            session.update(high, 11.0, 101.0)
            self.assertEqual(tracker.state, TrackerState.TENSION)
            unusable = ScoreSnapshot(None, None, None, 0.0, False, "лицо потеряно")
            session.update(unusable, 12.0, 102.0)
            episode = store.list_episodes(session_id)[0]
            self.assertEqual(episode["ended_at"], 101.0)
            self.assertEqual(episode["duration"], 0.0)
            self.assertTrue(episode["interrupted"])
            self.assertEqual(tracker.state, TrackerState.NORMAL)
            session.update(high, 13.0, 103.0)
            session.update(high, 14.0, 104.0)
            self.assertEqual(len(store.list_episodes(session_id)), 2)
            session.finish(105.0)
            store.close()

    def test_fresh_frame_after_long_gap_interrupts_episode_and_restarts_dwell(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                                   enter_seconds=1, exit_seconds=1)
            session = TrackingSession(store, tracker)
            session_id = session.start(100.0)
            high = ScoreSnapshot(80, 80, 80, 1.0, True)
            session.update(high, 10.0, 100.0)
            session.update(high, 11.0, 101.0)
            session.update(high, 20.0, 110.0)
            episodes = store.list_episodes(session_id)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0]["ended_at"], 101.0)
            self.assertTrue(episodes[0]["interrupted"])
            self.assertEqual(tracker.state, TrackerState.NORMAL)
            session.update(high, 21.0, 111.0)
            self.assertEqual(len(store.list_episodes(session_id)), 2)
            session.finish(112.0)
            store.close()

    def test_backward_wall_clock_is_clamped_to_monotonic_event_order(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            tracker = EventTracker(enter_threshold=60, exit_threshold=45,
                                   enter_seconds=1, exit_seconds=1)
            session = TrackingSession(store, tracker)
            session_id = session.start(1000.0)
            high = ScoreSnapshot(80, 80, 80, 1.0, True)
            session.update(high, 10.0, 1000.0)
            session.update(high, 11.0, 500.0)
            episode = store.list_episodes(session_id)[0]
            self.assertEqual(episode["started_at"], 1001.0)
            session.finish(502.0)
            summary = next(item for item in store.list_sessions() if item.session_id == session_id)
            self.assertGreaterEqual(summary.ended_at, episode["started_at"])
            store.close()

    def test_capture_ids_are_unique_across_restarted_sessions(self):
        with TemporaryDirectory() as folder:
            store = JournalStore(Path(folder) / "journal.sqlite3")
            session = TrackingSession(store, EventTracker())
            session.start(1)
            first_capture_id = session.capture_id
            session.finish(2)
            session.start(3)
            self.assertNotEqual(session.capture_id, first_capture_id)
            session.finish(4)
            store.close()


if __name__ == "__main__":
    unittest.main()
