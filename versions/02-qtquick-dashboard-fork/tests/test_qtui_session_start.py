"""User-visible feedback for blocked Qt session starts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication

from face_tracker.qtui.controller import TrackerController
from face_tracker.storage import JournalStore


class SessionStartFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        store = JournalStore(Path(self.temp_dir.name) / "journal.sqlite3")
        self.controller = TrackerController(store=store)

    def tearDown(self) -> None:
        self.controller.close()
        self.temp_dir.cleanup()

    def test_missing_profile_shows_visible_next_step_and_keeps_session_stopped(self) -> None:
        initial_title = self.controller.stateTitle
        initial_hint = self.controller.cameraHint
        self.controller.startSession()

        self.assertFalse(self.controller.tracking)
        self.assertIsNone(self.controller.session)
        self.assertNotEqual(self.controller.stateTitle, initial_title)
        self.assertTrue(self.controller.stateDescription)
        self.assertNotEqual(self.controller.cameraHint, initial_hint)

    def test_calibration_or_camera_worker_shows_wait_state(self) -> None:
        self.controller.profile = object()
        self.controller.calibrating = True
        self.controller.worker = SimpleNamespace(running=True, stop=lambda timeout: True)
        initial_title = self.controller.stateTitle
        initial_hint = self.controller.cameraHint

        self.controller.startSession()

        self.assertFalse(self.controller.tracking)
        self.assertIsNone(self.controller.session)
        self.assertNotEqual(self.controller.stateTitle, initial_title)
        self.assertTrue(self.controller.stateDescription)
        self.assertNotEqual(self.controller.cameraHint, initial_hint)

    def test_valid_idle_profile_still_starts_session(self) -> None:
        self.controller.profile = object()
        self.controller._start_worker = lambda: None

        with patch("face_tracker.qtui.controller.Scorer") as scorer:
            self.controller.startSession()

        self.assertTrue(self.controller.tracking)
        self.assertIsNotNone(self.controller.session)
        scorer.assert_called_once_with(self.controller.profile)


if __name__ == "__main__":
    unittest.main()
