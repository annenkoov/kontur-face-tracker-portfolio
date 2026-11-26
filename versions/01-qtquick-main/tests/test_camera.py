import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from face_tracker.camera import CameraError, CameraSource


class CameraSourceTests(unittest.TestCase):
    @patch("face_tracker.camera.cv2.VideoCapture")
    def test_read_returns_bgr_frame_and_stop_releases_camera(self, capture_factory):
        capture = Mock()
        capture.isOpened.return_value = True
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = (255, 0, 0)
        frame[1, 2] = (0, 255, 0)
        capture.read.return_value = (True, frame)
        capture_factory.return_value = capture

        camera = CameraSource(index=0)
        camera.start()
        capture.set.assert_called_once_with(cv2.CAP_PROP_BUFFERSIZE, 1)
        result = camera.read()
        np.testing.assert_array_equal(result.frame, frame)
        self.assertGreater(result.captured_at, 0)
        self.assertGreater(result.captured_monotonic, 0)
        camera.stop()
        capture.release.assert_called_once()

    @patch("face_tracker.camera.cv2.VideoCapture")
    def test_read_raises_clear_error_when_camera_frame_unavailable(self, capture_factory):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        capture_factory.return_value = capture

        camera = CameraSource(index=0)
        camera.start()
        with self.assertRaises(CameraError):
            camera.read()
        camera.stop()


if __name__ == "__main__":
    unittest.main()
