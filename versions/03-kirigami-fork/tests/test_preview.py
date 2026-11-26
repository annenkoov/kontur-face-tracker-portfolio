import unittest

import numpy as np

from face_tracker.ui import format_confidence_calibration, prepare_preview_frame
from face_tracker.vision import FACIAL_CONTOUR_CONNECTIONS


class PreviewFrameTests(unittest.TestCase):
    def test_confidence_calibration_format_handles_legacy_and_partial_reports(self):
        self.assertEqual(format_confidence_calibration({}), "—")
        self.assertEqual(format_confidence_calibration({"nll": 1.0}), "—")
        self.assertEqual(format_confidence_calibration({
            "top_label_ece": float("nan"), "brier_score": 0.4, "nll": 1.2
        }), "—")
        self.assertEqual(format_confidence_calibration({
            "top_label_ece": 0.1234, "brier_score": 0.4567, "nll": 1.234
        }), "ECE 0.123 · Brier 0.457 · NLL 1.23")

    def test_preview_copy_preserves_camera_left_right_order_and_source(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        source[0, 0] = (255, 0, 0)
        source[1, 2] = (0, 255, 0)
        preview = prepare_preview_frame(source)
        np.testing.assert_array_equal(preview, source)
        np.testing.assert_array_equal(source[0, 0], (255, 0, 0))
        np.testing.assert_array_equal(source[1, 2], (0, 255, 0))

    def test_landmark_overlay_uses_real_coordinates_without_mirroring_or_mutation(self):
        source = np.zeros((96, 96, 3), dtype=np.uint8)
        points = [(0.98, 0.98)] * 478
        points[10] = (0.25, 0.5)
        points[338] = (0.75, 0.5)
        self.assertIn((10, 338), FACIAL_CONTOUR_CONNECTIONS)

        preview = prepare_preview_frame(source, landmarks=points)

        self.assertFalse(np.any(source))
        self.assertTrue(np.any(preview[48, 24]))
        self.assertTrue(np.any(preview[48, 72]))
        self.assertFalse(np.any(preview[48, 8]))


if __name__ == "__main__":
    unittest.main()
