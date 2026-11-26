import unittest
from types import SimpleNamespace

import numpy as np

from face_tracker.preprocessing import mediapipe_face_crop


class MediaPipeFaceCropTests(unittest.TestCase):
    def test_returns_tight_rgb_crop_and_box_from_normalized_landmarks(self):
        # A 10x8 synthetic RGB image with red-channel coordinates makes the
        # selected crop and color-order contract easy to verify.
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(10, dtype=np.uint8)[None, :]
        landmarks = [SimpleNamespace(x=0.2, y=0.25), SimpleNamespace(x=0.6, y=0.75)]

        crop, box = mediapipe_face_crop(image, landmarks)

        self.assertEqual(box, (2, 2, 6, 6))
        self.assertEqual(crop.shape, (4, 4, 3))
        np.testing.assert_array_equal(crop, image[2:6, 2:6])
        self.assertTrue(crop.flags.c_contiguous)

    def test_clamps_landmarks_at_frame_boundaries(self):
        image = np.ones((5, 6, 3), dtype=np.uint8)
        landmarks = [SimpleNamespace(x=-0.1, y=0.2), SimpleNamespace(x=1.2, y=0.8)]

        crop, box = mediapipe_face_crop(image, landmarks)

        self.assertEqual(box, (0, 1, 6, 4))
        self.assertEqual(crop.shape, (3, 6, 3))


if __name__ == "__main__":
    unittest.main()
