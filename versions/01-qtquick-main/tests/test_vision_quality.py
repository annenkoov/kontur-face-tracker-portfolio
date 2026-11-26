import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import cv2

from face_tracker.vision import (
    REQUIRED_BLENDSHAPE_KEYS,
    _download_atomically,
    face_pose_degrees,
    face_sharpness,
    has_complete_blendshapes,
    pose_quality,
    softmax_probabilities,
)


class PoseQualityTests(unittest.TestCase):
    def test_blendshape_quality_requires_every_feature_key_and_finite_range(self):
        scores = {key: 0.1 for key in REQUIRED_BLENDSHAPE_KEYS}
        self.assertTrue(has_complete_blendshapes(scores))
        self.assertFalse(has_complete_blendshapes({}))
        missing = dict(scores)
        missing.pop(next(iter(REQUIRED_BLENDSHAPE_KEYS)))
        self.assertFalse(has_complete_blendshapes(missing))
        invalid = dict(scores)
        invalid[next(iter(REQUIRED_BLENDSHAPE_KEYS))] = float("nan")
        self.assertFalse(has_complete_blendshapes(invalid))

    def test_frontal_pose_has_full_quality(self):
        matrix = np.eye(4, dtype=np.float32)
        self.assertEqual(face_pose_degrees(matrix), (0.0, 0.0, 0.0))
        self.assertEqual(pose_quality(0.0, 0.0, 0.0), (1.0, None))

    def test_large_yaw_is_rejected_with_clear_reason(self):
        quality, reason = pose_quality(35.0, 0.0, 0.0)
        self.assertEqual(quality, 0.0)
        self.assertEqual(reason, "Повернись лицом к камере")

    def test_non_finite_pose_is_rejected(self):
        quality, reason = pose_quality(float("nan"), 0.0, 0.0)
        self.assertEqual(quality, 0.0)
        self.assertIsNotNone(reason)

    def test_softmax_rejects_non_finite_or_mismatched_logits(self):
        self.assertIsNone(softmax_probabilities(np.array([1.0, np.nan]), ("Neutral", "Anger")))
        self.assertIsNone(softmax_probabilities(np.array([1.0]), ("Neutral", "Anger")))

    def test_softmax_returns_finite_normalized_probabilities(self):
        result = softmax_probabilities(np.array([2.0, 1.0]), ("Neutral", "Anger"))
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertTrue(all(np.isfinite(value) and 0 <= value <= 1 for value in result.values()))

    def test_face_sharpness_drops_when_image_is_blurred_and_is_size_normalized(self):
        checker = np.indices((128, 128)).sum(axis=0) % 2 * 255
        sharp = face_sharpness(checker.astype(np.uint8))
        blurred = face_sharpness(cv2.GaussianBlur(checker.astype(np.uint8), (9, 9), 2.0))
        resized = face_sharpness(cv2.resize(checker.astype(np.uint8), (256, 256)))
        self.assertGreater(sharp, blurred)
        self.assertIsNotNone(resized)
        self.assertIsNone(face_sharpness(np.zeros((0, 0), dtype=np.uint8)))

    def test_invalid_download_is_removed_and_not_left_at_destination(self):
        with TemporaryDirectory() as folder:
            destination = Path(folder) / "model.onnx"
            def interrupted_download(_url, target):
                Path(target).write_bytes(b"partial")
                raise OSError("network interrupted")

            with patch("face_tracker.vision.urlretrieve", side_effect=interrupted_download):
                with self.assertRaises(OSError):
                    _download_atomically("https://example.invalid/model", destination, 10, lambda _p: None)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(folder).glob("*.part")), [])

    def test_invalid_cached_model_is_replaced_atomically(self):
        with TemporaryDirectory() as folder:
            destination = Path(folder) / "model.task"
            destination.write_bytes(b"corrupt old cache")

            def good_download(_url, target):
                Path(target).write_bytes(b"valid downloaded model")

            with patch("face_tracker.vision.urlretrieve", side_effect=good_download):
                _download_atomically("https://example.invalid/model", destination, 10,
                                     lambda path: self.assertEqual(path.read_bytes(), b"valid downloaded model"))
            self.assertEqual(destination.read_bytes(), b"valid downloaded model")
            self.assertEqual(list(Path(folder).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
