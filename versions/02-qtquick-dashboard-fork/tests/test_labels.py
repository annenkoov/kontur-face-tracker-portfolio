import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from face_tracker.labels import LabeledSampleStore, observation_is_fresh


class LabeledSampleStoreTests(unittest.TestCase):
    def test_observation_freshness_rejects_stale_or_future_frames(self):
        self.assertTrue(observation_is_fresh(10.0, 11.0))
        self.assertFalse(observation_is_fresh(10.0, 12.1))
        self.assertFalse(observation_is_fresh(10.1, 10.0))

    def test_freshness_uses_monotonic_time_when_capture_clock_is_available(self):
        self.assertTrue(observation_is_fresh(10.0, 99999.0, 1.0,
                                             monotonic_timestamp=20.0, monotonic_now=20.5))
        self.assertFalse(observation_is_fresh(10.0, 10.0, 1.0,
                                              monotonic_timestamp=20.0, monotonic_now=22.0))

    def test_saves_only_requested_face_crop_and_records_metadata(self):
        with TemporaryDirectory() as folder:
            store = LabeledSampleStore(folder)
            frame = np.zeros((100, 120, 3), dtype=np.uint8)
            frame[20:80, 30:90] = (10, 20, 30)
            sample = store.save_crop(frame, (30, 20, 90, 80), "Happiness", 0.9,
                                     "model-v1", "mesh-tight", "session-1")
            self.assertTrue(sample.path.exists())
            self.assertEqual(sample.path.parent, Path(folder) / "faces")
            self.assertEqual(sample.label, "Happiness")
            self.assertEqual(sample.face_box, (9, 9, 69, 69))
            self.assertEqual(store.counts()["Happiness"], 1)
            self.assertEqual(store.list_samples()[0].session_id, "session-1")
            self.assertEqual(store.list_samples()[0].face_box, sample.face_box)
            store.close()

    def test_rejects_low_quality_and_invalid_boxes_without_writing(self):
        with TemporaryDirectory() as folder:
            store = LabeledSampleStore(folder)
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            with self.assertRaises(ValueError):
                store.save_crop(frame, (5, 5, 30, 30), "Neutral", 0.2, "m", "p", "s")
            with self.assertRaises(ValueError):
                store.save_crop(frame, (-1, 5, 30, 30), "Neutral", 1.0, "m", "p", "s")
            self.assertEqual(store.list_samples(), [])
            store.close()

    def test_delete_all_removes_owned_images_and_manifest_rows(self):
        with TemporaryDirectory() as folder:
            store = LabeledSampleStore(folder)
            frame = np.ones((30, 30, 3), dtype=np.uint8)
            sample = store.save_crop(frame, (2, 2, 28, 28), "Neutral", 1.0, "m", "p", "s")
            orphan = store.faces_dir / ("a" * 32 + ".jpg")
            orphan.write_bytes(b"interrupted crop write")
            unrelated = store.faces_dir / "notes.jpg"
            unrelated.write_bytes(b"not a managed crop")
            store.delete_all()
            self.assertFalse(sample.path.exists())
            self.assertFalse(orphan.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(store.list_samples(), [])
            store.close()

    def test_relabel_and_delete_one_sample(self):
        with TemporaryDirectory() as folder:
            store = LabeledSampleStore(folder)
            frame = np.ones((30, 30, 3), dtype=np.uint8)
            sample = store.save_crop(frame, (2, 2, 28, 28), "Neutral", 1.0, "m", "p", "s")
            self.assertTrue(store.update_label(sample.sample_id, "Sadness"))
            self.assertEqual(store.list_samples()[0].label, "Sadness")
            self.assertTrue(store.delete_sample(sample.sample_id))
            self.assertFalse(sample.path.exists())
            self.assertEqual(store.list_samples(), [])
            store.close()

    def test_relabel_rejects_unknown_classes(self):
        with TemporaryDirectory() as folder:
            store = LabeledSampleStore(folder)
            with self.assertRaises(ValueError):
                store.update_label("missing", "InternalState")
            store.close()


if __name__ == "__main__":
    unittest.main()
