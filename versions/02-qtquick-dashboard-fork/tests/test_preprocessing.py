import unittest

import numpy as np

from face_tracker.model_registry import MODEL_SPECS, get_model_spec
from face_tracker.preprocessing import crop_variants, padded_face_crop, to_model_rgb


class PreprocessingTests(unittest.TestCase):
    def test_padded_face_crop_preserves_face_box_coordinates(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        crop, local_box = padded_face_crop(frame, (30, 20, 90, 80))
        self.assertEqual(crop.shape[:2], (78, 78))
        self.assertEqual(local_box, (9, 9, 69, 69))

    def test_crop_variants_make_reproducible_tight_variant_when_context_is_saved(self):
        image = np.zeros((130, 130, 3), dtype=np.uint8)
        variants = crop_variants(image, "mesh-bbox;saved=mesh-bbox+15pct-context-v1", (15, 15, 115, 115))
        self.assertEqual(set(variants), {"stored", "tight"})
        self.assertLess(variants["tight"].shape[0], variants["stored"].shape[0])

    def test_crop_variants_do_not_invent_context_for_old_tight_samples(self):
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        self.assertEqual(set(crop_variants(image, "mesh-bbox-v1")), {"stored"})

    def test_legacy_context_is_explicitly_marked_approximate(self):
        image = np.zeros((130, 130, 3), dtype=np.uint8)
        variants = crop_variants(image, "saved=mesh-bbox+15pct-context-v1")
        self.assertIn("tight_approx", variants)
        self.assertNotIn("tight", variants)

    def test_edge_clipped_padding_uses_exact_manifest_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        padded, local_box = padded_face_crop(frame, (0, 20, 60, 80))
        variants = crop_variants(padded, "mesh-bbox;saved=mesh-bbox+15pct-context-v1", local_box)
        self.assertEqual(variants["tight"].shape[:2], (60, 60))
        self.assertEqual(local_box, (0, 9, 60, 69))

    def test_opencv_bgr_image_is_converted_to_rgb_for_classifier(self):
        bgr = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        rgb = to_model_rgb(bgr)
        np.testing.assert_array_equal(rgb, np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8))

    def test_registry_only_exposes_supported_eight_class_models(self):
        self.assertEqual(len(MODEL_SPECS), 4)
        self.assertEqual(get_model_spec("enet_b0_8_va_mtl").expected_classes, 8)
        with self.assertRaises(ValueError):
            get_model_spec("unknown")


if __name__ == "__main__":
    unittest.main()
