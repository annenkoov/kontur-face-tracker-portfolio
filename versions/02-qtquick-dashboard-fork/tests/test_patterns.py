import unittest

from face_tracker.patterns import FacialPatternTracker
from face_tracker.scoring import CalibrationProfile
from face_tracker.vision import FaceObservation


def profile():
    names = ("brow_down", "eye_squint", "mouth_press", "mouth_frown", "jaw_open",
             "eye_wide", "upper_lip_raise", "smile", "brow_eye_coactivation", "mouth_coactivation",
             "brow_asymmetry", "eye_squint_asymmetry", "mouth_press_asymmetry")
    return CalibrationProfile(
        medians={"brow_down": 0.1}, scales={"brow_down": 0.04},
        pattern_medians={name: 0.1 for name in names}, pattern_scales={name: 0.04 for name in names},
    )


def values(active=0.1):
    return {name: active for name in profile().pattern_medians}


class FacialPatternTrackerTests(unittest.TestCase):
    def test_non_finite_window_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            FacialPatternTracker(profile(), window_seconds=float("nan"))

    def test_calibration_builds_separate_pattern_baselines(self):
        observations = [FaceObservation(
            timestamp=index * 0.1, present=True, quality=0.9,
            features={"brow_down": 0.1 + index * 0.01},
            patterns={"brow_down": 0.2 + index * 0.01, "eye_squint": 0.1},
        ) for index in range(5)]
        result = CalibrationProfile.from_observations(observations)
        self.assertAlmostEqual(result.pattern_medians["brow_down"], 0.22)
        self.assertAlmostEqual(result.pattern_medians["eye_squint"], 0.1)
        self.assertNotIn("eye_squint", result.medians)

    def test_one_frame_spike_does_not_create_a_sustained_pattern(self):
        tracker = FacialPatternTracker(profile())
        normal = values()
        for index in range(5):
            tracker.update(index * 0.2, 0.9, normal)
        spike = values()
        spike["eye_squint"] = 0.8
        result = tracker.update(1.0, 0.9, spike)
        self.assertFalse(result.signals["eye_squint"].above_baseline)

    def test_sustained_pattern_is_detected_after_dwell_and_coactivation(self):
        tracker = FacialPatternTracker(profile())
        normal = values()
        for index in range(8):
            tracker.update(index * 0.2, 0.9, normal)
        elevated = values()
        elevated.update({"eye_squint": 0.55, "brow_down": 0.6,
                         "brow_eye_coactivation": 0.55, "mouth_press": 0.1})
        for index in range(8, 20):
            result = tracker.update(index * 0.2, 0.9, elevated)
        self.assertTrue(result.signals["eye_squint"].above_baseline)
        self.assertTrue(result.signals["brow_eye_coactivation"].above_baseline)
        self.assertFalse(result.signals["mouth_press"].above_baseline)

    def test_bad_quality_and_long_gap_clear_temporal_pattern_state(self):
        tracker = FacialPatternTracker(profile())
        elevated = values(0.7)
        for index in range(20):
            result = tracker.update(index * 0.2, 0.9, elevated)
        self.assertTrue(result.signals["eye_squint"].above_baseline)
        lost = tracker.update(4.0, 0.0, None)
        self.assertFalse(lost.ready)
        self.assertFalse(lost.signals["eye_squint"].above_baseline)
        tracker.update(5.0, 0.9, elevated)  # long gap starts a fresh observation window
        fresh = tracker.update(5.2, 0.9, elevated)
        self.assertFalse(fresh.signals["eye_squint"].above_baseline)

    def test_invalid_pattern_values_are_rejected(self):
        tracker = FacialPatternTracker(profile())
        invalid = values()
        invalid["eye_squint"] = float("nan")
        result = tracker.update(1.0, 0.9, invalid)
        self.assertFalse(result.ready)
        self.assertIn("некоррект", result.reason.lower())

    def test_side_asymmetry_is_independent_and_compared_to_personal_baseline(self):
        tracker = FacialPatternTracker(profile())
        normal = values()
        for index in range(8):
            tracker.update(index * 0.2, 0.9, normal)
        elevated = values()
        elevated["brow_asymmetry"] = 0.65
        for index in range(8, 20):
            result = tracker.update(index * 0.2, 0.9, elevated)
        self.assertTrue(result.signals["brow_asymmetry"].above_baseline)
        self.assertFalse(result.signals["eye_squint_asymmetry"].above_baseline)


if __name__ == "__main__":
    unittest.main()
