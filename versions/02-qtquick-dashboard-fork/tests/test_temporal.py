import unittest

from face_tracker.temporal import ExpressionSmoother


class ExpressionSmootherTests(unittest.TestCase):
    def test_non_finite_timing_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpressionSmoother(window_seconds=float("nan"))

    @staticmethod
    def p(anger=0.05, neutral=0.90, sadness=0.05):
        return {"Anger": anger, "Neutral": neutral, "Sadness": sadness}

    def test_requires_several_frames_and_time_before_reporting_stable_expression(self):
        smoother = ExpressionSmoother()
        for index in range(3):
            result = smoother.update(index * 0.1, 0.9, self.p())
        self.assertFalse(result.ready)
        result = smoother.update(0.4, 0.9, self.p())
        self.assertTrue(result.ready)
        self.assertEqual(result.label, "Neutral")

    def test_single_frame_spike_does_not_flip_stable_label(self):
        smoother = ExpressionSmoother(window_seconds=2.0, min_duration_seconds=0.2)
        for index in range(5):
            state = smoother.update(index * 0.1, 0.9, self.p())
        self.assertEqual(state.label, "Neutral")
        spike = smoother.update(0.5, 0.9, self.p(anger=0.92, neutral=0.04, sadness=0.04))
        self.assertEqual(spike.label, "Neutral")

    def test_camera_gap_and_bad_quality_clear_stale_consensus(self):
        smoother = ExpressionSmoother()
        for index in range(5):
            state = smoother.update(index * 0.1, 0.9, self.p())
        self.assertEqual(state.label, "Neutral")
        lost = smoother.update(3.0, 0.0, None)
        self.assertIsNone(lost.label)
        self.assertEqual(smoother.update(3.1, 0.9, self.p()).probabilities["Neutral"], 0.9)

    def test_timestamp_reversal_resets_the_window(self):
        smoother = ExpressionSmoother()
        for index in range(5):
            smoother.update(1.0 + index * 0.1, 0.9, self.p())
        result = smoother.update(0.5, 0.9, self.p(anger=0.8, neutral=0.1, sadness=0.1))
        self.assertFalse(result.ready)
        self.assertIsNone(result.label)


if __name__ == "__main__":
    unittest.main()
