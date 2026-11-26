import unittest

from face_tracker.scoring import CalibrationProfile, CalibrationSampler, Scorer, apply_personal_sharpness_gate
from face_tracker.vision import EXPRESSION_SIGNAL_WEIGHTS, FaceObservation


def observation(t, features, expressions=None, quality=1.0, reason=None):
    return FaceObservation(t, True, quality, reason, features,
                           expressions if expressions is not None else full_expression("Neutral"))


def full_expression(label):
    return {name: float(name == label) for name in EXPRESSION_SIGNAL_WEIGHTS}


class ScoringTests(unittest.TestCase):
    def test_calibration_sampler_spaces_samples_and_requires_a_real_observation_window(self):
        sampler = CalibrationSampler(min_interval_seconds=0.1, minimum_duration_seconds=5.0)
        sampler.start(10.0)
        self.assertTrue(sampler.accept(10.0))
        self.assertFalse(sampler.accept(10.05))
        for index in range(1, 50):
            self.assertTrue(sampler.accept(10.0 + index * 0.1))
        self.assertFalse(sampler.ready(14.99, required_samples=50))
        self.assertTrue(sampler.ready(15.0, required_samples=50))

    def test_calibration_sampler_rejects_bad_time_and_resets_between_runs(self):
        sampler = CalibrationSampler()
        with self.assertRaises(ValueError):
            sampler.start(float("nan"))
        sampler.start(20.0)
        self.assertFalse(sampler.accept(float("nan")))
        self.assertFalse(sampler.accept(19.0))
        self.assertEqual(sampler.sample_count, 0)
        sampler.start(30.0)
        self.assertEqual(sampler.sample_count, 0)
        self.assertEqual(sampler.first_sample_at, None)

    def test_calibration_sampler_restarts_window_after_camera_gap(self):
        sampler = CalibrationSampler(min_interval_seconds=0.1, minimum_duration_seconds=5.0,
                                     max_gap_seconds=1.8)
        sampler.start(0.0)
        self.assertTrue(sampler.accept(0.0))
        for index in range(1, 50):
            self.assertTrue(sampler.accept(index * 0.1))
        self.assertTrue(sampler.ready(5.0, required_samples=50))
        self.assertTrue(sampler.accept(8.0))
        self.assertTrue(sampler.continuity_broken)
        self.assertEqual(sampler.sample_count, 1)
        self.assertEqual(sampler.first_sample_at, 8.0)
        self.assertFalse(sampler.ready(8.0, required_samples=50))

    def test_personal_sharpness_baseline_rejects_large_blur_drop(self):
        baseline = [
            FaceObservation(i, True, 0.9, features={"brow_down": 0.1},
                            expressions={"Neutral": 1.0}, sharpness=100.0)
            for i in range(5)
        ]
        profile = CalibrationProfile.from_observations(baseline)
        self.assertEqual(profile.sharpness_median, 100.0)
        self.assertAlmostEqual(profile.sharpness_threshold, 80.0)
        blurred = FaceObservation(6, True, 0.9, features={"brow_down": 0.5},
                                  expressions={"Anger": 1.0}, sharpness=30.0)
        gated = apply_personal_sharpness_gate(blurred, profile)
        self.assertEqual(gated.quality, 0.0)
        result = Scorer(profile).update(blurred, 6.0)
        self.assertFalse(result.usable)
        self.assertIn("смазан", result.reason.lower())

    def test_sharpness_gate_is_optional_for_legacy_profiles(self):
        profile = CalibrationProfile(medians={"brow_down": 0.1}, scales={"brow_down": 0.1})
        frame = observation(1, {"brow_down": 0.2})
        self.assertIs(apply_personal_sharpness_gate(frame, profile), frame)

    def test_non_finite_smoothing_configuration_is_rejected(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        with self.assertRaises(ValueError):
            Scorer(profile, max_gap_seconds=float("nan"))

    def test_calibration_is_median_based_and_scorer_fuses_both_components(self):
        baseline = [observation(i, {"brow_down": 0.1, "jaw_open": 0.05}) for i in range(5)]
        profile = CalibrationProfile.from_observations(baseline)
        scorer = Scorer(profile, smoothing_window=1, ema_alpha=1.0)
        result = scorer.update(observation(5, {"brow_down": 0.8, "jaw_open": 0.6},
                                              full_expression("Anger")), 5.0)
        self.assertEqual(result.expression_score, 100.0)
        self.assertGreater(result.deviation_score, 50.0)
        self.assertAlmostEqual(result.index, 0.5 * result.expression_score + 0.5 * result.deviation_score)
        self.assertTrue(result.usable)

    def test_unreliable_observation_pauses_scoring(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        result = Scorer(profile).update(observation(1, {}, quality=0.0, reason="Повернись лицом к камере"), 1.0)
        self.assertFalse(result.usable)
        self.assertEqual(result.reason, "Повернись лицом к камере")
        self.assertIsNone(result.index)

    def test_uniform_expression_probabilities_are_rejected_as_uncertain(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        uniform = {label: 0.125 for label in (
            "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"
        )}
        result = Scorer(profile).update(observation(6, {"brow_down": 0.8}, uniform), 6.0)
        self.assertFalse(result.usable)
        self.assertIn("неувер", result.reason.lower())

    def test_long_observation_gap_resets_score_smoothing(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        scorer = Scorer(profile, smoothing_window=5, ema_alpha=0.35)
        first = scorer.update(observation(1, {"brow_down": 0.1}, full_expression("Neutral")), 1.0)
        after_gap = scorer.update(observation(5, {"brow_down": 0.9}, full_expression("Anger")), 5.0)
        self.assertNotEqual(first.index, after_gap.index)
        self.assertEqual(after_gap.index, round(0.5 * after_gap.expression_score +
                                                0.5 * after_gap.deviation_score, 1))

    def test_non_finite_classifier_values_are_rejected(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        scorer = Scorer(profile)
        invalid = full_expression("Neutral")
        invalid["Anger"] = float("inf")
        result = scorer.update(observation(1, {"brow_down": 0.8}, invalid), 1.0)
        self.assertFalse(result.usable)
        self.assertIsNone(result.index)
        self.assertIn("некоррект", result.reason.lower())

    def test_non_finite_features_and_quality_are_rejected(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        scorer = Scorer(profile)
        bad_feature = scorer.update(observation(1, {"brow_down": float("nan")}), 1.0)
        bad_quality = scorer.update(observation(2, {"brow_down": 0.8}, quality=float("nan")), 2.0)
        self.assertFalse(bad_feature.usable)
        self.assertFalse(bad_quality.usable)

    def test_calibration_rejects_non_finite_feature_values(self):
        observations = [observation(i, {"brow_down": 0.1}) for i in range(4)]
        observations.append(observation(4, {"brow_down": float("nan")}))
        with self.assertRaisesRegex(ValueError, "надёжных наблюдений"):
            CalibrationProfile.from_observations(observations)

    def test_scorer_rejects_zero_or_non_finite_calibration_scales(self):
        for scale in (0.0, float("inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "калибров" ):
                Scorer(CalibrationProfile(medians={"brow_down": 0.1}, scales={"brow_down": scale}))

    def test_scorer_rejects_unknown_expression_labels(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1}) for i in range(5)]
        )
        result = Scorer(profile).update(observation(6, {"brow_down": 0.2}, {"Unknown": 1.0}), 6.0)
        self.assertFalse(result.usable)
        self.assertIn("класс", result.reason.lower())

    def test_scorer_rejects_incomplete_expression_and_calibration_feature_sets(self):
        profile = CalibrationProfile.from_observations(
            [observation(i, {"brow_down": 0.1, "jaw_open": 0.05}) for i in range(5)]
        )
        incomplete_classes = Scorer(profile).update(
            observation(6, {"brow_down": 0.8, "jaw_open": 0.6}, {"Neutral": 1.0}), 6.0
        )
        incomplete_features = Scorer(profile).update(
            observation(7, {"brow_down": 0.8}, full_expression("Anger")), 7.0
        )
        self.assertFalse(incomplete_classes.usable)
        self.assertIn("неполный", incomplete_classes.reason.lower())
        self.assertFalse(incomplete_features.usable)
        self.assertIn("не хватает", incomplete_features.reason.lower())


if __name__ == "__main__":
    unittest.main()
