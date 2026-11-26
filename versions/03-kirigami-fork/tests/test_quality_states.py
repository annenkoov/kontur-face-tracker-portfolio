import unittest

from face_tracker.quality_states import (
    BLINK_SCORE_THRESHOLD,
    CLOSED_SCORE_THRESHOLD,
    EyeState,
    EyeStateStabilizer,
    QualityStatusDebouncer,
    classify_eye_state,
)
from face_tracker.scoring import CalibrationProfile, Scorer, apply_personal_sharpness_gate
from face_tracker.vision import (
    FaceObservation, EXPRESSION_SIGNAL_WEIGHTS, REQUIRED_BLENDSHAPE_KEYS,
    has_complete_blendshapes,
)


class EyeStateTests(unittest.TestCase):
    def test_thresholds_separate_open_blink_and_closed_states(self):
        self.assertLess(BLINK_SCORE_THRESHOLD, CLOSED_SCORE_THRESHOLD)
        open_state = classify_eye_state({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.2})
        blink_state = classify_eye_state({"eyeBlinkLeft": 0.7, "eyeBlinkRight": 0.7})
        closed_state = classify_eye_state({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9})
        self.assertEqual(open_state.state, "open")
        self.assertEqual(blink_state.state, "blink")
        self.assertEqual(closed_state.state, "closed")

    def test_one_eye_blink_is_detected_but_not_classified_as_bilateral(self):
        eye_state = classify_eye_state({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1})
        self.assertEqual(eye_state.state, "blink")
        self.assertFalse(eye_state.bilateral_blink)

    def test_missing_or_invalid_blink_blendshape_is_unknown_and_invalid(self):
        missing = classify_eye_state({"eyeBlinkLeft": 0.1})
        invalid = classify_eye_state({"eyeBlinkLeft": 0.1, "eyeBlinkRight": float("nan")})
        self.assertEqual(missing, EyeState("unknown", False))
        self.assertFalse(invalid.valid)

    def test_high_score_closure_is_a_blink_until_it_persists(self):
        stabilizer = EyeStateStabilizer(confirmation_seconds=0.4)
        raw_closed = classify_eye_state({"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95})
        self.assertEqual(stabilizer.update(1.0, raw_closed).state, "blink")
        self.assertEqual(stabilizer.update(1.39, raw_closed).state, "blink")
        self.assertEqual(stabilizer.update(1.4, raw_closed).state, "closed")

    def test_open_frame_resets_persistent_closure_timer(self):
        stabilizer = EyeStateStabilizer(confirmation_seconds=0.4)
        closed = classify_eye_state({"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95})
        opened = classify_eye_state({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1})
        stabilizer.update(1.0, closed)
        self.assertEqual(stabilizer.update(1.2, opened).state, "open")
        self.assertEqual(stabilizer.update(1.21, closed).state, "blink")

    def test_required_blendshape_set_includes_both_eye_blink_scores(self):
        complete = {name: 0.2 for name in REQUIRED_BLENDSHAPE_KEYS}
        self.assertTrue(has_complete_blendshapes(complete))
        self.assertFalse(has_complete_blendshapes(complete | {"eyeBlinkRight": None}))
        self.assertFalse(has_complete_blendshapes({key: value for key, value in complete.items()
                                                   if key != "eyeBlinkRight"}))


class SharpnessGateTests(unittest.TestCase):
    def setUp(self):
        self.profile = CalibrationProfile(
            medians={"brow_down": 0.1}, scales={"brow_down": 0.1},
            sharpness_median=100.0, sharpness_scale=1.0,
        )

    def test_temporary_bilateral_blink_does_not_become_a_blur_warning(self):
        eye_state = classify_eye_state({"eyeBlinkLeft": 0.7, "eyeBlinkRight": 0.7})
        frame = FaceObservation(1.0, True, 0.9, features={"brow_down": 0.2},
                                expressions={name: float(name == "Neutral") for name in EXPRESSION_SIGNAL_WEIGHTS},
                                sharpness=10.0, eye_state=eye_state)
        gated = apply_personal_sharpness_gate(frame, self.profile)
        self.assertEqual(gated.quality, frame.quality)
        self.assertIsNone(gated.quality_reason)

    def test_open_eyes_still_reject_true_optical_blur(self):
        eye_state = classify_eye_state({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1})
        frame = FaceObservation(1.0, True, 0.9, features={"brow_down": 0.2},
                                expressions={name: float(name == "Neutral") for name in EXPRESSION_SIGNAL_WEIGHTS},
                                sharpness=10.0, eye_state=eye_state)
        result = Scorer(self.profile).update(frame, 1.0)
        self.assertFalse(result.usable)
        self.assertIn("смазан", result.reason.lower())

    def test_one_eye_blink_does_not_exempt_blur_gate(self):
        eye_state = classify_eye_state({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.1})
        frame = FaceObservation(1.0, True, 0.9, features={"brow_down": 0.2},
                                expressions={name: float(name == "Neutral") for name in EXPRESSION_SIGNAL_WEIGHTS},
                                sharpness=10.0, eye_state=eye_state)
        result = Scorer(self.profile).update(frame, 1.0)
        self.assertFalse(result.usable)
        self.assertIn("смазан", result.reason.lower())

    def test_persistent_closed_eyes_are_not_exempted_from_quality_checks(self):
        eye_state = classify_eye_state({"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95})
        frame = FaceObservation(1.0, True, 0.0, "Глаза закрыты",
                                features={"brow_down": 0.2}, sharpness=100.0,
                                eye_state=eye_state)
        result = Scorer(self.profile).update(frame, 1.0)
        self.assertFalse(result.usable)
        self.assertEqual(result.reason, "Глаза закрыты")
        self.assertIs(apply_personal_sharpness_gate(frame, self.profile), frame)


class QualityStatusDebouncerTests(unittest.TestCase):
    def test_transient_no_face_is_suppressed_until_it_persists_for_point_four_seconds(self):
        debouncer = QualityStatusDebouncer(confirmation_seconds=0.4)
        self.assertEqual(debouncer.update(0.0, True, True, None), "Камера работает локально")
        self.assertEqual(debouncer.update(0.1, False, True, "Лицо не найдено"), "Камера работает локально")
        self.assertEqual(debouncer.update(0.49, True, True, None), "Камера работает локально")

    def test_sustained_no_face_is_shown_after_confirmation_window(self):
        debouncer = QualityStatusDebouncer(confirmation_seconds=0.4)
        debouncer.update(0.0, True, True, None)
        debouncer.update(1.0, False, True, "Лицо не найдено")
        self.assertEqual(debouncer.update(1.39, False, True, "Лицо не найдено"),
                         "Камера работает локально")
        self.assertEqual(debouncer.update(1.4, False, True, "Лицо не найдено"), "Лицо не найдено")

    def test_stale_frame_is_not_reported_as_a_usable_score_state(self):
        debouncer = QualityStatusDebouncer(confirmation_seconds=0.4)
        debouncer.update(0.0, True, True, None)
        self.assertEqual(debouncer.update(1.0, True, False, "Нет свежего кадра"),
                         "Камера работает локально")
        self.assertEqual(debouncer.update(1.4, True, False, "Нет свежего кадра"),
                         "Нет свежего кадра")

    def test_fresh_frame_clears_confirmed_loss_before_a_new_transient_loss(self):
        debouncer = QualityStatusDebouncer(confirmation_seconds=0.4)
        debouncer.update(0.0, True, True, None)
        debouncer.update(1.0, False, True, "Лицо не найдено")
        debouncer.update(1.4, False, True, "Лицо не найдено")
        self.assertEqual(debouncer.update(1.41, True, True, None), "Камера работает локально")
        self.assertEqual(debouncer.update(1.42, False, True, "Лицо не найдено"),
                         "Камера работает локально")


if __name__ == "__main__":
    unittest.main()
