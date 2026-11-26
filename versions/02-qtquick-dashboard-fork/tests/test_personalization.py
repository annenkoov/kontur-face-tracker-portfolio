import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import json

from face_tracker.labels import LABELS
from face_tracker.personalization import PersonalAdapter, fit_personal_adapter, load_adapter, save_adapter


def sample_rows(session_count=4):
    targets = ("Anger", "Neutral", "Sadness")
    rows = []
    for session_number in range(session_count):
        for target in targets:
            probabilities = {label: 0.08 / 6 for label in LABELS}
            if target == "Neutral":
                probabilities[target] = 0.50
                probabilities["Happiness"] = 0.42
            else:
                probabilities[target] = 0.42
                probabilities["Neutral"] = 0.50
            rows.append({"true_label": target, "session_id": f"session-{session_number}",
                         "created_at": f"2026-01-0{session_number + 1}T12:00:00+00:00",
                         "model_version": "enet_b0_8_best_vgaf", "preprocessing": "tight",
                         "probabilities": probabilities})
    return rows


class PersonalizationTests(unittest.TestCase):
    def test_too_few_independent_sessions_refuses_adaptation(self):
        result = fit_personal_adapter(sample_rows(3))
        self.assertFalse(result.activated)
        self.assertIsNone(result.adapter)
        self.assertIn("независимых", result.reason)

    def test_adapter_is_accepted_only_after_temporal_session_holdout_gain(self):
        result = fit_personal_adapter(sample_rows(6))
        self.assertTrue(result.activated, result.reason)
        self.assertGreaterEqual(result.adapter.adapted_macro_f1,
                                result.adapter.baseline_macro_f1 + 0.03)
        self.assertGreaterEqual(result.adapter.validation_sessions, 3)

    def test_five_sessions_are_not_enough_for_three_session_holdout(self):
        result = fit_personal_adapter(sample_rows(5))
        self.assertFalse(result.activated)
        self.assertIsNone(result.adapter)

    def test_adapter_roundtrip_and_model_mismatch_fall_back_to_generic(self):
        adapter = PersonalAdapter("enet_b0_8_best_vgaf", "tight", {label: 0.0 for label in LABELS},
"2026-01-01T00:00:00+00:00", 5, 3, 0.1, 0.2, 0.3, 0.3,
                                  ("Anger", "Neutral", "Sadness"))
        with TemporaryDirectory() as folder:
            path = save_adapter(adapter, Path(folder) / "adapter.json")
            loaded = load_adapter(path, expected_model="enet_b0_8_best_vgaf")
            self.assertEqual(loaded, adapter)
            self.assertIsNone(load_adapter(path, expected_model="different-model"))
            self.assertIsNone(load_adapter(path, expected_preprocessing="stored"))
        probabilities = {label: 1 / 8 for label in LABELS}
        adapted = adapter.apply(probabilities)
        self.assertAlmostEqual(sum(adapted.values()), 1.0)

    def test_corrupt_non_finite_artifact_is_not_loaded(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.json"
            data = {"model_name": "enet_b0_8_best_vgaf", "preprocessing": "tight",
                    "biases": {label: 0.0 for label in LABELS}, "trained_at": "x",
                    "training_sessions": 0, "validation_sessions": 0,
                    "baseline_macro_f1": float("nan"), "adapted_macro_f1": float("nan"),
                    "baseline_accuracy": float("nan"), "adapted_accuracy": float("nan"),
                    "covered_labels": ["Anger", "Neutral", "Sadness"]}
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNone(load_adapter(path))

    def test_unvalidated_rare_class_probabilities_remain_generic(self):
        adapter = PersonalAdapter("enet_b0_8_best_vgaf", "tight", {label: 1.0 for label in LABELS},
                                  "2026-01-01T00:00:00+00:00", 5, 3, 0.1, 0.2, 0.3, 0.3,
                                  ("Anger", "Neutral", "Sadness"))
        probabilities = {label: 0.02 for label in LABELS}
        probabilities["Happiness"] = 0.86
        self.assertEqual(adapter.apply(probabilities), probabilities)

    def test_adapter_with_bias_for_unvalidated_class_is_rejected(self):
        adapter = PersonalAdapter("enet_b0_8_best_vgaf", "tight", {label: 0.0 for label in LABELS},
                                  "2026-01-01T00:00:00+00:00", 5, 3, 0.1, 0.2, 0.3, 0.3,
                                  ("Anger", "Neutral", "Sadness"))
        data = adapter.__dict__ | {"covered_labels": list(adapter.covered_labels),
                                   "biases": dict(adapter.biases, Happiness=0.3)}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNone(load_adapter(path))

    def test_malformed_bias_shape_or_session_count_falls_back_safely(self):
        adapter = PersonalAdapter("enet_b0_8_best_vgaf", "tight", {label: 0.0 for label in LABELS},
                                  "2026-01-01T00:00:00+00:00", 5, 3, 0.1, 0.2, 0.3, 0.3,
                                  ("Anger", "Neutral", "Sadness"))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.json"
            for data in (
                adapter.__dict__ | {"covered_labels": list(adapter.covered_labels), "biases": list(LABELS)},
                adapter.__dict__ | {"covered_labels": list(adapter.covered_labels), "training_sessions": float("nan")},
            ):
                path.write_text(json.dumps(data), encoding="utf-8")
                self.assertIsNone(load_adapter(path))


if __name__ == "__main__":
    unittest.main()
