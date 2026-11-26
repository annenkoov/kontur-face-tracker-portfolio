import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from face_tracker import emotion_evaluation
from face_tracker.emotion_evaluation import evaluate_predictions, top_two_summary


class EmotionEvaluationTests(unittest.TestCase):
    def test_sadness_metrics_are_label_keyed_when_label_order_is_permuted(self):
        result = evaluate_predictions(
            [("Sadness", "Neutral"), ("Sadness", "Sadness"), ("Neutral", "Neutral")],
            ["Neutral", "Sadness"],
        )
        self.assertEqual(result.confusion_matrix, ((1, 0), (1, 1)))
        self.assertEqual(result.per_class["Sadness"].recall, 0.5)

    def test_rejects_missing_or_duplicate_classes_and_bad_rows(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([("Sadness", "Sadness")], ["Sadness"])
        with self.assertRaises(ValueError):
            evaluate_predictions([("Sadness", "Other")], ["Sadness", "Neutral"])

    def test_top_two_tie_is_reported_as_ambiguous(self):
        self.assertEqual(top_two_summary({"Sadness": 0.4, "Neutral": 0.4, "Fear": 0.2}),
                         ("Sadness", "Neutral", True))

    def test_cli_returns_nonzero_when_no_model_was_evaluated(self):
        output = StringIO()
        with patch("sys.argv", ["emotion_evaluation", "--manifest", "unused.csv"]), \
             patch.object(emotion_evaluation, "_evaluate_manifest", return_value=0), \
             redirect_stdout(output):
            self.assertNotEqual(emotion_evaluation.main(), 0)
        self.assertIn("no model was evaluated", output.getvalue().lower())

    def test_cli_requires_full_frames_for_shared_production_crop_path(self):
        output = StringIO()
        with patch("sys.argv", ["emotion_evaluation", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                emotion_evaluation.main()
        self.assertEqual(result.exception.code, 0)
        help_text = " ".join(output.getvalue().lower().split())
        self.assertIn("full frames", help_text)
        self.assertIn("exactly one visible face", help_text)


if __name__ == "__main__":
    unittest.main()
