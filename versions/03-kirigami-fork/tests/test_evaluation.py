import unittest
import math

from face_tracker.evaluation import (
    classification_report,
    evaluate_rows,
    self_rating_summary,
    probability_quality_report,
    spearman_correlation,
    top_expressions,
)


class ClassificationReportTests(unittest.TestCase):
    def test_spearman_uses_average_ranks_for_ties(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]), 1.0)
        self.assertIsNone(spearman_correlation([1, 1], [2, 2]))

    def test_self_rating_summary_requires_multiple_sessions_and_quality(self):
        rows = [
            {"rating": index, "face_index": index * 10, "face_quality": 0.9,
             "session_id": 1 + (index % 3), "rating_mode": "blind"}
            for index in range(10)
        ]
        self.assertEqual(self_rating_summary(rows)["rho"], 1.0)
        self.assertIsNone(self_rating_summary(rows[:9])["rho"])
        self.assertIsNone(self_rating_summary([{**row, "face_quality": 0.2} for row in rows])["rho"])

    def test_self_rating_correlation_is_centered_within_sessions(self):
        rows = []
        for session_id, start in ((1, 70), (2, 20), (3, 40)):
            for offset in range(4):
                rows.append({
                    "rating": offset,
                    "face_index": start + offset * 10,
                    "face_quality": 0.9,
                    "session_id": session_id,
                    "rating_mode": "blind",
                })
        summary = self_rating_summary(rows)
        self.assertAlmostEqual(summary["rho"], 1.0)

    def test_constant_sessions_do_not_count_as_informative(self):
        rows = []
        for session_id in range(1, 4):
            for index in range(4):
                rows.append({"rating": 5, "face_index": 50, "face_quality": 0.9,
                             "session_id": session_id})
        summary = self_rating_summary(rows)
        self.assertEqual(summary["sessions"], 0)
        self.assertIsNone(summary["rho"])

    def test_reports_accuracy_macro_f1_and_confusion_matrix(self):
        report = classification_report(
            ["happy", "happy", "sad", "sad"],
            ["happy", "sad", "sad", "sad"],
            labels=["happy", "sad"],
        )
        self.assertAlmostEqual(report.accuracy, 0.75)
        self.assertAlmostEqual(report.per_class["happy"].precision, 1.0)
        self.assertAlmostEqual(report.per_class["happy"].recall, 0.5)
        self.assertAlmostEqual(report.macro_f1, (2 / 3 + 0.8) / 2)
        self.assertEqual(report.confusion_matrix, ((1, 1), (0, 2)))

    def test_probability_quality_metrics_report_confident_predictions(self):
        report = probability_quality_report(
            ["happy", "sad"],
            [{"happy": 0.8, "sad": 0.2}, {"happy": 0.2, "sad": 0.8}],
            ["happy", "sad"], bins=10,
        )
        self.assertEqual(report["units"], 2)
        self.assertAlmostEqual(report["nll"], -math.log(0.8))
        self.assertAlmostEqual(report["brier_score"], 0.08)
        self.assertAlmostEqual(report["top_label_ece"], 0.2)

    def test_probability_quality_metrics_reject_invalid_rows(self):
        with self.assertRaises(ValueError):
            probability_quality_report(["happy"], [{"happy": 0.6, "sad": 0.3}], ["happy", "sad"])
        with self.assertRaises(ValueError):
            probability_quality_report(["other"], [{"happy": 0.5, "sad": 0.5}], ["happy", "sad"])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            classification_report(["happy"], [], labels=["happy"])

    def test_top_expressions_are_ranked_and_normalized(self):
        self.assertEqual(top_expressions({"sad": 2, "happy": 6, "neutral": 2}),
                         [("happy", 0.6), ("sad", 0.2), ("neutral", 0.2)])

    def test_manifest_groups_metrics_by_model_and_preprocessing(self):
        rows = [
            {"true_label": "happy", "predicted_label": "happy", "model_version": "m1", "preprocessing": "crop-a", "session_id": "s1"},
            {"true_label": "sad", "predicted_label": "happy", "model_version": "m1", "preprocessing": "crop-a", "session_id": "s2"},
        ]
        result = evaluate_rows(rows, ["happy", "sad"])
        self.assertEqual(result["comparisons"][0]["examples"], 2)
        self.assertEqual(result["comparisons"][0]["sessions"], 2)
        self.assertEqual(result["comparisons"][0]["model_version"], "m1")


if __name__ == "__main__":
    unittest.main()
