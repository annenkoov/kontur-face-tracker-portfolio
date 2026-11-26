import unittest

import numpy as np
import pandas as pd

from face_tracker.action_units import AU_LABELS_RU, PyFeatActionUnitAnalyzer


class FakeDetector:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return self.result


class ActionUnitAnalyzerTests(unittest.TestCase):
    def make_analyzer(self, result):
        analyzer = PyFeatActionUnitAnalyzer.__new__(PyFeatActionUnitAnalyzer)
        analyzer._detector = FakeDetector(result)
        analyzer.device = "cpu"
        return analyzer

    def test_returns_finite_model_scores_and_clips_invalid_range(self):
        row = {key: 0.25 for key in AU_LABELS_RU}
        row["AU01"] = 1.4
        row["AU02"] = -0.3
        row["AU04"] = float("nan")
        analyzer = self.make_analyzer(pd.DataFrame([row]))

        values = analyzer.analyze(np.zeros((180, 200, 3), dtype=np.uint8))

        self.assertEqual(values["AU01"], 1.0)
        self.assertEqual(values["AU02"], 0.0)
        self.assertNotIn("AU04", values)
        self.assertTrue(all(0 <= value <= 1 for value in values.values()))
        self.assertEqual(analyzer._detector.calls[0][1]["progress_bar"], False)

    def test_invalid_crop_is_ignored_without_running_the_model(self):
        analyzer = self.make_analyzer(pd.DataFrame())

        self.assertIsNone(analyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertEqual(analyzer._detector.calls, [])

    def test_empty_prediction_returns_no_action_units(self):
        analyzer = self.make_analyzer(pd.DataFrame())

        self.assertIsNone(analyzer.analyze(np.zeros((180, 200, 3), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
