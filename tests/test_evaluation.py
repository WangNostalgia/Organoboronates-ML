import unittest
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RepeatedKFold

from src.evaluation import make_repeated_kfold_splits, repeated_kfold_evaluate


class RecordingZeroRegressor(BaseEstimator, RegressorMixin):
    fit_records = []

    def fit(self, X, y):
        self.__class__.fit_records.append(
            {
                "X_min": np.min(X, axis=0),
                "X_max": np.max(X, axis=0),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "n_train": len(X),
            }
        )
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class RepeatedKFoldEvaluationTests(unittest.TestCase):
    def setUp(self):
        RecordingZeroRegressor.fit_records = []

    def test_evaluation_source_preserves_original_unicode_text(self):
        source = Path("src/evaluation.py").read_text(encoding="utf-8")

        self.assertIn("Unified evaluation center (Single Source of Truth for 5×5 RepeatedKFold).", source)
        self.assertIn("All modules that need a rigorous, leakage-free 5×5 RepeatedKFold evaluation", source)
        self.assertIn("n_repeats : int, number of repeats (default 5 → 25 total evaluations)", source)

    def test_make_repeated_kfold_splits_matches_sklearn(self):
        X = np.arange(16, dtype=float).reshape(8, 2)

        splits = make_repeated_kfold_splits(
            X,
            n_splits=4,
            n_repeats=2,
            random_state=7,
        )
        expected = list(RepeatedKFold(n_splits=4, n_repeats=2, random_state=7).split(X))

        self.assertEqual(len(splits), 8)
        self.assertEqual(len(splits), len(expected))
        for (train_idx, test_idx), (expected_train, expected_test) in zip(splits, expected):
            np.testing.assert_array_equal(train_idx, expected_train)
            np.testing.assert_array_equal(test_idx, expected_test)

    def test_repeated_kfold_evaluate_reuses_supplied_splits_and_scores_in_original_units(self):
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([2, 3, 4, 5]), np.array([0, 1])),
        ]

        results = repeated_kfold_evaluate(
            RecordingZeroRegressor(),
            X,
            y,
            n_splits=5,
            n_repeats=5,
            random_state=999,
            splits=splits,
        )

        expected_maes = []
        expected_r2s = []
        for train_idx, test_idx in splits:
            y_train = y[train_idx]
            y_test = y[test_idx]
            y_pred = np.full(len(test_idx), y_train.min())

            expected_maes.append(np.mean(np.abs(y_test - y_pred)))

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            expected_r2s.append(1 - ss_res / ss_tot)

        self.assertEqual(results["n_evals"], len(splits))
        self.assertAlmostEqual(results["rkf_mae_mean"], float(np.mean(expected_maes)))
        self.assertAlmostEqual(results["rkf_mae_std"], float(np.std(expected_maes)))
        self.assertAlmostEqual(results["rkf_r2_mean"], float(np.mean(expected_r2s)))
        self.assertAlmostEqual(results["rkf_r2_std"], float(np.std(expected_r2s)))
        self.assertEqual(results["mae_mean"], results["rkf_mae_mean"])
        self.assertEqual(results["mae_std"], results["rkf_mae_std"])
        self.assertEqual(results["r2_mean"], results["rkf_r2_mean"])
        self.assertEqual(results["r2_std"], results["rkf_r2_std"])

        self.assertEqual(len(RecordingZeroRegressor.fit_records), len(splits))
        for record, (train_idx, _) in zip(RecordingZeroRegressor.fit_records, splits):
            self.assertEqual(record["n_train"], len(train_idx))
            np.testing.assert_allclose(record["X_min"], np.zeros(X.shape[1]))
            np.testing.assert_allclose(record["X_max"], np.ones(X.shape[1]))
            self.assertAlmostEqual(record["y_min"], 0.0)
            self.assertAlmostEqual(record["y_max"], 100.0)


if __name__ == "__main__":
    unittest.main()
