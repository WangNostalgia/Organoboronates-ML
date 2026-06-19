import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from src.applicability_domain import (
    _compute_knn_distance,
    _oof_residual_scale,
    _training_oof_predictions,
    applicability_domain_analysis,
    main,
)


class RecordingMinMaxScaler:
    records = []

    def __init__(self, *args, **kwargs):
        self.feature_range = kwargs.get("feature_range", (0, 1))
        self._delegate = SklearnMinMaxScaler(*args, **kwargs)

    @classmethod
    def reset(cls):
        cls.records = []

    def fit(self, X, y=None):
        self.__class__.records.append(
            {
                "feature_range": self.feature_range,
                "values": np.asarray(X, dtype=float).copy(),
            }
        )
        self._delegate.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        self.__class__.records.append(
            {
                "feature_range": self.feature_range,
                "values": np.asarray(X, dtype=float).copy(),
            }
        )
        return self._delegate.fit_transform(X, y)

    def transform(self, X):
        return self._delegate.transform(X)

    def inverse_transform(self, X):
        return self._delegate.inverse_transform(X)


class MeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def make_linear_model_info(X, y):
    scaler_X = SklearnMinMaxScaler()
    scaler_y = SklearnMinMaxScaler(feature_range=(0, 100))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    model = LinearRegression(n_jobs=1)
    model.fit(X_scaled, y_scaled)
    return {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "features": list(X.columns),
    }


class ApplicabilityDomainTests(unittest.TestCase):
    def setUp(self):
        self.loky_env = patch.dict(os.environ, {"LOKY_MAX_CPU_COUNT": "1"})
        self.loky_env.start()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name) / "outputs"

    def tearDown(self):
        self.tmpdir.cleanup()
        self.loky_env.stop()

    def test_oof_residual_scale_uses_exact_mad_formula(self):
        residuals = np.array([1.0, 2.0, 100.0])

        self.assertAlmostEqual(_oof_residual_scale(residuals), 1.4826)

    def test_oof_residual_scale_falls_back_to_sample_std_when_mad_zero(self):
        residuals = np.array([2.0, 2.0, 2.0, 4.0])

        self.assertAlmostEqual(_oof_residual_scale(residuals), 1.0)

    def test_oof_residual_scale_returns_epsilon_when_spread_is_zero(self):
        residuals = np.zeros(4, dtype=float)

        self.assertEqual(_oof_residual_scale(residuals), np.finfo(float).eps)

    def test_training_oof_predictions_cover_every_sample_and_fit_fold_local_scalers(self):
        X = pd.DataFrame({"f1": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
        y = pd.Series([3.0, 8.0, 14.0, 21.0, 29.0, 38.0], name="activation_energy")

        RecordingMinMaxScaler.reset()
        with patch("src.applicability_domain.MinMaxScaler", RecordingMinMaxScaler):
            preds = _training_oof_predictions(
                LinearRegression(n_jobs=1),
                X,
                y,
                n_splits=3,
                random_state=42,
            )

        self.assertEqual(preds.shape, (len(X),))
        self.assertTrue(np.isfinite(preds).all())

        x_fit_records = [
            record["values"]
            for record in RecordingMinMaxScaler.records
            if tuple(record["feature_range"]) == (0, 1)
        ]
        y_fit_records = [
            record["values"]
            for record in RecordingMinMaxScaler.records
            if tuple(record["feature_range"]) == (0, 100)
        ]

        self.assertEqual(len(x_fit_records), 3)
        self.assertEqual(len(y_fit_records), 3)

        expected_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(X))
        expected_x_trains = [X.iloc[train_idx].to_numpy() for train_idx, _ in expected_splits]
        expected_y_trains = [y.iloc[train_idx].to_numpy().reshape(-1, 1) for train_idx, _ in expected_splits]

        for observed, expected in zip(x_fit_records, expected_x_trains):
            np.testing.assert_allclose(observed, expected)
            self.assertLess(len(observed), len(X))

        for observed, expected in zip(y_fit_records, expected_y_trains):
            np.testing.assert_allclose(observed, expected)
            self.assertLess(len(observed), len(y))

    def test_training_oof_predictions_reduces_default_folds_for_small_training_sets(self):
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        y = pd.Series([4.0, 5.0, 6.0], name="activation_energy")

        preds = _training_oof_predictions(MeanRegressor(), X, y)

        self.assertEqual(preds.shape, (3,))
        self.assertTrue(np.isfinite(preds).all())

    def test_applicability_domain_requires_y_train_for_residual_calibration(self):
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 3.0, 5.0]})
        y_train = pd.Series([2.0, 4.0, 6.0], name="activation_energy")
        model_info = make_linear_model_info(X_train, y_train)
        X_external = pd.DataFrame({"f1": [1.5], "f2": [4.0]})
        y_external = np.array([5.0])

        with self.assertRaisesRegex(ValueError, "y_train"):
            applicability_domain_analysis(
                model_info=model_info,
                X_train=X_train,
                X_external=X_external,
                y_train=None,
                y_external=y_external,
                k_neighbors=2,
                output_dir=str(self.output_dir),
            )

    def test_external_standardized_residual_does_not_include_sqrt_leverage_factor(self):
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [0.5, 1.5, 2.5, 3.5]})
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0], name="activation_energy")
        model_info = make_linear_model_info(X_train, y_train)
        X_external = pd.DataFrame({"f1": [4.0], "f2": [4.5]})

        with patch("src.applicability_domain._training_oof_predictions", return_value=np.asarray(y_train, dtype=float)):
            with patch("src.applicability_domain._oof_residual_scale", return_value=2.0):
                results = applicability_domain_analysis(
                    model_info=model_info,
                    X_train=X_train,
                    X_external=X_external,
                    y_train=y_train,
                    y_external=np.array([10.0]),
                    y_pred_external=np.array([6.0]),
                    k_neighbors=3,
                    output_dir=str(self.output_dir),
                )

        self.assertAlmostEqual(results["ad_results"]["std_residual"].iloc[0], 2.0)

    def test_prediction_only_mode_renders_leverage_only_williams_plot(self):
        X_train = pd.DataFrame(
            {"f1": [0.0, 1.0, 2.0, 3.0, 4.0], "f2": [0.2, 0.4, 0.6, 0.8, 1.0]}
        )
        y_train = pd.Series([1.0, 2.0, 2.5, 3.5, 4.5], name="activation_energy")
        model_info = make_linear_model_info(X_train, y_train)
        X_external = pd.DataFrame({"f1": [4.5, 5.0], "f2": [1.1, 1.3]})

        results = applicability_domain_analysis(
            model_info=model_info,
            X_train=X_train,
            X_external=X_external,
            y_train=y_train,
            y_external=None,
            k_neighbors=4,
            output_dir=str(self.output_dir),
            model_name="LinearRegression",
        )

        self.assertTrue(results["ad_results"]["std_residual"].isna().all())
        self.assertEqual(results["ad_summary"]["n_williams_high_residual"], 0)
        self.assertTrue(any(path.endswith(".png") and "williams_plot" in path for path in results["output_files"]))
        summary_path = next(
            path
            for path in results["output_files"]
            if "ad_summary" in Path(path).name and path.endswith(".txt")
        )
        summary_text = Path(summary_path).read_text(encoding="utf-8")
        self.assertIn("Total external samples:          2", summary_text)
        self.assertRegex(summary_text, r"Compounds flagged:\s+\d+")
        self.assertRegex(summary_text, r"Percentage flagged:\s+\d+\.\d%")
        interpretation = summary_text.split("--- Interpretation ---", 1)[1]
        self.assertIn("leverage", interpretation.lower())
        self.assertIn("k-NN", interpretation)
        self.assertNotIn("residual", interpretation.lower())

    def test_knn_distance_uses_single_job_estimator_to_avoid_loky_probe_warning(self):
        X_train = np.array([[0.0], [0.5], [1.0], [1.5]])
        X_external = np.array([[0.25], [1.25]])

        with patch("src.applicability_domain.NearestNeighbors") as nn_class:
            nn_class.return_value.fit.return_value = nn_class.return_value
            nn_class.return_value.kneighbors.side_effect = [
                (
                    np.array(
                        [
                            [0.0, 0.5, 1.0],
                            [0.0, 0.5, 0.5],
                            [0.0, 0.5, 0.5],
                            [0.0, 0.5, 1.0],
                        ]
                    ),
                    np.zeros((4, 3), dtype=int),
                ),
                (
                    np.array([[0.25, 0.25], [0.25, 0.25]]),
                    np.zeros((2, 2), dtype=int),
                ),
            ]

            _compute_knn_distance(
                X_train_scaled=X_train,
                X_ext_scaled=X_external,
                k_neighbors=2,
                z_threshold=3.0,
            )

        self.assertEqual(nn_class.call_args.kwargs["n_jobs"], 1)

    def test_cli_extracts_y_train_and_excludes_target_from_training_features(self):
        training_csv = Path(self.tmpdir.name) / "training.csv"
        external_csv = Path(self.tmpdir.name) / "external.csv"

        pd.DataFrame(
            {
                "sub_H": ["a", "b", "c"],
                "sub_B": ["x", "y", "z"],
                "f1": [0.0, 1.0, 2.0],
                "f2": [1.0, 2.0, 3.0],
                "activation_energy": [5.0, 6.0, 7.0],
            }
        ).to_csv(training_csv, index=False)
        pd.DataFrame({"sub_H": ["d"], "sub_B": ["w"], "f1": [1.5], "f2": [2.5]}).to_csv(
            external_csv,
            index=False,
        )

        captured = {}

        def fake_analysis(**kwargs):
            captured.update(kwargs)
            return {
                "ad_summary": {
                    "n_total": 1,
                    "h_star": 0.1,
                    "n_williams_high_leverage": 0,
                    "n_williams_high_residual": 0,
                    "n_williams_warning": 0,
                    "knn_training_mean": 0.1,
                    "knn_threshold": 0.2,
                    "n_knn_warning": 0,
                    "n_combined_warning": 0,
                },
                "output_files": [],
            }

        with patch("src.external_validation.load_model", return_value={"features": ["f1", "f2"]}):
            with patch("src.applicability_domain.applicability_domain_analysis", side_effect=fake_analysis):
                argv = [
                    "src.applicability_domain",
                    "--model",
                    "SVR",
                    "--training",
                    str(training_csv),
                    "--external",
                    str(external_csv),
                    "--output-dir",
                    str(self.output_dir),
                ]
                with patch.object(sys, "argv", argv):
                    main()

        self.assertIn("y_train", captured)
        np.testing.assert_allclose(np.asarray(captured["y_train"], dtype=float), np.array([5.0, 6.0, 7.0]))
        self.assertNotIn("activation_energy", captured["X_train"].columns)

    def test_cli_raises_clear_error_when_training_target_column_is_missing(self):
        training_csv = Path(self.tmpdir.name) / "training_missing_target.csv"
        external_csv = Path(self.tmpdir.name) / "external.csv"

        pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 3.0]}).to_csv(training_csv, index=False)
        pd.DataFrame({"f1": [1.5], "f2": [2.5]}).to_csv(external_csv, index=False)

        with patch("src.external_validation.load_model", return_value={"features": ["f1", "f2"]}):
            argv = [
                "src.applicability_domain",
                "--model",
                "SVR",
                "--training",
                str(training_csv),
                "--external",
                str(external_csv),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "activation_energy"):
                    main()

    def test_applicability_domain_validates_inputs_and_neighbor_count(self):
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 2.0, 3.0]})
        y_train = pd.Series([2.0, 3.0, 4.0], name="activation_energy")
        model_info = make_linear_model_info(X_train, y_train)
        X_external = pd.DataFrame({"f1": [1.5], "f2": [2.5]})

        with self.subTest("non-finite training feature"):
            bad_train = X_train.copy()
            bad_train.iloc[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "training"):
                applicability_domain_analysis(
                    model_info=model_info,
                    X_train=bad_train,
                    X_external=X_external,
                    y_train=y_train,
                    y_external=None,
                    output_dir=str(self.output_dir),
                )

        with self.subTest("non-finite external feature"):
            bad_external = X_external.copy()
            bad_external.iloc[0, 1] = np.inf
            with self.assertRaisesRegex(ValueError, "external"):
                applicability_domain_analysis(
                    model_info=model_info,
                    X_train=X_train,
                    X_external=bad_external,
                    y_train=y_train,
                    y_external=None,
                    output_dir=str(self.output_dir),
                )

        with self.subTest("too few training samples"):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                applicability_domain_analysis(
                    model_info=model_info,
                    X_train=X_train.iloc[:1],
                    X_external=X_external,
                    y_train=y_train.iloc[:1],
                    y_external=None,
                    output_dir=str(self.output_dir),
                )

        with self.subTest("invalid y_train length"):
            with self.assertRaisesRegex(ValueError, "y_train"):
                applicability_domain_analysis(
                    model_info=model_info,
                    X_train=X_train,
                    X_external=X_external,
                    y_train=y_train.iloc[:2],
                    y_external=None,
                    k_neighbors=2,
                    output_dir=str(self.output_dir),
                )

        with self.subTest("invalid k_neighbors lower bound"):
            with self.assertRaisesRegex(ValueError, "k_neighbors"):
                applicability_domain_analysis(
                    model_info=model_info,
                    X_train=X_train,
                    X_external=X_external,
                    y_train=y_train,
                    y_external=None,
                    k_neighbors=0,
                    output_dir=str(self.output_dir),
                )

        with self.subTest("invalid k_neighbors upper bound"):
            with self.assertRaisesRegex(ValueError, "k_neighbors"):
                applicability_domain_analysis(
                    model_info=model_info,
                    X_train=X_train,
                    X_external=X_external,
                    y_train=y_train,
                    y_external=None,
                    k_neighbors=3,
                    output_dir=str(self.output_dir),
                )


if __name__ == "__main__":
    unittest.main()
