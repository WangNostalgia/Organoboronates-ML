import unittest
from unittest.mock import patch
import sys
import types
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.utils.validation import check_is_fitted


def install_training_selection_test_stubs():
    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")

        class FakeTrial:
            def __init__(self):
                self.params = {}
                self.user_attrs = {}

            def suggest_float(self, name, low, high, log=False):
                value = low
                self.params[name] = value
                return value

            def suggest_int(self, name, low, high, step=1):
                value = low
                self.params[name] = value
                return value

            def suggest_categorical(self, name, choices):
                value = choices[0]
                self.params[name] = value
                return value

            def set_user_attr(self, name, value):
                self.user_attrs[name] = value

        class FakeStudy:
            def __init__(self):
                self.best_value = None
                self.best_params = {}
                self.best_trial = None

            def optimize(self, objective, n_jobs=1, n_trials=1):
                best_trial = None
                best_value = None
                for _ in range(n_trials):
                    trial = FakeTrial()
                    value = objective(trial)
                    if best_value is None or value < best_value:
                        best_value = value
                        best_trial = trial
                self.best_value = best_value
                self.best_params = dict(best_trial.params)
                self.best_trial = best_trial

        class FakeTPESampler:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        optuna.samplers = types.SimpleNamespace(TPESampler=FakeTPESampler)
        optuna.create_study = lambda sampler=None, direction="minimize": FakeStudy()
        sys.modules["optuna"] = optuna

    def install_optional_regressor_module(module_name, class_name):
        if module_name in sys.modules:
            return
        module = types.ModuleType(module_name)

        class OptionalRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, X, y):
                self._mean = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), getattr(self, "_mean", 0.0))

            def get_params(self, deep=False):
                return dict(self.kwargs)

        setattr(module, class_name, OptionalRegressor)
        sys.modules[module_name] = module

    install_optional_regressor_module("xgboost", "XGBRegressor")
    install_optional_regressor_module("lightgbm", "LGBMRegressor")
    install_optional_regressor_module("catboost", "CatBoostRegressor")


install_training_selection_test_stubs()

from src.fixed_params import get_fixed_params
from src.hyperparameter_optimization_and_training import (
    hyperparameter_optimization_and_training,
)
from src.train_and_evaluate import _select_alpha_via_inner_cv, train_and_evaluate


def make_development_dataset():
    X = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0],
            "f2": [10.0, 9.0, 7.0, 6.0, 5.0, 3.0, 2.0, 1.0, -1.0, -2.0],
        }
    )
    y = pd.Series(
        [12.0, 15.0, 20.0, 23.0, 28.0, 31.0, 36.0, 40.0, 45.0, 49.0],
        name="activation_energy",
    )
    return X, y


def manual_alpha_maes(model_class, X, y, candidate_alphas, random_state, inner_splits, extra_params=None):
    extra_params = extra_params or {}
    kfold = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    maes = {}

    for alpha in candidate_alphas:
        fold_maes = []
        for train_idx, test_idx in kfold.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            scaler_X = SklearnMinMaxScaler()
            scaler_y = SklearnMinMaxScaler(feature_range=(0, 100))
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()

            model = model_class(alpha=alpha, **extra_params)
            model.fit(X_train_scaled, y_train_scaled)

            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            fold_maes.append(float(np.mean(np.abs(y_test.to_numpy() - y_pred))))

        maes[alpha] = float(np.mean(fold_maes))

    return maes


class RecordingMinMaxScaler:
    fit_shapes = []

    def __init__(self, *args, **kwargs):
        self._delegate = SklearnMinMaxScaler(*args, **kwargs)

    @classmethod
    def reset(cls):
        cls.fit_shapes = []

    def fit_transform(self, X, y=None):
        self.__class__.fit_shapes.append(tuple(np.asarray(X).shape))
        return self._delegate.fit_transform(X, y)

    def transform(self, X):
        return self._delegate.transform(X)

    def inverse_transform(self, X):
        return self._delegate.inverse_transform(X)


class TrainingSelectionTests(unittest.TestCase):
    def test_train_and_evaluate_source_preserves_unicode_log_text(self):
        source = Path("src/train_and_evaluate.py").read_text(encoding="utf-8")

        self.assertIn("5×5 RepeatedKFold: MAE = %.4f ± %.4f | R² = %.4f ± %.4f (%d evaluations)", source)
        self.assertIn('"100-split stability MAE: %.4f ± %.4f"', source)
        self.assertNotIn("卤", source)
        self.assertNotIn("虏", source)
        self.assertNotIn("5x5 RepeatedKFold", source)

    def test_select_alpha_helper_no_longer_exposes_unused_base_params_argument(self):
        signature = inspect.signature(_select_alpha_via_inner_cv)

        self.assertNotIn("base_params", signature.parameters)

    def test_ridge_alpha_selection_uses_fold_local_scaling_and_original_unit_mae(self):
        X, y = make_development_dataset()
        candidate_alphas = np.array([0.1, 1.0])
        expected_maes = manual_alpha_maes(
            Ridge,
            X,
            y,
            candidate_alphas=candidate_alphas,
            random_state=7,
            inner_splits=2,
        )

        RecordingMinMaxScaler.reset()
        with patch("src.train_and_evaluate.MinMaxScaler", RecordingMinMaxScaler):
            best_alpha, best_mae = _select_alpha_via_inner_cv(
                Ridge,
                X,
                y,
                candidate_alphas=candidate_alphas,
                random_state=7,
                inner_splits=2,
            )

        self.assertIn(best_alpha, candidate_alphas)
        self.assertAlmostEqual(best_mae, min(expected_maes.values()))
        self.assertEqual(
            RecordingMinMaxScaler.fit_shapes,
            [(5, 2), (5, 1), (5, 2), (5, 1), (5, 2), (5, 1), (5, 2), (5, 1)],
        )

    def test_lasso_alpha_selection_uses_fold_local_scaling_and_original_unit_mae(self):
        X, y = make_development_dataset()
        candidate_alphas = np.array([0.01, 0.1])
        lasso_params = get_fixed_params(Lasso, n_jobs=1)
        extra_params = {
            "max_iter": lasso_params["max_iter"],
            "selection": lasso_params["selection"],
            "random_state": lasso_params["random_state"],
            "tol": 1e-4,
        }
        expected_maes = manual_alpha_maes(
            Lasso,
            X,
            y,
            candidate_alphas=candidate_alphas,
            random_state=11,
            inner_splits=2,
            extra_params=extra_params,
        )

        RecordingMinMaxScaler.reset()
        with patch("src.train_and_evaluate.MinMaxScaler", RecordingMinMaxScaler):
            best_alpha, best_mae = _select_alpha_via_inner_cv(
                Lasso,
                X,
                y,
                tuned_params={"tol": 1e-4},
                candidate_alphas=candidate_alphas,
                random_state=11,
                inner_splits=2,
            )

        self.assertIn(best_alpha, candidate_alphas)
        self.assertAlmostEqual(best_mae, min(expected_maes.values()))
        self.assertEqual(
            RecordingMinMaxScaler.fit_shapes,
            [(5, 2), (5, 1), (5, 2), (5, 1), (5, 2), (5, 1), (5, 2), (5, 1)],
        )

    def test_train_and_evaluate_returns_development_only_artifacts(self):
        X, y = make_development_dataset()
        split_random_states = []
        recorded_internal_cv = {}

        def recording_split(*args, **kwargs):
            split_random_states.append(kwargs["random_state"])
            return train_test_split(*args, **kwargs)

        def fake_internal_cv(model, X_arg, y_arg, **kwargs):
            recorded_internal_cv["model"] = model
            recorded_internal_cv["X"] = X_arg
            recorded_internal_cv["y"] = y_arg
            recorded_internal_cv["kwargs"] = kwargs
            return {"rkf_mae_mean": 1.2, "rkf_mae_std": 0.3, "rkf_r2_mean": 0.4, "rkf_r2_std": 0.5, "n_evals": 25}

        with patch("src.train_and_evaluate.train_test_split", side_effect=recording_split), patch(
            "src.train_and_evaluate.repeated_kfold_evaluate",
            side_effect=fake_internal_cv,
        ):
            artifacts = train_and_evaluate(
                LinearRegression,
                X,
                y,
                random_state=13,
                n_trials=1,
                n_jobs=1,
            )

        self.assertEqual(split_random_states, list(range(100)))
        self.assertIs(recorded_internal_cv["X"], X)
        self.assertIs(recorded_internal_cv["y"], y)
        self.assertIsInstance(artifacts["estimator"], LinearRegression)
        self.assertEqual(artifacts["complete_params"], get_fixed_params(LinearRegression, 1))
        self.assertEqual(artifacts["selection_mae"], artifacts["development_cv_mae"])
        self.assertEqual(artifacts["internal_cv"]["rkf_mae_mean"], 1.2)
        self.assertEqual(artifacts["stability"]["mae_mean"], artifacts["stability_mae_mean"])
        self.assertEqual(artifacts["stability"]["mae_std"], artifacts["stability_mae_std"])
        self.assertEqual(artifacts["stability"]["n_splits"], 100)
        self.assertNotIn("X_train", artifacts)
        self.assertNotIn("X_test", artifacts)
        self.assertNotIn("y_train", artifacts)
        self.assertNotIn("y_test", artifacts)

        with self.assertRaises(NotFittedError):
            check_is_fitted(artifacts["estimator"])

    def test_ridge_selected_alpha_is_carried_into_estimator_and_complete_params(self):
        X, y = make_development_dataset()

        with patch("src.train_and_evaluate._select_alpha_via_inner_cv", return_value=(0.25, 1.5)), patch(
            "src.train_and_evaluate._stability_analysis",
            return_value={"mae_mean": 1.0, "mae_std": 0.1, "n_splits": 100},
        ), patch(
            "src.train_and_evaluate.repeated_kfold_evaluate",
            return_value={"rkf_mae_mean": 1.2, "rkf_mae_std": 0.2, "rkf_r2_mean": 0.3, "rkf_r2_std": 0.4, "n_evals": 25},
        ):
            artifacts = train_and_evaluate(
                Ridge,
                X,
                y,
                random_state=13,
                n_trials=1,
                n_jobs=1,
            )

        self.assertEqual(artifacts["complete_params"]["alpha"], 0.25)
        self.assertEqual(artifacts["estimator"].get_params(deep=False)["alpha"], 0.25)

    def test_lasso_selected_alpha_is_carried_into_estimator_and_complete_params(self):
        X, y = make_development_dataset()

        with patch("src.train_and_evaluate._select_alpha_via_inner_cv", return_value=(0.125, 1.75)), patch(
            "src.train_and_evaluate._stability_analysis",
            return_value={"mae_mean": 1.0, "mae_std": 0.1, "n_splits": 100},
        ), patch(
            "src.train_and_evaluate.repeated_kfold_evaluate",
            return_value={"rkf_mae_mean": 1.2, "rkf_mae_std": 0.2, "rkf_r2_mean": 0.3, "rkf_r2_std": 0.4, "n_evals": 25},
        ):
            artifacts = train_and_evaluate(
                Lasso,
                X,
                y,
                random_state=13,
                n_trials=1,
                n_jobs=1,
            )

        self.assertEqual(artifacts["complete_params"]["alpha"], 0.125)
        self.assertEqual(artifacts["estimator"].get_params(deep=False)["alpha"], 0.125)

    def test_hyperparameter_optimization_and_training_returns_unfitted_complete_estimator_without_splits(self):
        X, y = make_development_dataset()

        artifacts = hyperparameter_optimization_and_training(
            LinearRegression,
            X,
            y,
            n_trials=1,
            random_state=3,
            n_jobs=1,
        )

        self.assertEqual(
            set(artifacts),
            {
                "estimator",
                "complete_params",
                "selection_mae",
                "development_cv_mae",
                "stability",
                "stability_mae_mean",
                "stability_mae_std",
                "internal_cv",
            },
        )
        self.assertIsInstance(artifacts["estimator"], LinearRegression)
        self.assertEqual(artifacts["complete_params"], get_fixed_params(LinearRegression, 1))
        self.assertNotIn("X_train", artifacts)
        self.assertNotIn("X_test", artifacts)
        self.assertNotIn("y_train", artifacts)
        self.assertNotIn("y_test", artifacts)

        with self.assertRaises(NotFittedError):
            check_is_fitted(artifacts["estimator"])


if __name__ == "__main__":
    unittest.main()
