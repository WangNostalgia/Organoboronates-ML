import inspect
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error as sklearn_mean_absolute_error
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def install_optional_dependency_stubs():
    def install_regressor_module(module_name, class_name):
        if module_name in sys.modules:
            return
        module = types.ModuleType(module_name)

        class OptionalRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_params(self, deep=False):
                return dict(self.kwargs)

        OptionalRegressor.__name__ = class_name
        setattr(module, class_name, OptionalRegressor)
        sys.modules[module_name] = module

    install_regressor_module("xgboost", "XGBRegressor")
    install_regressor_module("lightgbm", "LGBMRegressor")
    install_regressor_module("catboost", "CatBoostRegressor")

    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")

        class FakeTrial:
            def __init__(self):
                self.params = {}
                self.user_attrs = {}

            def suggest_float(self, name, low, high, log=False):
                self.params[name] = low
                return low

            def suggest_int(self, name, low, high, step=1):
                self.params[name] = low
                return low

            def suggest_categorical(self, name, choices):
                self.params[name] = choices[0]
                return choices[0]

            def set_user_attr(self, name, value):
                self.user_attrs[name] = value

        class FakeStudy:
            def optimize(self, objective, n_jobs=1, n_trials=1):
                candidates = []
                for _ in range(n_trials):
                    trial = FakeTrial()
                    candidates.append((objective(trial), trial))
                self.best_value, self.best_trial = min(
                    candidates, key=lambda candidate: candidate[0]
                )
                self.best_params = dict(self.best_trial.params)

        class FakeTPESampler:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        optuna.logging = types.SimpleNamespace(
            WARNING=30,
            ERROR=40,
            set_verbosity=lambda level: None,
        )
        optuna.samplers = types.SimpleNamespace(TPESampler=FakeTPESampler)
        optuna.create_study = lambda sampler=None, direction="minimize": FakeStudy()
        sys.modules["optuna"] = optuna


install_optional_dependency_stubs()

import main
import src.iterative_optimization as iterative_module
from src.iterative_optimization import iterative_optimization
from src.visualization import (
    add_plot_labels,
    add_plot_labels_standard,
    plot_scatter_standard,
)


def make_sentinel_dataset():
    index = pd.Index(range(100, 120), name="sentinel_id")
    X = pd.DataFrame(
        {
            "f1": np.linspace(0.0, 19.0, 20),
            "f2": np.linspace(20.0, 1.0, 20),
            "f3": np.sin(np.linspace(0.0, 3.0, 20)),
        },
        index=index,
    )
    y = pd.Series(10.0 + 1.5 * X["f1"] - 0.2 * X["f2"], index=index, name="activation_energy")
    return X, y


def make_artifacts(model_class, n_features, n_jobs=1):
    if model_class is SVR:
        complete_params = {
            "kernel": "rbf",
            "tol": 1e-3,
            "max_iter": 10000,
            "cache_size": 1000,
            "C": 4.5,
            "epsilon": 0.15,
            "gamma": 0.4,
        }
    else:
        complete_params = {"marker": 73}

    return {
        "estimator": model_class(**complete_params),
        "complete_params": complete_params,
        "selection_mae": float(n_features) + 0.25,
        "development_cv_mae": float(n_features) + 0.25,
        "stability": {
            "mae_mean": float(n_features) + 0.5,
            "mae_std": 0.2,
            "n_splits": 100,
        },
        "stability_mae_mean": float(n_features) + 0.5,
        "stability_mae_std": 0.2,
        "internal_cv": {
            "rkf_mae_mean": float(n_features),
            "rkf_mae_std": 0.1,
            "rkf_r2_mean": 0.9 - n_features * 0.01,
            "rkf_r2_std": 0.02,
            "n_evals": 25,
        },
    }


class FakeGPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, marker=73):
        self.marker = marker

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.formula_ = "mean(y)"
        return self

    def predict(self, X):
        values = np.asarray(X)
        return self.mean_ + 0.01 * values[:, 0]


class TaggedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, marker=73):
        self.marker = marker

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        values = np.asarray(X)
        return self.mean_ + 0.02 * values[:, 0]


class IterativeBoundaryTests(unittest.TestCase):
    def test_shared_final_test_is_never_seen_by_selection_and_is_evaluated_once(self):
        X, y = make_sentinel_dataset()
        X_dev, X_final_test, y_dev, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=40,
        )
        development_indices = set(X_dev.index)
        final_test_indices = set(X_final_test.index)
        selection_calls = {"tuning": [], "shap": [], "loo": []}
        saved_payloads = {}

        def fake_tuning(model_class, X_arg, y_arg, **kwargs):
            selection_calls["tuning"].append(set(X_arg.index))
            self.assertEqual(set(X_arg.index), set(y_arg.index))
            return make_artifacts(model_class, X_arg.shape[1], n_jobs=kwargs["n_jobs"])

        def fake_shap(model, X_arg, y_arg, model_name, cv_folds):
            selection_calls["shap"].append(set(X_arg.index))
            self.assertEqual(set(X_arg.index), set(y_arg.index))
            return X_arg.columns[-1], [(name, 100.0 / X_arg.shape[1]) for name in X_arg.columns], "low_importance"

        def fake_loo(model, X_arg, y_arg):
            selection_calls["loo"].append(set(X_arg.index))
            self.assertEqual(set(X_arg.index), set(y_arg.index))
            return 0.55, 1.25

        def capture_dump(payload, path):
            saved_payloads[Path(path).name] = payload

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "src.iterative_optimization.os.getcwd",
            return_value=tmpdir,
        ), patch(
            "src.iterative_optimization.setup_logger",
            return_value=logging.getLogger("iterative-boundary-test"),
        ), patch(
            "src.iterative_optimization.clean_old_versions",
        ), patch(
            "src.iterative_optimization.hyperparameter_optimization_and_training",
            side_effect=fake_tuning,
        ), patch(
            "src.feature_selection.shap_rfecv_select_worst_feature",
            side_effect=fake_shap,
        ), patch(
            "src.iterative_optimization.leave_one_out_validation",
            side_effect=fake_loo,
        ), patch(
            "src.iterative_optimization.plot_performance_history",
        ), patch(
            "src.iterative_optimization.save_performance_history",
        ), patch(
            "src.iterative_optimization.plot_scatter",
        ), patch(
            "src.iterative_optimization.joblib.dump",
            side_effect=capture_dump,
        ), patch(
            "src.iterative_optimization.mean_absolute_error",
            wraps=sklearn_mean_absolute_error,
        ) as mae_mock, patch(
            "src.iterative_optimization.r2_score",
            wraps=sklearn_r2_score,
        ) as r2_mock:
            results, best_models = iterative_optimization(
                {"SVR": SVR},
                X,
                y,
                n_trials=1,
                n_jobs=1,
                keep_versions=1,
                min_features=2,
            )

        self.assertEqual(len(selection_calls["tuning"]), 2)
        self.assertEqual(len(selection_calls["shap"]), 1)
        self.assertEqual(len(selection_calls["loo"]), 2)
        for call_group in selection_calls.values():
            for observed_indices in call_group:
                self.assertEqual(observed_indices, development_indices)
                self.assertTrue(observed_indices.isdisjoint(final_test_indices))

        self.assertEqual(mae_mock.call_count, 1)
        self.assertEqual(r2_mock.call_count, 1)
        self.assertIsInstance(best_models["SVR"], SVR)

        final_payload = next(
            payload for filename, payload in saved_payloads.items() if "_final_" in filename
        )
        self.assertEqual(final_payload["features"], ["f1", "f2"])
        self.assertEqual(final_payload["complete_params"]["kernel"], "rbf")
        self.assertEqual(final_payload["complete_params"]["max_iter"], 10000)
        self.assertEqual(final_payload["model"].get_params(deep=False)["kernel"], "rbf")
        self.assertEqual(final_payload["model"].get_params(deep=False)["C"], 4.5)
        self.assertEqual(final_payload["scaler_X"].n_samples_seen_, len(X_dev))
        self.assertEqual(final_payload["scaler_y"].n_samples_seen_, len(y_dev))
        np.testing.assert_allclose(
            final_payload["scaler_X"].data_min_,
            X_dev[["f1", "f2"]].min().to_numpy(),
        )
        np.testing.assert_allclose(
            final_payload["scaler_y"].data_min_,
            np.array([y_dev.min()]),
        )

        protocol = final_payload["evaluation_protocol"]
        self.assertEqual(protocol["random_state"], 40)
        self.assertEqual(protocol["test_size"], 0.2)
        self.assertEqual(protocol["development_indices"], sorted(development_indices))
        self.assertEqual(protocol["final_test_indices"], sorted(final_test_indices))
        self.assertEqual(protocol["development_size"], 16)
        self.assertEqual(protocol["final_test_size"], 4)
        self.assertEqual(protocol["final_test_evaluations"], 1)
        self.assertEqual(protocol["selection_scope"], "development_only")

        metrics = final_payload["metrics"]
        self.assertEqual(
            metrics["primary"]["final_test"]["test_mae"],
            results["SVR"]["test_mae"],
        )
        self.assertEqual(
            metrics["primary"]["final_test"]["test_r2"],
            results["SVR"]["test_r2"],
        )
        self.assertEqual(
            metrics["secondary"]["internal_cv"]["rkf_mae_mean"],
            2.0,
        )
        self.assertEqual(metrics["secondary"]["stability"]["n_splits"], 100)
        self.assertEqual(metrics["secondary"]["loo"]["mae"], 1.25)
        self.assertEqual(metrics["mae_test_avg"], metrics["test_mae"])
        self.assertEqual(metrics["r2_test_avg"], metrics["test_r2"])
        self.assertEqual(
            metrics["rkf_mae_opt_mean"],
            metrics["secondary"]["internal_cv"]["rkf_mae_mean"],
        )
        self.assertEqual(
            metrics["mae_mean"],
            metrics["secondary"]["stability"]["mae_mean"],
        )
        self.assertEqual(results["SVR"]["mae_test_avg"], results["SVR"]["test_mae"])
        self.assertEqual(results["SVR"]["r2_test_avg"], results["SVR"]["test_r2"])
        self.assertEqual(results["SVR"]["best_params_avg"], final_payload["complete_params"])

        for path_entry in results["SVR"]["shap_rfecv_path"]:
            self.assertIn("internal_cv", path_entry["metrics"])
            self.assertNotIn("test_mae", path_entry["metrics"])
            self.assertNotIn("test_r2", path_entry["metrics"])

    def test_gplearn_single_pass_uses_same_development_final_test_boundary(self):
        X, y = make_sentinel_dataset()
        X_dev, X_final_test, _, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=40,
        )
        observed_tuning_indices = []
        observed_loo_indices = []
        saved_payloads = {}

        def fake_tuning(model_class, X_arg, y_arg, **kwargs):
            observed_tuning_indices.append(set(X_arg.index))
            return make_artifacts(model_class, X_arg.shape[1], n_jobs=kwargs["n_jobs"])

        def fake_loo(model, X_arg, y_arg):
            observed_loo_indices.append(set(X_arg.index))
            return 0.1, 2.0

        def capture_dump(payload, path):
            saved_payloads[Path(path).name] = payload

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            iterative_module,
            "GPLearnRegressor",
            FakeGPRegressor,
        ), patch(
            "src.iterative_optimization.os.getcwd",
            return_value=tmpdir,
        ), patch(
            "src.iterative_optimization.setup_logger",
            return_value=logging.getLogger("gplearn-boundary-test"),
        ), patch(
            "src.iterative_optimization.clean_old_versions",
        ), patch(
            "src.iterative_optimization.hyperparameter_optimization_and_training",
            side_effect=fake_tuning,
        ), patch(
            "src.feature_selection.shap_rfecv_select_worst_feature",
        ) as shap_mock, patch(
            "src.iterative_optimization.leave_one_out_validation",
            side_effect=fake_loo,
        ), patch(
            "src.iterative_optimization.plot_performance_history",
        ), patch(
            "src.iterative_optimization.save_performance_history",
        ), patch(
            "src.iterative_optimization.plot_scatter",
        ), patch(
            "src.iterative_optimization.joblib.dump",
            side_effect=capture_dump,
        ):
            results, _ = iterative_optimization(
                {"GPlearn": FakeGPRegressor},
                X,
                y,
                n_trials=1,
                n_jobs=1,
            )

        self.assertEqual(observed_tuning_indices, [set(X_dev.index)])
        self.assertEqual(observed_loo_indices, [set(X_dev.index)])
        self.assertTrue(set(X_dev.index).isdisjoint(set(X_final_test.index)))
        shap_mock.assert_not_called()
        final_payload = next(
            payload for filename, payload in saved_payloads.items() if "_final_" in filename
        )
        self.assertEqual(final_payload["features"], list(X.columns))
        self.assertEqual(final_payload["complete_params"], {"marker": 73})
        self.assertEqual(
            final_payload["metrics"]["primary"]["final_test"]["test_mae"],
            results["GPlearn"]["test_mae"],
        )
        self.assertEqual(
            final_payload["metrics"]["secondary"]["internal_cv"]["rkf_mae_mean"],
            3.0,
        )

    def test_gplearn_rejects_forced_feature_count_before_training_or_persistence(self):
        X, y = make_sentinel_dataset()

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            iterative_module,
            "GPLearnRegressor",
            FakeGPRegressor,
        ), patch(
            "src.iterative_optimization.os.getcwd",
            return_value=tmpdir,
        ), patch(
            "src.iterative_optimization.setup_logger",
            return_value=logging.getLogger("gplearn-force-test"),
        ), patch(
            "src.iterative_optimization.clean_old_versions",
        ) as clean_mock, patch(
            "src.iterative_optimization.hyperparameter_optimization_and_training",
            return_value=make_artifacts(FakeGPRegressor, X.shape[1], n_jobs=1),
        ) as tuning_mock, patch(
            "src.iterative_optimization.leave_one_out_validation",
            return_value=(0.1, 2.0),
        ), patch(
            "src.iterative_optimization.plot_performance_history",
        ), patch(
            "src.iterative_optimization.save_performance_history",
        ), patch(
            "src.iterative_optimization.plot_scatter",
        ), patch(
            "src.iterative_optimization.joblib.dump",
        ) as dump_mock:
            with self.assertRaisesRegex(
                ValueError,
                r"GPlearn.*force_n_features.*no SHAP-RFECV path",
            ):
                iterative_optimization(
                    {"GPlearn": FakeGPRegressor},
                    X,
                    y,
                    n_trials=1,
                    n_jobs=1,
                    force_n_features=2,
                )

        clean_mock.assert_not_called()
        tuning_mock.assert_not_called()
        dump_mock.assert_not_called()

    def test_iteration_checkpoints_cover_complete_three_to_two_feature_path(self):
        X, y = make_sentinel_dataset()
        saved_payloads = {}

        def fake_tuning(model_class, X_arg, y_arg, **kwargs):
            return make_artifacts(model_class, X_arg.shape[1], n_jobs=kwargs["n_jobs"])

        def fake_shap(model, X_arg, y_arg, model_name, cv_folds):
            return X_arg.columns[-1], list(zip(X_arg.columns, [1.0] * X_arg.shape[1])), "low_importance"

        def capture_dump(payload, path):
            saved_payloads[Path(path).name] = payload

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "src.iterative_optimization.os.getcwd",
            return_value=tmpdir,
        ), patch(
            "src.iterative_optimization.setup_logger",
            return_value=logging.getLogger("checkpoint-path-test"),
        ), patch(
            "src.iterative_optimization.clean_old_versions",
        ), patch(
            "src.iterative_optimization.hyperparameter_optimization_and_training",
            side_effect=fake_tuning,
        ), patch(
            "src.feature_selection.shap_rfecv_select_worst_feature",
            side_effect=fake_shap,
        ), patch(
            "src.iterative_optimization.leave_one_out_validation",
            return_value=(0.55, 1.25),
        ), patch(
            "src.iterative_optimization.plot_performance_history",
        ), patch(
            "src.iterative_optimization.save_performance_history",
        ), patch(
            "src.iterative_optimization.plot_scatter",
        ), patch(
            "src.iterative_optimization.joblib.dump",
            side_effect=capture_dump,
        ):
            iterative_optimization(
                {"SVR": SVR},
                X,
                y,
                n_trials=1,
                n_jobs=1,
                min_features=2,
            )

        checkpoints = [
            payload
            for filename, payload in sorted(saved_payloads.items())
            if "_iteration_" in filename
        ]
        self.assertEqual(
            [len(payload["features"]) for payload in checkpoints],
            [3, 2],
        )
        for payload in checkpoints:
            feature_count = len(payload["features"])
            self.assertEqual(payload["model"].n_features_in_, feature_count)
            self.assertEqual(payload["scaler_X"].n_features_in_, feature_count)
            self.assertEqual(
                payload["metrics"]["internal_cv"]["rkf_mae_mean"],
                float(feature_count),
            )

    def test_iterative_signature_and_main_parser_expose_only_current_selection_controls(self):
        signature = inspect.signature(iterative_optimization)
        self.assertNotIn("mae_threshold", signature.parameters)
        self.assertEqual(signature.parameters["min_features"].default, 5)

        parser = main.build_argument_parser()
        actions = {action.dest: action for action in parser._actions}
        self.assertNotIn("mae_threshold", actions)
        self.assertEqual(actions["min_features"].default, 5)
        force_help = actions["force_n_features"].help.lower()
        self.assertIn("evaluate", force_help)
        self.assertIn("path", force_help)
        self.assertIn("exact", force_help)

    def test_scatter_labels_mark_final_test_primary_and_validation_metrics_secondary(self):
        metrics = {
            "r_train": 0.91,
            "r_test": 0.82,
            "rmse_test": 1.8,
            "r2_test": 0.67,
            "mae_test": 1.4,
        }

        with patch("src.visualization.plt.text") as text_mock, patch(
            "src.visualization.plt.legend"
        ):
            add_plot_labels(
                metrics,
                mae_mean=1.7,
                model_name="SVR",
                r2_loo=0.61,
                rkf_mae=1.5,
                rkf_r2=0.64,
            )

        labels = "\n".join(str(call.args[2]) for call in text_mock.call_args_list)
        self.assertIn("Final Test MAE", labels)
        self.assertIn("PRIMARY", labels)
        self.assertIn("Stability MAE", labels)
        self.assertIn("Internal CV MAE", labels)
        self.assertIn("LOOCV", labels)
        self.assertIn("secondary", labels)

    def test_standard_scatter_labels_use_the_same_primary_secondary_semantics(self):
        metrics = {
            "r_train": 0.91,
            "r_test": 0.82,
            "rmse_test": 1.8,
            "r2_test": 0.67,
            "mae_test": 1.4,
        }

        with patch("src.visualization.plt.text") as text_mock:
            add_plot_labels_standard(
                metrics,
                mae_mean=1.7,
                model_name="SVR",
                r2_loo=0.61,
                rkf_mae=1.5,
                rkf_r2=0.64,
            )

        labels = "\n".join(str(call.args[2]) for call in text_mock.call_args_list)
        self.assertIn("Final Test MAE", labels)
        self.assertIn("PRIMARY", labels)
        self.assertIn("Stability MAE", labels)
        self.assertIn("Internal CV MAE", labels)
        self.assertIn("LOOCV", labels)
        self.assertIn("secondary", labels)

    def test_publication_scatter_legend_uses_development_and_final_test_labels(self):
        y_development = np.array([1.0, 2.0, 3.0])
        y_pred_development = np.array([1.1, 1.9, 3.1])
        y_final_test = np.array([1.5, 2.5])
        y_pred_final_test = np.array([1.4, 2.6])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "src.visualization.plt.scatter"
        ) as scatter_mock, patch(
            "src.visualization.calculate_metrics",
            return_value={
                "r_train": 0.9,
                "r_test": 0.8,
                "r2_train": 0.7,
                "rmse_test": 0.2,
                "r2_test": 0.6,
                "mae_test": 0.1,
            },
        ), patch(
            "src.visualization.plt.savefig",
        ):
            plot_scatter_standard(
                y_development,
                y_pred_development,
                y_final_test,
                y_pred_final_test,
                model_name="SVR",
                mae_mean=None,
                output_dir=tmpdir + os.sep,
                output_name="publication.png",
            )

        self.assertEqual(
            [call.kwargs["label"] for call in scatter_mock.call_args_list],
            ["Development", "Final Test"],
        )

    def test_two_models_share_one_split_and_each_evaluate_final_test_once(self):
        X, y = make_sentinel_dataset()
        X_development, X_final_test, _, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=40,
        )
        real_split = train_test_split
        real_evaluate = iterative_module._evaluate_final_test_once
        real_inverse_predict = iterative_module._inverse_predict
        split_calls = []
        evaluation_calls = []
        prediction_calls = []

        def recording_split(*args, **kwargs):
            split_calls.append(kwargs.copy())
            return real_split(*args, **kwargs)

        def fake_tuning(model_class, X_arg, y_arg, **kwargs):
            return make_artifacts(model_class, X_arg.shape[1], n_jobs=kwargs["n_jobs"])

        def recording_evaluate(
            estimator,
            scaler_X,
            scaler_y,
            X_development_arg,
            y_development_arg,
            X_final_test_arg,
            y_final_test_arg,
        ):
            evaluation_calls.append(
                (type(estimator), set(X_final_test_arg.index))
            )
            return real_evaluate(
                estimator,
                scaler_X,
                scaler_y,
                X_development_arg,
                y_development_arg,
                X_final_test_arg,
                y_final_test_arg,
            )

        def recording_inverse_predict(estimator, scaler_y, X_scaled):
            prediction_calls.append((type(estimator), len(X_scaled)))
            return real_inverse_predict(estimator, scaler_y, X_scaled)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "src.iterative_optimization.os.getcwd",
            return_value=tmpdir,
        ), patch(
            "src.iterative_optimization.setup_logger",
            return_value=logging.getLogger("two-model-boundary-test"),
        ), patch(
            "src.iterative_optimization.clean_old_versions",
        ), patch(
            "src.iterative_optimization.train_test_split",
            side_effect=recording_split,
        ), patch(
            "src.iterative_optimization.hyperparameter_optimization_and_training",
            side_effect=fake_tuning,
        ), patch(
            "src.iterative_optimization.leave_one_out_validation",
            return_value=(0.55, 1.25),
        ), patch(
            "src.iterative_optimization._evaluate_final_test_once",
            side_effect=recording_evaluate,
        ), patch(
            "src.iterative_optimization._inverse_predict",
            side_effect=recording_inverse_predict,
        ), patch(
            "src.iterative_optimization.plot_performance_history",
        ), patch(
            "src.iterative_optimization.save_performance_history",
        ), patch(
            "src.iterative_optimization.plot_scatter",
        ), patch(
            "src.iterative_optimization.joblib.dump",
        ):
            iterative_optimization(
                {"SVR": SVR, "Tagged": TaggedRegressor},
                X,
                y,
                n_trials=1,
                n_jobs=1,
                min_features=3,
            )

        self.assertEqual(split_calls, [{"test_size": 0.2, "random_state": 40}])
        self.assertEqual(
            [model_class for model_class, _ in evaluation_calls],
            [SVR, TaggedRegressor],
        )
        for _, observed_final_indices in evaluation_calls:
            self.assertEqual(observed_final_indices, set(X_final_test.index))
            self.assertTrue(
                observed_final_indices.isdisjoint(set(X_development.index))
            )

        for model_class in (SVR, TaggedRegressor):
            final_test_predictions = [
                call
                for call in prediction_calls
                if call == (model_class, len(X_final_test))
            ]
            self.assertEqual(final_test_predictions, [(model_class, 4)])


if __name__ == "__main__":
    unittest.main()
