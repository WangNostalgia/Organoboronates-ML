import logging
import os
import re
import tempfile
from datetime import datetime
from numbers import Integral
from pprint import pformat

import joblib
import matplotlib

matplotlib.use("Agg")  # Must be set before importing pyplot.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.gplearn_wrapper import GPLearnRegressor
from src.hyperparameter_optimization_and_training import (
    hyperparameter_optimization_and_training,
)
from src.leave_one_out_validation import leave_one_out_validation
from src.logger_config import setup_logger
from src.visualization import plot_scatter

plt.rcParams["font.family"] = "DejaVu Sans"


INTERNAL_CV_LABEL = "Internal 5×5 RepeatedKFold CV"
RKF_PLOT_LABEL = "5×5 RKFold"
STABILITY_PLOT_LABEL = "100-split MAE"


def _validate_keep_versions(keep_versions):
    if (
        not isinstance(keep_versions, Integral)
        or isinstance(keep_versions, bool)
        or keep_versions < 1
    ):
        raise ValueError("keep_versions must be a positive integer")
    return int(keep_versions)


def clean_old_versions(model_dir, keep_versions=2):
    """Keep only the newest final-model runs and matching iteration checkpoints."""
    keep_versions = _validate_keep_versions(keep_versions)
    logger = logging.getLogger(__name__)
    model_name = os.path.basename(os.path.normpath(model_dir))
    escaped_model_name = re.escape(model_name)
    timestamp_pattern = r"(?P<timestamp>\d{8}_\d{6})"
    final_pattern = re.compile(
        rf"^{escaped_model_name}_final_{timestamp_pattern}\.joblib$"
    )
    iteration_pattern = re.compile(
        rf"^{escaped_model_name}_iteration_\d+_{timestamp_pattern}\.joblib$"
    )
    artifact_patterns = (
        final_pattern,
        re.compile(rf"^{escaped_model_name}_final_{timestamp_pattern}_metrics\.txt$"),
        iteration_pattern,
        re.compile(
            rf"^{escaped_model_name}_iteration_\d+_"
            rf"{timestamp_pattern}_metrics\.txt$"
        ),
        re.compile(rf"^performance_history_{timestamp_pattern}\.(?:csv|png)$"),
        re.compile(rf"^final_scatter_{timestamp_pattern}\.png$"),
        re.compile(rf"^final_scatter_{timestamp_pattern}_outliers\.csv$"),
    )

    matched_artifacts = []
    final_timestamps = set()
    for name in os.listdir(model_dir):
        filepath = os.path.join(model_dir, name)
        if not os.path.isfile(filepath):
            continue
        for pattern in artifact_patterns:
            match = pattern.fullmatch(name)
            if not match:
                continue
            timestamp = match.group("timestamp")
            matched_artifacts.append((filepath, timestamp))
            if pattern is final_pattern:
                final_timestamps.add(timestamp)
            break

    retained_timestamps = set(sorted(final_timestamps, reverse=True)[:keep_versions])

    for filepath, timestamp in matched_artifacts:
        if timestamp in retained_timestamps:
            continue
        try:
            os.remove(filepath)
            logger.info("Deleted old run artifact: %s", os.path.basename(filepath))
        except OSError as exc:
            logger.info("Failed to delete file %s: %s", filepath, exc)


def _fit_on_development(estimator, X_development, y_development):
    """Fit a fresh artifact estimator and both scalers on all development data."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 100))
    X_scaled = scaler_X.fit_transform(X_development)
    y_scaled = scaler_y.fit_transform(
        np.asarray(y_development).reshape(-1, 1)
    ).ravel()
    fitted_estimator = clone(estimator)
    fitted_estimator.fit(X_scaled, y_scaled)
    return fitted_estimator, scaler_X, scaler_y, X_scaled


def _inverse_predict(estimator, scaler_y, X_scaled):
    predictions_scaled = np.asarray(estimator.predict(X_scaled)).reshape(-1, 1)
    return scaler_y.inverse_transform(predictions_scaled).ravel()


def _evaluate_final_test_once(
    estimator,
    scaler_X,
    scaler_y,
    X_development,
    y_development,
    X_final_test,
    y_final_test,
):
    """Generate final predictions and calculate the two primary metrics once."""
    X_development_scaled = scaler_X.transform(X_development)
    X_final_test_scaled = scaler_X.transform(X_final_test)
    y_pred_development = _inverse_predict(estimator, scaler_y, X_development_scaled)
    y_pred_final_test = _inverse_predict(estimator, scaler_y, X_final_test_scaled)

    test_mae = float(mean_absolute_error(y_final_test, y_pred_final_test))
    test_r2 = float(r2_score(y_final_test, y_pred_final_test))

    y_development_array = np.asarray(y_development)
    y_final_test_array = np.asarray(y_final_test)
    scatter_metrics = {
        "r_train": float(np.corrcoef(y_development_array, y_pred_development)[0, 1]),
        "r_test": float(np.corrcoef(y_final_test_array, y_pred_final_test)[0, 1]),
        "r2_train": float(
            1.0
            - np.sum((y_development_array - y_pred_development) ** 2)
            / np.sum((y_development_array - np.mean(y_development_array)) ** 2)
        ),
        "rmse_test": float(
            np.sqrt(np.mean((y_final_test_array - y_pred_final_test) ** 2))
        ),
        "r2_test": test_r2,
        "mae_test": test_mae,
    }
    return y_pred_development, y_pred_final_test, test_mae, test_r2, scatter_metrics


def _safe_loo(estimator, X_development, y_development, logger, context):
    try:
        return leave_one_out_validation(estimator, X_development, y_development)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("LOOCV failed for %s: %s", context, exc, exc_info=True)
        return float("nan"), float("nan")


def _params_one_line(params):
    """Return a compact one-line representation without dropping any parameter."""
    return pformat(dict(params), compact=True, width=1_000_000, sort_dicts=True)


def _write_params_block(handle, title, params):
    handle.write(f"\n--- {title} ---\n")
    handle.write(_params_one_line(params))
    handle.write("\n")


def _write_iteration_path_table(handle, shap_rfecv_path, selected_n_features=None):
    """Write one machine-readable-ish row per evaluated feature-count checkpoint."""
    if not shap_rfecv_path:
        handle.write("\n--- Iteration Parameter Path ---\n")
        handle.write("No iterative SHAP-RFECV path was recorded for this model.\n")
        return

    columns = [
        "Iter",
        "Feat",
        "Selected",
        f"{INTERNAL_CV_LABEL} MAE",
        f"{INTERNAL_CV_LABEL} R2",
        "LOOCV R2",
        "LOOCV MAE",
        "100-split MAE",
        "Complete Parameters",
        "Removed_before",
    ]
    handle.write("\n--- Iteration Parameter Path ---\n")
    handle.write("\t".join(columns) + "\n")
    for entry in shap_rfecv_path:
        metrics = entry["metrics"]
        internal_cv = metrics["internal_cv"]
        stability = metrics["stability"]
        loo = metrics["loo"]
        selected = "yes" if entry["n_features"] == selected_n_features else ""
        row = [
            str(entry["iteration"]),
            str(entry["n_features"]),
            selected,
            f"{internal_cv['rkf_mae_mean']:.4f} ± {internal_cv['rkf_mae_std']:.4f}",
            f"{internal_cv['rkf_r2_mean']:.4f} ± {internal_cv['rkf_r2_std']:.4f}",
            f"{loo['r2']:.4f}",
            f"{loo['mae']:.4f}",
            f"{stability['mae_mean']:.4f} ± {stability['mae_std']:.4f}",
            _params_one_line(entry["complete_params"]),
            ", ".join(entry.get("removed_features", [])),
        ]
        handle.write("\t".join(row) + "\n")


def _make_path_entry(
    features,
    artifacts,
    loo_r2,
    loo_mae,
    removed_features,
    iteration,
):
    internal_cv = dict(artifacts["internal_cv"])
    stability = dict(artifacts["stability"])
    loo = {"r2": float(loo_r2), "mae": float(loo_mae)}
    metrics = {
        "development_cv_mae": float(artifacts["development_cv_mae"]),
        "selection_mae": float(artifacts["selection_mae"]),
        "internal_cv": internal_cv,
        "stability": stability,
        "loo": loo,
        # Compatibility aliases. These remain development-only metrics.
        "mae_mean": float(artifacts["stability_mae_mean"]),
        "rkf_mae_mean": float(internal_cv["rkf_mae_mean"]),
        "rkf_mae_std": float(internal_cv["rkf_mae_std"]),
        "rkf_r2_mean": float(internal_cv["rkf_r2_mean"]),
        "rkf_r2_std": float(internal_cv["rkf_r2_std"]),
        "loo_r2": float(loo_r2),
        "loo_mae": float(loo_mae),
    }
    return {
        "features": list(features),
        "complete_params": dict(artifacts["complete_params"]),
        "hyperparameters": dict(artifacts["complete_params"]),
        "metrics": metrics,
        "removed_features": list(removed_features),
        "iteration": iteration,
        "n_features": len(features),
    }


def _append_performance_history(history, entry, removed_feature):
    metrics = entry["metrics"]
    internal_cv = metrics["internal_cv"]
    stability = metrics["stability"]
    loo = metrics["loo"]

    history["iteration"].append(entry["iteration"])
    history["remaining_features"].append(entry["n_features"])
    history["removed_feature"].append(removed_feature)
    history["development_cv_mae"].append(metrics["development_cv_mae"])
    history["stability_mae_mean"].append(stability["mae_mean"])
    history["stability_mae_std"].append(stability["mae_std"])
    history["internal_cv_mae_mean"].append(internal_cv["rkf_mae_mean"])
    history["internal_cv_mae_std"].append(internal_cv["rkf_mae_std"])
    history["internal_cv_r2_mean"].append(internal_cv["rkf_r2_mean"])
    history["internal_cv_r2_std"].append(internal_cv["rkf_r2_std"])
    history["loo_r2"].append(loo["r2"])
    history["loo_mae"].append(loo["mae"])
    history["final_test_mae"].append(float("nan"))
    history["final_test_r2"].append(float("nan"))

    # Backward-compatible CSV columns, with development-only semantics.
    history["mae"].append(stability["mae_mean"])
    history["r2"].append(internal_cv["rkf_r2_mean"])
    history["rkf_mae_mean"].append(internal_cv["rkf_mae_mean"])
    history["rkf_mae_std"].append(internal_cv["rkf_mae_std"])
    history["rkf_r2_mean"].append(internal_cv["rkf_r2_mean"])
    history["rkf_r2_std"].append(internal_cv["rkf_r2_std"])


def _select_path_entry(path, force_n_features, logger):
    if force_n_features is not None:
        matches = [entry for entry in path if entry["n_features"] == force_n_features]
        if not matches:
            available = sorted(entry["n_features"] for entry in path)
            raise ValueError(
                f"force_n_features={force_n_features} is not on the evaluated "
                f"SHAP-RFECV path; available counts: {available}"
            )
        logger.info(
            "Manual feature-count selection: exact evaluated path entry %d",
            force_n_features,
        )
        return matches[0]

    minimum_entry = min(
        path,
        key=lambda entry: entry["metrics"]["internal_cv"]["rkf_mae_mean"],
    )
    minimum_mae = minimum_entry["metrics"]["internal_cv"]["rkf_mae_mean"]
    minimum_std = minimum_entry["metrics"]["internal_cv"].get("rkf_mae_std", 0.0)
    threshold = minimum_mae + 0.25 * max(minimum_std, 1e-8)
    candidates = [
        entry
        for entry in path
        if entry["metrics"]["internal_cv"]["rkf_mae_mean"] <= threshold
    ]
    selected = min(candidates, key=lambda entry: entry["n_features"])
    logger.info(
        "Development %s selection: minimum MAE %.4f ± %.4f; "
        "fractional 1-SE threshold %.4f; selected %d features",
        INTERNAL_CV_LABEL,
        minimum_mae,
        minimum_std,
        threshold,
        selected["n_features"],
    )
    return selected


def _make_result(
    selected_entry,
    final_features,
    shap_rfecv_path,
    test_mae,
    test_r2,
):
    internal_cv = dict(selected_entry["metrics"]["internal_cv"])
    stability = dict(selected_entry["metrics"]["stability"])
    loo = dict(selected_entry["metrics"]["loo"])
    complete_params = dict(selected_entry["complete_params"])
    metrics = {
        "primary": {
            "final_test": {
                "test_mae": test_mae,
                "test_r2": test_r2,
            }
        },
        "secondary": {
            "development_cv_mae": selected_entry["metrics"]["development_cv_mae"],
            "internal_cv": internal_cv,
            "stability": stability,
            "loo": loo,
        },
        "test_mae": test_mae,
        "test_r2": test_r2,
    }
    compatibility_aliases = {
        "mae_mean": stability["mae_mean"],
        "r2_test_avg": test_r2,
        "mae_test_avg": test_mae,
        "best_params_avg": complete_params,
        "r2_loo_avg": loo["r2"],
        "mae_loo_avg": loo["mae"],
        "rkf_mae_opt_mean": internal_cv["rkf_mae_mean"],
        "rkf_r2_opt_mean": internal_cv["rkf_r2_mean"],
    }
    metrics.update(compatibility_aliases)
    return {
        "test_mae": test_mae,
        "test_r2": test_r2,
        "complete_params": complete_params,
        "internal_cv": internal_cv,
        "stability": stability,
        "loo": loo,
        "metrics": metrics,
        "final_features": list(final_features),
        "optimal_features": list(final_features),
        "optimal_n_features": len(final_features),
        "shap_rfecv_path": shap_rfecv_path,
        # Legacy aliases with explicit current semantics.
        **compatibility_aliases,
    }


def _write_final_metrics(
    path,
    model_name,
    result,
    evaluation_protocol,
    force_n_features,
    estimator,
):
    internal_cv = result["internal_cv"]
    stability = result["stability"]
    loo = result["loo"]
    selection_mode = "exact forced count" if force_n_features is not None else "auto-selected"

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Selection mode: Final — {selection_mode}\n")
        handle.write(f"Optimal feature count: {len(result['final_features'])}\n")
        handle.write(f"Features: {', '.join(result['final_features'])}\n")
        handle.write("\n--- PRIMARY: Untouched Final Test ---\n")
        handle.write(f"Final Test MAE: {result['test_mae']:.4f} kcal/mol\n")
        handle.write(f"Final Test R²:  {result['test_r2']:.4f}\n")
        handle.write(
            f"Split: random_state={evaluation_protocol['random_state']}, "
            f"test_size={evaluation_protocol['test_size']}, "
            f"development={evaluation_protocol['development_size']}, "
            f"final_test={evaluation_protocol['final_test_size']}\n"
        )
        handle.write("\n--- SECONDARY: Development-Only Validation ---\n")
        handle.write(
            f"{INTERNAL_CV_LABEL} MAE: {internal_cv['rkf_mae_mean']:.4f} "
            f"± {internal_cv['rkf_mae_std']:.4f}\n"
        )
        handle.write(
            f"{INTERNAL_CV_LABEL} R²:  {internal_cv['rkf_r2_mean']:.4f} "
            f"± {internal_cv['rkf_r2_std']:.4f}\n"
        )
        handle.write(
            f"100-split MAE: {stability['mae_mean']:.4f} "
            f"± {stability['mae_std']:.4f}\n"
        )
        handle.write(f"LOOCV MAE: {loo['mae']:.4f}\n")
        handle.write(f"LOOCV R²:  {loo['r2']:.4f}\n")
        handle.write("\n--- Compatibility aliases ---\n")
        handle.write("mae_test_avg = Final Test MAE\n")
        handle.write("r2_test_avg = Final Test R²\n")
        handle.write("mae_mean = 100-split development MAE\n")
        handle.write("rkf_* = development Internal 5×5 RepeatedKFold CV\n")
        _write_params_block(handle, "Selected Complete Parameters", result["complete_params"])
        _write_iteration_path_table(
            handle,
            result["shap_rfecv_path"],
            selected_n_features=len(result["final_features"]),
        )
        if getattr(estimator, "formula_", None):
            handle.write(f"\nGPlearn Formula:\n  {estimator.formula_}\n")


def _temporary_sibling_path(final_path):
    directory = os.path.dirname(final_path) or "."
    basename = os.path.basename(final_path)
    handle, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{basename}.tmp-",
        suffix=".tmp",
    )
    os.close(handle)
    return temp_path


def _remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _write_final_artifacts_atomically(
    final_model_path,
    final_model_info,
    model_name,
    result,
    evaluation_protocol,
    force_n_features,
    estimator,
):
    final_metrics_path = f"{final_model_path}_metrics.txt"
    final_joblib_path = f"{final_model_path}.joblib"
    temp_metrics_path = _temporary_sibling_path(final_metrics_path)
    temp_joblib_path = _temporary_sibling_path(final_joblib_path)
    created_canonical_paths = []

    try:
        _write_final_metrics(
            temp_metrics_path,
            model_name,
            result,
            evaluation_protocol,
            force_n_features,
            estimator,
        )
        joblib.dump(final_model_info, temp_joblib_path)

        os.replace(temp_metrics_path, final_metrics_path)
        created_canonical_paths.append(final_metrics_path)

        os.replace(temp_joblib_path, final_joblib_path)
        created_canonical_paths.append(final_joblib_path)
    except Exception:
        _remove_if_exists(temp_metrics_path)
        _remove_if_exists(temp_joblib_path)
        for path in created_canonical_paths:
            _remove_if_exists(path)
        raise


def _write_iteration_metrics(path, model_name, entry, artifacts, loo_r2, loo_mae):
    internal_cv = entry["metrics"]["internal_cv"]
    stability = entry["metrics"]["stability"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Iteration: {entry['iteration']}\n")
        handle.write(f"Feature count: {entry['n_features']}\n")
        handle.write(f"Features: {', '.join(entry['features'])}\n")
        handle.write("All metrics below are SECONDARY and development-only.\n")
        handle.write(
            f"{INTERNAL_CV_LABEL} MAE: {internal_cv['rkf_mae_mean']:.4f} "
            f"± {internal_cv['rkf_mae_std']:.4f}\n"
        )
        handle.write(
            f"{INTERNAL_CV_LABEL} R²: {internal_cv['rkf_r2_mean']:.4f} "
            f"± {internal_cv['rkf_r2_std']:.4f}\n"
        )
        handle.write(f"LOOCV MAE: {loo_mae:.4f}\n")
        handle.write(f"LOOCV R²: {loo_r2:.4f}\n")
        handle.write(
            f"100-split MAE: {stability['mae_mean']:.4f} "
            f"± {stability['mae_std']:.4f}\n"
        )
        _write_params_block(handle, "Complete Parameters", artifacts["complete_params"])


def iterative_optimization(
    models,
    X,
    y,
    n_trials=100,
    n_jobs=-1,
    optuna_jobs=1,
    keep_versions=2,
    min_features=5,
    custom_min_features=None,
    force_n_features=None,
):
    """
    Select models and features using development data, then evaluate once.

    A single random_state=40 split is created before the model loop. Every
    tuning, stability, internal-CV, SHAP-RFECV, feature-count selection, and
    LOOCV operation receives only the development partition. The untouched
    final-test partition is transformed and scored exactly once per model,
    after the feature set and complete estimator configuration are fixed.
    """
    keep_versions = _validate_keep_versions(keep_versions)
    if not isinstance(optuna_jobs, Integral) or isinstance(optuna_jobs, bool) or optuna_jobs < 1:
        raise ValueError("optuna_jobs must be a positive integer")
    optuna_jobs = int(optuna_jobs)

    if force_n_features is not None and any(
        model_class == GPLearnRegressor for model_class in models.values()
    ):
        raise ValueError(
            "GPlearn cannot use force_n_features because inherent-selection "
            "mode has no SHAP-RFECV path to force."
        )

    if min_features < 1:
        raise ValueError("min_features must be at least 1")

    X_development, X_final_test, y_development, y_final_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=40,
    )
    evaluation_protocol = {
        "name": "single_untouched_final_test",
        "random_state": 40,
        "test_size": 0.2,
        "selection_scope": "development_only",
        "development_size": len(X_development),
        "final_test_size": len(X_final_test),
        "development_indices": sorted(X_development.index.tolist()),
        "final_test_indices": sorted(X_final_test.index.tolist()),
        "final_test_evaluations": 1,
    }

    results = {}
    best_models = {}
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    logger = setup_logger(models_dir)

    for model_name, model_class in models.items():
        logger.info("Starting model: %s", model_name)
        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        clean_old_versions(model_dir, keep_versions)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        effective_min_features = min_features
        if custom_min_features and model_name in custom_min_features:
            effective_min_features = custom_min_features[model_name]
        if effective_min_features < 1:
            raise ValueError("effective minimum feature count must be at least 1")
        if effective_min_features > X_development.shape[1]:
            effective_min_features = X_development.shape[1]

        X_model = X_development.copy()
        removed_features = []
        shap_rfecv_path = []
        candidate_estimators = {}
        performance_history = {
            "iteration": [],
            "remaining_features": [],
            "removed_feature": [],
            "development_cv_mae": [],
            "stability_mae_mean": [],
            "stability_mae_std": [],
            "internal_cv_mae_mean": [],
            "internal_cv_mae_std": [],
            "internal_cv_r2_mean": [],
            "internal_cv_r2_std": [],
            "loo_r2": [],
            "loo_mae": [],
            "final_test_mae": [],
            "final_test_r2": [],
            "mae": [],
            "r2": [],
            "rkf_mae_mean": [],
            "rkf_mae_std": [],
            "rkf_r2_mean": [],
            "rkf_r2_std": [],
        }

        if model_class == GPLearnRegressor:
            logger.info(
                "GPlearn uses a single development-only pass because genetic "
                "programming performs inherent feature selection."
            )
            artifacts = hyperparameter_optimization_and_training(
                model_class,
                X_model,
                y_development,
                n_trials=n_trials,
                n_jobs=n_jobs,
                optuna_jobs=optuna_jobs,
                random_state=40,
            )
            loo_r2, loo_mae = _safe_loo(
                artifacts["estimator"],
                X_model,
                y_development,
                logger,
                model_name,
            )
            selected_entry = _make_path_entry(
                X_model.columns,
                artifacts,
                loo_r2,
                loo_mae,
                removed_features,
                iteration=1,
            )
            shap_rfecv_path = []
            candidate_estimator = artifacts["estimator"]
            _append_performance_history(
                performance_history, selected_entry, "None single pass"
            )
        else:
            iteration = 0
            while True:
                iteration += 1
                artifacts = hyperparameter_optimization_and_training(
                    model_class,
                    X_model,
                    y_development,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    optuna_jobs=optuna_jobs,
                    random_state=40,
                )
                loo_r2, loo_mae = _safe_loo(
                    artifacts["estimator"],
                    X_model,
                    y_development,
                    logger,
                    f"{model_name} iteration {iteration}",
                )
                entry = _make_path_entry(
                    X_model.columns,
                    artifacts,
                    loo_r2,
                    loo_mae,
                    removed_features,
                    iteration,
                )
                shap_rfecv_path.append(entry)
                candidate_estimators[entry["n_features"]] = artifacts["estimator"]
                _append_performance_history(
                    performance_history,
                    entry,
                    removed_features[-1] if removed_features else "Initial",
                )

                internal_cv = entry["metrics"]["internal_cv"]
                logger.info(
                    "Iteration %d: %d features | %s MAE %.4f ± %.4f",
                    iteration,
                    entry["n_features"],
                    INTERNAL_CV_LABEL,
                    internal_cv["rkf_mae_mean"],
                    internal_cv["rkf_mae_std"],
                )
                logger.info(
                    "Iteration %d complete parameters: %s",
                    iteration,
                    _params_one_line(artifacts["complete_params"]),
                )

                checkpoint_model, checkpoint_scaler_X, checkpoint_scaler_y, _ = (
                    _fit_on_development(artifacts["estimator"], X_model, y_development)
                )
                checkpoint_path = os.path.join(
                    model_dir,
                    f"{model_name}_iteration_{iteration}_{run_timestamp}",
                )
                checkpoint_info = {
                    "model": checkpoint_model,
                    "estimator": checkpoint_model,
                    "scaler_X": checkpoint_scaler_X,
                    "scaler_y": checkpoint_scaler_y,
                    "features": list(X_model.columns),
                    "complete_params": dict(artifacts["complete_params"]),
                    "hyperparameters": dict(artifacts["complete_params"]),
                    "metrics": entry["metrics"],
                    "removed_features": list(removed_features),
                    "evaluation_protocol": {
                        **evaluation_protocol,
                        "final_test_evaluations": 0,
                        "artifact_scope": "development_checkpoint",
                    },
                }
                joblib.dump(checkpoint_info, f"{checkpoint_path}.joblib")
                _write_iteration_metrics(
                    f"{checkpoint_path}_metrics.txt",
                    model_name,
                    entry,
                    artifacts,
                    loo_r2,
                    loo_mae,
                )

                if X_model.shape[1] <= effective_min_features:
                    break

                from src.feature_selection import shap_rfecv_select_worst_feature

                use_consensus = X_model.shape[1] <= max(10, effective_min_features + 3)
                worst_feature, ranking, removal_reason = shap_rfecv_select_worst_feature(
                    artifacts["estimator"],
                    X_model,
                    y_development,
                    model_name,
                    cv_folds=5 if use_consensus else 0,
                )
                logger.info(
                    "SHAP-RFECV removes %s (%s); ranking=%s",
                    worst_feature,
                    removal_reason,
                    ranking,
                )

                removed_features.append(worst_feature)
                X_model = X_model.drop(columns=[worst_feature])

            selected_entry = _select_path_entry(shap_rfecv_path, force_n_features, logger)
            candidate_estimator = candidate_estimators[selected_entry["n_features"]]

        final_features = list(selected_entry["features"])
        final_estimator, scaler_X, scaler_y, _ = _fit_on_development(
            candidate_estimator,
            X_development[final_features],
            y_development,
        )
        (
            y_pred_development,
            y_pred_final_test,
            test_mae,
            test_r2,
            scatter_metrics,
        ) = _evaluate_final_test_once(
            final_estimator,
            scaler_X,
            scaler_y,
            X_development[final_features],
            y_development,
            X_final_test[final_features],
            y_final_test,
        )

        result = _make_result(
            selected_entry,
            final_features,
            shap_rfecv_path,
            test_mae,
            test_r2,
        )
        results[model_name] = result
        best_models[model_name] = final_estimator

        selected_history_index = performance_history["remaining_features"].index(
            len(final_features)
        )
        performance_history["final_test_mae"][selected_history_index] = test_mae
        performance_history["final_test_r2"][selected_history_index] = test_r2

        plot_performance_history(
            performance_history,
            model_name,
            os.path.join(model_dir, f"performance_history_{run_timestamp}.png"),
        )
        save_performance_history(
            performance_history,
            os.path.join(model_dir, f"performance_history_{run_timestamp}.csv"),
        )

        final_model_info = {
            "model": final_estimator,
            "estimator": final_estimator,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "features": final_features,
            "complete_params": dict(selected_entry["complete_params"]),
            # Legacy alias now intentionally stores the complete configuration.
            "hyperparameters": dict(selected_entry["complete_params"]),
            "metrics": result["metrics"],
            "primary_metrics": result["metrics"]["primary"],
            "secondary_metrics": result["metrics"]["secondary"],
            "test_mae": test_mae,
            "test_r2": test_r2,
            "optimal_n_features": len(final_features),
            "force_n_features": force_n_features,
            "removed_features": list(selected_entry["removed_features"]),
            "all_path_removed_features": list(removed_features),
            "shap_rfecv_path_summary": shap_rfecv_path,
            "evaluation_protocol": dict(evaluation_protocol),
        }
        final_model_path = os.path.join(model_dir, f"{model_name}_final_{run_timestamp}")

        internal_cv = result["internal_cv"]
        plot_scatter(
            y_train=y_development,
            y_pred_train=y_pred_development,
            y_test=y_final_test,
            y_pred_test=y_pred_final_test,
            model_name=f"{model_name} — {len(final_features)} features",
            mae_mean=result["stability"]["mae_mean"],
            output_dir=model_dir + os.sep,
            output_name=f"final_scatter_{run_timestamp}.png",
            X_train=X_development[final_features],
            X_test=X_final_test[final_features],
            r2_loo=result["loo"]["r2"],
            rkf_mae=internal_cv["rkf_mae_mean"],
            rkf_r2=internal_cv["rkf_r2_mean"],
            precomputed_metrics=scatter_metrics,
        )

        _write_final_artifacts_atomically(
            final_model_path,
            final_model_info,
            model_name,
            result,
            evaluation_protocol,
            force_n_features,
            final_estimator,
        )

        clean_old_versions(model_dir, keep_versions)

        logger.info(
            "%s complete | Final Test MAE %.4f, R² %.4f | "
            "Development %s MAE %.4f ± %.4f",
            model_name,
            test_mae,
            test_r2,
            INTERNAL_CV_LABEL,
            internal_cv["rkf_mae_mean"],
            internal_cv["rkf_mae_std"],
        )

    return results, best_models


def plot_performance_history(history, model_name, output_path):
    """Plot development-only selection metrics and the one final-test result."""
    iterations = np.asarray(history["iteration"])
    internal_cv_mae = np.asarray(
        history.get("internal_cv_mae_mean", history.get("rkf_mae_mean", [])),
        dtype=float,
    )
    internal_cv_mae_std = np.asarray(
        history.get("internal_cv_mae_std", history.get("rkf_mae_std", [])),
        dtype=float,
    )
    internal_cv_r2 = np.asarray(
        history.get("internal_cv_r2_mean", history.get("rkf_r2_mean", [])),
        dtype=float,
    )
    internal_cv_r2_std = np.asarray(
        history.get("internal_cv_r2_std", history.get("rkf_r2_std", [])),
        dtype=float,
    )
    stability_mae = np.asarray(
        history.get("stability_mae_mean", history.get("mae", [])),
        dtype=float,
    )
    loo_mae = np.asarray(history.get("loo_mae", []), dtype=float)
    loo_r2 = np.asarray(history.get("loo_r2", []), dtype=float)
    final_test_mae = np.asarray(
        history.get("final_test_mae", [np.nan] * len(iterations)), dtype=float
    )
    final_test_r2 = np.asarray(
        history.get("final_test_r2", [np.nan] * len(iterations)), dtype=float
    )

    fig, (ax_mae, ax_r2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    ax_mae.plot(
        iterations,
        internal_cv_mae,
        "b-o",
        linewidth=2,
        label=f"{RKF_PLOT_LABEL} MAE",
    )
    ax_mae.fill_between(
        iterations,
        internal_cv_mae - internal_cv_mae_std,
        internal_cv_mae + internal_cv_mae_std,
        alpha=0.15,
        color="blue",
    )
    ax_mae.plot(
        iterations,
        stability_mae,
        "s--",
        color="gray",
        label=STABILITY_PLOT_LABEL,
    )
    if len(loo_mae):
        ax_mae.plot(
            iterations,
            loo_mae,
            "d-",
            color="green",
            label="LOOCV MAE",
        )
    final_mae_mask = np.isfinite(final_test_mae)
    if final_mae_mask.any():
        ax_mae.scatter(
            iterations[final_mae_mask],
            final_test_mae[final_mae_mask],
            marker="*",
            s=220,
            color="black",
            label="Final Test MAE",
            zorder=5,
        )

    ax_r2.plot(
        iterations,
        internal_cv_r2,
        "r-o",
        linewidth=2,
        label=f"{RKF_PLOT_LABEL} R²",
    )
    ax_r2.fill_between(
        iterations,
        internal_cv_r2 - internal_cv_r2_std,
        internal_cv_r2 + internal_cv_r2_std,
        alpha=0.15,
        color="red",
    )
    if len(loo_r2):
        ax_r2.plot(
            iterations,
            loo_r2,
            "d-",
            color="orange",
            label="LOOCV R²",
        )
    final_r2_mask = np.isfinite(final_test_r2)
    if final_r2_mask.any():
        ax_r2.scatter(
            iterations[final_r2_mask],
            final_test_r2[final_r2_mask],
            marker="*",
            s=220,
            color="black",
            label="Final Test R²",
            zorder=5,
        )

    ax_mae.set_ylabel("MAE\nkcal/mol")
    ax_mae.set_title(f"{model_name} — Feature Selection Path", fontweight="bold")
    ax_r2.set_xlabel("Iteration")
    ax_r2.set_ylabel("R²")

    for axis in (ax_mae, ax_r2):
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    for index, (feature, remaining) in enumerate(
        zip(history["removed_feature"], history["remaining_features"])
    ):
        ax_mae.annotate(
            f"{feature}\n{remaining} left",
            (iterations[index], internal_cv_mae[index]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            rotation=45,
            ha="left",
            alpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_performance_history(history, output_path):
    """Save performance history to CSV."""
    pd.DataFrame(history).to_csv(output_path, index=False)
    logging.getLogger(__name__).info("Performance history saved to: %s", output_path)
