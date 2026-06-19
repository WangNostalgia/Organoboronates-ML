# -*- coding: utf-8 -*-
"""
Checkpoint loading guide for SHAP-RFECV training outputs.

This example mirrors the selection rules implemented in
``src.external_validation.load_model()``:

- search both final and iteration checkpoints
- require an exact feature count by default
- prefer exact final checkpoints over exact iteration checkpoints
- break ties by newest filename timestamp (YYYYMMDD_HHMMSS)
- only allow nearest-match loading when ``allow_closest=True``
"""

from __future__ import annotations

import glob
import math
import os
from numbers import Real
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.external_validation import load_model


_MISSING = object()


def _is_missing_metric_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Real) and not isinstance(value, bool):
        return not math.isfinite(float(value))
    return False


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric_value(metrics: dict[str, Any] | None, *candidates: Any) -> Any:
    """
    Return the first metric value found in ordered candidates.

    Each candidate may be either:
    - a top-level string key, or
    - a tuple/list path for nested dictionaries
    """
    if not isinstance(metrics, dict):
        return None

    for candidate in candidates:
        if isinstance(candidate, str):
            value = metrics.get(candidate, _MISSING)
        else:
            value = metrics
            for part in candidate:
                if not isinstance(value, dict) or part not in value:
                    value = _MISSING
                    break
                value = value[part]

        if value is not _MISSING and not _is_missing_metric_value(value):
            return value

    return None


def _format_metric(
    metrics: dict[str, Any] | None,
    *candidates: Any,
    precision: int = 4,
) -> str:
    """
    Format a metric safely for console output.

    - numeric finite values -> fixed precision
    - strings -> returned unchanged
    - missing / None / NaN -> ``N/A``
    """
    value = _metric_value(metrics, *candidates)
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f"{numeric:.{precision}f}"
        return "N/A"
    return str(value)


def _print_metric_line(label: str, metrics: dict[str, Any] | None, *candidates: Any) -> None:
    print(f"  {label:<18}: {_format_metric(metrics, *candidates)}")


def _extract_model_context(model_dir: str) -> tuple[str, str]:
    model_dir = os.path.normpath(model_dir)
    model_name = os.path.basename(model_dir)
    models_dir = os.path.dirname(model_dir) or "."
    return model_name, models_dir


def _load_checkpoint_info(model_dir: str, n_features: int | None = None, allow_closest: bool = False) -> dict[str, Any]:
    model_name, models_dir = _extract_model_context(model_dir)
    return load_model(
        model_name=model_name,
        n_features=n_features,
        models_dir=models_dir,
        allow_closest=allow_closest,
    )


def _print_summary(model_info: dict[str, Any]) -> None:
    metrics = model_info.get("metrics", {})
    features = model_info.get("features", [])

    print(f"  Loaded from: {os.path.basename(model_info.get('_loaded_from', '?'))}")
    print(f"  Checkpoint type: {model_info.get('_checkpoint_type', 'unknown')}")
    print(
        f"  Requested features: {model_info.get('_requested_n_features', 'latest')} | "
        f"actual features: {model_info.get('_actual_n_features', len(features))}"
    )
    print(f"  Features ({len(features)}): {features}")
    _print_metric_line(
        "Final Test MAE",
        metrics,
        "test_mae",
        ("primary", "final_test", "test_mae"),
        "mae_test_avg",
        "mae_test",
    )
    _print_metric_line(
        "Final Test R²",
        metrics,
        "test_r2",
        ("primary", "final_test", "test_r2"),
        "r2_test_avg",
        "r2_test",
    )
    _print_metric_line(
        "Internal CV MAE",
        metrics,
        ("secondary", "internal_cv", "rkf_mae_mean"),
        ("internal_cv", "rkf_mae_mean"),
        "rkf_mae_mean",
        "rkf_mae_opt_mean",
    )
    _print_metric_line(
        "Internal CV R²",
        metrics,
        ("secondary", "internal_cv", "rkf_r2_mean"),
        ("internal_cv", "rkf_r2_mean"),
        "rkf_r2_mean",
        "rkf_r2_opt_mean",
    )
    _print_metric_line(
        "100-split MAE",
        metrics,
        ("secondary", "stability", "mae_mean"),
        ("stability", "mae_mean"),
        "mae_mean",
    )
    _print_metric_line(
        "LOOCV MAE",
        metrics,
        ("secondary", "loo", "mae"),
        ("loo", "mae"),
        "loo_mae",
        "mae_loo_avg",
    )
    _print_metric_line(
        "LOOCV R²",
        metrics,
        ("secondary", "loo", "r2"),
        ("loo", "r2"),
        "loo_r2",
        "r2_loo_avg",
    )


def list_checkpoints(model_dir: str) -> None:
    """
    List final and iteration checkpoints in ``models/<ModelName>/``.

    Usage:
        list_checkpoints("models/SVR")
    """
    iteration_files = sorted(glob.glob(os.path.join(model_dir, "*_iteration_*.joblib")))
    final_files = sorted(glob.glob(os.path.join(model_dir, "*_final_*.joblib")))

    print(f"\n{'=' * 72}")
    print(f"  Checkpoints in: {model_dir}")
    print(f"{'=' * 72}")

    if final_files:
        print("\n  Final checkpoints:")
        for path in final_files:
            info = joblib.load(path)
            metrics = info.get("metrics", {})
            print(f"    {os.path.basename(path)}")
            print(f"      Features ({len(info.get('features', []))}): {info.get('features', [])}")
            print(
                "      Final Test MAE / R²: "
                f"{_format_metric(metrics, 'test_mae', ('primary', 'final_test', 'test_mae'), 'mae_test_avg', 'mae_test')} / "
                f"{_format_metric(metrics, 'test_r2', ('primary', 'final_test', 'test_r2'), 'r2_test_avg', 'r2_test')}"
            )

    if iteration_files:
        print(f"\n  Iteration checkpoints ({len(iteration_files)} total):")
        for path in iteration_files:
            info = joblib.load(path)
            removed = info.get("removed_features", [])
            last_removed = removed[-1] if removed else "Initial"
            metrics = info.get("metrics", {})
            print(f"    {os.path.basename(path)}")
            print(
                f"      Features={len(info.get('features', []))} | "
                f"last removed={last_removed} | "
                f"Internal CV MAE={_format_metric(metrics, ('internal_cv', 'rkf_mae_mean'), 'rkf_mae_mean', 'rkf_mae_opt_mean')}"
            )
    else:
        print("\n  (No iteration checkpoints found)")


def load_final_model(model_dir: str):
    """
    Load the newest preferred final checkpoint for prediction.

    Usage:
        model, scaler_X, scaler_y, features = load_final_model("models/SVR")
    """
    info = _load_checkpoint_info(model_dir)
    print("\n  Final checkpoint selection summary")
    _print_summary(info)
    return info["model"], info["scaler_X"], info["scaler_y"], info["features"]


def load_by_feature_count(model_dir: str, target_n_features: int, allow_closest: bool = False):
    """
    Load a checkpoint by desired feature count.

    By default this requires an exact feature count. Set ``allow_closest=True``
    only when you explicitly want nearest-match fallback.
    """
    info = _load_checkpoint_info(
        model_dir,
        n_features=target_n_features,
        allow_closest=allow_closest,
    )
    print("\n  Feature-count checkpoint selection summary")
    _print_summary(info)
    return info["model"], info["scaler_X"], info["scaler_y"], info["features"]


def compare_checkpoints(model_dir: str, n_features_a: int, n_features_b: int) -> None:
    """
    Compare two exact feature-count checkpoints side by side.

    Usage:
        compare_checkpoints("models/SVR", 7, 5)
    """
    info_a = _load_checkpoint_info(model_dir, n_features=n_features_a)
    info_b = _load_checkpoint_info(model_dir, n_features=n_features_b)
    metrics_a = info_a.get("metrics", {})
    metrics_b = info_b.get("metrics", {})

    rows = [
        ("Final Test MAE", ("test_mae", ("primary", "final_test", "test_mae"), "mae_test_avg", "mae_test"), "lower"),
        ("Final Test R²", ("test_r2", ("primary", "final_test", "test_r2"), "r2_test_avg", "r2_test"), "higher"),
        ("Internal CV MAE", (("secondary", "internal_cv", "rkf_mae_mean"), ("internal_cv", "rkf_mae_mean"), "rkf_mae_mean", "rkf_mae_opt_mean"), "lower"),
        ("Internal CV R²", (("secondary", "internal_cv", "rkf_r2_mean"), ("internal_cv", "rkf_r2_mean"), "rkf_r2_mean", "rkf_r2_opt_mean"), "higher"),
        ("100-split MAE", (("secondary", "stability", "mae_mean"), ("stability", "mae_mean"), "mae_mean"), "lower"),
        ("LOOCV MAE", (("secondary", "loo", "mae"), ("loo", "mae"), "loo_mae", "mae_loo_avg"), "lower"),
        ("LOOCV R²", (("secondary", "loo", "r2"), ("loo", "r2"), "loo_r2", "r2_loo_avg"), "higher"),
    ]

    print(f"\n{'=' * 84}")
    print(f"  Manual comparison: {n_features_a} features vs {n_features_b} features")
    print(f"{'=' * 84}")
    print(f"  {'Metric':<18} {f'{n_features_a} features':>18} {f'{n_features_b} features':>18} {'Better':>10}")
    print(f"  {'-' * 70}")

    for label, candidates, direction in rows:
        value_a = _metric_value(metrics_a, *candidates)
        value_b = _metric_value(metrics_b, *candidates)
        formatted_a = _format_metric(metrics_a, *candidates)
        formatted_b = _format_metric(metrics_b, *candidates)

        better = "N/A"
        if _is_finite_number(value_a) and _is_finite_number(value_b):
            if direction == "lower":
                better = "<-" if value_a <= value_b else "->"
            else:
                better = "<-" if value_a >= value_b else "->"

        print(f"  {label:<18} {formatted_a:>18} {formatted_b:>18} {better:>10}")

    set_a = set(info_a.get("features", []))
    set_b = set(info_b.get("features", []))
    print(f"\n  Features ({n_features_a}): {info_a.get('features', [])}")
    print(f"  Features ({n_features_b}): {info_b.get('features', [])}")
    print(f"  Only in {n_features_a}: {sorted(set_a - set_b)}")
    print(f"  Only in {n_features_b}: {sorted(set_b - set_a)}")


def predict_with_checkpoint(model_dir: str, csv_path: str, n_features: int | None = None, allow_closest: bool = False):
    """
    Predict on new data with either the latest final checkpoint or a selected
    feature-count checkpoint.
    """
    if n_features is None:
        info = _load_checkpoint_info(model_dir)
    else:
        info = _load_checkpoint_info(
            model_dir,
            n_features=n_features,
            allow_closest=allow_closest,
        )

    data = pd.read_csv(csv_path)
    features = info["features"]
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X_new = data[features]
    X_scaled = info["scaler_X"].transform(X_new)
    y_pred_scaled = np.asarray(info["model"].predict(X_scaled)).reshape(-1, 1)
    y_pred = info["scaler_y"].inverse_transform(y_pred_scaled).ravel()

    result = data.copy()
    result["predicted_activation_energy"] = y_pred

    out_path = csv_path.replace(".csv", f"_predicted_{len(features)}feat.csv")
    result.to_csv(out_path, index=False)

    print("\n  Prediction summary")
    _print_summary(info)
    print(f"  New data rows: {len(result)}")
    print(f"  Output file: {out_path}")
    print(f"  Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f} kcal/mol")
    return result


if __name__ == "__main__":
    list_checkpoints("models/SVR")

    # model, scaler_X, scaler_y, features = load_final_model("models/SVR")
    # model, scaler_X, scaler_y, features = load_by_feature_count("models/SVR", 5)
    # compare_checkpoints("models/SVR", 7, 5)
    # predict_with_checkpoint("models/SVR", "new_data.csv", n_features=5)
