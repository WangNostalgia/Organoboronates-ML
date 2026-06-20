# -*- coding: utf-8 -*-
"""
===============================================================================
  Manual Feature-Count Selection & Plotting (Standalone)
===============================================================================

After training completes, inspect each model's SHAP-RFECV path summary, decide
manual per-model feature counts using development-only evidence, and use this
script to create strict manual-final artifacts.

This script now mirrors the main.py finalization protocol:
  1. Read manual_feature_selection.csv with columns: model_name,n_features
  2. Exactly load the iteration checkpoint with the requested feature count
  3. Refuse missing feature-count matches instead of falling back to closest
  4. Read evaluation_protocol development_indices/final_test_indices
  5. Refit selected features + complete_params on the development rows
  6. Evaluate the final-test rows exactly once
  7. Save a manual-final checkpoint, metrics txt, and final scatter plot

The CSV format:
    model_name,n_features
    SVR,5
    RandomForest,7
    XGBoost,4
    ...

Important:
    Do not choose manual n_features after inspecting final-test plots. Manual
    selection should be based on development-only metrics, path stability, and
    chemical interpretability so that the final test remains untouched.
===============================================================================
"""

from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import plot_scatter


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Configuration — EDIT THESE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MANUAL_SELECTION_CSV = os.path.join(os.path.dirname(__file__), 'manual_feature_selection.csv')
DATA_PATH = 'example/B_dataset.csv'
OUTPUT_DIR = 'models/manual_selection_plots'
TARGET_COL = 'activation_energy'
MODELS_DIR = 'models'
MODEL_JOBS = -1


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Core logic                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def load_manual_selections(csv_path):
    """
    Read the manual feature-count selection CSV.

    Expected format:
        model_name,n_features
        SVR,5
        RandomForest,7

    Returns
    -------
    dict
        {model_name: n_features}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Manual selection CSV not found: {csv_path}\n"
            f"Create one with columns: model_name, n_features\n"
            f"Example:\n"
            f"  model_name,n_features\n"
            f"  SVR,5\n"
            f"  XGBoost,4\n"
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'model_name' not in df.columns or 'n_features' not in df.columns:
        raise ValueError("CSV must have columns: model_name, n_features")

    selections = {}
    for _, row in df.iterrows():
        model_name = str(row['model_name']).strip()
        n_features = int(row['n_features'])
        if not model_name:
            raise ValueError("model_name values must be non-empty")
        if n_features < 1:
            raise ValueError(f"n_features must be positive for {model_name}")
        selections[model_name] = n_features

    print(f"Loaded {len(selections)} manual selections from {csv_path}")
    for name, n in selections.items():
        print(f"  {name}: {n} features")
    return selections


def _checkpoint_timestamp(filepath):
    match = re.search(r"_iteration_\d+_(\d{8}_\d{6})\.joblib$", os.path.basename(filepath))
    if not match:
        return ""
    return match.group(1)


def _internal_cv_mae(checkpoint_info):
    metrics = checkpoint_info.get('metrics', {})
    internal_cv = metrics.get('internal_cv', {})
    value = internal_cv.get('rkf_mae_mean')
    if value is None:
        value = metrics.get('rkf_mae_mean')
    if value is None:
        value = metrics.get('mae_mean', float('inf'))
    return float(value)


def find_checkpoint(model_name, n_features):
    """
    Exactly find the iteration checkpoint for a model and feature count.

    No closest fallback is allowed. If multiple exact checkpoints exist from
    different runs, choose the newest timestamp; if the newest timestamp is
    ambiguous, raise an error instead of silently cherry-picking by final-test
    performance.
    """
    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pattern = os.path.join(model_dir, '*_iteration_*.joblib')
    available_counts = []
    candidates = []
    for filepath in glob.glob(pattern):
        info = joblib.load(filepath)
        feature_count = len(info['features'])
        available_counts.append(feature_count)
        if feature_count == n_features:
            candidates.append((_checkpoint_timestamp(filepath), filepath, info))

    if not candidates:
        available = sorted(set(available_counts))
        raise FileNotFoundError(
            f"{model_name}: no iteration checkpoint with exactly {n_features} features. "
            f"Available feature counts: {available}. Re-run main.py if the requested "
            "feature count is not on the evaluated SHAP-RFECV path."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    newest_timestamp = candidates[0][0]
    newest = [item for item in candidates if item[0] == newest_timestamp]
    if len(newest) > 1:
        names = [os.path.basename(item[1]) for item in newest]
        raise ValueError(
            f"{model_name}: ambiguous exact {n_features}-feature checkpoints with "
            f"timestamp {newest_timestamp}: {names}"
        )

    timestamp, source_file, info = newest[0]
    print(
        f"  {model_name}: loaded exact {n_features}-feature checkpoint "
        f"(internal_cv_mae={_internal_cv_mae(info):.4f}, file: {os.path.basename(source_file)})"
    )
    return info, source_file


def _require_protocol_indices(checkpoint_info, source_file):
    protocol = checkpoint_info.get('evaluation_protocol') or {}
    development_indices = protocol.get('development_indices')
    final_test_indices = protocol.get('final_test_indices')

    if not development_indices or not final_test_indices:
        raise ValueError(
            f"Checkpoint {source_file} does not contain both "
            "evaluation_protocol['development_indices'] and "
            "evaluation_protocol['final_test_indices']. Re-run main.py to create "
            "protocol-aware iteration checkpoints."
        )
    return protocol, list(development_indices), list(final_test_indices)


def _split_data_by_checkpoint_protocol(data, checkpoint_info, source_file, features):
    protocol, development_indices, final_test_indices = _require_protocol_indices(
        checkpoint_info,
        source_file,
    )

    missing_dev = [idx for idx in development_indices if idx not in data.index]
    missing_test = [idx for idx in final_test_indices if idx not in data.index]
    if missing_dev or missing_test:
        raise ValueError(
            f"Checkpoint split indices do not match {DATA_PATH}. "
            f"Missing development labels: {missing_dev[:10]}; "
            f"missing final-test labels: {missing_test[:10]}. Use the same, "
            "unreordered source CSV used by main.py."
        )

    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Training data missing selected features: {missing_features}")
    if TARGET_COL not in data.columns:
        raise ValueError(f"Training data missing target column: {TARGET_COL}")

    X_development = data.loc[development_indices, features].copy()
    y_development = data.loc[development_indices, TARGET_COL].copy()
    X_final_test = data.loc[final_test_indices, features].copy()
    y_final_test = data.loc[final_test_indices, TARGET_COL].copy()
    return protocol, X_development, y_development, X_final_test, y_final_test


def _make_estimator_from_checkpoint(checkpoint_info):
    estimator_template = checkpoint_info.get('estimator', checkpoint_info.get('model'))
    if estimator_template is None:
        raise ValueError("Checkpoint missing both 'estimator' and 'model'.")

    estimator = clone(estimator_template)
    complete_params = dict(
        checkpoint_info.get('complete_params')
        or checkpoint_info.get('hyperparameters')
        or estimator.get_params()
    )
    try:
        estimator.set_params(**complete_params)
    except ValueError as exc:
        raise ValueError(
            "Could not apply checkpoint complete_params to the estimator. "
            "This manual-final flow requires a checkpoint with a reproducible "
            "complete estimator configuration."
        ) from exc
    return estimator, complete_params


def _fit_and_predict(estimator, X_development, y_development, X_final_test):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 100))
    X_development_scaled = scaler_X.fit_transform(X_development)
    X_final_test_scaled = scaler_X.transform(X_final_test)
    y_development_scaled = scaler_y.fit_transform(
        np.asarray(y_development).reshape(-1, 1)
    ).ravel()

    fitted_estimator = clone(estimator)
    fitted_estimator.fit(X_development_scaled, y_development_scaled)

    y_pred_dev_scaled = np.asarray(fitted_estimator.predict(X_development_scaled)).reshape(-1, 1)
    y_pred_test_scaled = np.asarray(fitted_estimator.predict(X_final_test_scaled)).reshape(-1, 1)
    y_pred_development = scaler_y.inverse_transform(y_pred_dev_scaled).ravel()
    y_pred_final_test = scaler_y.inverse_transform(y_pred_test_scaled).ravel()
    return fitted_estimator, scaler_X, scaler_y, y_pred_development, y_pred_final_test


def _pearson_r(y_true, y_pred):
    if len(y_true) > 1:
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    return 0.0


def _scatter_metrics(y_development, y_pred_development, y_final_test, y_pred_final_test):
    return {
        'r_train': _pearson_r(y_development, y_pred_development),
        'r_test': _pearson_r(y_final_test, y_pred_final_test),
        'r2_train': float(r2_score(y_development, y_pred_development)),
        'rmse_test': float(np.sqrt(mean_squared_error(y_final_test, y_pred_final_test))),
        'r2_test': float(r2_score(y_final_test, y_pred_final_test)),
        'mae_test': float(mean_absolute_error(y_final_test, y_pred_final_test)),
    }


def _secondary_metrics(checkpoint_info):
    metrics = checkpoint_info.get('metrics', {})
    internal_cv = metrics.get('internal_cv', {})
    stability = metrics.get('stability', {})
    loo = metrics.get('loo', {})
    return {
        'development_cv_mae': metrics.get('development_cv_mae'),
        'internal_cv': internal_cv,
        'stability': stability,
        'loo': loo,
        'mae_mean': metrics.get('mae_mean'),
        'rkf_mae_mean': internal_cv.get('rkf_mae_mean', metrics.get('rkf_mae_mean')),
        'rkf_r2_mean': internal_cv.get('rkf_r2_mean', metrics.get('rkf_r2_mean')),
        'loo_r2': loo.get('r2', metrics.get('loo_r2')),
    }


def _write_metrics_txt(path, model_name, n_features, features, complete_params,
                       source_file, protocol, metrics, secondary):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write("Selection mode: manual feature-count selection\n")
        handle.write(f"Manual feature count: {n_features}\n")
        handle.write(f"Features ({n_features}): {', '.join(features)}\n")
        handle.write(f"Source iteration checkpoint: {source_file}\n")
        handle.write("\n--- Evaluation Protocol ---\n")
        handle.write("Split source: checkpoint evaluation_protocol indices\n")
        handle.write(f"Protocol name: {protocol.get('name', 'N/A')}\n")
        handle.write(f"Random state: {protocol.get('random_state', 'N/A')}\n")
        handle.write(f"Test size: {protocol.get('test_size', 'N/A')}\n")
        handle.write(f"Development rows: {len(protocol.get('development_indices', []))}\n")
        handle.write(f"Final-test rows: {len(protocol.get('final_test_indices', []))}\n")
        handle.write("Final-test evaluations in this manual-final artifact: 1\n")
        handle.write("\n--- PRIMARY: Untouched Final Test ---\n")
        handle.write(f"Final Test MAE: {metrics['mae_test']:.4f} kcal/mol\n")
        handle.write(f"Final Test R²:  {metrics['r2_test']:.4f}\n")
        handle.write(f"Final Test RMSE: {metrics['rmse_test']:.4f} kcal/mol\n")
        handle.write(f"Final Test Pearson R: {metrics['r_test']:.4f}\n")
        handle.write("\n--- SECONDARY: Development-Only Metrics from Source Checkpoint ---\n")
        if secondary.get('rkf_mae_mean') is not None:
            handle.write(f"Internal CV MAE: {secondary['rkf_mae_mean']:.4f}\n")
        if secondary.get('rkf_r2_mean') is not None:
            handle.write(f"Internal CV R²:  {secondary['rkf_r2_mean']:.4f}\n")
        if secondary.get('mae_mean') is not None:
            handle.write(f"Stability MAE:   {secondary['mae_mean']:.4f}\n")
        if secondary.get('loo_r2') is not None:
            handle.write(f"LOOCV R²:        {secondary['loo_r2']:.4f}\n")
        handle.write("\nComplete Parameters:\n")
        handle.write(f"{complete_params}\n")


def generate_manual_final_artifacts(model_name, n_features, checkpoint_info, source_file):
    """
    Generate a strict manual-final checkpoint, metrics txt, and scatter plot.
    """
    features = list(checkpoint_info['features'])
    if len(features) != n_features:
        raise ValueError(
            f"Loaded checkpoint feature count {len(features)} does not match requested {n_features}."
        )

    data = pd.read_csv(DATA_PATH).dropna(axis=1, how='all')
    protocol, X_dev, y_dev, X_test, y_test = _split_data_by_checkpoint_protocol(
        data,
        checkpoint_info,
        source_file,
        features,
    )
    estimator, complete_params = _make_estimator_from_checkpoint(checkpoint_info)
    fitted_estimator, scaler_X, scaler_y, y_pred_dev, y_pred_test = _fit_and_predict(
        estimator,
        X_dev,
        y_dev,
        X_test,
    )
    metrics = _scatter_metrics(y_dev, y_pred_dev, y_test, y_pred_test)
    secondary = _secondary_metrics(checkpoint_info)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(MODELS_DIR, model_name)
    plot_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    safe_model_label = f"{model_name} ({n_features} feat; manual feature-count selection)"
    scatter_name = f"manual_final_scatter_{n_features}feat_{timestamp}.png"
    plot_scatter(
        y_train=y_dev,
        y_pred_train=y_pred_dev,
        y_test=y_test,
        y_pred_test=y_pred_test,
        model_name=safe_model_label,
        mae_mean=secondary.get('mae_mean'),
        output_dir=plot_dir + os.sep,
        output_name=scatter_name,
        X_train=X_dev,
        X_test=X_test,
        r2_loo=secondary.get('loo_r2'),
        rkf_mae=secondary.get('rkf_mae_mean'),
        rkf_r2=secondary.get('rkf_r2_mean'),
        precomputed_metrics=metrics,
    )

    manual_protocol = {
        **protocol,
        'artifact_scope': 'manual_final_checkpoint',
        'selection_mode': 'manual_feature_count_selection',
        'manual_n_features': n_features,
        'source_iteration_checkpoint': source_file,
        'final_test_evaluations': 1,
    }
    manual_metrics = {
        'primary': {
            'final_test': {
                'test_mae': metrics['mae_test'],
                'test_r2': metrics['r2_test'],
                'rmse_test': metrics['rmse_test'],
                'r_test': metrics['r_test'],
            }
        },
        'secondary': secondary,
        'test_mae': metrics['mae_test'],
        'test_r2': metrics['r2_test'],
        'rmse_test': metrics['rmse_test'],
        'mae_test_avg': metrics['mae_test'],
        'r2_test_avg': metrics['r2_test'],
        'rkf_mae_opt_mean': secondary.get('rkf_mae_mean'),
        'rkf_r2_opt_mean': secondary.get('rkf_r2_mean'),
        'mae_mean': secondary.get('mae_mean'),
    }
    manual_info = {
        'model': fitted_estimator,
        'estimator': fitted_estimator,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': features,
        'complete_params': complete_params,
        'hyperparameters': complete_params,
        'metrics': manual_metrics,
        'primary_metrics': manual_metrics['primary'],
        'secondary_metrics': secondary,
        'test_mae': metrics['mae_test'],
        'test_r2': metrics['r2_test'],
        'rmse_test': metrics['rmse_test'],
        'optimal_n_features': n_features,
        'manual_n_features': n_features,
        'selection_mode': 'manual_feature_count_selection',
        'source_iteration_checkpoint': source_file,
        'evaluation_protocol': manual_protocol,
    }

    artifact_stem = os.path.join(model_dir, f"{model_name}_manual_final_{n_features}feat_{timestamp}")
    checkpoint_path = f"{artifact_stem}.joblib"
    metrics_path = f"{artifact_stem}_metrics.txt"
    joblib.dump(manual_info, checkpoint_path)
    _write_metrics_txt(
        metrics_path,
        model_name,
        n_features,
        features,
        complete_params,
        source_file,
        manual_protocol,
        metrics,
        secondary,
    )

    print(f"  {model_name}: manual-final checkpoint saved to {checkpoint_path}")
    print(f"  {model_name}: metrics saved to {metrics_path}")
    print(f"  {model_name}: scatter saved to {os.path.join(plot_dir, scatter_name)}")
    print(
        f"  {model_name}: Final Test MAE={metrics['mae_test']:.4f}, "
        f"R²={metrics['r2_test']:.4f}, RMSE={metrics['rmse_test']:.4f}"
    )
    return manual_info, checkpoint_path, metrics_path


def predict_external(checkpoint_info, csv_path, output_path=None):
    """Use a manual-final checkpoint to predict on external data."""
    model = checkpoint_info['model']
    scaler_X = checkpoint_info['scaler_X']
    scaler_y = checkpoint_info['scaler_y']
    features = checkpoint_info['features']

    data = pd.read_csv(csv_path)
    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Missing features in external data: {missing}")

    X_new = data[features]
    X_scaled = scaler_X.transform(X_new)
    y_pred_s = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    data['predicted_activation_energy'] = y_pred
    if output_path is None:
        base = os.path.splitext(csv_path)[0]
        output_path = f"{base}_predicted_{len(features)}feat.csv"
    data.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return data


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    selections = load_manual_selections(MANUAL_SELECTION_CSV)

    for model_name, n_features in selections.items():
        print(f"\n{'=' * 60}")
        print(f"  Manual-final: {model_name} ({n_features} features)")
        print(f"{'=' * 60}")
        try:
            info, src = find_checkpoint(model_name, n_features)
            generate_manual_final_artifacts(model_name, n_features, info, src)
        except Exception as exc:
            print(f"  ERROR: {model_name}: {exc}")
            continue

    print("\nDone. Manual-final artifacts saved under models/<ModelName>/ and plots under "
          f"{OUTPUT_DIR}/")
