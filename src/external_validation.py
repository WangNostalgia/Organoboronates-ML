"""
External Validation Module for Organoboronate ML Models
========================================================

Provides a reusable CLI and Python API for evaluating trained `.joblib` models
on completely independent, unseen data.

Key capabilities:
  - List all trained models with their metrics and feature counts
  - Load final, manual-final, or iteration checkpoints by name and feature count
  - Align external CSV columns to the model's expected feature set
  - Preserve metadata columns such as ID, SMILES, filename, conformer labels, and source files
  - Predict activation energies on new data
  - Calculate MAE, R², RMSE when ground-truth `activation_energy` is present
  - Generate prediction-vs-experiment scatter plots
  - Export predictions to CSV
"""

import argparse
import glob
import logging
import os
import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from numbers import Integral

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    'ignore', message='X does not have valid feature names', category=UserWarning
)
warnings.filterwarnings(
    'ignore', message='BaseEstimator._validate_data', category=FutureWarning
)

DEFAULT_MODELS_DIR = 'models'
DEFAULT_TARGET_COL = 'activation_energy'
DEFAULT_OUTPUT_DIR = 'external_validation_results'
DEFAULT_ID_COLS = ('ID', 'SMILES', 'filename')


class EnsembleValidationError(RuntimeError):
    """Raised when ensemble members cannot produce a valid aggregate."""


FEATURE_ALIASES = {
    # Add aliases here as needed. Example:
    # 'pKa': 'pka',
    # 'HOMO': 'homo_energy',
}


def _normalise_column_name(col: str) -> str:
    """Map a column name to its canonical form using FEATURE_ALIASES."""
    return FEATURE_ALIASES.get(col, col)


def _coerce_id_cols(id_cols=None) -> tuple[str, ...]:
    if id_cols is None:
        return DEFAULT_ID_COLS
    if isinstance(id_cols, str):
        return tuple(col.strip() for col in id_cols.split(',') if col.strip())
    return tuple(str(col).strip() for col in id_cols if str(col).strip())


def _unique_preserving_order(columns: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for col in columns:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def _metadata_columns(df: pd.DataFrame,
                      expected_features: Sequence[str],
                      target_col: str | None = DEFAULT_TARGET_COL,
                      id_cols=None,
                      preserve_all_metadata: bool = True) -> list[str]:
    """
    Return metadata columns to preserve in prediction outputs.

    Preferred identity columns, such as ID/SMILES/filename, are placed first.
    By default, all additional non-feature, non-target columns are kept as
    metadata so external predictions remain traceable even when users add
    conformer IDs, source paths, batch labels, or other annotations.
    """
    feature_set = set(expected_features)
    excluded = set(feature_set)
    if target_col is not None:
        excluded.add(target_col)

    preferred = [col for col in _coerce_id_cols(id_cols) if col in df.columns and col not in excluded]
    if not preserve_all_metadata:
        return _unique_preserving_order(preferred)

    extra = [col for col in df.columns if col not in excluded and col not in preferred]
    return _unique_preserving_order(preferred + extra)


def _metadata_target_for_output(df: pd.DataFrame, target_col: str | None) -> str | None:
    """Exclude an existing target column from metadata even in predict-only mode."""
    if target_col is not None:
        return target_col
    if DEFAULT_TARGET_COL in df.columns:
        logger.warning(
            "Predict-only mode requested but column '%s' is present; preserving final-test discipline by not using it for metrics and not treating it as metadata.",
            DEFAULT_TARGET_COL,
        )
        return DEFAULT_TARGET_COL
    return None


def _nested_metric(mapping: dict, path: Sequence[str]):
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _available_model_metric(metrics: dict, metric_name: str, legacy_name: str):
    metric_paths = (
        ('secondary', 'internal_cv', metric_name),
        ('internal_cv', metric_name),
        (metric_name,),
        (legacy_name,),
    )
    for path in metric_paths:
        value = _nested_metric(metrics, path)
        if value is not None:
            return value
    return None


def list_available_models(models_dir: str = DEFAULT_MODELS_DIR) -> pd.DataFrame:
    """Scan the models directory and return trained final/manual-final checkpoints."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    records = []
    for model_dir in sorted(glob.glob(os.path.join(models_dir, '*'))):
        if not os.path.isdir(model_dir):
            continue
        for checkpoint in _discover_model_checkpoints(model_dir):
            if checkpoint['checkpoint_type'] not in {'final', 'manual_final'}:
                continue
            info = checkpoint['model_info']
            features = info.get('features', [])
            metrics = info.get('metrics', {})
            records.append({
                'model_name': os.path.basename(model_dir),
                'checkpoint_type': checkpoint['checkpoint_type'],
                'n_features': checkpoint['actual_n_features'],
                'features': ', '.join(features) if features else 'N/A',
                'rkf_mae': _available_model_metric(metrics, 'rkf_mae_mean', 'rkf_mae_opt_mean'),
                'rkf_r2': _available_model_metric(metrics, 'rkf_r2_mean', 'rkf_r2_opt_mean'),
                'filepath': checkpoint['path'],
            })

    if not records:
        raise FileNotFoundError(
            f"No *_final_*.joblib or *_manual_final_*feat_*.joblib files found under {models_dir}. Run main.py or manual_selection_and_plot.py first."
        )

    return pd.DataFrame(records).sort_values(['model_name', 'checkpoint_type']).reset_index(drop=True)


def external_validation(model_info: dict,
                        external_data,
                        target_col: str = DEFAULT_TARGET_COL,
                        output_dir: str = DEFAULT_OUTPUT_DIR,
                        output_prefix: str = None,
                        id_cols=None,
                        preserve_all_metadata: bool = True) -> dict:
    """Evaluate one trained model on an external single-fragment dataset."""
    if isinstance(external_data, str):
        df = pd.read_csv(external_data)
        logger.info(
            "Loaded external data from %s (%d rows, %d columns)",
            external_data, len(df), len(df.columns)
        )
    elif isinstance(external_data, pd.DataFrame):
        df = external_data.copy()
    else:
        raise TypeError(
            f"external_data must be a file path (str) or DataFrame, got {type(external_data).__name__}"
        )

    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info("Dropped %d unnamed column(s): %s", len(unnamed_cols), unnamed_cols)

    df = df.rename(columns=_normalise_column_name)

    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']
    expected_features = list(model_info['features'])

    if output_prefix is None:
        if '_loaded_from' in model_info:
            model_name = os.path.basename(os.path.dirname(model_info['_loaded_from']))
        else:
            model_name = 'Model'
    else:
        model_name = output_prefix

    features_present = [feature for feature in expected_features if feature in df.columns]
    features_missing = [feature for feature in expected_features if feature not in df.columns]

    if features_missing:
        available = [col for col in df.columns if col not in set(expected_features) and col != target_col]
        logger.error(
            "Missing features in external data: %s\n  Expected: %s\n  Available candidate columns: %s",
            features_missing, expected_features, available
        )
        raise ValueError(
            f"External data is missing {len(features_missing)} required feature(s): {features_missing}. "
            "Ensure the input CSV contains all features the model was trained on."
        )

    X_external = df[expected_features].copy()
    nan_mask = X_external.isna().any(axis=1)
    if nan_mask.any():
        logger.warning(
            "Dropping %d row(s) with NaN feature values (out of %d total).",
            nan_mask.sum(), len(X_external)
        )
        X_external = X_external[~nan_mask]
        df = df[~nan_mask].reset_index(drop=True)
        X_external = X_external.reset_index(drop=True)

    X_scaled = scaler_X.transform(X_external)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    metadata_target_col = _metadata_target_for_output(df, target_col)
    metadata_cols = _metadata_columns(
        df,
        expected_features=expected_features,
        target_col=metadata_target_col,
        id_cols=id_cols,
        preserve_all_metadata=preserve_all_metadata,
    )
    result_cols = _unique_preserving_order(metadata_cols + expected_features)
    results_df = df[result_cols].copy()
    results_df['predicted_activation_energy'] = y_pred

    mae = r2 = rmse = None
    has_ground_truth = target_col is not None and target_col in df.columns

    if has_ground_truth:
        y_true = df[target_col].values
        valid_mask = ~np.isnan(y_true)
        if not valid_mask.all():
            logger.info(
                "Dropping %d row(s) with NaN %s (no ground truth).",
                (~valid_mask).sum(), target_col
            )
            y_true_eval = y_true[valid_mask]
            y_pred_eval = y_pred[valid_mask]
        else:
            y_true_eval = y_true
            y_pred_eval = y_pred

        if len(y_true_eval) > 0:
            mae = float(mean_absolute_error(y_true_eval, y_pred_eval))
            r2 = float(r2_score(y_true_eval, y_pred_eval))
            rmse = float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval)))
            results_df[target_col] = df[target_col].values
            results_df['absolute_error'] = np.abs(y_pred - df[target_col].values)
            logger.info(
                "External validation metrics for %s (%d features): MAE=%.4f, R²=%.4f, RMSE=%.4f, N=%d",
                model_name, len(expected_features), mae, r2, rmse, len(y_true_eval)
            )
        else:
            logger.warning("No valid ground truth values found — prediction-only mode.")
            has_ground_truth = False
    else:
        logger.info("Column '%s' not found in external data — running in prediction-only mode.", target_col)

    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_model_name = model_name.replace(' ', '_')

    preds_path = os.path.join(output_dir, f"{safe_model_name}_external_validation_{timestamp}.csv")
    results_df.to_csv(preds_path, index=False, encoding='utf-8-sig')
    output_files.append(preds_path)
    logger.info("Predictions saved to: %s", preds_path)

    if has_ground_truth and mae is not None:
        plot_path = os.path.join(output_dir, f"{safe_model_name}_external_scatter_{timestamp}.png")
        _plot_external_scatter(
            y_true_eval, y_pred_eval, model_name, len(expected_features), mae, r2, rmse, plot_path
        )
        output_files.append(plot_path)

    summary_path = os.path.join(output_dir, f"{safe_model_name}_external_summary_{timestamp}.txt")
    _write_external_summary(
        summary_path, model_name, model_info, len(expected_features), expected_features,
        features_missing, len(X_external), mae, r2, rmse, metadata_cols
    )
    output_files.append(summary_path)

    return {
        'predictions': results_df,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'n_samples': len(X_external),
        'n_features_used': len(expected_features),
        'features_used': expected_features,
        'features_missing': features_missing,
        'features_present': features_present,
        'metadata_columns': metadata_cols,
        'output_files': output_files,
    }


def _plot_external_scatter(y_true, y_pred, model_name, n_features, mae, r2, rmse, output_path):
    """Create a prediction-vs-experiment scatter plot for external validation."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(y_true, y_pred, alpha=0.8, edgecolors='black', linewidths=0.5)
    y_min = float(min(np.min(y_true), np.min(y_pred)))
    y_max = float(max(np.max(y_true), np.max(y_pred)))
    padding = max((y_max - y_min) * 0.05, 1e-6)
    lo = y_min - padding
    hi = y_max + padding
    ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Experimental activation energy')
    ax.set_ylabel('Predicted activation energy')
    ax.set_title(f'{model_name} external validation ({n_features} features)')
    ax.text(
        0.05, 0.95,
        f'MAE = {mae:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}',
        transform=ax.transAxes,
        va='top',
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Scatter plot saved to: %s", output_path)


def _write_external_summary(path, model_name, model_info, n_features,
                            features_used, features_missing, n_samples,
                            mae, r2, rmse, metadata_cols=None):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("External Validation Summary\n")
        f.write("===========================\n\n")
        f.write(f"Model:       {model_name}\n")
        f.write(f"Checkpoint:  {model_info.get('_loaded_from', 'N/A')}\n")
        f.write(f"Features:    {n_features}\n")
        f.write(f"Feature list: {', '.join(features_used)}\n")
        if metadata_cols:
            f.write(f"Metadata columns preserved: {', '.join(metadata_cols)}\n")
        if features_missing:
            f.write(f"Missing:     {', '.join(features_missing)}\n")
        f.write(f"Hyperparameters: {model_info.get('hyperparameters', 'N/A')}\n")
        f.write("\n--- External Validation ---\n")
        f.write(f"Samples:     {n_samples}\n")
        if mae is not None:
            f.write(f"MAE:         {mae:.4f} kcal/mol\n")
            f.write(f"R²:          {r2:.4f}\n")
            f.write(f"RMSE:        {rmse:.4f} kcal/mol\n")
        else:
            f.write("Mode:        Prediction-only (no ground truth)\n")
    logger.info("Summary saved to: %s", path)


def _checkpoint_filename_match(filepath: str, model_name: str = None):
    """Match canonical automatic, manual-final, or iteration checkpoints."""
    canonical_name = model_name or os.path.basename(os.path.dirname(filepath))
    pattern = re.compile(
        rf"^{re.escape(canonical_name)}_"
        r"(?:"
        r"(?P<final>final)_"
        r"|manual_final_(?P<manual_n_features>\d+)feat_"
        r"|iteration_(?P<iteration>\d+)_"
        r")"
        r"(?P<timestamp>\d{8}_\d{6})\.joblib$"
    )
    return pattern.fullmatch(os.path.basename(filepath))


def _checkpoint_timestamp_from_path(filepath: str, model_name: str = None) -> str:
    match = _checkpoint_filename_match(filepath, model_name)
    if not match:
        raise ValueError(f"Unrecognised checkpoint filename format: {filepath}")
    return match.group('timestamp')


def _checkpoint_type_from_path(filepath: str, model_name: str = None) -> str:
    match = _checkpoint_filename_match(filepath, model_name)
    if not match:
        raise ValueError(f"Unrecognised checkpoint filename format: {filepath}")
    if match.group('final'):
        return 'final'
    if match.group('manual_n_features'):
        return 'manual_final'
    return 'iteration'


def _loaded_feature_count(model_info: dict, filepath: str = None) -> int:
    """Use len(features) as truth and reject contradictory saved metadata."""
    source = filepath if filepath else '<unknown>'
    features = model_info.get('features')
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError(
            f"Invalid checkpoint features schema in {source}: features must be a non-string sequence."
        )

    invalid_features = [
        feature for feature in features
        if not isinstance(feature, str) or not feature.strip()
    ]
    if invalid_features:
        raise ValueError(
            f"Invalid checkpoint features schema in {source}: every feature must be a non-empty string; "
            f"invalid values={invalid_features!r}."
        )
    if len(set(features)) != len(features):
        raise ValueError(
            f"Invalid checkpoint features schema in {source}: feature names must be unique."
        )

    actual_count = len(features)
    metadata_count = model_info.get('optimal_n_features')
    metadata_is_valid = isinstance(metadata_count, Integral) and not isinstance(metadata_count, bool)
    if metadata_count is not None and (not metadata_is_valid or int(metadata_count) != actual_count):
        raise ValueError(
            f"Invalid or inconsistent checkpoint feature metadata in {source}: "
            f"optimal_n_features={metadata_count!r}, len(features)={actual_count}."
        )

    match = _checkpoint_filename_match(source) if filepath else None
    if match and match.group('manual_n_features'):
        filename_count = int(match.group('manual_n_features'))
        if filename_count != actual_count:
            raise ValueError(
                f"Invalid manual-final checkpoint filename in {source}: "
                f"filename feature count={filename_count}, len(features)={actual_count}."
            )
    return actual_count


def _discover_model_checkpoints(model_dir: str) -> list[dict]:
    """Load all readable canonical checkpoints under one model directory."""
    model_name = os.path.basename(model_dir)
    checkpoints = []
    for filepath in sorted(glob.glob(os.path.join(model_dir, '*.joblib'))):
        if not _checkpoint_filename_match(filepath, model_name):
            continue
        try:
            info = joblib.load(filepath)
        except Exception as exc:
            logger.warning(
                "Skipping unreadable checkpoint %s: %s", filepath, exc.__class__.__name__
            )
            continue

        actual_n_features = _loaded_feature_count(info, filepath=filepath)
        checkpoints.append({
            'path': filepath,
            'model_info': info,
            'checkpoint_type': _checkpoint_type_from_path(filepath, model_name),
            'timestamp': _checkpoint_timestamp_from_path(filepath, model_name),
            'actual_n_features': actual_n_features,
        })
    return checkpoints


def _checkpoint_priority(checkpoint: dict) -> int:
    priorities = {
        'final': 0,
        'manual_final': 1,
        'iteration': 2,
    }
    return priorities.get(checkpoint['checkpoint_type'], 99)


def _best_checkpoint(checkpoints: list[dict]) -> dict:
    if not checkpoints:
        raise FileNotFoundError("No final, manual-final, or iteration checkpoints found")
    top_priority = min(_checkpoint_priority(item) for item in checkpoints)
    priority_candidates = [item for item in checkpoints if _checkpoint_priority(item) == top_priority]
    latest_timestamp = max(item['timestamp'] for item in priority_candidates)
    latest_candidates = [item for item in priority_candidates if item['timestamp'] == latest_timestamp]
    if len(latest_candidates) > 1:
        names = [os.path.basename(item['path']) for item in latest_candidates]
        raise ValueError(f"Ambiguous checkpoint candidates: {names}")
    return latest_candidates[0]


def _select_checkpoint(checkpoints: list[dict], n_features: int = None, allow_closest: bool = False) -> dict:
    if not checkpoints:
        raise FileNotFoundError("No final, manual-final, or iteration checkpoints found")
    if n_features is None:
        return _best_checkpoint(checkpoints)
    exact = [item for item in checkpoints if item['actual_n_features'] == n_features]
    if exact:
        return _best_checkpoint(exact)
    available_counts = sorted({item['actual_n_features'] for item in checkpoints})
    if not allow_closest:
        raise ValueError(
            f"No checkpoint with exactly {n_features} features. Available feature counts: {available_counts}"
        )
    nearest_distance = min(abs(count - n_features) for count in available_counts)
    nearest_counts = [count for count in available_counts if abs(count - n_features) == nearest_distance]
    nearest = [item for item in checkpoints if item['actual_n_features'] in set(nearest_counts)]
    return _best_checkpoint(nearest)


def load_model(model_name: str, n_features: int = None,
               models_dir: str = DEFAULT_MODELS_DIR,
               allow_closest: bool = False) -> dict:
    """Load final, manual-final, or iteration checkpoints by model name and feature count."""
    model_dir = os.path.join(models_dir, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    checkpoints = _discover_model_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No final, manual-final, or iteration checkpoints found for {model_name} in {model_dir}"
        )
    selected = _select_checkpoint(checkpoints, n_features=n_features, allow_closest=allow_closest)
    info = selected['model_info']
    required_keys = ['model', 'scaler_X', 'scaler_y', 'features']
    missing = [key for key in required_keys if key not in info]
    if missing:
        raise ValueError(f"Checkpoint {selected['path']} is missing required key(s): {missing}")
    loaded = dict(info)
    loaded['_loaded_from'] = selected['path']
    loaded['_checkpoint_type'] = selected['checkpoint_type']
    loaded['_requested_n_features'] = n_features
    loaded['_actual_n_features'] = selected['actual_n_features']
    return loaded


def _metric_from_paths(metrics: dict, paths: Sequence[Sequence[str]]):
    for path in paths:
        value = _nested_metric(metrics, path)
        if value is not None:
            return value
    return None


def _ensemble_weight_mae(model_info: dict) -> float:
    metrics = model_info.get('metrics', {})
    mae = _metric_from_paths(
        metrics,
        (
            ('secondary', 'internal_cv', 'rkf_mae_mean'),
            ('internal_cv', 'rkf_mae_mean'),
            ('rkf_mae_mean',),
            ('rkf_mae_opt_mean',),
        ),
    )
    if mae is None:
        raise ValueError("No internal-CV MAE is available for ensemble weighting.")
    mae = float(mae)
    if not np.isfinite(mae) or mae <= 0:
        raise ValueError(f"Internal-CV MAE must be finite and > 0; got {mae!r}")
    return mae


def ensemble_validation(ensemble_csv: str,
                        external_data,
                        target_col: str = DEFAULT_TARGET_COL,
                        output_dir: str = DEFAULT_OUTPUT_DIR,
                        models_dir: str = DEFAULT_MODELS_DIR,
                        allow_closest: bool = False,
                        id_cols=None,
                        preserve_all_metadata: bool = True) -> dict:
    """Run CSV-driven ensemble external validation."""
    spec = pd.read_csv(ensemble_csv)
    required_columns = {'model_name', 'n_features'}
    missing_spec = required_columns - set(spec.columns)
    if missing_spec:
        raise ValueError(
            f"Ensemble CSV must contain columns {sorted(required_columns)}; missing {sorted(missing_spec)}"
        )

    if isinstance(external_data, str):
        df_external = pd.read_csv(external_data)
    elif isinstance(external_data, pd.DataFrame):
        df_external = external_data.copy()
    else:
        raise TypeError(
            f"external_data must be a file path (str) or DataFrame, got {type(external_data).__name__}"
        )

    unnamed_cols = [col for col in df_external.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df_external = df_external.drop(columns=unnamed_cols)
    df_external = df_external.rename(columns=_normalise_column_name)

    original_index = df_external.index.copy()
    original_index_col = 'original_index'
    while original_index_col in df_external.columns:
        original_index_col = f"_{original_index_col}"
    df_external[original_index_col] = list(original_index)
    df_external = df_external.reset_index(drop=True)
    df_external.index.name = '_ensemble_row_id'

    all_predictions = {}
    model_weights = {}
    individual_results = []
    ensemble_members = []
    ensemble_errors = []
    resolved_checkpoints = {}
    all_expected_features = set()

    for spec_position, (_, row) in enumerate(spec.iterrows(), start=1):
        model_name = row['model_name']
        requested_n_features = int(row['n_features'])
        requested_label = f"{model_name} ({requested_n_features} feat)"

        try:
            model_info = load_model(
                model_name,
                n_features=requested_n_features,
                models_dir=models_dir,
                allow_closest=allow_closest,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", requested_label, exc)
            ensemble_errors.append((model_name, requested_n_features, str(exc)))
            continue

        loaded_from = os.path.normcase(os.path.realpath(os.path.abspath(model_info['_loaded_from'])))
        if loaded_from in resolved_checkpoints:
            first_label = resolved_checkpoints[loaded_from]
            error = (
                f"Resolved to the same checkpoint as {first_label}: "
                f"{os.path.basename(model_info['_loaded_from'])}"
            )
            logger.warning("Skipping %s: %s", requested_label, error)
            ensemble_errors.append((model_name, requested_n_features, error))
            continue
        resolved_checkpoints[loaded_from] = requested_label

        model = model_info['model']
        scaler_X = model_info['scaler_X']
        scaler_y = model_info['scaler_y']
        expected_features = list(model_info['features'])
        all_expected_features.update(expected_features)
        actual_n_features = model_info['_actual_n_features']
        label = f"{model_name} ({actual_n_features} feat)"
        member_id = f"member_{spec_position}"

        missing = [feature for feature in expected_features if feature not in df_external.columns]
        if missing:
            logger.warning("Skipping %s - missing features: %s", label, missing)
            ensemble_errors.append((model_name, requested_n_features, f"Missing features: {missing}"))
            continue

        try:
            rkf_mae = _ensemble_weight_mae(model_info)
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                member_weight = float(1.0 / (rkf_mae ** 2))
            if not np.isfinite(member_weight) or member_weight <= 0:
                raise ValueError(
                    f"Internal-CV MAE {rkf_mae!r} produces a non-finite or non-positive ensemble weight."
                )
        except ValueError as exc:
            logger.warning("Skipping %s: %s", label, exc)
            ensemble_errors.append((model_name, requested_n_features, str(exc)))
            continue

        X_ext = df_external[expected_features].copy()
        nan_mask = X_ext.isna().any(axis=1)
        valid_index = X_ext.index[~nan_mask]
        if nan_mask.any():
            logger.warning("%s: dropping %d NaN row(s)", label, nan_mask.sum())
        if len(valid_index) == 0:
            error = f"{label} has no complete rows for its required features."
            logger.warning("Skipping %s: %s", label, error)
            ensemble_errors.append((model_name, requested_n_features, error))
            continue

        X_valid = X_ext.loc[valid_index]
        X_scaled = scaler_X.transform(X_valid)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        prediction_series = pd.Series(y_pred, index=valid_index, name=member_id)
        all_predictions[member_id] = prediction_series
        ensemble_members.append((model_name, actual_n_features))
        model_weights[member_id] = member_weight
        individual_results.append({
            'member_id': member_id,
            'model_name': model_name,
            'n_features': actual_n_features,
            'label': label,
            'features': expected_features,
            'mae': None,
            'r2': None,
            'rmse': None,
            'weight': member_weight,
        })
        logger.info("%s: prediction complete", label)

    if not all_predictions:
        raise EnsembleValidationError(
            f"No valid ensemble members produced predictions. Errors: {ensemble_errors}"
        )

    prediction_frame = pd.concat(all_predictions.values(), axis=1, join='inner').dropna(how='any')
    if prediction_frame.empty:
        raise EnsembleValidationError(
            "No common complete rows remain in the member prediction intersection."
        )

    member_ids = list(prediction_frame.columns)
    result_by_member_id = {result['member_id']: result for result in individual_results}
    labels = [result_by_member_id[member_id]['label'] for member_id in member_ids]
    weight_array = np.array([model_weights[member_id] for member_id in member_ids], dtype=float)
    weight_sum = float(weight_array.sum())
    if not np.isfinite(weight_array).all() or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise EnsembleValidationError("Ensemble member weights must be finite and have a positive sum.")
    weight_array = weight_array / weight_sum

    y_pred_mean = prediction_frame.mean(axis=1)
    y_pred_weighted = prediction_frame.dot(weight_array)
    excluded_rows = len(df_external) - len(prediction_frame)

    metadata_target_col = _metadata_target_for_output(df_external, target_col)
    metadata_cols = _metadata_columns(
        df_external,
        expected_features=all_expected_features,
        target_col=metadata_target_col,
        id_cols=id_cols,
        preserve_all_metadata=preserve_all_metadata,
    )
    results_df = df_external.loc[
        prediction_frame.index,
        _unique_preserving_order([original_index_col] + metadata_cols),
    ].copy()

    used_prediction_columns = set()
    for member_id, label in zip(member_ids, labels):
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
        prediction_column = f'pred_{safe_label}'
        if prediction_column in used_prediction_columns:
            prediction_column = f'{prediction_column}_{member_id}'
        used_prediction_columns.add(prediction_column)
        results_df[prediction_column] = prediction_frame[member_id]
        result_by_member_id[member_id]['prediction_column'] = prediction_column

    results_df['predicted_activation_energy'] = y_pred_weighted
    results_df['predicted_mean'] = y_pred_mean
    results_df['predicted_weighted'] = y_pred_weighted

    mae = r2 = rmse = None
    has_gt = target_col is not None and target_col in df_external.columns
    if has_gt:
        y_true = df_external.loc[prediction_frame.index, target_col]
        valid = ~y_true.isna()
        if valid.any():
            y_true_valid = y_true.loc[valid]
            y_pred_weighted_valid = y_pred_weighted.loc[valid]
            mae = float(mean_absolute_error(y_true_valid, y_pred_weighted_valid))
            r2 = float(r2_score(y_true_valid, y_pred_weighted_valid))
            rmse = float(np.sqrt(mean_squared_error(y_true_valid, y_pred_weighted_valid)))
            results_df[target_col] = y_true
            results_df['absolute_error_weighted'] = np.abs(y_pred_weighted - y_true)
            for result in individual_results:
                if result['member_id'] not in prediction_frame.columns:
                    continue
                preds = prediction_frame.loc[valid, result['member_id']]
                result['mae'] = float(mean_absolute_error(y_true_valid, preds))
                result['r2'] = float(r2_score(y_true_valid, preds))
                result['rmse'] = float(np.sqrt(mean_squared_error(y_true_valid, preds)))

    original_values = results_df[original_index_col].copy()
    results_df.index = pd.Index(original_values, name=original_index.name)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_files = []

    csv_path = os.path.join(output_dir, f'ensemble_validation_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    output_files.append(csv_path)

    if has_gt and mae is not None:
        plot_path = os.path.join(output_dir, f'ensemble_scatter_{timestamp}.png')
        _plot_external_scatter(
            y_true_valid, y_pred_weighted_valid, 'Ensemble', len(member_ids), mae, r2, rmse, plot_path
        )
        output_files.append(plot_path)

    summary_path = os.path.join(output_dir, f'ensemble_summary_{timestamp}.txt')
    _write_ensemble_summary(
        summary_path, ensemble_members, individual_results, ensemble_errors,
        len(prediction_frame), excluded_rows, mae, r2, rmse, labels, weight_array,
    )
    output_files.append(summary_path)

    return {
        'predictions': results_df,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'n_samples': len(prediction_frame),
        'excluded_rows': excluded_rows,
        'ensemble_members': ensemble_members,
        'individual_results': individual_results,
        'ensemble_errors': ensemble_errors,
        'metadata_columns': metadata_cols,
        'output_files': output_files,
    }


def _write_ensemble_summary(path, ensemble_members, individual_results, ensemble_errors,
                            n_samples, excluded_rows, mae, r2, rmse, labels,
                            normalised_weights):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Ensemble External Validation Summary\n")
        f.write("====================================\n\n")
        f.write(f"Members:       {len(labels)}\n")
        f.write(f"Samples:       {n_samples}\n")
        f.write(f"Excluded rows: {excluded_rows}\n")
        f.write("\n--- Members ---\n")
        for label, weight in zip(labels, normalised_weights):
            f.write(f"{label}: weight={weight:.6f}\n")
        if mae is not None:
            f.write("\n--- Weighted Ensemble Metrics ---\n")
            f.write(f"MAE:  {mae:.4f} kcal/mol\n")
            f.write(f"R²:   {r2:.4f}\n")
            f.write(f"RMSE: {rmse:.4f} kcal/mol\n")
        else:
            f.write("\nMode: Prediction-only (no ground truth)\n")
        if individual_results:
            f.write("\n--- Individual Results ---\n")
            for result in individual_results:
                f.write(
                    f"{result['label']}: weight={result['weight']:.6f}, MAE={result['mae']}\n"
                )
        if ensemble_errors:
            f.write("\n--- Skipped Members ---\n")
            for model_name, n_features, error in ensemble_errors:
                f.write(f"{model_name} ({n_features} feat): {error}\n")


def main():
    """Command-line interface for external validation."""
    parser = argparse.ArgumentParser(
        description='External Validation for Organoboronate ML Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_features', type=int, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--target-col', type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--ensemble', type=str, default=None)
    parser.add_argument('--allow-closest', action='store_true')
    parser.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        '--id-cols',
        type=str,
        default=None,
        help='Comma-separated metadata columns to prioritize in output CSVs. Defaults to ID,SMILES,filename.',
    )
    parser.add_argument(
        '--only-id-cols',
        action='store_true',
        help='Only preserve prioritized --id-cols metadata instead of all non-feature metadata.',
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if args.list_models:
        models_df = list_available_models(args.models_dir)
        print(models_df.to_string(index=False))
        return models_df

    if args.data is None:
        parser.error('--data is required unless --list-models is used')

    target_col = None if args.predict_only else args.target_col
    preserve_all_metadata = not args.only_id_cols

    if args.ensemble:
        results = ensemble_validation(
            ensemble_csv=args.ensemble,
            external_data=args.data,
            target_col=target_col,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            allow_closest=args.allow_closest,
            id_cols=args.id_cols,
            preserve_all_metadata=preserve_all_metadata,
        )
        print(f"\n{'=' * 60}")
        print("  Ensemble External Validation Complete")
        print(f"{'=' * 60}")
        print(f"  Members:     {len(results['ensemble_members'])}")
        print(f"  Samples:     {results['n_samples']}")
        print(f"  Excluded:    {results['excluded_rows']}")
        if results['mae'] is not None:
            print(f"  MAE:         {results['mae']:.4f} kcal/mol")
            print(f"  R2:          {results['r2']:.4f}")
            print(f"  RMSE:        {results['rmse']:.4f} kcal/mol")
        for path in results['output_files']:
            print(f"  - {path}")
        return results

    if args.model is None:
        parser.error('--model is required unless --list-models or --ensemble is used')

    model_info = load_model(
        args.model,
        n_features=args.n_features,
        models_dir=args.models_dir,
        allow_closest=args.allow_closest,
    )
    results = external_validation(
        model_info,
        args.data,
        target_col=target_col,
        output_dir=args.output_dir,
        output_prefix=args.model,
        id_cols=args.id_cols,
        preserve_all_metadata=preserve_all_metadata,
    )

    print(f"\n{'=' * 60}")
    print("  External Validation Complete")
    print(f"{'=' * 60}")
    print(f"  Model:       {args.model}")
    print(f"  Features:    {results['n_features_used']}")
    print(f"  Samples:     {results['n_samples']}")
    if results['mae'] is not None:
        print(f"  MAE:         {results['mae']:.4f} kcal/mol")
        print(f"  R2:          {results['r2']:.4f}")
        print(f"  RMSE:        {results['rmse']:.4f} kcal/mol")
    else:
        print("  Mode:        Prediction-only")
    print("\n  Output files:")
    for path in results['output_files']:
        print(f"  - {path}")
    print(f"{'=' * 60}\n")
    return results


if __name__ == '__main__':
    main()
