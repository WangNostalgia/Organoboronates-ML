"""
External Validation Module for Organoboronate ML Models
========================================================

Provides a reusable CLI and Python API for evaluating trained `.joblib` models
on completely independent, unseen data (external validation).

Key capabilities:
  - List all trained models with their metrics and feature counts
  - Load any final model by name and (optionally) desired feature count
  - Align external CSV columns to the model's expected feature set
  - Predict activation energies on new data
  - Calculate MAE, R², RMSE when ground-truth `activation_energy` is present
  - Generate publication-quality prediction-vs-experiment scatter plots
  - Export predictions to CSV

Usage (CLI):
  # List all trained models and their feature counts
  python src/external_validation.py --list-models

  # Run external validation with a specific model
  python src/external_validation.py --model SVR --data external_data.csv

  # Run with a specific feature count (optional)
  python src/external_validation.py --model SVR --n_features 5 --data external_data.csv

  # Prediction-only mode (no ground truth column in the CSV)
  python src/external_validation.py --model SVR --data new_compounds.csv --predict-only

Usage (Python API):
  >>> from src.external_validation import list_available_models, load_model, external_validation
  >>> models_df = list_available_models()
  >>> model_info = load_model('SVR', n_features=4)
  >>> results = external_validation(model_info, 'external_data.csv')
  >>> print(f"MAE={results['mae']:.2f}, R²={results['r2']:.3f}")
"""

import os
import glob
import argparse
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Suppress noisy third-party warnings (same filters as main.py).
# LightGBM internally converts DataFrames → numpy, triggering a sklearn
# feature-names validation mismatch on every predict() call.  Harmless.
warnings.filterwarnings('ignore', message='X does not have valid feature names',
                        category=UserWarning)
# sklearn 1.6 deprecation of BaseEstimator._validate_data — CatBoost triggers it.
warnings.filterwarnings('ignore', message='BaseEstimator._validate_data',
                        category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODELS_DIR = 'models'
DEFAULT_TARGET_COL = 'activation_energy'
DEFAULT_OUTPUT_DIR = 'external_validation_results'

# Feature aliases — some external CSVs may use alternative column names for the
# same physicochemical property.  This mapping normalises them to the canonical
# names used during model training (derived from B_dataset.csv).
FEATURE_ALIASES = {
    # Add aliases here as needed.  Example:
    # 'pKa': 'pka',
    # 'HOMO': 'homo_energy',
}


def _normalise_column_name(col: str) -> str:
    """Map a column name to its canonical form using FEATURE_ALIASES."""
    return FEATURE_ALIASES.get(col, col)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def list_available_models(models_dir: str = DEFAULT_MODELS_DIR) -> pd.DataFrame:
    """
    Scan the models directory and return a DataFrame of all trained final models.

    Parameters
    ----------
    models_dir : str
        Path to the directory containing per-model subdirectories.

    Returns
    -------
    pd.DataFrame
        Columns: model_name, n_features, features, rkf_mae, rkf_r2, filepath
    """
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    records = []
    pattern = os.path.join(models_dir, '*', '*_final_*.joblib')
    for filepath in sorted(glob.glob(pattern)):
        try:
            info = joblib.load(filepath)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", filepath, exc)
            continue

        model_name = os.path.basename(os.path.dirname(filepath))
        n_features = info.get('optimal_n_features', len(info.get('features', [])))
        features = info.get('features', [])
        metrics = info.get('metrics', {})

        records.append({
            'model_name': model_name,
            'n_features': n_features,
            'features': ', '.join(features) if features else 'N/A',
            'rkf_mae': metrics.get('rkf_mae_opt_mean'),
            'rkf_r2': metrics.get('rkf_r2_opt_mean'),
            'filepath': filepath,
        })

    if not records:
        raise FileNotFoundError(
            f"No *_final_*.joblib files found under {models_dir}. "
            f"Run main.py first to train models."
        )

    return pd.DataFrame(records).sort_values('model_name').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str,
               n_features: int = None,
               models_dir: str = DEFAULT_MODELS_DIR) -> dict:
    """
    Load a trained final model by name and optional feature count.

    If multiple final versions exist for the same model, the most recent
    (by file modification time) is returned.  When `n_features` is specified,
    the closest-matching feature count among available versions is selected.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'SVR', 'RandomForest').  Case-sensitive —
        must match the subdirectory name under models/.
    n_features : int, optional
        Desired feature count.  When None, the most recent version is used.
        When specified, the version whose feature count is closest to the
        requested value is selected (ties go to the most recent).

    Returns
    -------
    dict
        The full model_info dictionary from the .joblib file, with an extra
        key `_loaded_from` recording the source filepath.
    """
    model_dir = os.path.join(models_dir, model_name)
    if not os.path.isdir(model_dir):
        # Try case-insensitive match (user-friendly)
        candidates = [
            d for d in os.listdir(models_dir)
            if d.lower() == model_name.lower()
            and os.path.isdir(os.path.join(models_dir, d))
        ]
        if not candidates:
            available = ', '.join(
                d for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))
            )
            raise FileNotFoundError(
                f"Model '{model_name}' not found. Available models: {available}"
            )
        model_dir = os.path.join(models_dir, candidates[0])
        model_name = candidates[0]  # use canonical casing

    pattern = os.path.join(model_dir, '*_final_*.joblib')
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(
            f"No *_final_*.joblib found in {model_dir}. "
            f"Run main.py to train this model first."
        )

    if n_features is not None:
        # Select the version closest to the requested feature count;
        # ties broken by recency (already sorted by mtime desc).
        def _dist(filepath):
            try:
                info = joblib.load(filepath)
                return abs(info.get('optimal_n_features',
                                    len(info.get('features', []))) - n_features)
            except Exception:
                return float('inf')

        candidates.sort(key=_dist)  # stable sort preserves mtime ordering

    best_path = candidates[0]
    model_info = joblib.load(best_path)
    model_info['_loaded_from'] = best_path

    # Validate required keys
    required = ['model', 'scaler_X', 'scaler_y', 'features']
    missing = [k for k in required if k not in model_info]
    if missing:
        raise KeyError(
            f"Model file {best_path} is missing required keys: {missing}. "
            f"Available keys: {list(model_info.keys())}"
        )

    logger.info("Loaded %s from %s (%d features: %s)",
                model_name, os.path.basename(best_path),
                len(model_info['features']),
                ', '.join(model_info['features']))

    return model_info


# ---------------------------------------------------------------------------
# Core external validation
# ---------------------------------------------------------------------------

def external_validation(model_info: dict,
                        external_data,
                        target_col: str = DEFAULT_TARGET_COL,
                        output_dir: str = DEFAULT_OUTPUT_DIR,
                        output_prefix: str = None) -> dict:
    """
    Evaluate a trained model on an external dataset.

    Parameters
    ----------
    model_info : dict
        The dictionary loaded from a .joblib file.  Must contain keys:
        'model', 'scaler_X', 'scaler_y', 'features'.
    external_data : str or pd.DataFrame
        Path to a CSV file, or a DataFrame, containing feature columns and
        (optionally) a target column for evaluation.
    target_col : str
        Name of the column containing ground-truth activation energies.
        When this column is absent, the function runs in prediction-only
        mode (no evaluation metrics are computed).
    output_dir : str
        Directory for saving output files (predictions CSV, scatter plot).
    output_prefix : str, optional
        Prefix for output filenames.  Defaults to the model name extracted
        from `_loaded_from`.

    Returns
    -------
    dict
        Keys: 'predictions' (DataFrame), 'mae', 'r2', 'rmse' (all None in
        prediction-only mode), 'n_samples', 'n_features_used', 'features_used',
        'features_missing', 'output_files' (list of saved file paths).
    """
    # ── Load data ──────────────────────────────────────────────────────
    if isinstance(external_data, str):
        df = pd.read_csv(external_data)
        logger.info("Loaded external data from %s (%d rows, %d columns)",
                    external_data, len(df), len(df.columns))
    elif isinstance(external_data, pd.DataFrame):
        df = external_data.copy()
    else:
        raise TypeError(
            f"external_data must be a file path (str) or DataFrame, "
            f"got {type(external_data).__name__}"
        )

    # ── Normalise column names ─────────────────────────────────────────
    # Some CSVs may contain "Unnamed: N" columns from Excel exports;
    # these are never real features and are dropped silently.
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info("Dropped %d unnamed column(s): %s",
                    len(unnamed_cols), unnamed_cols)

    # Apply alias mapping for alternative feature names
    df = df.rename(columns=_normalise_column_name)

    # ── Extract model components ───────────────────────────────────────
    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']
    expected_features = model_info['features']

    # Determine model name for labelling
    if output_prefix is None:
        if '_loaded_from' in model_info:
            model_name = os.path.basename(
                os.path.dirname(model_info['_loaded_from'])
            )
        else:
            model_name = 'Model'
    else:
        model_name = output_prefix

    # ── Feature alignment ──────────────────────────────────────────────
    # Check which expected features are present in the external data
    features_present = [f for f in expected_features if f in df.columns]
    features_missing = [f for f in expected_features if f not in df.columns]

    if features_missing:
        logger.error(
            "Missing features in external data: %s\n"
            "  Expected: %s\n"
            "  Available: %s",
            features_missing, expected_features,
            [c for c in df.columns if c not in ('sub_H', 'sub_B', target_col)]
        )
        raise ValueError(
            f"External data is missing {len(features_missing)} required "
            f"feature(s): {features_missing}. "
            f"Ensure the input CSV contains all features the model was "
            f"trained on."
        )

    # Subset to the exact feature set (in the order the scaler expects)
    X_external = df[expected_features].copy()

    # Handle NaN in feature columns — drop affected rows with a warning
    nan_mask = X_external.isna().any(axis=1)
    if nan_mask.any():
        logger.warning(
            "Dropping %d row(s) with NaN feature values (out of %d total).",
            nan_mask.sum(), len(X_external)
        )
        X_external = X_external[~nan_mask]
        df = df[~nan_mask].reset_index(drop=True)
        X_external = X_external.reset_index(drop=True)

    # ── Predict ────────────────────────────────────────────────────────
    X_scaled = scaler_X.transform(X_external)
    y_pred_scaled = model.predict(X_scaled)
    # Inverse-transform predictions back to real kcal/mol
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    # IMPORTANT: activation_energy has no physical upper bound, so we do NOT
    # clip predictions.  Clipping would artificially truncate high-energy
    # predictions and hide the model's true behaviour on extreme cases.

    # ── Build output DataFrame ─────────────────────────────────────────
    # Preserve identifier columns (sub_H, sub_B) if they exist
    id_cols = [c for c in ['sub_H', 'sub_B'] if c in df.columns]
    results_df = df[id_cols + expected_features].copy()
    results_df['predicted_activation_energy'] = y_pred

    # ── Evaluation (only when ground truth exists) ─────────────────────
    mae = r2 = rmse = None
    has_ground_truth = target_col in df.columns

    if has_ground_truth:
        y_true = df[target_col].values

        # Drop rows where ground truth is NaN (prediction-only rows in a
        # mixed validation set)
        valid_mask = ~np.isnan(y_true)
        if not valid_mask.all():
            logger.info(
                "Dropping %d row(s) with NaN %s (no ground truth).",
                (~valid_mask).sum(), target_col
            )
            y_true = y_true[valid_mask]
            y_pred_eval = y_pred[valid_mask]
        else:
            y_pred_eval = y_pred

        if len(y_true) > 0:
            mae = float(mean_absolute_error(y_true, y_pred_eval))
            r2 = float(r2_score(y_true, y_pred_eval))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_eval)))

            results_df[target_col] = df[target_col].values
            results_df['absolute_error'] = np.abs(y_pred - df[target_col].values)

            logger.info(
                "External validation metrics for %s (%d features):\n"
                "  MAE  = %.4f kcal/mol\n"
                "  R²   = %.4f\n"
                "  RMSE = %.4f kcal/mol\n"
                "  N    = %d",
                model_name, len(expected_features), mae, r2, rmse, len(y_true)
            )
        else:
            logger.warning("No valid ground truth values found — prediction-only mode.")
            has_ground_truth = False
    else:
        logger.info("Column '%s' not found in external data — running in "
                    "prediction-only mode.", target_col)

    # ── Save outputs ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(' ', '_')

    # Predictions CSV
    preds_path = os.path.join(
        output_dir,
        f"{safe_model_name}_external_validation_{timestamp}.csv"
    )
    results_df.to_csv(preds_path, index=False, encoding='utf-8-sig')
    output_files.append(preds_path)
    logger.info("Predictions saved to: %s", preds_path)

    # Scatter plot (only when ground truth is available)
    if has_ground_truth:
        plot_path = os.path.join(
            output_dir,
            f"{safe_model_name}_external_scatter_{timestamp}.png"
        )
        _plot_external_scatter(
            y_true=y_true,
            y_pred=y_pred_eval,
            model_name=model_name,
            n_features=len(expected_features),
            mae=mae,
            r2=r2,
            rmse=rmse,
            output_path=plot_path,
        )
        output_files.append(plot_path)
        logger.info("Scatter plot saved to: %s", plot_path)

    # Summary text file
    summary_path = os.path.join(
        output_dir,
        f"{safe_model_name}_external_summary_{timestamp}.txt"
    )
    _write_summary(
        summary_path, model_name, model_info, expected_features,
        features_missing, mae, r2, rmse, len(X_external),
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
        'output_files': output_files,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_external_scatter(y_true, y_pred, model_name, n_features,
                           mae, r2, rmse, output_path):
    """
    Generate a publication-quality scatter plot of predicted vs experimental
    activation energies for external validation.
    """
    import matplotlib as mpl
    plt.rcParams['font.family'] = 'DejaVu Sans'

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Scatter points
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='#2c3e50',
               facecolors='#3498db', s=60, linewidth=0.5, zorder=5)

    # Perfect-prediction diagonal
    all_vals = np.concatenate([y_true, y_pred])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = 0.05 * (val_max - val_min) if val_max > val_min else 1.0
    ax.plot([val_min - margin, val_max + margin],
            [val_min - margin, val_max + margin],
            '--', color='#e74c3c', linewidth=1.5, label='Perfect prediction',
            zorder=4)

    # ±2 kcal/mol band (common acceptance window in computational chemistry)
    band_x = np.linspace(val_min - margin, val_max + margin, 100)
    ax.fill_between(band_x, band_x - 2.0, band_x + 2.0,
                    alpha=0.08, color='#2ecc71', label=r'$\pm$2 kcal/mol band')

    ax.set_xlabel('Experimental Activation Energy (kcal/mol)', fontsize=13)
    ax.set_ylabel('Predicted Activation Energy (kcal/mol)', fontsize=13)
    ax.set_title(
        f'{model_name} — External Validation\n'
        f'({n_features} features, {len(y_true)} samples)',
        fontsize=14, fontweight='bold'
    )

    # Metrics text box
    metrics_text = (
        f"MAE  = {mae:.2f} kcal/mol\n"
        f"RMSE = {rmse:.2f} kcal/mol\n"
        f"R²   = {r2:.4f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlim(val_min - margin, val_max + margin)
    ax.set_ylim(val_min - margin, val_max + margin)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _write_summary(summary_path, model_name, model_info, features_used,
                   features_missing, mae, r2, rmse, n_samples):
    """Write a human-readable summary of the external validation run."""
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"External Validation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model:        {model_name}\n")
        f.write(f"Source:       {model_info.get('_loaded_from', 'N/A')}\n")
        f.write(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- Model Info ---\n")
        f.write(f"Features ({len(features_used)}): {', '.join(features_used)}\n")
        if features_missing:
            f.write(f"Missing:      {', '.join(features_missing)}\n")
        f.write(f"Hyperparameters: {model_info.get('hyperparameters', 'N/A')}\n")
        f.write(f"\n--- External Validation ---\n")
        f.write(f"Samples:      {n_samples}\n")
        if mae is not None:
            f.write(f"MAE:          {mae:.4f} kcal/mol\n")
            f.write(f"R²:           {r2:.4f}\n")
            f.write(f"RMSE:         {rmse:.4f} kcal/mol\n")
        else:
            f.write(f"Mode:         Prediction-only (no ground truth)\n")


# ---------------------------------------------------------------------------
# Ensemble external validation
# ---------------------------------------------------------------------------

def ensemble_validation(ensemble_csv,
                        external_data,
                        target_col=DEFAULT_TARGET_COL,
                        output_dir=DEFAULT_OUTPUT_DIR,
                        models_dir=DEFAULT_MODELS_DIR):
    """
    Load multiple models specified in a CSV file, run predictions on external
    data, and aggregate results using simple-mean and weighted-mean strategies.

    The CSV format follows the same convention as manual_feature_selection.py:

        model_name,n_features
        SVR,4
        RandomForest,4
        KRR,4

    Each row specifies a model and the desired feature count.  The function
    loads the closest-matching checkpoint for each row, predicts independently,
    then aggregates:
      - **simple mean** — arithmetic average of all model predictions
      - **weighted mean** — weight ∝ 1 / RKfold_MAE² (more accurate models
        contribute more to the final prediction)

    Parameters
    ----------
    ensemble_csv : str
        Path to a CSV file with columns ``model_name`` and ``n_features``.
    external_data : str or pd.DataFrame
        Path to the external CSV, or a DataFrame.
    target_col : str or None
        Name of the ground-truth column.  Pass ``None`` for prediction-only.
    output_dir : str
        Directory for output files.
    models_dir : str
        Root directory containing per-model subdirectories.

    Returns
    -------
    dict
        Keys: 'predictions' (DataFrame with all individual + aggregated columns),
        'individual_results' (list of per-model result dicts),
        'mae', 'r2', 'rmse' (aggregated weighted-mean metrics, or None),
        'ensemble_members' (list of (model_name, n_features) tuples loaded),
        'ensemble_errors' (list of (model_name, n_features, error_msg) for
        any models that failed to load), 'output_files'.
    """
    # ── Read ensemble specification ─────────────────────────────────────
    spec = pd.read_csv(ensemble_csv)
    required_cols = {'model_name', 'n_features'}
    missing_cols = required_cols - set(spec.columns)
    if missing_cols:
        raise ValueError(
            f"Ensemble CSV must contain columns: {required_cols}. "
            f"Missing: {missing_cols}. Found: {list(spec.columns)}"
        )
    # Drop rows with NaN in required columns
    spec = spec.dropna(subset=['model_name', 'n_features'])
    spec['n_features'] = spec['n_features'].astype(int)

    logger.info("Ensemble specification: %d models from %s",
                len(spec), ensemble_csv)
    for _, row in spec.iterrows():
        logger.info("  - %s (%d features)", row['model_name'], row['n_features'])

    # ── Load external data once (shared across all ensemble members) ────
    if isinstance(external_data, str):
        df_external = pd.read_csv(external_data)
        logger.info("Loaded external data from %s (%d rows)", external_data, len(df_external))
    elif isinstance(external_data, pd.DataFrame):
        df_external = external_data.copy()
    else:
        raise TypeError(f"external_data must be str or DataFrame, got {type(external_data).__name__}")

    # Drop "Unnamed" artefact columns
    unnamed_cols = [c for c in df_external.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df_external = df_external.drop(columns=unnamed_cols)
        logger.info("Dropped %d unnamed column(s)", len(unnamed_cols))

    # Normalise column names
    df_external = df_external.rename(columns=_normalise_column_name)

    # ── Predict with each ensemble member ────────────────────────────────
    all_predictions = {}      # label → np.array of predictions
    model_weights = {}        # label → weight for weighted-mean aggregation
    individual_results = []   # per-model result dicts
    ensemble_members = []     # successfully loaded (model_name, n_features)
    ensemble_errors = []      # failed loads

    for _, row in spec.iterrows():
        model_name = row['model_name']
        n_features = int(row['n_features'])
        label = f"{model_name} ({n_features} feat)"

        try:
            model_info = load_model(model_name, n_features=n_features,
                                    models_dir=models_dir)
        except Exception as exc:
            logger.warning("Skipping %s: %s", label, exc)
            ensemble_errors.append((model_name, n_features, str(exc)))
            continue

        model = model_info['model']
        scaler_X = model_info['scaler_X']
        scaler_y = model_info['scaler_y']
        expected_features = model_info['features']

        # Check feature availability
        missing = [f for f in expected_features if f not in df_external.columns]
        if missing:
            logger.warning("Skipping %s — missing features: %s", label, missing)
            ensemble_errors.append(
                (model_name, n_features, f"Missing features: {missing}")
            )
            continue

        X_ext = df_external[expected_features].copy()
        # Handle NaN rows
        nan_mask = X_ext.isna().any(axis=1)
        if nan_mask.any():
            logger.warning("%s: dropping %d NaN row(s)", label, nan_mask.sum())
            X_ext = X_ext[~nan_mask].reset_index(drop=True)

        X_scaled = scaler_X.transform(X_ext)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

        all_predictions[label] = y_pred
        ensemble_members.append((model_name, n_features))

        # Weight for weighted-mean: 1 / RKfold_MAE²
        # (higher weight = lower error).  Fall back to 1 / MAE² from saved metrics.
        metrics = model_info.get('metrics', {})
        rkf_mae = metrics.get('rkf_mae_opt_mean')
        if rkf_mae is None or rkf_mae <= 0:
            rkf_mae = metrics.get('mae_mean', 2.0)
        model_weights[label] = 1.0 / (rkf_mae ** 2)

        # Evaluate individually if ground truth present
        ind_mae = ind_r2 = ind_rmse = None
        has_gt = (target_col is not None and target_col in df_external.columns)
        if has_gt:
            y_true = df_external[target_col].values[:len(y_pred)]
            valid = ~np.isnan(y_true)
            if valid.any():
                ind_mae = float(mean_absolute_error(y_true[valid], y_pred[valid]))
                ind_r2 = float(r2_score(y_true[valid], y_pred[valid]))
                ind_rmse = float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))

        individual_results.append({
            'model_name': model_name,
            'n_features': n_features,
            'label': label,
            'features': expected_features,
            'mae': ind_mae,
            'r2': ind_r2,
            'rmse': ind_rmse,
            'weight': model_weights[label],
        })
        logger.info("%s: prediction complete (MAE=%s)",
                    label, f"{ind_mae:.2f}" if ind_mae is not None else "N/A")

    if not all_predictions:
        raise RuntimeError(
            f"No ensemble members loaded successfully. "
            f"Errors: {ensemble_errors}"
        )

    # ── Aggregate predictions ───────────────────────────────────────────
    # Align all prediction arrays to the same length (use the minimum
    # length in case some models dropped different numbers of NaN rows)
    min_len = min(len(p) for p in all_predictions.values())
    aligned_preds = {}
    for label, preds in all_predictions.items():
        aligned_preds[label] = preds[:min_len]

    pred_matrix = np.column_stack(list(aligned_preds.values()))
    labels = list(aligned_preds.keys())

    # Simple mean
    y_pred_mean = pred_matrix.mean(axis=1)

    # Weighted mean
    weight_array = np.array([model_weights.get(l, 1.0) for l in labels])
    weight_array = weight_array / weight_array.sum()
    y_pred_weighted = np.dot(pred_matrix, weight_array)

    logger.info("Ensemble aggregation: %d members, %d predictions each",
                len(labels), min_len)
    logger.info("Weights (normalised): %s",
                {l: f"{w:.3f}" for l, w in zip(labels, weight_array)})

    # ── Build output DataFrame ──────────────────────────────────────────
    id_cols = [c for c in ['sub_H', 'sub_B'] if c in df_external.columns]
    results_df = df_external[id_cols].iloc[:min_len].copy().reset_index(drop=True)

    # Add individual model predictions
    for label, preds in aligned_preds.items():
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
        results_df[f'pred_{safe_label}'] = preds

    results_df['predicted_activation_energy'] = y_pred_weighted
    results_df['predicted_mean'] = y_pred_mean
    results_df['predicted_weighted'] = y_pred_weighted

    # ── Evaluation ──────────────────────────────────────────────────────
    mae = r2 = rmse = None
    has_gt = (target_col is not None and target_col in df_external.columns)
    if has_gt:
        y_true = df_external[target_col].values[:min_len]
        valid = ~np.isnan(y_true)
        if valid.any():
            mae = float(mean_absolute_error(y_true[valid], y_pred_weighted[valid]))
            r2 = float(r2_score(y_true[valid], y_pred_weighted[valid]))
            rmse = float(np.sqrt(mean_squared_error(y_true[valid], y_pred_weighted[valid])))
            results_df[target_col] = y_true
            results_df['absolute_error_weighted'] = np.abs(y_pred_weighted - y_true)

            logger.info(
                "Ensemble weighted-mean metrics:\n"
                "  MAE  = %.4f kcal/mol\n"
                "  R²   = %.4f\n"
                "  RMSE = %.4f kcal/mol",
                mae, r2, rmse
            )

    # ── Save outputs ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Predictions CSV
    preds_path = os.path.join(output_dir, f"ensemble_validation_{timestamp}.csv")
    results_df.to_csv(preds_path, index=False, encoding='utf-8-sig')
    output_files.append(preds_path)
    logger.info("Ensemble predictions saved to: %s", preds_path)

    # Scatter plot
    if has_gt and mae is not None:
        plot_path = os.path.join(output_dir, f"ensemble_scatter_{timestamp}.png")
        _plot_external_scatter(
            y_true=y_true[valid],
            y_pred=y_pred_weighted[valid],
            model_name=f"Ensemble ({len(labels)} models)",
            n_features=sum(len(r['features']) for r in individual_results),
            mae=mae, r2=r2, rmse=rmse,
            output_path=plot_path,
        )
        output_files.append(plot_path)

    # Summary file
    summary_path = os.path.join(output_dir, f"ensemble_summary_{timestamp}.txt")
    _write_ensemble_summary(summary_path, individual_results, ensemble_errors,
                            labels, weight_array, mae, r2, rmse, min_len)
    output_files.append(summary_path)

    return {
        'predictions': results_df,
        'individual_results': individual_results,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'n_samples': min_len,
        'ensemble_members': ensemble_members,
        'ensemble_errors': ensemble_errors,
        'output_files': output_files,
    }


def _write_ensemble_summary(summary_path, individual_results, ensemble_errors,
                            labels, weight_array, mae, r2, rmse, n_samples):
    """Write a human-readable summary of the ensemble validation run."""
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Ensemble External Validation Summary\n")
        f.write(f"{'='*55}\n")
        f.write(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Members:    {len(labels)}\n")
        f.write(f"Samples:    {n_samples}\n")
        f.write(f"\n--- Ensemble Members ---\n")
        f.write(f"{'Model':<22} {'Feat':<6} {'Weight':<10} {'MAE':<10} {'R2':<10}\n")
        f.write(f"{'-'*58}\n")
        for r in individual_results:
            w = r['weight']
            w_norm = w / sum(r2['weight'] for r2 in individual_results)
            mae_s = f"{r['mae']:.2f}" if r['mae'] is not None else 'N/A'
            r2_s = f"{r['r2']:.4f}" if r['r2'] is not None else 'N/A'
            f.write(f"{r['model_name']:<22} {r['n_features']:<6} "
                    f"{w_norm:<10.4f} {mae_s:<10} {r2_s:<10}\n")
        f.write(f"\n--- Aggregated Metrics (weighted mean) ---\n")
        if mae is not None:
            f.write(f"MAE:          {mae:.4f} kcal/mol\n")
            f.write(f"R2:           {r2:.4f}\n")
            f.write(f"RMSE:         {rmse:.4f} kcal/mol\n")
        else:
            f.write("Mode:         Prediction-only\n")
        if ensemble_errors:
            f.write(f"\n--- Errors / Skipped Models ---\n")
            for name, nf, err in ensemble_errors:
                f.write(f"  {name} ({nf} feat): {err}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for external validation."""
    parser = argparse.ArgumentParser(
        description='External Validation for Organoboronate ML Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available trained models
  python src/external_validation.py --list-models

  # Validate SVR model on external data
  python src/external_validation.py --model SVR --data my_external_data.csv

  # Validate RandomForest with 5 features
  python src/external_validation.py --model RandomForest --n_features 5 --data external.csv

  # Prediction-only (no ground truth column)
  python src/external_validation.py --model SVR --data new_compounds.csv --predict-only

  # Ensemble external validation (CSV-driven, multiple models)
  python src/external_validation.py --ensemble ensemble_spec.csv --data external.csv
        """
    )
    parser.add_argument('--list-models', action='store_true',
                        help='List all available trained models and exit.')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name to use for external validation (e.g., SVR).')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Desired feature count for the model. '
                             'When omitted, the most recent version is used.')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the external CSV file with features and '
                             '(optionally) activation_energy column.')
    parser.add_argument('--target-col', type=str, default=DEFAULT_TARGET_COL,
                        help=f'Name of the target column. '
                             f'Default: {DEFAULT_TARGET_COL}')
    parser.add_argument('--predict-only', action='store_true',
                        help='Force prediction-only mode (skip evaluation even '
                             'if target column exists).')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--ensemble', type=str, default=None,
                        help='Path to a CSV file specifying ensemble members '
                             '(columns: model_name, n_features). '
                             'When provided, runs ensemble validation instead '
                             'of single-model validation.')
    parser.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR,
                        help=f'Directory containing trained models. '
                             f'Default: {DEFAULT_MODELS_DIR}')

    args = parser.parse_args()

    # ── Setup logging ──────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # ── --list-models ──────────────────────────────────────────────────
    if args.list_models:
        models_df = list_available_models(args.models_dir)
        print(f"\n{'='*90}")
        print(f"  Available Trained Models")
        print(f"{'='*90}")
        print(f"  {'Model':<20} {'Feat':<5} {'RKfold MAE':<14} {'RKfold R2':<12} Features")
        print(f"  {'-'*86}")
        for _, row in models_df.iterrows():
            mae_str = f"{row['rkf_mae']:.4f}" if pd.notna(row['rkf_mae']) else 'N/A'
            r2_str = f"{row['rkf_r2']:.4f}" if pd.notna(row['rkf_r2']) else 'N/A'
            print(f"  {row['model_name']:<20} {row['n_features']:<5} "
                  f"{mae_str:<14} {r2_str:<12} {row['features']}")
        print(f"{'='*90}\n")
        return

    # ── Ensemble validation mode ────────────────────────────────────────
    if args.ensemble:
        if not args.data:
            parser.error("--data is required for ensemble validation.")
        target_col = None if args.predict_only else args.target_col
        results = ensemble_validation(
            ensemble_csv=args.ensemble,
            external_data=args.data,
            target_col=target_col,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
        )
        # Print summary
        print(f"\n{'='*60}")
        print(f"  Ensemble External Validation Complete")
        print(f"{'='*60}")
        print(f"  Members:     {len(results['individual_results'])}")
        if results['ensemble_errors']:
            print(f"  Errors:      {len(results['ensemble_errors'])}")
        print(f"  Samples:     {results['n_samples']}")
        if results['mae'] is not None:
            print(f"  MAE:         {results['mae']:.4f} kcal/mol  (weighted mean)")
            print(f"  R2:          {results['r2']:.4f}  (weighted mean)")
            print(f"  RMSE:        {results['rmse']:.4f} kcal/mol  (weighted mean)")
        else:
            print(f"  Mode:        Prediction-only")
        print(f"\n  Individual member performance:")
        for r in results['individual_results']:
            mae_s = f"{r['mae']:.2f}" if r['mae'] is not None else 'N/A'
            print(f"    - {r['label']}: MAE={mae_s}")
        if results['ensemble_errors']:
            print(f"\n  Skipped members:")
            for name, nf, err in results['ensemble_errors']:
                print(f"    - {name} ({nf} feat): {err}")
        print(f"\n  Output files:")
        for fp in results['output_files']:
            print(f"    - {fp}")
        print(f"{'='*60}\n")
        return results

    # ── Single-model validation mode ────────────────────────────────────
    if not args.model:
        parser.error("Either --list-models, --ensemble, or --model is required.")
    if not args.data:
        parser.error("--data is required for external validation.")

    # Load model
    model_info = load_model(
        args.model,
        n_features=args.n_features,
        models_dir=args.models_dir,
    )

    # Determine target column
    target_col = None if args.predict_only else args.target_col

    # Run external validation
    results = external_validation(
        model_info,
        external_data=args.data,
        target_col=target_col,
        output_dir=args.output_dir,
    )

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  External Validation Complete")
    print(f"{'='*60}")
    print(f"  Model:       {args.model}")
    print(f"  Features:    {results['n_features_used']} "
          f"({', '.join(results['features_used'])})")
    print(f"  Samples:     {results['n_samples']}")
    if results['mae'] is not None:
        print(f"  MAE:         {results['mae']:.4f} kcal/mol")
        print(f"  R2:          {results['r2']:.4f}")
        print(f"  RMSE:        {results['rmse']:.4f} kcal/mol")
    else:
        print(f"  Mode:        Prediction-only")
    print(f"\n  Output files:")
    for fp in results['output_files']:
        print(f"    - {fp}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    main()
