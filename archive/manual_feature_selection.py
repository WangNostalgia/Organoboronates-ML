# -*- coding: utf-8 -*-
# ==============================================================================
# DEPRECATED: This module has been archived on 2026-06-11.
# Its functionality is fully superseded by example/manual_selection_and_plot.py,
# which provides the same checkpoint-loading and prediction capabilities plus
# integrated scatter-plot generation with RKfold metrics, all backed by a
# CSV-driven selection file (manual_feature_selection.csv).
# DO NOT import or call from active workflows.
# ==============================================================================
raise DeprecationWarning(
    "This legacy module has been archived. "
    "Use example/manual_selection_and_plot.py instead."
)

"""
===============================================================================
  Manual Feature-Count Selection & External Prediction (DEPRECATED)
===============================================================================

When the SHAP-RFECV auto-selection picks a feature count (e.g., 7) based on
minimum RKfold MAE, but your manual inspection of the Path Summary shows that
another feature count (e.g., 5) is better — considering R², interpretability,
or chemical plausibility — use this script to load the corresponding checkpoint
and make predictions on external data.

All code is standalone; no source modifications needed.

Usage:
    python example/manual_feature_selection.py
===============================================================================
"""

import joblib
import glob
import os
import pandas as pd
import numpy as np


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 1: Inspect the SHAP-RFECV Path Summary                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# The console output (or *_final_*_metrics.txt) prints a table like:
#
#   SHAP-RFECV Path Summary for SVR
#   Feat  RKfold MAE ± std      RKfold R² ± std     LOOCV R²   LOOCV MAE  ...
#   9     2.3400 ± 0.1500       0.7700 ± 0.0400     0.8112     2.1800     ...
#   8     2.2800 ± 0.1300       0.7850 ± 0.0350     0.8201     2.0500     ...
#   7     2.1500 ± 0.1100       0.8100 ± 0.0300     0.8450     1.9200     ... ← auto
#   6     2.3100 ± 0.1400       0.7800 ± 0.0380     0.8180     2.1400     ...
#   5     2.5200 ± 0.1700       0.7400 ± 0.0450     0.7900     2.3500     ...
#
# The auto-selection picks n=7 (lowest MAE).  But after reviewing R², LOOCV
# stability, and the chemical meaning of the retained features, you decide
# n=5 is actually better for your use case (more interpretable, nearly the
# same RKfold R² after accounting for std).


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 2: Load a checkpoint with the desired feature count               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_by_feature_count(model_dir, target_n_features):
    """
    Find and load the iteration checkpoint whose feature set has exactly
    `target_n_features` features and whose MAE is lowest among all such
    checkpoints.

    Parameters
    ----------
    model_dir : str, path to the model subdirectory (e.g. 'models/SVR')
    target_n_features : int, desired number of features

    Returns
    -------
    info : dict with keys 'model', 'scaler_X', 'scaler_y', 'features',
          'hyperparameters', 'metrics'
    source_file : str, filename of the selected checkpoint
    """
    candidates = []
    pattern = os.path.join(model_dir, '*_iteration_*.joblib')

    for filepath in glob.glob(pattern):
        info = joblib.load(filepath)
        n_feat = len(info['features'])
        if n_feat == target_n_features:
            # Use mae_mean (100-split average) as tiebreaker for same feature count
            mae = info['metrics'].get('mae_mean', float('inf'))
            candidates.append((mae, filepath, info))

    if not candidates:
        # Fallback: find the closest feature count available
        all_entries = []
        for filepath in glob.glob(pattern):
            info = joblib.load(filepath)
            all_entries.append((abs(len(info['features']) - target_n_features),
                                info['metrics'].get('mae_mean', float('inf')),
                                filepath, info))
        all_entries.sort()
        _, _, closest_file, closest_info = all_entries[0]
        print(f"  WARNING: No checkpoint with exactly {target_n_features} features.")
        print(f"  Using closest: {len(closest_info['features'])} features"
              f"  (file: {os.path.basename(closest_file)})")
        return closest_info, closest_file

    # Sort by MAE ascending and pick the best
    candidates.sort(key=lambda x: x[0])
    best_mae, best_file, best_info = candidates[0]

    print(f"  Selected: {os.path.basename(best_file)}")
    print(f"  Features ({len(best_info['features'])}): {best_info['features']}")
    m = best_info['metrics']
    print(f"  Metrics: MAE_mean={m.get('mae_mean','?'):.4f}, "
          f"R²_test={m.get('r2_test','?'):.4f}, "
          f"MAE_test={m.get('mae_test','?'):.4f}")
    print(f"  Best hyperparameters: {best_info['hyperparameters']}")

    return best_info, best_file


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 3: Predict on external data                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def predict_external(checkpoint_info, csv_path, output_path=None):
    """
    Use a loaded checkpoint to predict activation energies on new data.

    Parameters
    ----------
    checkpoint_info : dict from joblib.load() or load_by_feature_count()
    csv_path : str, path to the external CSV file
    output_path : str, optional output path (default: auto-generated)

    Returns
    -------
    DataFrame with predictions appended as 'predicted_activation_energy'
    """
    model = checkpoint_info['model']
    scaler_X = checkpoint_info['scaler_X']
    scaler_y = checkpoint_info['scaler_y']
    features = checkpoint_info['features']

    # Load external data
    data = pd.read_csv(csv_path)
    print(f"\n  External data: {len(data)} samples from {csv_path}")

    # Check that required feature columns exist
    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required feature columns in external data: {missing}"
        )
    extra = set(features) - set(data.columns)
    if extra:
        print(f"  NOTE: {len(extra)} features not found in CSV, will be ignored.")

    # Subset to required features (order matters for the scaler)
    X_new = data[features]

    # Scale and predict
    X_scaled = scaler_X.transform(X_new)
    y_pred_scaled = model.predict(X_scaled)

    # Inverse-transform to original kcal/mol units
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Append predictions to the original DataFrame
    data['predicted_activation_energy'] = y_pred

    if output_path is None:
        base = os.path.splitext(csv_path)[0]
        output_path = f"{base}_predicted_{len(features)}feat.csv"
    data.to_csv(output_path, index=False)

    print(f"  Predictions saved to: {output_path}")
    print(f"  Predicted range: {y_pred.min():.2f} ~ {y_pred.max():.2f} kcal/mol")
    print(f"  Mean prediction: {y_pred.mean():.2f} kcal/mol")

    return data


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 4: Compare two feature counts side-by-side                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compare_feature_counts(model_dir, n_a, n_b):
    """
    Print a side-by-side comparison of all available metrics for two
    different feature counts, to help make a manual selection decision.

    Parameters
    ----------
    model_dir : str, e.g. 'models/SVR'
    n_a, n_b : int, the two feature counts to compare
    """
    info_a, _ = load_by_feature_count(model_dir, n_a)
    info_b, _ = load_by_feature_count(model_dir, n_b)

    ma = info_a['metrics']
    mb = info_b['metrics']

    # Build comparison table
    metric_keys = [
        ('mae_mean',       '100-split MAE',       'lower'),
        ('r2_test',        'Test R²',             'higher'),
        ('mae_test',       'Test MAE',            'lower'),
        ('rkf_mae_mean',   'RKfold MAE',          'lower'),
        ('rkf_mae_std',    'RKfold MAE std',      'lower'),
        ('rkf_r2_mean',    'RKfold R²',           'higher'),
        ('loo_r2',         'LOOCV R²',            'higher'),
        ('loo_mae',        'LOOCV MAE',           'lower'),
    ]

    print(f"\n{'='*70}")
    print(f"  Manual Comparison: {n_a} features vs {n_b} features")
    print(f"  Model directory: {model_dir}")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {str(n_a)+' feat':>20} {str(n_b)+' feat':>20} {'Favors':>8}")
    print(f"  {'-'*68}")

    for key, label, direction in metric_keys:
        va = ma.get(key)
        vb = mb.get(key)
        if va is None or vb is None:
            continue
        if direction == 'lower':
            favors = f'{n_a} feat' if va < vb else (f'{n_b} feat' if vb < va else 'tie')
        else:
            favors = f'{n_a} feat' if va > vb else (f'{n_b} feat' if vb > va else 'tie')
        print(f"  {label:<20} {va:>20.4f} {vb:>20.4f} {favors:>8}")

    # Print feature lists
    print(f"\n  Features ({n_a}): {info_a['features']}")
    print(f"  Features ({n_b}): {info_b['features']}")
    only_a = set(info_a['features']) - set(info_b['features'])
    only_b = set(info_b['features']) - set(info_a['features'])
    if only_a:
        print(f"  Only in {n_a} feat: {only_a}")
    if only_b:
        print(f"  Only in {n_b} feat: {only_b}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Interactive examples (uncomment to use)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # --- Example 1: Compare auto-selected (7 features) vs your pick (5 features) ---
    # compare_feature_counts('models/SVR', 7, 5)

    # --- Example 2: Load the 5-feature checkpoint and predict on new data ---
    # info_5, src = load_by_feature_count('models/SVR', 5)
    # predict_external(info_5, 'external_data.csv')

    # --- Example 3: One-liner: find the best checkpoint for any feature count ---
    # model_dir, target_n = 'models/SVR', 5
    # mi = min(
    #     [joblib.load(f) for f in glob.glob(f'{model_dir}/*_iteration_*.joblib')
    #      if len(joblib.load(f)['features']) == target_n],
    #     key=lambda x: x['metrics']['mae_mean']
    # )
    # model, sX, sY, feats = mi['model'], mi['scaler_X'], mi['scaler_y'], mi['features']

    print("manual_feature_selection.py loaded.  Uncomment examples above to run.")
