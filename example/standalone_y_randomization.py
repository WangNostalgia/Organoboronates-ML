# -*- coding: utf-8 -*-
"""
===============================================================================
  Standalone y-Randomization Test
===============================================================================

Runs y-randomization independently on models and feature counts specified in
a CSV file (same format as manual_selection_and_plot.py).

Designed for final validation runs — after you have locked in the optimal
model and feature set for each model, launch this script overnight to obtain
the statistical p-value needed for publication.

The CSV format (same file used by manual_selection_and_plot.py):
    model_name,n_features
    SVR,5
    RandomForest,7
    XGBoost,4

Usage:
    1. Ensure the CSV exists (default: 'example/manual_feature_selection.csv')
    2. Run: python example/standalone_y_randomization.py
    3. Check models/<ModelName>/y_randomization_<ModelName>.png for histograms
===============================================================================
"""

import joblib
import glob
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.y_randomization import y_randomization_test

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — EDIT THESE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Path to the CSV specifying per-model feature counts
# (same file as used by manual_selection_and_plot.py)
SELECTION_CSV = os.path.join(os.path.dirname(__file__), 'manual_feature_selection.csv')

N_PERMS = 100          # Number of y-shuffling permutations (reduce to 30 for quick test)
RANDOM_SEED = 42       # Base random seed for reproducibility
DATA_PATH = 'example/B_dataset.csv'  # Path to training data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_selections(csv_path):
    """Read the model selection CSV. Returns dict: {model_name: n_features}."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Selection CSV not found: {csv_path}\n"
            f"Create one with columns: model_name, n_features\n"
            f"Example:\n  model_name,n_features\n  SVR,5\n  XGBoost,4"
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'model_name' not in df.columns or 'n_features' not in df.columns:
        raise ValueError("CSV must have columns: model_name, n_features")
    selections = {}
    for _, row in df.iterrows():
        selections[str(row['model_name']).strip()] = int(row['n_features'])
    return selections


def find_checkpoint(model_dir, n_features):
    """Find the best iteration checkpoint with exactly n_features features."""
    pattern = os.path.join(model_dir, '*_iteration_*.joblib')
    candidates = []
    for fp in glob.glob(pattern):
        info = joblib.load(fp)
        if len(info['features']) == n_features:
            mae = info['metrics'].get('mae_mean', float('inf'))
            candidates.append((mae, fp, info))

    if not candidates:
        # Fallback: closest available feature count
        all_entries = []
        for fp in glob.glob(pattern):
            info = joblib.load(fp)
            all_entries.append((abs(len(info['features']) - n_features),
                                info['metrics'].get('mae_mean', float('inf')),
                                fp, info))
        all_entries.sort()
        _, _, closest_file, closest_info = all_entries[0]
        logger.warning("No checkpoint with %d features; using closest: %d (%s)",
                       n_features, len(closest_info['features']),
                       os.path.basename(closest_file))
        return closest_info

    candidates.sort(key=lambda x: x[0])
    best_mae, best_file, best_info = candidates[0]
    logger.info("Loaded checkpoint: %s (MAE_mean=%.4f)",
                os.path.basename(best_file), best_mae)
    logger.info("Features (%d): %s", len(best_info['features']), best_info['features'])
    return best_info


def detect_model_class(checkpoint_info):
    """Auto-detect the model class from the stored model object."""
    cls = type(checkpoint_info['model'])
    logger.info("Auto-detected model class: %s", cls.__name__)
    return cls


def main():
    # 1. Load selections from CSV
    selections = load_selections(SELECTION_CSV)
    logger.info("Loaded %d model(s) from %s", len(selections), SELECTION_CSV)

    # 2. Load the full training data once
    data = pd.read_csv(DATA_PATH).dropna(axis=1, how='all')
    numeric_cols = data.select_dtypes(include=['number']).columns
    X = data[numeric_cols].drop('activation_energy', axis=1)
    y = data['activation_energy']

    # 3. Process each model
    for model_name, n_features in selections.items():
        model_dir = os.path.join('models', model_name)
        if not os.path.isdir(model_dir):
            logger.warning("Model directory not found: %s — skipping", model_dir)
            continue

        logger.info("=" * 60)
        logger.info("  y-Randomization: %s (%d features)", model_name, n_features)
        logger.info("=" * 60)

        try:
            info = find_checkpoint(model_dir, n_features)
            features = info['features']
            best_params = info['hyperparameters']
            model_class = detect_model_class(info)

            logger.info("Starting %d permutations (5x5 RepeatedKFold each)...", N_PERMS)
            results = y_randomization_test(
                model_class=model_class,
                best_params=best_params,
                X=X, y=y,
                selected_features=features,
                n_permutations=N_PERMS,
                random_state=RANDOM_SEED,
            )

            logger.info("  Original MAE:  %.4f ± %.4f", results['original_mae_mean'], results['original_mae_std'])
            logger.info("  Original R²:   %.4f ± %.4f", results['original_r2_mean'], results['original_r2_std'])
            logger.info("  Random MAE:    %.4f ± %.4f", results['random_mae_mean'], results['random_mae_std'])
            logger.info("  Random R²:     %.4f ± %.4f", results['random_r2_mean'], results['random_r2_std'])
            logger.info("  p-value (MAE): %.4f", results['p_value_mae'])
            logger.info("  Result:        %s", "PASSED" if results['passed'] else "FAILED")

        except Exception as e:
            logger.error("%s: y-randomization failed — %s", model_name, e, exc_info=True)
            continue

    logger.info("All models processed. Check models/<ModelName>/ for y_randomization_*.png")


if __name__ == '__main__':
    main()
