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
    3. Check models/y_randomization_<ModelName>.png for histograms
===============================================================================
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.external_validation import load_model
from src.y_randomization import y_randomization_test

# ============================================================================
# CONFIGURATION — EDIT THESE
# ============================================================================

SELECTION_CSV = os.path.join(os.path.dirname(__file__), 'manual_feature_selection.csv')
MODELS_DIR = 'models'
ALLOW_CLOSEST = False
N_PERMS = 100
RANDOM_SEED = 42
DATA_PATH = 'example/B_dataset.csv'

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


def main():
    selections = load_selections(SELECTION_CSV)
    logger.info("Loaded %d model(s) from %s", len(selections), SELECTION_CSV)

    data = pd.read_csv(DATA_PATH).dropna(axis=1, how='all')
    numeric_cols = data.select_dtypes(include=['number']).columns
    X = data[numeric_cols].drop('activation_energy', axis=1)
    y = data['activation_energy']

    for model_name, n_features in selections.items():
        logger.info("=" * 60)
        logger.info("  y-Randomization: %s (%d features)", model_name, n_features)
        logger.info("=" * 60)

        try:
            info = load_model(
                model_name,
                n_features=n_features,
                models_dir=MODELS_DIR,
                allow_closest=ALLOW_CLOSEST,
            )
            features = info['features']

            logger.info(
                "Loaded checkpoint from %s (%d actual features)",
                info.get('_loaded_from', '<unknown>'),
                info.get('_actual_n_features', len(features)),
            )
            if not ALLOW_CLOSEST and info.get('_requested_n_features') is not None:
                logger.info("Exact checkpoint loading enforced (allow_closest=%s)", ALLOW_CLOSEST)

            logger.info("Starting %d permutations (5x5 RepeatedKFold each)...", N_PERMS)
            results = y_randomization_test(
                model=info['model'],
                X=X,
                y=y,
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

        except Exception as exc:
            logger.error("%s: y-randomization failed — %s", model_name, exc, exc_info=True)
            continue

    logger.info("All models processed. Check models/ for y_randomization_*.png")


if __name__ == '__main__':
    main()
