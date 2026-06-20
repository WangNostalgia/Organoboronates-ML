# -*- coding: utf-8 -*-
"""
===============================================================================
  Standalone y-Randomization Test
===============================================================================

Runs y-randomization independently on locked models and feature counts specified
in a CSV file (same format as manual_selection_and_plot.py).

By default, the test is run only on the development rows recorded in each model
checkpoint's evaluation_protocol["development_indices"]. This preserves the
untouched final-test split created by the main training pipeline.

The CSV format:
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
TARGET_COL = 'activation_energy'
METADATA_COLUMNS = ('ID', 'SMILES', 'filename')
ALLOW_FULL_DATA_WITHOUT_PROTOCOL = False

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


def _development_data_from_checkpoint(data, model_info, model_name):
    protocol = model_info.get('evaluation_protocol') or {}
    development_indices = protocol.get('development_indices')

    if not development_indices:
        if ALLOW_FULL_DATA_WITHOUT_PROTOCOL:
            logger.warning(
                "%s checkpoint has no evaluation_protocol.development_indices; "
                "falling back to full data because ALLOW_FULL_DATA_WITHOUT_PROTOCOL=True.",
                model_name,
            )
            return data.copy()
        raise ValueError(
            f"{model_name} checkpoint does not contain "
            "evaluation_protocol['development_indices']. Re-run main.py to create "
            "protocol-aware checkpoints, or set ALLOW_FULL_DATA_WITHOUT_PROTOCOL=True "
            "only for legacy exploratory analysis."
        )

    missing = [idx for idx in development_indices if idx not in data.index]
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"{model_name} development_indices do not match {DATA_PATH}. "
            f"Missing index labels: {preview}. Ensure the CSV is the same, "
            "unreordered source file used during main.py training."
        )

    subset = data.loc[development_indices].copy()
    logger.info(
        "%s: using %d development rows for y-randomization; final-test rows remain untouched.",
        model_name,
        len(subset),
    )
    return subset


def _make_xy(data):
    if TARGET_COL not in data.columns:
        raise ValueError(f"Input data must contain target column '{TARGET_COL}'.")
    numeric_cols = data.select_dtypes(include=['number']).columns
    excluded = set(METADATA_COLUMNS) | {TARGET_COL}
    feature_cols = [col for col in numeric_cols if col not in excluded]
    if not feature_cols:
        raise ValueError("No numeric descriptor feature columns found for y-randomization.")
    X = data[feature_cols]
    y = data[TARGET_COL]
    return X, y


def main():
    selections = load_selections(SELECTION_CSV)
    logger.info("Loaded %d model(s) from %s", len(selections), SELECTION_CSV)

    data = pd.read_csv(DATA_PATH).dropna(axis=1, how='all')

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
            development_data = _development_data_from_checkpoint(data, info, model_name)
            X, y = _make_xy(development_data)

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
