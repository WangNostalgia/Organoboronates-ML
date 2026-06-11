# -*- coding: utf-8 -*-
"""
===============================================================================
  Manual Feature-Count Selection & Plotting (Standalone)
===============================================================================

After training completes, inspect each model's SHAP-RFECV Path Summary
(printed in the console log or saved in *_final_*_metrics.txt), decide the
best feature count manually, and use this script to:

  1. Read a CSV specifying per-model manual feature counts
  2. Load the corresponding iteration checkpoint for that feature count
  3. Generate final_scatter and final_scatter_outliers for each model
  4. (Future) Predict on external data — commented out for now

Usage:
    1. Create a CSV file (e.g. 'manual_feature_selection.csv') with columns:
       model_name, n_features
    2. Run: python example/manual_selection_and_plot.py

The CSV format:
    model_name,n_features
    SVR,5
    RandomForest,7
    XGBoost,4
    ...
===============================================================================
"""

import joblib
import glob
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import plot_scatter
from sklearn.model_selection import train_test_split


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Configuration — EDIT THESE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Path to the CSV specifying manual feature counts per model
MANUAL_SELECTION_CSV = os.path.join(os.path.dirname(__file__), 'manual_feature_selection.csv')

# Path to the training data (same as used in main.py)
DATA_PATH = 'example/B_dataset.csv'

# Output directory for plots
OUTPUT_DIR = 'models/manual_selection_plots'

# Target column name
TARGET_COL = 'activation_energy'


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
    dict: {model_name: n_features}
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
        selections[str(row['model_name']).strip()] = int(row['n_features'])
    print(f"Loaded {len(selections)} manual selections from {csv_path}")
    for name, n in selections.items():
        print(f"  {name}: {n} features")
    return selections


def find_checkpoint(model_name, n_features):
    """
    Find the iteration checkpoint for a given model and feature count.

    Searches models/<model_name>/*_iteration_*.joblib for the checkpoint
    whose feature set has exactly n_features and has the lowest MAE among
    all matches.

    Parameters
    ----------
    model_name : str, e.g. 'SVR'
    n_features : int, desired feature count

    Returns
    -------
    info : dict from joblib.load()
    source_file : str, path to the checkpoint file
    """
    model_dir = os.path.join('models', model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pattern = os.path.join(model_dir, '*_iteration_*.joblib')
    candidates = []
    for filepath in glob.glob(pattern):
        info = joblib.load(filepath)
        if len(info['features']) == n_features:
            mae = info['metrics'].get('mae_mean', float('inf'))
            candidates.append((mae, filepath, info))

    if not candidates:
        # Closest available
        all_entries = []
        for filepath in glob.glob(pattern):
            info = joblib.load(filepath)
            all_entries.append((abs(len(info['features']) - n_features),
                                info['metrics'].get('mae_mean', float('inf')),
                                filepath, info))
        all_entries.sort()
        _, _, closest_file, closest_info = all_entries[0]
        actual_n = len(closest_info['features'])
        print(f"  WARNING: {model_name}: no checkpoint with {n_features} features. "
              f"Using closest: {actual_n} features (file: {os.path.basename(closest_file)})")
        return closest_info, closest_file

    candidates.sort(key=lambda x: x[0])
    best_mae, best_file, best_info = candidates[0]
    print(f"  {model_name}: loaded {n_features}-feature checkpoint "
          f"(MAE_mean={best_mae:.4f}, file: {os.path.basename(best_file)})")
    return best_info, best_file


def generate_plots(model_name, checkpoint_info):
    """
    Generate final_scatter and final_scatter_outliers for a manually
    selected model, matching the output of iterative_optimization.py.

    Parameters
    ----------
    model_name : str, e.g. 'SVR'
    checkpoint_info : dict from joblib.load()
    """
    model = checkpoint_info['model']
    features = checkpoint_info['features']
    best_params = checkpoint_info['hyperparameters']

    # Load the training data and subset to the selected features
    data = pd.read_csv(DATA_PATH).dropna(axis=1, how='all')
    X = data[features]
    y = data[TARGET_COL]

    # Train/test split (MUST match iterative_optimization.py which uses random_state=40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )

    # Scale and fit (fresh scalers on this split)
    from sklearn.preprocessing import MinMaxScaler
    sX = MinMaxScaler()
    sY = MinMaxScaler(feature_range=(0, 100))
    X_train_s = sX.fit_transform(X_train)
    X_test_s = sX.transform(X_test)
    y_train_s = sY.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model.fit(X_train_s, y_train_s)
    y_pred_train_s = model.predict(X_train_s)
    y_pred_test_s = model.predict(X_test_s)
    y_pred_train = sY.inverse_transform(y_pred_train_s.reshape(-1, 1)).ravel()
    y_pred_test = sY.inverse_transform(y_pred_test_s.reshape(-1, 1)).ravel()

    # Use the checkpoint's stored metrics (computed during the original
    # training iteration with the correct feature set).
    stored_metrics = checkpoint_info.get('metrics', {})
    mae_mean = stored_metrics.get('mae_mean')     # 100-split average MAE
    rkf_mae = stored_metrics.get('rkf_mae_mean')   # 5×5 RepeatedKFold MAE
    rkf_r2 = stored_metrics.get('rkf_r2_mean')     # 5×5 RepeatedKFold R²

    # Output directories
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Generate plot (same signature as iterative_optimization.py)
    n_feat = len(features)
    plot_scatter(
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        model_name=f"{model_name} ({n_feat} feat)",
        mae_mean=mae_mean,
        output_dir=out_dir + '/',
        output_name=f'final_scatter_{n_feat}feat.png',
        X_train=X_train,
        X_test=X_test,
        rkf_mae=rkf_mae,
        rkf_r2=rkf_r2,
    )
    print(f"  {model_name}: plots saved to {out_dir}/")


def predict_external(checkpoint_info, csv_path, output_path=None):
    """
    (FUTURE USE — currently commented out in main)

    Use a manually selected checkpoint to predict on external data.

    Parameters
    ----------
    checkpoint_info : dict from joblib.load()
    csv_path : str, path to external CSV
    output_path : str, optional output path
    """
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
    # 1. Load manual selections
    selections = load_manual_selections(MANUAL_SELECTION_CSV)

    # 2. For each model, load the checkpoint and generate plots
    for model_name, n_features in selections.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {model_name} ({n_features} features)")
        print(f"{'='*60}")
        try:
            info, src = find_checkpoint(model_name, n_features)
            generate_plots(model_name, info)
        except Exception as e:
            print(f"  ERROR: {model_name}: {e}")
            continue

    print(f"\nDone. Plots saved to {OUTPUT_DIR}/")

    # ═══════════════════════════════════════════════════════════════════════
    # FUTURE: External prediction (uncomment when ready)
    # ═══════════════════════════════════════════════════════════════════════
    # for model_name, n_features in selections.items():
    #     info, src = find_checkpoint(model_name, n_features)
    #     predict_external(info, 'external_data.csv')
