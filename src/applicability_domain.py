"""
Applicability-domain analysis for trained organoboronate ML models.
"""

import argparse
import logging
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_K_NEIGHBORS = 5
DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_MODELS_DIR = 'models'
DEFAULT_TARGET_COL = 'activation_energy'
DEFAULT_OUTPUT_DIR = 'applicability_domain_results'

LINEAR_LEVERAGE_MODELS = {
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
}


def applicability_domain_analysis(model_info, X_train, X_external,
                                  y_train=None, y_external=None,
                                  y_pred_external=None,
                                  k_neighbors=DEFAULT_K_NEIGHBORS,
                                  z_threshold=DEFAULT_Z_THRESHOLD,
                                  output_dir=DEFAULT_OUTPUT_DIR,
                                  model_name='Model'):
    """
    Assess the applicability domain of a trained model for an external set.
    """
    X_train = _to_dataframe(X_train, 'training')
    X_external = _to_dataframe(X_external, 'external')

    expected_features = list(model_info['features'])
    missing_train = [f for f in expected_features if f not in X_train.columns]
    missing_ext = [f for f in expected_features if f not in X_external.columns]
    if missing_train:
        raise ValueError(f"Training data missing features: {missing_train}")
    if missing_ext:
        raise ValueError(f"External data missing features: {missing_ext}")

    X_train_feat = X_train[expected_features]
    X_ext_feat = X_external[expected_features]

    _validate_feature_matrix(X_train_feat, 'training')
    _validate_feature_matrix(X_ext_feat, 'external')

    n_train = len(X_train_feat)
    n_ext = len(X_ext_feat)
    if n_train < 2:
        raise ValueError("Training data must contain at least 2 usable samples for applicability-domain analysis.")
    if n_ext < 1:
        raise ValueError("External data must contain at least 1 usable sample for applicability-domain analysis.")

    if not isinstance(k_neighbors, (int, np.integer)):
        raise ValueError("k_neighbors must be a positive integer.")
    if k_neighbors < 1 or k_neighbors > (n_train - 1):
        raise ValueError(f"k_neighbors must be between 1 and {n_train - 1} for {n_train} training samples.")

    y_train_arr = None
    if y_train is not None:
        y_train_arr = _validate_target_array(y_train, n_train, 'y_train')

    y_external_arr = None
    if y_external is not None:
        y_external_arr = _validate_target_array(y_external, n_ext, 'y_external')
        if y_train_arr is None:
            raise ValueError(
                "y_train is required to calibrate Williams standardized residuals when y_external is provided."
            )

    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']

    X_train_values = X_train_feat.to_numpy(dtype=float)
    X_ext_values = X_ext_feat.to_numpy(dtype=float)

    if y_pred_external is None:
        X_ext_scaled_for_model = scaler_X.transform(X_ext_feat)
        y_pred_scaled = model.predict(X_ext_scaled_for_model)
        y_pred_external = scaler_y.inverse_transform(
            np.asarray(y_pred_scaled, dtype=float).reshape(-1, 1)
        ).ravel()
    else:
        y_pred_external = _validate_target_array(y_pred_external, n_ext, 'y_pred_external')

    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train_values)
    X_ext_scaled_for_ad = feat_scaler.transform(X_ext_values)

    williams_data = _compute_williams(
        X_train_scaled=X_train_scaled,
        X_ext_scaled=X_ext_scaled_for_ad,
        model=model,
        y_train=y_train_arr,
        y_external=y_external_arr,
        y_pred_external=y_pred_external,
        p=len(expected_features),
        n_train=n_train,
        X_train_unscaled=X_train_feat,
    )

    knn_data = _compute_knn_distance(
        X_train_scaled=X_train_scaled,
        X_ext_scaled=X_ext_scaled_for_ad,
        k_neighbors=k_neighbors,
        z_threshold=z_threshold,
    )

    id_cols = [c for c in ['sub_H', 'sub_B'] if c in X_external.columns]
    results_df = X_external[id_cols + expected_features].copy().reset_index(drop=True)
    results_df['predicted_activation_energy'] = y_pred_external
    if y_external_arr is not None:
        results_df['activation_energy'] = y_external_arr
        results_df['absolute_error'] = np.abs(y_pred_external - y_external_arr)

    results_df['leverage'] = williams_data['leverage_ext']
    results_df['std_residual'] = williams_data['std_residuals_ext']
    results_df['williams_warning'] = williams_data['warnings_ext']
    results_df['knn_distance'] = knn_data['distances_ext']
    results_df['knn_z_score'] = knn_data['z_scores_ext']
    results_df['knn_warning'] = knn_data['warnings_ext']
    results_df['ad_combined_warning'] = (
        results_df['williams_warning'] | results_df['knn_warning']
    )

    ad_summary = {
        'n_total': n_ext,
        'n_williams_warning': int(williams_data['warnings_ext'].sum()),
        'n_williams_high_leverage': int((williams_data['leverage_ext'] > williams_data['h_star']).sum()),
        'n_williams_high_residual': int(
            (np.abs(williams_data['std_residuals_ext']) > williams_data['residual_critical']).sum()
        ),
        'n_knn_warning': int(knn_data['warnings_ext'].sum()),
        'n_combined_warning': int(results_df['ad_combined_warning'].sum()),
        'h_star': float(williams_data['h_star']),
        'residual_critical': float(williams_data['residual_critical']),
        'knn_threshold': float(knn_data['threshold']),
        'knn_training_mean': float(knn_data['training_mean']),
        'knn_training_std': float(knn_data['training_std']),
        'leverage_note': williams_data['leverage_note'],
        'prediction_only': y_external_arr is None,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(' ', '_')

    ad_csv_path = os.path.join(output_dir, f"{safe_name}_ad_results_{timestamp}.csv")
    results_df.to_csv(ad_csv_path, index=False, encoding='utf-8-sig')
    output_files.append(ad_csv_path)

    williams_path = os.path.join(output_dir, f"{safe_name}_williams_plot_{timestamp}.png")
    _plot_williams(williams_data, model_name, williams_path)
    output_files.append(williams_path)

    knn_path = os.path.join(output_dir, f"{safe_name}_knn_distance_{timestamp}.png")
    _plot_knn_distance(knn_data, k_neighbors, model_name, knn_path)
    output_files.append(knn_path)

    summary_path = os.path.join(output_dir, f"{safe_name}_ad_summary_{timestamp}.txt")
    _write_ad_summary(summary_path, model_name, ad_summary, expected_features)
    output_files.append(summary_path)

    return {
        'ad_results': results_df,
        'williams_data': williams_data,
        'knn_data': knn_data,
        'ad_summary': ad_summary,
        'output_files': output_files,
    }


def _compute_williams(X_train_scaled, X_ext_scaled, model, y_train, y_external,
                      y_pred_external, p, n_train, X_train_unscaled):
    """
    Compute descriptor-space leverage and calibrated standardized residuals.
    """
    X_with_intercept = np.column_stack([np.ones(n_train), X_train_scaled])
    try:
        leverage_train = _hat_matrix_diag(X_with_intercept)
    except np.linalg.LinAlgError:
        logger.warning("Hat matrix computation failed (singular). Using pseudo-inverse fallback.")
        leverage_train = _hat_matrix_diag_pinv(X_with_intercept)

    X_ext_design = np.column_stack([np.ones(len(X_ext_scaled)), X_ext_scaled])
    try:
        xtx_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
    leverage_ext = np.array([x_i @ xtx_inv @ x_i for x_i in X_ext_design], dtype=float)

    h_star = 3.0 * (p + 1) / n_train

    if y_external is not None:
        residuals_ext = np.asarray(y_external, dtype=float) - np.asarray(y_pred_external, dtype=float)
    else:
        residuals_ext = np.full(len(X_ext_scaled), np.nan, dtype=float)

    residuals_train = None
    residual_scale = np.finfo(float).eps
    if y_train is not None:
        y_pred_train_oof = _training_oof_predictions(model, X_train_unscaled, y_train)
        residuals_train = np.asarray(y_train, dtype=float).ravel() - y_pred_train_oof
        residual_scale = _oof_residual_scale(residuals_train)

    with np.errstate(divide='ignore', invalid='ignore'):
        std_residuals_ext = np.where(
            np.isfinite(residuals_ext),
            residuals_ext / residual_scale,
            np.nan,
        )

    residual_critical = 3.0
    warnings_ext = (
        (leverage_ext > h_star) |
        (np.abs(std_residuals_ext) > residual_critical)
    )
    nan_resid_mask = np.isnan(std_residuals_ext)
    warnings_ext[nan_resid_mask] = leverage_ext[nan_resid_mask] > h_star

    leverage_is_approximate = _uses_descriptor_space_leverage_approximation(model)
    return {
        'leverage_train': leverage_train,
        'leverage_ext': leverage_ext,
        'std_residuals_ext': std_residuals_ext,
        'h_star': h_star,
        'residual_critical': residual_critical,
        'residuals_ext': residuals_ext,
        'residuals_train': residuals_train,
        'residual_scale': residual_scale,
        'warnings_ext': warnings_ext,
        'n_train': n_train,
        'p': p,
        'residuals_available': bool(np.isfinite(std_residuals_ext).any()),
        'leverage_note': (
            "Descriptor-space leverage is an approximation for nonlinear estimators."
            if leverage_is_approximate
            else "Descriptor-space leverage follows the standard linear Williams-plot formulation."
        ),
        'leverage_xlabel': (
            'Descriptor-space leverage h (approx.)'
            if leverage_is_approximate
            else 'Leverage (h)'
        ),
    }


def _hat_matrix_diag(X):
    xtx_inv = np.linalg.inv(X.T @ X)
    return np.array([x @ xtx_inv @ x for x in X], dtype=float)


def _hat_matrix_diag_pinv(X):
    xtx_inv = np.linalg.pinv(X.T @ X)
    return np.array([x @ xtx_inv @ x for x in X], dtype=float)


def _training_oof_predictions(model, X, y, n_splits=5, random_state=42):
    """
    Generate one OOF prediction per training sample in kcal/mol.
    """
    X_df = _to_dataframe(X, 'training')
    y_arr = _validate_target_array(y, len(X_df), 'y_train')
    _validate_feature_matrix(X_df, 'training')

    if len(X_df) < 2:
        raise ValueError("Training data must contain at least 2 usable samples for OOF prediction.")

    effective_splits = min(int(n_splits), len(X_df))
    if effective_splits < 2:
        raise ValueError("OOF prediction requires at least 2 folds.")

    oof_predictions = np.full(len(X_df), np.nan, dtype=float)
    splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in splitter.split(X_df):
        X_train_fold = X_df.iloc[train_idx].to_numpy(dtype=float)
        X_test_fold = X_df.iloc[test_idx].to_numpy(dtype=float)
        y_train_fold = y_arr[train_idx]

        fold_scaler_X = MinMaxScaler()
        fold_scaler_y = MinMaxScaler(feature_range=(0, 100))
        X_train_scaled = fold_scaler_X.fit_transform(X_train_fold)
        X_test_scaled = fold_scaler_X.transform(X_test_fold)
        y_train_scaled = fold_scaler_y.fit_transform(y_train_fold.reshape(-1, 1)).ravel()

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = np.asarray(fold_model.predict(X_test_scaled), dtype=float).reshape(-1, 1)
        y_pred_fold = fold_scaler_y.inverse_transform(y_pred_scaled).ravel()
        oof_predictions[test_idx] = y_pred_fold

    if len(oof_predictions) != len(X_df):
        raise ValueError("OOF prediction output length does not match the number of training samples.")
    if not np.isfinite(oof_predictions).all():
        raise ValueError("OOF predictions must contain one finite prediction for every training sample.")

    return oof_predictions


def _oof_residual_scale(residuals):
    """
    Robust Williams residual scale from finite 1-D OOF residuals.
    """
    residuals_arr = np.asarray(residuals, dtype=float)
    if residuals_arr.ndim != 1 or residuals_arr.size == 0:
        raise ValueError("residuals must be a non-empty 1-D array.")
    if not np.isfinite(residuals_arr).all():
        raise ValueError("residuals must contain only finite values.")

    m = np.median(residuals_arr)
    scale = 1.4826 * np.median(np.abs(residuals_arr - m))
    if scale == 0:
        scale = np.std(residuals_arr, ddof=1)
    if not np.isfinite(scale) or scale == 0:
        scale = np.finfo(float).eps
    return float(scale)


def _compute_knn_distance(X_train_scaled, X_ext_scaled, k_neighbors, z_threshold):
    """
    Compute average Euclidean distance to the k nearest training neighbours.
    """
    n_train = len(X_train_scaled)
    if not isinstance(k_neighbors, (int, np.integer)):
        raise ValueError("k_neighbors must be an integer.")
    if k_neighbors < 1 or k_neighbors > (n_train - 1):
        raise ValueError(f"k_neighbors must be between 1 and {n_train - 1} for {n_train} training samples.")

    nn = NearestNeighbors(
        n_neighbors=k_neighbors + 1,
        metric='euclidean',
        n_jobs=1,
    )
    nn.fit(X_train_scaled)

    train_distances, _ = nn.kneighbors(X_train_scaled)
    train_knn_avg = train_distances[:, 1:k_neighbors + 1].mean(axis=1)
    train_mean = float(np.mean(train_knn_avg))
    train_std = float(np.std(train_knn_avg, ddof=1)) if len(train_knn_avg) > 1 else 0.0
    threshold = train_mean + z_threshold * max(train_std, 1e-8)

    ext_distances, _ = nn.kneighbors(X_ext_scaled, n_neighbors=k_neighbors)
    ext_knn_avg = ext_distances.mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores_ext = np.where(
            train_std > 0,
            (ext_knn_avg - train_mean) / train_std,
            0.0,
        )
    warnings_ext = ext_knn_avg > threshold

    return {
        'distances_train': train_knn_avg,
        'distances_ext': ext_knn_avg,
        'z_scores_ext': z_scores_ext,
        'warnings_ext': warnings_ext,
        'training_mean': train_mean,
        'training_std': train_std,
        'threshold': threshold,
        'k_effective': k_neighbors,
    }


def _plot_williams(williams_data, model_name, output_path):
    leverage_ext = np.asarray(williams_data['leverage_ext'], dtype=float)
    std_residuals = np.asarray(williams_data['std_residuals_ext'], dtype=float)
    h_star = float(williams_data['h_star'])
    residual_critical = float(williams_data['residual_critical'])
    warnings = np.asarray(williams_data['warnings_ext'], dtype=bool)
    residuals_available = bool(williams_data.get('residuals_available', np.isfinite(std_residuals).any()))

    plot_residuals = np.where(np.isfinite(std_residuals), std_residuals, 0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    normal_mask = ~warnings
    if normal_mask.any():
        ax.scatter(
            leverage_ext[normal_mask],
            plot_residuals[normal_mask],
            alpha=0.6,
            edgecolors='#2c3e50',
            facecolors='#3498db',
            s=50,
            linewidth=0.3,
            label='Within AD',
            zorder=5,
        )
    if warnings.any():
        ax.scatter(
            leverage_ext[warnings],
            plot_residuals[warnings],
            alpha=0.8,
            edgecolors='#922b21',
            facecolors='#e74c3c',
            s=70,
            linewidth=0.5,
            marker='^',
            label=f'Outside AD ({warnings.sum()})',
            zorder=6,
        )

    x_max = max(float(np.max(leverage_ext)) if len(leverage_ext) else 0.0, h_star * 1.5, 1e-8) * 1.1
    finite_abs = np.abs(std_residuals[np.isfinite(std_residuals)])
    y_max = max(float(np.max(finite_abs)) if finite_abs.size else 0.0, residual_critical) * 1.3

    ax.axvline(x=h_star, color='#e67e22', linestyle='--', linewidth=1.5, label=f'h* = {h_star:.4f}')
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

    if residuals_available:
        ax.axhline(y=residual_critical, color='#e67e22', linestyle=':', linewidth=1.2)
        ax.axhline(y=-residual_critical, color='#e67e22', linestyle=':', linewidth=1.2,
                   label=f'±{residual_critical:.0f}σ residual')
        ax.fill_between([h_star, x_max], -residual_critical, residual_critical, alpha=0.04, color='orange')
        ax.fill_between([0, x_max], residual_critical, y_max, alpha=0.04, color='orange')
        ax.fill_between([0, x_max], -y_max, -residual_critical, alpha=0.04, color='orange')
        ax.set_ylabel('Standardized Residual', fontsize=13)
        title_suffix = 'Applicability Domain Assessment'
    else:
        ax.set_ylabel('Residual unavailable (prediction-only)', fontsize=13)
        title_suffix = 'Leverage-only View'

    ax.set_xlabel(williams_data.get('leverage_xlabel', 'Leverage (h)'), fontsize=13)
    ax.set_title(f'{model_name} — Williams Plot\n{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_knn_distance(knn_data, k_neighbors, model_name, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    train_dist = np.asarray(knn_data['distances_train'], dtype=float)
    ext_dist = np.asarray(knn_data['distances_ext'], dtype=float)
    threshold = float(knn_data['threshold'])

    ax1.hist(train_dist, bins=30, alpha=0.6, color='#3498db',
             label=f'Training (n={len(train_dist)})', edgecolor='#2c3e50', linewidth=0.3)
    ax1.hist(ext_dist, bins=30, alpha=0.7, color='#e74c3c',
             label=f'External (n={len(ext_dist)})', edgecolor='#922b21', linewidth=0.3)
    ax1.axvline(x=threshold, color='#e67e22', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    ax1.set_xlabel(f'Average Distance to {k_neighbors} Nearest Neighbours', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('k-NN Distance Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)

    indices = np.arange(len(ext_dist))
    colors = ['#e74c3c' if d > threshold else '#3498db' for d in ext_dist]
    ax2.bar(indices, ext_dist, color=colors, alpha=0.8, width=0.8)
    ax2.axhline(y=threshold, color='#e67e22', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    ax2.set_xlabel('External Compound Index', fontsize=11)
    ax2.set_ylabel(f'Avg Distance to {k_neighbors} NN', fontsize=11)
    ax2.set_title('Per-Compound k-NN Distance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle(f'{model_name} — k-NN Applicability Domain', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _write_ad_summary(summary_path, model_name, ad_summary, features):
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Applicability Domain Analysis Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Model:        {model_name}\n")
        f.write(f"Features:     {', '.join(features)}\n")
        f.write(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total external samples:          {ad_summary['n_total']}\n")
        f.write("\n--- Williams Plot ---\n")
        f.write(f"Leverage threshold (h*):        {ad_summary['h_star']:.6f}\n")
        f.write(f"Residual threshold:             ±{ad_summary['residual_critical']:.0f}σ\n")
        f.write(f"High leverage (>h*):            {ad_summary['n_williams_high_leverage']}\n")
        f.write(f"High residual:                  {ad_summary['n_williams_high_residual']}\n")
        f.write(f"Total Williams warnings:        {ad_summary['n_williams_warning']}\n")
        f.write(f"Leverage note:                  {ad_summary['leverage_note']}\n")
        f.write("\n--- k-NN Distance ---\n")
        f.write(f"Training mean distance:         {ad_summary['knn_training_mean']:.4f}\n")
        f.write(f"Training std distance:          {ad_summary['knn_training_std']:.4f}\n")
        f.write(f"Warning threshold:              {ad_summary['knn_threshold']:.4f}\n")
        f.write(f"k-NN warnings:                  {ad_summary['n_knn_warning']}\n")
        f.write("\n--- Combined ---\n")
        f.write(f"Compounds flagged:              {ad_summary['n_combined_warning']}\n")
        flagged_percentage = (
            100.0 * ad_summary['n_combined_warning'] / max(ad_summary['n_total'], 1)
        )
        f.write(f"Percentage flagged:             {flagged_percentage:.1f}%\n")
        f.write("\n--- Interpretation ---\n")
        if ad_summary.get('prediction_only', False):
            f.write(
                "Prediction-only mode: flagged samples fall outside the "
                "leverage and/or k-NN distance criteria.\n"
            )
        elif ad_summary['n_combined_warning']:
            f.write(
                "Flagged samples exceed at least one Williams-plot or k-NN "
                "applicability-domain criterion.\n"
            )
        else:
            f.write(
                "All external samples satisfy the configured Williams-plot "
                "and k-NN applicability-domain criteria.\n"
            )


def _to_dataframe(data, label='data'):
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"{label} file not found: {data}")
        df = pd.read_csv(data)
        unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"{label} array must be 2-D.")
        return pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])
    raise TypeError(f"{label} must be str, DataFrame, or ndarray, got {type(data).__name__}")


def _validate_feature_matrix(X, label):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{label} feature matrix must be a non-empty 2-D array.")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} feature matrix contains non-finite values.")


def _validate_target_array(y, expected_length, label):
    values = np.asarray(y, dtype=float).ravel()
    if len(values) != expected_length:
        raise ValueError(f"{label} length ({len(values)}) does not match expected length ({expected_length}).")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")
    return values


def _uses_descriptor_space_leverage_approximation(model):
    return model.__class__.__name__ not in LINEAR_LEVERAGE_MODELS


def main():
    parser = argparse.ArgumentParser(
        description='Applicability Domain Analysis for Organoboronate ML Models',
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., SVR, RandomForest).')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Feature count for the model version.')
    parser.add_argument('--training', type=str, required=True,
                        help='Path to training data CSV.')
    parser.add_argument('--external', type=str, required=True,
                        help='Path to external compounds CSV to assess.')
    parser.add_argument('--target-col', type=str, default=DEFAULT_TARGET_COL,
                        help='Target column name in training/external CSV files.')
    parser.add_argument('-k', '--k-neighbors', type=int, default=DEFAULT_K_NEIGHBORS,
                        help=f'Number of nearest neighbours (default: {DEFAULT_K_NEIGHBORS}).')
    parser.add_argument('-z', '--z-threshold', type=float, default=DEFAULT_Z_THRESHOLD,
                        help=f'z-score threshold for k-NN warning (default: {DEFAULT_Z_THRESHOLD}).')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR,
                        help=f'Models directory (default: {DEFAULT_MODELS_DIR}).')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    from src.external_validation import load_model

    model_info = load_model(args.model, n_features=args.n_features, models_dir=args.models_dir)
    X_train = _to_dataframe(args.training, 'training')
    X_external = _to_dataframe(args.external, 'external')

    if args.target_col not in X_train.columns:
        raise ValueError(
            f"Training CSV must contain target column '{args.target_col}' so Williams residuals can be calibrated."
        )

    y_train = X_train[args.target_col].to_numpy(dtype=float)
    X_train = X_train.drop(columns=[args.target_col])

    y_external = None
    if args.target_col in X_external.columns:
        y_external = X_external[args.target_col].to_numpy(dtype=float)

    results = applicability_domain_analysis(
        model_info=model_info,
        X_train=X_train,
        X_external=X_external,
        y_train=y_train,
        y_external=y_external,
        k_neighbors=args.k_neighbors,
        z_threshold=args.z_threshold,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    s = results['ad_summary']
    print(f"\n{'=' * 55}")
    print(f"  Applicability Domain Analysis — {args.model}")
    print(f"{'=' * 55}")
    print(f"  Total compounds assessed:  {s['n_total']}")
    print("")
    print("  Williams Plot:")
    print(f"    Leverage threshold h*:   {s['h_star']:.6f}")
    print(f"    High leverage:           {s['n_williams_high_leverage']}")
    print(f"    High residual:           {s['n_williams_high_residual']}")
    print(f"    Total warnings:          {s['n_williams_warning']}")
    print("")
    print(f"  k-NN Distance (k={args.k_neighbors}):")
    print(f"    Training dist (mean):    {s['knn_training_mean']:.4f}")
    print(f"    Warning threshold:       {s['knn_threshold']:.4f}")
    print(f"    Warnings:                {s['n_knn_warning']}")
    print("")
    print(f"  Combined warnings:         {s['n_combined_warning']}")
    print(f"  Percent flagged:           {100.0 * s['n_combined_warning'] / max(s['n_total'], 1):.1f}%")
    print("\n  Output files:")
    for fp in results['output_files']:
        print(f"    - {fp}")
    print(f"{'=' * 55}\n")


if __name__ == '__main__':
    main()
