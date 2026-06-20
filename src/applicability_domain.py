"""
Applicability-domain analysis for trained organoboronate ML models.

The command-line entry point calibrates the AD reference set from the
checkpoint's saved development_indices by default, so final-test rows are not
silently reused as part of the training domain.
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
DEFAULT_METADATA_COLUMNS = ('ID', 'SMILES', 'filename')

LINEAR_LEVERAGE_MODELS = {
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
}


def _identifier_columns(df):
    return [col for col in DEFAULT_METADATA_COLUMNS if col in df.columns]


def _unique_preserving_order(columns):
    seen = set()
    result = []
    for col in columns:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def applicability_domain_analysis(model_info, X_train, X_external,
                                  y_train=None, y_external=None,
                                  y_pred_external=None,
                                  k_neighbors=DEFAULT_K_NEIGHBORS,
                                  z_threshold=DEFAULT_Z_THRESHOLD,
                                  output_dir=DEFAULT_OUTPUT_DIR,
                                  model_name='Model'):
    """Assess the applicability domain of a trained model for an external set."""
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

    result_cols = _unique_preserving_order(_identifier_columns(X_external) + expected_features)
    results_df = X_external[result_cols].copy().reset_index(drop=True)
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
    """Compute descriptor-space leverage and calibrated standardized residuals."""
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
    """Generate one OOF prediction per training sample in kcal/mol."""
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
    """Robust Williams residual scale from finite 1-D OOF residuals."""
    residuals_arr = np.asarray(residuals, dtype=float)
    residuals_arr = residuals_arr[np.isfinite(residuals_arr)]
    if residuals_arr.size == 0:
        raise ValueError("Cannot calibrate residual scale from empty residuals.")
    median = np.median(residuals_arr)
    mad = np.median(np.abs(residuals_arr - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(residuals_arr, ddof=1)) if residuals_arr.size > 1 else 0.0
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = np.finfo(float).eps
    return float(scale)


def _compute_knn_distance(X_train_scaled, X_ext_scaled, k_neighbors, z_threshold):
    """Compute k-NN descriptor-space distance warnings."""
    train_neighbor_count = min(k_neighbors + 1, len(X_train_scaled))
    train_nn = NearestNeighbors(n_neighbors=train_neighbor_count, n_jobs=1)
    train_nn.fit(X_train_scaled)
    train_distances, _ = train_nn.kneighbors(X_train_scaled)
    if train_neighbor_count > 1:
        train_reference_dist = train_distances[:, 1:].mean(axis=1)
    else:
        train_reference_dist = train_distances[:, 0]

    training_mean = float(np.mean(train_reference_dist))
    training_std = float(np.std(train_reference_dist, ddof=1)) if len(train_reference_dist) > 1 else 0.0
    if not np.isfinite(training_std):
        training_std = 0.0
    threshold = training_mean + float(z_threshold) * training_std

    ext_neighbor_count = min(k_neighbors, len(X_train_scaled))
    ext_nn = NearestNeighbors(n_neighbors=ext_neighbor_count, n_jobs=1)
    ext_nn.fit(X_train_scaled)
    ext_distances, _ = ext_nn.kneighbors(X_ext_scaled)
    distances_ext = ext_distances.mean(axis=1)

    if training_std > np.finfo(float).eps:
        z_scores_ext = (distances_ext - training_mean) / training_std
    else:
        z_scores_ext = np.where(distances_ext > training_mean, np.inf, 0.0)
    warnings_ext = distances_ext > threshold

    return {
        'train_distances': train_reference_dist,
        'distances_ext': distances_ext,
        'z_scores_ext': z_scores_ext,
        'threshold': threshold,
        'training_mean': training_mean,
        'training_std': training_std,
        'warnings_ext': warnings_ext,
    }


def _plot_williams(williams_data, model_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    leverage_ext = williams_data['leverage_ext']
    std_resid = williams_data['std_residuals_ext']
    y_values = np.nan_to_num(std_resid, nan=0.0, posinf=0.0, neginf=0.0)
    ax.scatter(leverage_ext, y_values, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.axvline(williams_data['h_star'], linestyle='--', linewidth=1.2, label='h*')
    ax.axhline(williams_data['residual_critical'], linestyle=':', linewidth=1.0)
    ax.axhline(-williams_data['residual_critical'], linestyle=':', linewidth=1.0)
    ax.set_xlabel(williams_data['leverage_xlabel'])
    ax.set_ylabel('Standardized residual')
    ax.set_title(f'Williams plot — {model_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_knn_distance(knn_data, k_neighbors, model_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(knn_data['distances_ext']))
    ax.scatter(x, knn_data['distances_ext'], alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.axhline(knn_data['threshold'], linestyle='--', linewidth=1.2, label='threshold')
    ax.set_xlabel('External sample index')
    ax.set_ylabel(f'Mean distance to {k_neighbors} nearest training neighbours')
    ax.set_title(f'k-NN AD distance — {model_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _write_ad_summary(path, model_name, ad_summary, features):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Applicability Domain Summary\n")
        f.write("============================\n\n")
        f.write(f"Model:                          {model_name}\n")
        f.write(f"Features ({len(features)}):              {', '.join(features)}\n")
        f.write(f"Total external samples:          {ad_summary['n_total']}\n")
        f.write("\n--- Williams Plot ---\n")
        f.write(f"Leverage threshold h*:           {ad_summary['h_star']:.6f}\n")
        f.write(f"High leverage:                   {ad_summary['n_williams_high_leverage']}\n")
        f.write(f"High residual:                   {ad_summary['n_williams_high_residual']}\n")
        f.write(f"Williams warnings:               {ad_summary['n_williams_warning']}\n")
        f.write(f"Note:                            {ad_summary['leverage_note']}\n")
        f.write("\n--- k-NN Distance ---\n")
        f.write(f"Training mean distance:         {ad_summary['knn_training_mean']:.4f}\n")
        f.write(f"Training std distance:          {ad_summary['knn_training_std']:.4f}\n")
        f.write(f"Warning threshold:              {ad_summary['knn_threshold']:.4f}\n")
        f.write(f"k-NN warnings:                  {ad_summary['n_knn_warning']}\n")
        f.write("\n--- Combined ---\n")
        f.write(f"Compounds flagged:              {ad_summary['n_combined_warning']}\n")
        flagged_percentage = 100.0 * ad_summary['n_combined_warning'] / max(ad_summary['n_total'], 1)
        f.write(f"Percentage flagged:             {flagged_percentage:.1f}%\n")
        f.write("\n--- Interpretation ---\n")
        if ad_summary.get('prediction_only', False):
            f.write(
                "Prediction-only mode: flagged samples fall outside the leverage "
                "and/or k-NN distance criteria.\n"
            )
        elif ad_summary['n_combined_warning']:
            f.write(
                "Flagged samples exceed at least one Williams-plot or k-NN "
                "applicability-domain criterion.\n"
            )
        else:
            f.write(
                "All external samples satisfy the configured Williams-plot and "
                "k-NN applicability-domain criteria.\n"
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


def _subset_training_to_checkpoint_development(training_df, model_info, allow_full_training_csv_for_ad):
    protocol = model_info.get('evaluation_protocol') or {}
    development_indices = protocol.get('development_indices')

    if allow_full_training_csv_for_ad:
        logger.warning(
            "Using the full --training CSV for AD calibration because "
            "--allow-full-training-csv-for-ad was provided. This can mix final-test "
            "rows into the AD reference domain."
        )
        return training_df.copy()

    if not development_indices:
        raise ValueError(
            "Checkpoint does not contain evaluation_protocol['development_indices']; "
            "AD calibration refuses to use the full training CSV by default. Re-run "
            "main.py to create protocol-aware checkpoints, or pass "
            "--allow-full-training-csv-for-ad for explicitly acknowledged legacy use."
        )

    missing = [idx for idx in development_indices if idx not in training_df.index]
    if missing:
        preview = missing[:10]
        raise ValueError(
            "Checkpoint development_indices do not match the supplied --training CSV. "
            f"Missing index labels: {preview}. Use the same, unreordered source CSV "
            "used during main.py training, or pass --allow-full-training-csv-for-ad "
            "only if you intentionally accept full-data AD calibration."
        )

    subset = training_df.loc[development_indices].copy()
    logger.info(
        "AD calibration uses %d checkpoint development rows; final-test rows remain outside the training domain.",
        len(subset),
    )
    return subset


def main():
    parser = argparse.ArgumentParser(
        description='Applicability Domain Analysis for Organoboronate ML Models',
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., SVR, RandomForest).')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Feature count for the model version.')
    parser.add_argument('--training', type=str, required=True,
                        help='Path to the original training-data CSV.')
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
                        help=f'Directory containing trained models. Default: {DEFAULT_MODELS_DIR}')
    parser.add_argument(
        '--allow-full-training-csv-for-ad',
        action='store_true',
        help='Explicitly allow AD calibration on the full --training CSV. By default, '
             'the checkpoint development_indices are used and final-test rows are excluded.',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    from src.external_validation import load_model

    model_info = load_model(args.model, n_features=args.n_features, models_dir=args.models_dir)
    training_df = _to_dataframe(args.training, 'training')
    training_reference = _subset_training_to_checkpoint_development(
        training_df,
        model_info,
        allow_full_training_csv_for_ad=args.allow_full_training_csv_for_ad,
    )
    X_external = _to_dataframe(args.external, 'external')

    if args.target_col not in training_reference.columns:
        raise ValueError(
            f"Training CSV must contain target column '{args.target_col}' so Williams residuals can be calibrated."
        )

    y_train = training_reference[args.target_col].to_numpy(dtype=float)
    X_train = training_reference.drop(columns=[args.target_col])

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
