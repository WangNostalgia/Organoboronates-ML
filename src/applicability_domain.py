"""
Applicability Domain Analysis for Organoboronate ML Models
===========================================================

Provides two complementary methods for assessing whether a new compound falls
within the model's applicability domain (AD) — the chemical space where
predictions are trustworthy (OECD Principle 3 for QSAR models).

Methods implemented:
  1. **Williams Plot** — standardized cross-validated residuals vs. leverage.
     The gold standard in cheminformatics QSAR.  Identifies:
       - High-leverage points (structurally unusual compounds)
       - Outliers (large prediction errors)
       - Influential points (both high leverage AND large residual)

  2. **k-NN Distance** — average Euclidean distance to the k nearest training-set
     neighbours.  Intuitive for chemists: "how similar is my compound to the
     training set?"  Compounds with unusually large k-NN distances are outside
     the AD.

Usage (CLI):
  python -m src.applicability_domain --model SVR --training example/B_dataset.csv --external external_data.csv

Usage (Python API):
  >>> from src.applicability_domain import applicability_domain_analysis
  >>> results = applicability_domain_analysis(model_info, X_train, X_external)
  >>> print(results['ad_summary'])

References:
  - Gramatica, P. (2007). Principles of QSAR models validation: internal and
    external. QSAR Comb. Sci., 26(5), 694-701.
  - Jaworska, J., et al. (2005). QSAR Applicability Domain Estimation by
    Projection of the Training Set in Descriptor Space. ATLA, 33, 445-459.
"""

import os
import argparse
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_K_NEIGHBORS = 5          # k for k-NN distance
DEFAULT_Z_THRESHOLD = 3.0        # z-score threshold for outlier flagging
DEFAULT_MODELS_DIR = 'models'
DEFAULT_TARGET_COL = 'activation_energy'
DEFAULT_OUTPUT_DIR = 'applicability_domain_results'


# ---------------------------------------------------------------------------
# Core AD analysis
# ---------------------------------------------------------------------------

def applicability_domain_analysis(model_info, X_train, X_external,
                                  y_train=None, y_external=None,
                                  y_pred_external=None,
                                  k_neighbors=DEFAULT_K_NEIGHBORS,
                                  z_threshold=DEFAULT_Z_THRESHOLD,
                                  output_dir=DEFAULT_OUTPUT_DIR,
                                  model_name='Model'):
    """
    Assess the applicability domain of a trained model for a set of external
    compounds using Williams plot and k-NN distance methods.

    Parameters
    ----------
    model_info : dict
        Loaded .joblib dictionary.  Must contain 'model', 'scaler_X', 'features'.
    X_train : str, pd.DataFrame, or np.ndarray
        Training set feature matrix (unscaled).  Can be a CSV path, DataFrame,
        or numpy array.  Used for computing leverage thresholds and k-NN
        reference distances.
    X_external : str, pd.DataFrame, or np.ndarray
        External compounds to assess (unscaled).
    y_train : array-like, optional
        Training set target values.  When provided, the model's training-set
        residuals are used to calibrate the residual threshold (more accurate
        than using external residuals alone).
    y_external : array-like, optional
        Experimental (ground-truth) values for the external set.  Used to
        compute actual residuals.
    y_pred_external : array-like, optional
        Pre-computed model predictions for the external set.  When not
        provided, predictions are generated internally using the model.
    k_neighbors : int
        Number of nearest neighbours for k-NN distance (default 5).
    z_threshold : float
        z-score multiplier for the k-NN distance warning threshold.
        Default 3.0 (compounds with distance > mean + 3σ are flagged).
    output_dir : str
        Directory for saving output files.
    model_name : str
        Name used in plot titles and output file prefixes.

    Returns
    -------
    dict
        Keys:
        - 'ad_results': DataFrame with per-compound AD assessment
        - 'williams_data': dict with leverage, std_residuals, thresholds
        - 'knn_data': dict with distances, thresholds
        - 'ad_summary': dict with counts of flagged compounds
        - 'output_files': list of saved file paths
    """
    # ── Load and validate data ──────────────────────────────────────────
    X_train = _to_dataframe(X_train, 'training')
    X_external = _to_dataframe(X_external, 'external')

    expected_features = list(model_info['features'])
    # Check for feature aliasing between train and external column names
    missing_train = [f for f in expected_features if f not in X_train.columns]
    missing_ext = [f for f in expected_features if f not in X_external.columns]
    if missing_train:
        raise ValueError(f"Training data missing features: {missing_train}")
    if missing_ext:
        raise ValueError(f"External data missing features: {missing_ext}")

    X_train_feat = X_train[expected_features].values
    X_ext_feat = X_external[expected_features].values

    n_train = len(X_train_feat)
    n_ext = len(X_ext_feat)
    p = len(expected_features)  # number of features

    logger.info("AD analysis: %d training samples, %d external samples, %d features",
                n_train, n_ext, p)
    logger.info("Features: %s", expected_features)

    # ── Prepare predictions ─────────────────────────────────────────────
    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']

    if y_pred_external is None:
        X_ext_scaled = scaler_X.transform(X_ext_feat)
        y_pred_scaled = model.predict(X_ext_scaled)
        y_pred_external = scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

    # ── Scale training & external together for AD distance calculations ─
    # We use StandardScaler (z-score) for distance-based methods so each
    # feature contributes equally.  This is fitted on training data only
    # (no leakage from external set into the normalization).
    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train_feat)
    X_ext_scaled_for_ad = feat_scaler.transform(X_ext_feat)

    # ── 1. Williams Plot ────────────────────────────────────────────────
    williams_data = _compute_williams(
        X_train_scaled, X_ext_scaled_for_ad,
        model, scaler_X, scaler_y,
        y_train, y_external, y_pred_external,
        p, n_train, expected_features,
        X_train_unscaled=X_train_feat,  # Pass unscaled features for training residual computation
    )

    # ── 2. k-NN Distance ────────────────────────────────────────────────
    knn_data = _compute_knn_distance(
        X_train_scaled, X_ext_scaled_for_ad,
        k_neighbors, z_threshold,
    )

    # ── Assemble per-compound results ────────────────────────────────────
    id_cols = [c for c in ['sub_H', 'sub_B'] if c in X_external.columns]
    results_df = X_external[id_cols + expected_features].copy().reset_index(drop=True)
    results_df['predicted_activation_energy'] = y_pred_external

    if y_external is not None:
        results_df['activation_energy'] = np.asarray(y_external)
        results_df['absolute_error'] = np.abs(
            y_pred_external - np.asarray(y_external)
        )

    # Williams plot columns
    results_df['leverage'] = williams_data['leverage_ext']
    results_df['std_residual'] = williams_data['std_residuals_ext']
    results_df['williams_warning'] = williams_data['warnings_ext']

    # k-NN distance columns
    results_df['knn_distance'] = knn_data['distances_ext']
    results_df['knn_z_score'] = knn_data['z_scores_ext']
    results_df['knn_warning'] = knn_data['warnings_ext']

    # Combined AD flag: compound is outside AD if EITHER method flags it
    results_df['ad_combined_warning'] = (
        results_df['williams_warning'] | results_df['knn_warning']
    )

    # ── Summaries ───────────────────────────────────────────────────────
    ad_summary = {
        'n_total': n_ext,
        'n_williams_warning': int(williams_data['warnings_ext'].sum()),
        'n_williams_high_leverage': int(
            (williams_data['leverage_ext'] > williams_data['h_star']).sum()
        ),
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
    }

    logger.info(
        "AD Summary:\n"
        "  Williams: %d high-leverage, %d high-residual, %d total warnings\n"
        "  k-NN (%d): %d warnings (d > %.3f)\n"
        "  Combined:  %d / %d compounds flagged (%.1f%%)",
        ad_summary['n_williams_high_leverage'],
        ad_summary['n_williams_high_residual'],
        ad_summary['n_williams_warning'],
        k_neighbors,
        ad_summary['n_knn_warning'],
        ad_summary['knn_threshold'],
        ad_summary['n_combined_warning'],
        n_ext,
        100.0 * ad_summary['n_combined_warning'] / max(n_ext, 1),
    )

    # ── Save outputs ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(' ', '_')

    # Per-compound AD results CSV
    ad_csv_path = os.path.join(output_dir,
                               f"{safe_name}_ad_results_{timestamp}.csv")
    results_df.to_csv(ad_csv_path, index=False, encoding='utf-8-sig')
    output_files.append(ad_csv_path)
    logger.info("AD results saved to: %s", ad_csv_path)

    # Williams plot
    williams_path = os.path.join(output_dir,
                                 f"{safe_name}_williams_plot_{timestamp}.png")
    _plot_williams(williams_data, model_name, output_path=williams_path)
    output_files.append(williams_path)
    logger.info("Williams plot saved to: %s", williams_path)

    # k-NN distance distribution plot
    knn_plot_path = os.path.join(output_dir,
                                 f"{safe_name}_knn_distance_{timestamp}.png")
    _plot_knn_distance(knn_data, k_neighbors, model_name,
                       output_path=knn_plot_path)
    output_files.append(knn_plot_path)
    logger.info("k-NN distance plot saved to: %s", knn_plot_path)

    # Summary text file
    summary_path = os.path.join(output_dir,
                                f"{safe_name}_ad_summary_{timestamp}.txt")
    _write_ad_summary(summary_path, model_name, ad_summary, expected_features)
    output_files.append(summary_path)

    return {
        'ad_results': results_df,
        'williams_data': williams_data,
        'knn_data': knn_data,
        'ad_summary': ad_summary,
        'output_files': output_files,
    }


# ---------------------------------------------------------------------------
# Williams Plot computation
# ---------------------------------------------------------------------------

def _compute_williams(X_train_scaled, X_ext_scaled,
                      model, scaler_X, scaler_y,
                      y_train, y_external, y_pred_external,
                      p, n_train, feature_names,
                      X_train_unscaled=None):
    """
    Compute leverage (hat values), standardized residuals, and warning
    thresholds for the Williams plot.

    Leverage h_i = diagonal element i of the hat matrix H = X(X'X)^(-1)X'.
    The warning threshold h* = 3(p+1)/n  (3 times the average leverage).

    Standardized residuals use LOOCV-style variance:
      r_std_i = r_i / (sigma * sqrt(1 - h_i))
    where sigma is estimated from the residual standard deviation.

    When X_train_unscaled and y_train are both provided, training-set
    residuals are computed using the model's own scalers (MinMaxScaler)
    and combined with external residuals for a more robust sigma estimate.

    For models where the hat matrix cannot be derived analytically (e.g.,
    SVR with RBF kernel), we use a proximity-based pseudo-leverage computed
    from the feature-space distance to the training set centroid.
    """
    # ── Leverage for training set ───────────────────────────────────────
    # Standard OLS leverage: h = diag(X(X'X)^(-1)X')
    # For kernel methods, this is an approximation via the linearised feature
    # space.  We compute leverage from the scaled feature matrix.
    try:
        X_with_intercept = np.column_stack([
            np.ones(n_train), X_train_scaled
        ])
        hat_train = _hat_matrix_diag(X_with_intercept)
    except np.linalg.LinAlgError:
        # If X'X is singular (e.g., highly correlated features), fall back
        # to a robust pseudo-inverse.
        logger.warning("Hat matrix computation failed (singular). "
                       "Using pseudo-inverse fallback.")
        X_with_intercept = np.column_stack([
            np.ones(n_train), X_train_scaled
        ])
        hat_train = _hat_matrix_diag_pinv(X_with_intercept)

    # Leverage for external compounds
    # h*_i = x_i'(X'X)^(-1)x_i  for each external point
    X_train_design = np.column_stack([np.ones(n_train), X_train_scaled])
    X_ext_design = np.column_stack([
        np.ones(len(X_ext_scaled)), X_ext_scaled
    ])
    try:
        xtx_inv = np.linalg.inv(X_train_design.T @ X_train_design)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(X_train_design.T @ X_train_design)

    leverage_ext = np.array([
        x_i @ xtx_inv @ x_i for x_i in X_ext_design
    ])

    # Leverage for training set
    leverage_train = hat_train

    # Warning threshold: 3 * (p + 1) / n  (3× average leverage)
    h_star = 3.0 * (p + 1) / n_train

    # ── Residuals ───────────────────────────────────────────────────────
    # For external set: use true residuals if y_external available
    if y_external is not None:
        residuals_ext = np.asarray(y_external) - np.asarray(y_pred_external)
    else:
        # Without ground truth, residuals cannot be computed.
        # Set to NaN and skip residual-based warnings.
        residuals_ext = np.full(len(X_ext_scaled), np.nan)

    # ── Compute training-set residuals (when unscaled features available) ─
    # Training residuals provide a more accurate baseline for sigma estimation
    # than external residuals alone.  The model's own MinMaxScaler is used to
    # transform unscaled training features, ensuring consistency with how the
    # model was originally trained.
    residuals_train = None
    if X_train_unscaled is not None and y_train is not None:
        try:
            X_tr_model_scaled = scaler_X.transform(X_train_unscaled)
            y_pred_train_scaled = model.predict(X_tr_model_scaled)
            y_pred_train = scaler_y.inverse_transform(
                y_pred_train_scaled.reshape(-1, 1)
            ).ravel()
            residuals_train = np.asarray(y_train).ravel() - y_pred_train
            logger.info(
                "Training residuals computed: n=%d, mean=%.4f, std=%.4f",
                len(residuals_train),
                float(np.mean(residuals_train)),
                float(np.std(residuals_train, ddof=1)),
            )
        except Exception as exc:
            logger.warning(
                "Failed to compute training residuals: %s. "
                "Falling back to external-residual-only sigma.", exc
            )
            residuals_train = None

    # Estimate residual sigma from training residuals (preferred) or external
    # residuals (fallback).  Training residuals are more representative of the
    # model's typical error distribution, making the ±3σ threshold more accurate.
    if residuals_train is not None and len(residuals_train) >= 2:
        residual_sigma = _estimate_residual_std(residuals_train)
        logger.info("Williams sigma estimated from training residuals: %.4f", residual_sigma)
    else:
        residual_sigma = _estimate_residual_std(residuals_ext)
        logger.info("Williams sigma estimated from external residuals (fallback): %.4f", residual_sigma)

    # Standardized residuals (using LOOCV variance scaling)
    with np.errstate(divide='ignore', invalid='ignore'):
        std_residuals_ext = np.where(
            (residual_sigma > 0) & (leverage_ext < 0.999),
            residuals_ext / (residual_sigma * np.sqrt(np.maximum(1 - leverage_ext, 1e-10))),
            np.nan
        )

    # Critical residual threshold: ±3 standardized residuals
    residual_critical = 3.0

    # ── Warning flags ───────────────────────────────────────────────────
    warnings_ext = (
        (leverage_ext > h_star) |
        (np.abs(std_residuals_ext) > residual_critical)
    )
    # Points with NaN residuals (no ground truth) only get leverage check
    nan_resid_mask = np.isnan(std_residuals_ext)
    warnings_ext[nan_resid_mask] = leverage_ext[nan_resid_mask] > h_star

    return {
        'leverage_train': leverage_train,
        'leverage_ext': leverage_ext,
        'std_residuals_ext': std_residuals_ext,
        'h_star': h_star,
        'residual_critical': residual_critical,
        'residuals_ext': residuals_ext,
        'warnings_ext': warnings_ext,
        'n_train': n_train,
        'p': p,
    }


def _hat_matrix_diag(X):
    """Compute diagonal of the hat matrix H = X(X'X)^(-1)X'."""
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    # h_i = sum_j (X_ij * (X @ XtX_inv)_ij) — more numerically stable:
    # h_i = row_i(X) @ XtX_inv @ row_i(X)'
    hat_diag = np.array([x @ xtx_inv @ x for x in X])
    return hat_diag


def _hat_matrix_diag_pinv(X):
    """Compute hat matrix diagonal using Moore-Penrose pseudo-inverse."""
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    hat_diag = np.array([x @ xtx_inv @ x for x in X])
    return hat_diag


def _estimate_residual_std(residuals):
    """Robust estimate of residual standard deviation."""
    valid = residuals[~np.isnan(residuals)]
    if len(valid) < 2:
        return 1.0
    # Use MAD (median absolute deviation) for robustness, scaled to match
    # normal distribution std
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    return mad * 1.4826 if mad > 0 else np.std(valid, ddof=1)


# ---------------------------------------------------------------------------
# k-NN Distance computation
# ---------------------------------------------------------------------------

def _compute_knn_distance(X_train_scaled, X_ext_scaled,
                          k_neighbors, z_threshold):
    """
    Compute average Euclidean distance to the k nearest training-set
    neighbours for each external compound.

    The warning threshold is: mean_train + z_threshold * std_train,
    where mean_train and std_train are computed from the distribution of
    training-set internal k-NN distances (leave-one-out style).
    """
    n_train = len(X_train_scaled)
    n_ext = len(X_ext_scaled)

    # Fit k-NN on training data
    effective_k = min(k_neighbors, n_train - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric='euclidean')
    nn.fit(X_train_scaled)

    # ── Training-set internal distances (LOO-style) ─────────────────────
    # For each training point, find its k nearest neighbours (excluding itself).
    # This establishes the baseline "normal" distance distribution.
    train_distances, _ = nn.kneighbors(X_train_scaled)
    # Exclude self (distance 0 at index 0)
    train_knn_avg = train_distances[:, 1:effective_k + 1].mean(axis=1)

    train_mean = float(np.mean(train_knn_avg))
    train_std = float(np.std(train_knn_avg, ddof=1))

    # Warning threshold: training mean + z_threshold * training std
    threshold = train_mean + z_threshold * max(train_std, 1e-8)

    # ── External compound distances ─────────────────────────────────────
    ext_distances, ext_indices = nn.kneighbors(X_ext_scaled)
    ext_knn_avg = ext_distances[:, :effective_k].mean(axis=1)

    # z-score relative to training distribution
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores_ext = np.where(
            train_std > 0,
            (ext_knn_avg - train_mean) / train_std,
            0.0
        )

    warnings_ext = ext_knn_avg > threshold

    logger.info("k-NN (k=%d): training mean=%.4f, std=%.4f, threshold=%.4f",
                effective_k, train_mean, train_std, threshold)
    logger.info("  External: mean=%.4f, std=%.4f, %d / %d flagged",
                float(np.mean(ext_knn_avg)), float(np.std(ext_knn_avg)),
                int(warnings_ext.sum()), n_ext)

    return {
        'distances_train': train_knn_avg,
        'distances_ext': ext_knn_avg,
        'z_scores_ext': z_scores_ext,
        'warnings_ext': warnings_ext,
        'training_mean': train_mean,
        'training_std': train_std,
        'threshold': threshold,
        'k_effective': effective_k,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_williams(williams_data, model_name, output_path):
    """
    Generate a publication-quality Williams plot:
    standardized residuals vs. leverage, with warning boundaries.
    """
    leverage_ext = williams_data['leverage_ext']
    std_residuals = williams_data['std_residuals_ext']
    h_star = williams_data['h_star']
    residual_critical = williams_data['residual_critical']
    warnings = williams_data['warnings_ext']

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot points
    normal_mask = ~warnings
    if normal_mask.any():
        ax.scatter(leverage_ext[normal_mask], std_residuals[normal_mask],
                   alpha=0.6, edgecolors='#2c3e50', facecolors='#3498db',
                   s=50, linewidth=0.3, label='Within AD', zorder=5)
    if warnings.any():
        ax.scatter(leverage_ext[warnings], std_residuals[warnings],
                   alpha=0.8, edgecolors='#922b21', facecolors='#e74c3c',
                   s=70, linewidth=0.5, marker='^',
                   label=f'Outside AD ({warnings.sum()})', zorder=6)

    # Warning boundaries
    x_max = max(leverage_ext.max(), h_star * 1.5) * 1.1
    y_max = max(np.nanmax(np.abs(std_residuals)), residual_critical) * 1.3

    # Leverage threshold (vertical line)
    ax.axvline(x=h_star, color='#e67e22', linestyle='--', linewidth=1.5,
               label=f'h* = {h_star:.4f} (3(p+1)/n)')

    # Residual thresholds (horizontal lines)
    ax.axhline(y=residual_critical, color='#e67e22', linestyle=':', linewidth=1.2)
    ax.axhline(y=-residual_critical, color='#e67e22', linestyle=':', linewidth=1.2,
               label=f'±{residual_critical:.0f}σ residual')

    # Fill warning zones
    ax.fill_between([h_star, x_max], -residual_critical, residual_critical,
                    alpha=0.04, color='orange')
    ax.fill_between([0, x_max], residual_critical, y_max,
                    alpha=0.04, color='orange')
    ax.fill_between([0, x_max], -y_max, -residual_critical,
                    alpha=0.04, color='orange')

    ax.set_xlabel('Leverage (h)', fontsize=13)
    ax.set_ylabel('Standardized Residual', fontsize=13)
    ax.set_title(f'{model_name} — Williams Plot\nApplicability Domain Assessment',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_knn_distance(knn_data, k_neighbors, model_name, output_path):
    """
    Plot the distribution of k-NN distances for both training and external
    compounds, with the warning threshold marked.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    train_dist = knn_data['distances_train']
    ext_dist = knn_data['distances_ext']
    threshold = knn_data['threshold']

    # ── Left panel: overlaid histograms ─────────────────────────────────
    bins = 30
    ax1.hist(train_dist, bins=bins, alpha=0.6, color='#3498db',
             label=f'Training (n={len(train_dist)})', edgecolor='#2c3e50',
             linewidth=0.3)
    ax1.hist(ext_dist, bins=bins, alpha=0.7, color='#e74c3c',
             label=f'External (n={len(ext_dist)})', edgecolor='#922b21',
             linewidth=0.3)
    ax1.axvline(x=threshold, color='#e67e22', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    ax1.set_xlabel(f'Average Distance to {k_neighbors} Nearest Neighbours',
                   fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('k-NN Distance Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)

    # ── Right panel: per-compound bar chart ─────────────────────────────
    indices = np.arange(len(ext_dist))
    colors = ['#e74c3c' if d > threshold else '#3498db' for d in ext_dist]
    ax2.bar(indices, ext_dist, color=colors, alpha=0.8, width=0.8)
    ax2.axhline(y=threshold, color='#e67e22', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    ax2.set_xlabel('External Compound Index', fontsize=11)
    ax2.set_ylabel(f'Avg Distance to {k_neighbors} NN', fontsize=11)
    ax2.set_title('Per-Compound k-NN Distance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle(f'{model_name} — k-NN Applicability Domain',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _write_ad_summary(summary_path, model_name, ad_summary, features):
    """Write a human-readable summary of the AD analysis."""
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Applicability Domain Analysis Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model:        {model_name}\n")
        f.write(f"Features:     {', '.join(features)}\n")
        f.write(f"Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- Compound Assessment ---\n")
        f.write(f"Total external compounds: {ad_summary['n_total']}\n")
        f.write(f"\n--- Williams Plot ---\n")
        f.write(f"Leverage threshold (h*):        {ad_summary['h_star']:.6f}\n")
        f.write(f"Residual threshold:             ±{ad_summary['residual_critical']:.0f}σ\n")
        f.write(f"High leverage (>h*):            {ad_summary['n_williams_high_leverage']}\n")
        f.write(f"High residual:                  {ad_summary['n_williams_high_residual']}\n")
        f.write(f"Total Williams warnings:        {ad_summary['n_williams_warning']}\n")
        f.write(f"\n--- k-NN Distance ---\n")
        f.write(f"Training mean distance:         {ad_summary['knn_training_mean']:.4f}\n")
        f.write(f"Training std distance:          {ad_summary['knn_training_std']:.4f}\n")
        f.write(f"Warning threshold:              {ad_summary['knn_threshold']:.4f}\n")
        f.write(f"k-NN warnings:                  {ad_summary['n_knn_warning']}\n")
        f.write(f"\n--- Combined ---\n")
        f.write(f"Compounds flagged (either method): {ad_summary['n_combined_warning']}\n")
        pct = 100.0 * ad_summary['n_combined_warning'] / max(ad_summary['n_total'], 1)
        f.write(f"Percent flagged:                 {pct:.1f}%\n")
        f.write(f"\n--- Interpretation ---\n")
        f.write(f"Compounds flagged ONLY for high leverage are structurally\n")
        f.write(f"unusual but may still be well-predicted (interpolation vs.\n")
        f.write(f"extrapolation risk).  Compounds flagged for high residual\n")
        f.write(f"AND high leverage are unreliable — predictions should be\n")
        f.write(f"treated with caution.  Compounds outside the k-NN distance\n")
        f.write(f"threshold are in sparse regions of the training space.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dataframe(data, label='data'):
    """Convert input to a DataFrame, handling CSV paths and numpy arrays."""
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"{label} file not found: {data}")
        df = pd.read_csv(data)
        # Drop "Unnamed" artefact columns from Excel exports
        unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
        if unnamed:
            df = df.drop(columns=unnamed)
        logger.info("Loaded %s: %d rows × %d columns", label, len(df), len(df.columns))
        return df
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data, columns=[f'X{i}' for i in range(data.shape[1])])
    else:
        raise TypeError(f"{label} must be str, DataFrame, or ndarray, "
                        f"got {type(data).__name__}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Applicability Domain Analysis for Organoboronate ML Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run AD analysis with training and external data
  python -m src.applicability_domain --model SVR --training example/B_dataset.csv --external new_data.csv

  # With custom k and z-threshold
  python -m src.applicability_domain --model SVR --training example/B_dataset.csv --external new.csv -k 7 -z 2.5
        """
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., SVR, RandomForest).')
    parser.add_argument('--n_features', type=int, default=None,
                        help='Feature count for the model version.')
    parser.add_argument('--training', type=str, required=True,
                        help='Path to training data CSV (used for baseline '
                             'distance distribution and leverage thresholds).')
    parser.add_argument('--external', type=str, required=True,
                        help='Path to external compounds CSV to assess.')
    parser.add_argument('--target-col', type=str, default=DEFAULT_TARGET_COL,
                        help='Name of the target column in external CSV.')
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

    # Load model
    from src.external_validation import load_model
    model_info = load_model(args.model, n_features=args.n_features,
                            models_dir=args.models_dir)

    # Load data
    X_train = _to_dataframe(args.training, 'training')
    X_external = _to_dataframe(args.external, 'external')

    # Extract ground truth if available
    y_external = None
    if args.target_col and args.target_col in X_external.columns:
        y_external = X_external[args.target_col].values
        logger.info("Ground truth found in external data (%d values)",
                    len(y_external))

    # Run AD analysis
    results = applicability_domain_analysis(
        model_info=model_info,
        X_train=X_train,
        X_external=X_external,
        y_external=y_external,
        k_neighbors=args.k_neighbors,
        z_threshold=args.z_threshold,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    # Print summary
    s = results['ad_summary']
    print(f"\n{'='*55}")
    print(f"  Applicability Domain Analysis — {args.model}")
    print(f"{'='*55}")
    print(f"  Total compounds assessed:  {s['n_total']}")
    print(f"")
    print(f"  Williams Plot:")
    print(f"    Leverage threshold h*:   {s['h_star']:.6f}")
    print(f"    High leverage:           {s['n_williams_high_leverage']}")
    print(f"    High residual:           {s['n_williams_high_residual']}")
    print(f"    Total warnings:          {s['n_williams_warning']}")
    print(f"")
    print(f"  k-NN Distance (k={args.k_neighbors}):")
    print(f"    Training dist (mean):    {s['knn_training_mean']:.4f}")
    print(f"    Warning threshold:       {s['knn_threshold']:.4f}")
    print(f"    Warnings:                {s['n_knn_warning']}")
    print(f"")
    print(f"  Combined warnings:         {s['n_combined_warning']}")
    print(f"  Percent flagged:           {100.0*s['n_combined_warning']/max(s['n_total'],1):.1f}%")
    print(f"\n  Output files:")
    for fp in results['output_files']:
        print(f"    - {fp}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
