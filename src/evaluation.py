"""
Unified evaluation center (Single Source of Truth for 5脳5 RepeatedKFold).

All modules that need a rigorous, leakage-free 5脳5 RepeatedKFold evaluation
MUST call repeated_kfold_evaluate() from here.  This eliminates code duplication
and ensures consistent per-fold scaling + inverse-transform to kcal/mol.

Used by: train_and_evaluate.py, y_randomization.py, iterative_optimization.py
"""

import logging

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def make_repeated_kfold_splits(X, n_splits=5, n_repeats=5, random_state=42):
    """
    Generate reusable RepeatedKFold train/test index pairs.
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return list(rkf.split(X_arr))


def repeated_kfold_evaluate(model, X, y, n_splits=5, n_repeats=5, random_state=42, splits=None):
    """
    PRIMARY evaluation center for the entire project.

    Every fold independently fits its own MinMaxScaler(feature_range=(0,100))
    on the training portion ONLY, then inverse-transforms predictions back to
    the original kcal/mol scale before computing MAE.  This prevents data
    leakage and ensures MAE is reported in physically meaningful units.

    Parameters
    ----------
    model : sklearn model instance (cloned per fold)
    X : DataFrame or array, feature matrix (unscaled)
    y : Series or array, target vector
    n_splits : int, number of folds (default 5)
    n_repeats : int, number of repeats (default 5 -> 25 total evaluations)
    random_state : int
    splits : iterable of (train_idx, test_idx) or None
        Optional precomputed splits to reuse as-is.

    Returns
    -------
    dict with keys:
        'rkf_mae_mean', 'rkf_mae_std', 'rkf_r2_mean', 'rkf_r2_std', 'n_evals'
        (plus y-randomization compatible aliases: 'mae_mean', 'mae_std', 'r2_mean', 'r2_std')
    """
    y_orig = y.values.ravel() if hasattr(y, 'values') else np.asarray(y).ravel()
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    mae_scores, r2_scores = [], []

    split_iterable = splits
    if split_iterable is None:
        split_iterable = make_repeated_kfold_splits(
            X_arr,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    for train_idx, test_idx in split_iterable:
        X_tr_raw, X_te_raw = X_arr[train_idx], X_arr[test_idx]
        y_tr_orig, y_te_orig = y_orig[train_idx], y_orig[test_idx]

        fold_sX = MinMaxScaler()
        X_tr = fold_sX.fit_transform(X_tr_raw)
        X_te = fold_sX.transform(X_te_raw)

        fold_sY = MinMaxScaler(feature_range=(0, 100))
        y_tr_s = fold_sY.fit_transform(y_tr_orig.reshape(-1, 1)).ravel()

        m = clone(model)
        m.fit(X_tr, y_tr_s)

        y_pred_s = m.predict(X_te)
        y_pred_orig = fold_sY.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        mae_scores.append(mean_absolute_error(y_te_orig, y_pred_orig))
        r2_scores.append(r2_score(y_te_orig, y_pred_orig))

    return {
        'rkf_mae_mean': float(np.mean(mae_scores)),
        'rkf_mae_std': float(np.std(mae_scores)),
        'rkf_r2_mean': float(np.mean(r2_scores)),
        'rkf_r2_std': float(np.std(r2_scores)),
        'n_evals': len(mae_scores),
        'mae_mean': float(np.mean(mae_scores)),
        'mae_std': float(np.std(mae_scores)),
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
    }
