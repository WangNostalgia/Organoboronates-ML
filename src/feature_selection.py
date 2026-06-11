import logging
import numpy as np
logger = logging.getLogger(__name__)


def shap_rfecv_select_worst_feature(model, X, y, model_name, corr_threshold=0.8, cv_folds=5):
    """
    SHAP-RFECV with multi-fold CV consensus SHAP.

    Instead of relying on a single data split (vulnerable to random noise in
    small-N settings), SHAP importance is averaged across `cv_folds` folds.
    Each fold: model trained on (k-1)/k of X, SHAP computed on held-out 1/k.
    The consensus (mean) importance across folds drives the feature ranking.

    Strategy (two-phase):
      1. HIGH-CORRELATION: If any pair has |r| > corr_threshold, remove the
         less consensus-important one.
      2. LOW-IMPORTANCE: Otherwise, remove the globally least important feature.

    NOTE: This function manages its own internal MinMaxScaler instances per fold
    to guarantee zero data leakage between training and validation splits.

    Parameters
    ----------
    model : sklearn model class or instance
    X : DataFrame, current feature matrix (unscaled)
    y : Series, target vector
    model_name : str, model name for explainer selection
    corr_threshold : float, correlation threshold for collinearity (default 0.8)
    cv_folds : int, number of CV folds for SHAP consensus (0=single-fit, default 5)

    Returns
    -------
    worst_feature : str
    importance_ranking : list of (feature_name, importance_pct) sorted desc
    removal_reason : str, 'high_correlation' or 'low_importance'
    """
    import shap
    from sklearn.model_selection import KFold
    from sklearn.base import clone
    from sklearn.preprocessing import MinMaxScaler

    n_features = X.shape[1]
    feature_names = list(X.columns)
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    y_arr = y.values.ravel() if hasattr(y, 'values') else np.asarray(y).ravel()

    tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'DecisionTree', 'LightGBM', 'CatBoost']
    linear_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']

    # ── SHAP computation: two-tier strategy ──
    # cv_folds=0 → simple single-fit (fast, for coarse filtering with many features)
    # cv_folds≥2 → multi-fold CV consensus (robust, for fine-grained decisions with few features)
    if cv_folds <= 0:
        # ── Simple path: fit once on all data, explain on same data ──
        sX = MinMaxScaler()
        sY = MinMaxScaler(feature_range=(0, 100))
        X_scaled = sX.fit_transform(X_arr)
        y_scaled = sY.fit_transform(y_arr.reshape(-1, 1)).ravel()
        m = clone(model)
        m.fit(X_scaled, y_scaled)

        if model_name in tree_models:
            explainer = shap.TreeExplainer(m)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[0]
        elif model_name in linear_models:
            explainer = shap.LinearExplainer(m, X_scaled)
            sv = explainer.shap_values(X_scaled)
        else:
            # Clamp cluster/sample counts to available data size (prevents ValueError
            # when n_samples > len(X_scaled) on small feature subsets)
            n_clust_desired = max(10, min(15, int(len(X) * 0.15)))
            n_clusters = max(1, min(n_clust_desired, len(X_scaled)))
            n_samp_desired = max(20, min(int(len(X) * 0.3), 100))
            n_samples = max(1, min(n_samp_desired, len(X_scaled)))
            # Local RandomState isolates SHAP from global numpy seed (no pollution)
            rng = np.random.RandomState(42)
            background = shap.kmeans(X_scaled, n_clusters)
            explainer = shap.KernelExplainer(m.predict, background)
            sample_indices = rng.choice(len(X_scaled), n_samples, replace=False)
            sv = explainer.shap_values(X_scaled[sample_indices])

        importances = np.abs(sv).mean(axis=0)
        n_folds_used = 1
    else:
        # ── Multi-fold consensus path ──
        # Accumulate |SHAP| per feature across folds, then average.
        importances = np.zeros(n_features)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        n_folds_used = 0
        # Local RandomState avoids resetting the global numpy seed inside the loop,
        # which would produce identical pseudo-random sequences across folds.
        rng = np.random.RandomState(42)

        for train_idx, test_idx in kf.split(X_arr):
            X_tr_raw, X_te_raw = X_arr[train_idx], X_arr[test_idx]
            y_tr_raw, y_te_raw = y_arr[train_idx], y_arr[test_idx]

            fold_sX = MinMaxScaler()
            fold_sY = MinMaxScaler(feature_range=(0, 100))
            X_tr_s = fold_sX.fit_transform(X_tr_raw)
            X_te_s = fold_sX.transform(X_te_raw)
            y_tr_s = fold_sY.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()

            m = clone(model)
            m.fit(X_tr_s, y_tr_s)

            try:
                if model_name in tree_models:
                    explainer = shap.TreeExplainer(m)
                    sv = explainer.shap_values(X_te_s)
                    if isinstance(sv, list):
                        sv = sv[0]
                elif model_name in linear_models:
                    explainer = shap.LinearExplainer(m, X_te_s)
                    sv = explainer.shap_values(X_te_s)
                else:
                    n_clust_desired = max(5, min(10, int(len(X_te_s) * 0.3)))
                    n_clusters = max(1, min(n_clust_desired, len(X_te_s)))
                    n_samp_desired = max(5, min(int(len(X_te_s) * 0.5), len(X_te_s)))
                    n_samples = max(1, min(n_samp_desired, len(X_te_s)))
                    bg = shap.kmeans(X_te_s, n_clusters)  # shap.kmeans signature: kmeans(X, k, ...) — no random_state param
                    explainer = shap.KernelExplainer(m.predict, bg)
                    sample_idx = rng.choice(len(X_te_s), n_samples, replace=False)
                    sv = explainer.shap_values(X_te_s[sample_idx])

                importances += np.abs(sv).mean(axis=0)
                n_folds_used += 1
            except Exception as e:
                logger.warning("SHAP failed in fold %d: %s. Skipping.", n_folds_used + 1, str(e))
                continue

        if n_folds_used == 0:
            raise RuntimeError("SHAP computation failed in all CV folds.")
        importances /= n_folds_used

    # Normalize to percentages
    total = importances.sum()
    if total > 0:
        importance_pct = (importances / total) * 100
    else:
        importance_pct = np.ones(n_features) * (100.0 / n_features)

    # Build sorted ranking (highest → lowest)
    ranking = sorted(
        zip(feature_names, importance_pct),
        key=lambda x: x[1], reverse=True
    )

    if cv_folds >= 2:
        logger.info("  Multi-fold (n=%d) consensus SHAP ranking:", n_folds_used)
    else:
        logger.info("  Single-fit SHAP ranking:")
    for rank, (fn, fi) in enumerate(ranking, 1):
        logger.info("    [%d] %s: %.2f%%", rank, fn, fi)

    # ── Phase 1: High-correlation pairs ──
    corr_matrix = X.corr().abs()
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if corr_matrix.iloc[i, j] > corr_threshold:
                fi, fj = feature_names[i], feature_names[j]
                imp_i, imp_j = importance_pct[i], importance_pct[j]
                if imp_i < imp_j:
                    high_corr_pairs.append((fi, fj, corr_matrix.iloc[i, j], imp_i, imp_j))
                else:
                    high_corr_pairs.append((fj, fi, corr_matrix.iloc[i, j], imp_j, imp_i))

    if high_corr_pairs:
        high_corr_pairs.sort(key=lambda x: x[3])
        worst_feat, correlated_with, corr_val, worst_imp, other_imp = high_corr_pairs[0]
        logger.info("  High-correlation: '%s' (%.2f%%) ↔ '%s' (%.2f%%) |r|=%.2f → removing '%s'",
                    worst_feat, worst_imp, correlated_with, other_imp, corr_val, worst_feat)
        return worst_feat, ranking, 'high_correlation'

    # ── Phase 2: Globally least important ──
    worst_feature = ranking[-1][0]
    logger.info("  No high-correlation pairs. Removing least important: '%s' (%.2f%%)",
                worst_feature, ranking[-1][1])
    return worst_feature, ranking, 'low_importance'
