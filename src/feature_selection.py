import logging
import numpy as np
logger = logging.getLogger(__name__)


def shap_rfecv_select_worst_feature(model, X, y, model_name, corr_threshold=0.8, cv_folds=5):
    """
    SHAP-RFECV with multi-fold CV consensus SHAP.

    Instead of relying on a single data split (vulnerable to random noise in
    small-N settings), SHAP importance is averaged across `cv_folds` folds.
    Each fold trains the model on the training fold and computes SHAP values
    for the held-out fold. For LinearExplainer and KernelExplainer, the
    explainer background/masker is built from the training fold, not the
    held-out fold.

    Strategy (two-phase):
      1. HIGH-CORRELATION: If any pair has |r| > corr_threshold, remove the
         less consensus-important one.
      2. LOW-IMPORTANCE: Otherwise, remove the globally least important feature.

    NOTE: This function manages its own internal MinMaxScaler instances per fold
    to guarantee zero data leakage between training and validation splits.
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

    if cv_folds <= 0:
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
            n_clust_desired = max(10, min(15, int(len(X) * 0.15)))
            n_clusters = max(1, min(n_clust_desired, len(X_scaled)))
            n_samp_desired = max(20, min(int(len(X) * 0.3), 100))
            n_samples = max(1, min(n_samp_desired, len(X_scaled)))
            rng = np.random.RandomState(42)
            _stashed = np.random.get_state()
            np.random.seed(42)
            background = shap.kmeans(X_scaled, n_clusters)
            np.random.set_state(_stashed)
            explainer = shap.KernelExplainer(m.predict, background)
            sample_indices = rng.choice(len(X_scaled), n_samples, replace=False)
            sv = explainer.shap_values(X_scaled[sample_indices])

        importances = np.abs(sv).mean(axis=0)
        n_folds_used = 1
    else:
        importances = np.zeros(n_features)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        n_folds_used = 0
        rng = np.random.RandomState(42)

        for train_idx, test_idx in kf.split(X_arr):
            X_tr_raw, X_te_raw = X_arr[train_idx], X_arr[test_idx]
            y_tr_raw = y_arr[train_idx]

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
                    explainer = shap.LinearExplainer(m, X_tr_s)
                    sv = explainer.shap_values(X_te_s)
                else:
                    n_clust_desired = max(5, min(10, int(len(X_tr_s) * 0.3)))
                    n_clusters = max(1, min(n_clust_desired, len(X_tr_s)))
                    n_samp_desired = max(5, min(int(len(X_te_s) * 0.5), len(X_te_s)))
                    n_samples = max(1, min(n_samp_desired, len(X_te_s)))
                    _stashed = np.random.get_state()
                    np.random.seed(42)
                    background = shap.kmeans(X_tr_s, n_clusters)
                    np.random.set_state(_stashed)
                    explainer = shap.KernelExplainer(m.predict, background)
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

    total = importances.sum()
    if total > 0:
        importance_pct = (importances / total) * 100
    else:
        importance_pct = np.ones(n_features) * (100.0 / n_features)

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
        logger.info(
            "  High-correlation: '%s' (%.2f%%) ↔ '%s' (%.2f%%) |r|=%.2f → removing '%s'",
            worst_feat, worst_imp, correlated_with, other_imp, corr_val, worst_feat
        )
        return worst_feat, ranking, 'high_correlation'

    worst_feature = ranking[-1][0]
    logger.info(
        "  No high-correlation pairs. Removing least important: '%s' (%.2f%%)",
        worst_feature, ranking[-1][1]
    )
    return worst_feature, ranking, 'low_importance'
