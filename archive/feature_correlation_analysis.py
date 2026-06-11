# ==============================================================================
# DEPRECATED: This module has been archived. Its functionality (SHAP-based
# feature importance and Pearson correlation analysis) is now fully embedded
# in src/feature_selection.py (shap_rfecv_select_worst_feature) and the
# iterative SHAP-RFECV pipeline in src/iterative_optimization.py.
# DO NOT import or call from active workflows.
# ==============================================================================
raise DeprecationWarning("This legacy analysis module has been archived. Use src/feature_selection.py instead.")


import seaborn as sns

import matplotlib.pyplot as plt


def feature_correlation_analysis(X_model, mae_mean, mae_threshold):
    """
    Analyze the feature correlation and plot the correlation matrix.

    Parameters:
    X_model (DataFrame): The feature data.

    Returns:
    corr_matrix (DataFrame): The correlation matrix.
    """
    corr_matrix = X_model.corr().abs()
    if mae_mean < mae_threshold:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.show()
    return corr_matrix
