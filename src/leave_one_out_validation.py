from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
import numpy as np


def leave_one_out_validation(best_model, X_model, y):
    """
    Perform leave-one-out validation.

    Each LOO fold independently creates fresh MinMaxScaler instances and
    clones the model, preventing (a) cross-fold parameter contamination and
    (b) mutation of the caller's scaler state.

    Parameters:
    best_model (object): The trained best model (cloned per fold).
    X_model (DataFrame): The feature data.
    y (Series): The target data.

    Returns:
    r2_loo (float): The R^2 score from leave-one-out validation.
    mae_loo (float): The MAE from leave-one-out validation.
    """
    loo = LeaveOneOut()
    y_pred_loo = []
    y_true_loo = []

    for train_index, test_index in loo.split(X_model):
        X_train_loo, X_test_loo = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train_loo, y_test_loo = y.iloc[train_index], y.iloc[test_index]

        # Per-fold independent scalers — prevents polluting the caller's scaler state
        fold_sX = MinMaxScaler()
        fold_sY = MinMaxScaler(feature_range=(0, 100))

        X_train_loo_scaled = fold_sX.fit_transform(X_train_loo)
        X_test_loo_scaled = fold_sX.transform(X_test_loo)
        y_train_loo_scaled = fold_sY.fit_transform(y_train_loo.values.reshape(-1, 1)).ravel()

        # Clone model for clean per-fold initialization (blocks cross-fold leakage)
        fold_model = clone(best_model)
        fold_model.fit(X_train_loo_scaled, y_train_loo_scaled)

        y_pred_loo_scaled = fold_model.predict(X_test_loo_scaled)
        y_pred = fold_sY.inverse_transform(y_pred_loo_scaled.reshape(-1, 1)).ravel()

        y_pred_loo.append(y_pred[0])
        y_true_loo.append(y_test_loo.values[0])

    return r2_score(y_true_loo, y_pred_loo), mean_absolute_error(y_true_loo, y_pred_loo)
