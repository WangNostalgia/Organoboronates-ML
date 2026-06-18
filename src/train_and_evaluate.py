import logging

import numpy as np
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.evaluation import repeated_kfold_evaluate
from src.fixed_params import get_fixed_params
from src.gplearn_wrapper import GPLearnRegressor
from src.model_utils import build_model

logger = logging.getLogger(__name__)


def _split_features_and_target(X, y, train_idx, test_idx):
    X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
    X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
    y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
    y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
    return X_train, X_test, y_train, y_test


def _real_unit_kfold_mae(model, X, y, random_state=42, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_maes = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test, y_train, y_test = _split_features_and_target(X, y, train_idx, test_idx)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler(feature_range=(0, 100))
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = fold_model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).ravel()
        fold_maes.append(mean_absolute_error(np.asarray(y_test), y_pred))

    return float(np.mean(fold_maes))


def _stability_analysis(model, X, y):
    fold_maes = []

    for split_seed in range(100):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=split_seed,
        )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler(feature_range=(0, 100))
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = fold_model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).ravel()
        fold_maes.append(mean_absolute_error(np.asarray(y_test), y_pred))

    return {
        "mae_mean": float(np.mean(fold_maes)),
        "mae_std": float(np.std(fold_maes)),
        "n_splits": len(fold_maes),
    }


def _select_alpha_via_inner_cv(
    model_class,
    X,
    y,
    base_params,
    tuned_params=None,
    candidate_alphas=None,
    random_state=42,
    inner_splits=5,
    n_jobs=-1,
):
    tuned_params = dict(tuned_params or {})
    if candidate_alphas is None:
        candidate_alphas = (
            np.logspace(-3, 3, 50)
            if model_class == Ridge
            else np.logspace(-4, 1, 50)
        )

    kfold = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    best_alpha = None
    best_mae = None

    for alpha in candidate_alphas:
        candidate_model, _ = build_model(
            model_class,
            best_params={**tuned_params, "alpha": float(alpha)},
            n_jobs=n_jobs,
        )

        fold_maes = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test, y_train, y_test = _split_features_and_target(X, y, train_idx, test_idx)

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler(feature_range=(0, 100))
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

            fold_model = clone(candidate_model)
            fold_model.fit(X_train_scaled, y_train_scaled)

            y_pred_scaled = fold_model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).ravel()
            fold_maes.append(mean_absolute_error(np.asarray(y_test), y_pred))

        alpha_mae = float(np.mean(fold_maes))
        if best_mae is None or alpha_mae < best_mae:
            best_alpha = float(alpha)
            best_mae = alpha_mae

    return best_alpha, best_mae


def _suggest_trial_params(model_class, trial, n_jobs):
    if model_class == LinearRegression:
        return {}
    if model_class == Ridge:
        return {}
    if model_class == Lasso:
        return {"tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True)}
    if model_class == ElasticNet:
        return {
            "alpha": trial.suggest_float("alpha", 1e-1, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "tol": trial.suggest_float("tol", 1e-3, 1e-1, log=True),
        }
    if model_class == SVR:
        return {
            "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 0.5, log=True),
            "gamma": trial.suggest_float("gamma", 0.1, 10, log=True),
        }
    if model_class == DecisionTreeRegressor:
        return {
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),
        }
    if model_class == RandomForestRegressor:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    if model_class == GradientBoostingRegressor:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_samples_split": trial.suggest_int("min_samples_split", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "max_features": trial.suggest_float("max_features", 0.5, 0.9),
        }
    if model_class == XGBRegressor:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.8),
            "gamma": trial.suggest_float("gamma", 0.1, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        }
    if model_class == LGBMRegressor:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "num_leaves": trial.suggest_int("num_leaves", 2, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
    if model_class == CatBoostRegressor:
        return {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
    if model_class == MLPRegressor:
        hidden_layer_choices = ["20", "50", "20_10", "50_25"]
        choice_str = trial.suggest_categorical("hidden_layer_sizes", hidden_layer_choices)
        if choice_str == "20_10":
            parsed = (20, 10)
        elif choice_str == "50_25":
            parsed = (50, 25)
        else:
            parsed = (int(choice_str),)
        return {
            "hidden_layer_sizes": parsed,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-3, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-2, 1e-1, log=True),
        }
    if model_class == AdaBoostRegressor:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        }
    if model_class == GPLearnRegressor:
        return {
            "population_size": trial.suggest_int("population_size", 1000, 5000, step=500),
            "generations": trial.suggest_int("generations", 8, 25),
            "parsimony_coefficient": trial.suggest_float("parsimony_coefficient", 1e-4, 1e-2, log=True),
        }
    if model_class == KNeighborsRegressor:
        metric = trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"])
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 10),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"]),
            "metric": metric,
            "leaf_size": trial.suggest_int("leaf_size", 20, 60),
        }
        params["p"] = trial.suggest_int("p", 1, 3) if metric == "minkowski" else 2
        return params
    if model_class == GaussianProcessRegressor:
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

        kernel_choice = trial.suggest_categorical("kernel", ["RBF", "Matern", "RBF+White"])
        length_scale = trial.suggest_float("length_scale", 1e-3, 1e3, log=True)
        if kernel_choice == "RBF":
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_choice == "Matern":
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale)
        else:
            noise_level = trial.suggest_float("noise_level", 1e-5, 1.0, log=True)
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        return {
            "kernel": kernel,
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-1, log=True),
        }
    if model_class == KernelRidge:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "polynomial"]),
        }

    raise ValueError(f"Unsupported model class: {model_class}")


def _normalize_best_params(model_class, best_params):
    best_params = dict(best_params)

    if model_class == GaussianProcessRegressor:
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

        kernel_choice = best_params.pop("kernel", "RBF")
        length_scale = best_params.pop("length_scale", 1.0)
        noise_level = best_params.pop("noise_level", 1e-5)

        if kernel_choice == "RBF":
            best_params["kernel"] = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_choice == "Matern":
            best_params["kernel"] = ConstantKernel(1.0) * Matern(length_scale=length_scale)
        else:
            best_params["kernel"] = (
                ConstantKernel(1.0) * RBF(length_scale=length_scale)
                + WhiteKernel(noise_level=noise_level)
            )

    if model_class == MLPRegressor:
        raw_hidden_layers = best_params.get("hidden_layer_sizes", "20")
        if isinstance(raw_hidden_layers, str):
            if "_" in raw_hidden_layers:
                first, second = raw_hidden_layers.split("_", maxsplit=1)
                best_params["hidden_layer_sizes"] = (int(first), int(second))
            else:
                best_params["hidden_layer_sizes"] = (int(raw_hidden_layers),)

    return best_params


def train_and_evaluate(model_class, X, y, random_state=42, n_trials=100, n_jobs=-1):
    """
    Select model hyperparameters using development data only.

    The returned estimator is intentionally unfitted. Downstream stages can fit
    it on whatever fold or feature subset they need without inheriting any
    hidden train/test split from this selection stage.
    """

    logger.info("n_jobs: %s", n_jobs)

    if model_class == Ridge:
        best_alpha, development_cv_mae = _select_alpha_via_inner_cv(
            Ridge,
            X,
            y,
            base_params=get_fixed_params(Ridge, n_jobs),
            random_state=random_state,
            n_jobs=n_jobs,
        )
        best_params = {"alpha": best_alpha}
    else:
        def objective(trial):
            tuned_params = _suggest_trial_params(model_class, trial, n_jobs=n_jobs)

            if model_class == Lasso:
                best_alpha, alpha_mae = _select_alpha_via_inner_cv(
                    Lasso,
                    X,
                    y,
                    base_params=get_fixed_params(Lasso, n_jobs),
                    tuned_params=tuned_params,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                trial.set_user_attr("best_alpha", best_alpha)
                return alpha_mae

            model, _ = build_model(model_class, best_params=tuned_params, n_jobs=n_jobs)
            return _real_unit_kfold_mae(
                model,
                X,
                y,
                random_state=random_state,
                n_splits=5,
            )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5,
            n_ei_candidates=24,
            seed=42,
        )
        study = optuna.create_study(
            sampler=sampler,
            direction="minimize",
        )
        study.optimize(objective, n_jobs=n_jobs, n_trials=n_trials)

        development_cv_mae = float(study.best_value)
        best_params = _normalize_best_params(model_class, study.best_params)
        if model_class == Lasso:
            best_params["alpha"] = float(study.best_trial.user_attrs["best_alpha"])

    estimator, complete_params = build_model(model_class, best_params=best_params, n_jobs=n_jobs)
    stability = _stability_analysis(estimator, X, y)
    internal_cv = repeated_kfold_evaluate(estimator, X, y)

    logger.info("Development-data selection params: %s", complete_params)
    logger.info(
        "100-split stability MAE: %.4f ± %.4f",
        stability["mae_mean"],
        stability["mae_std"],
    )
    logger.info(
        "5x5 RepeatedKFold: MAE = %.4f ± %.4f | R² = %.4f ± %.4f (%d evaluations)",
        internal_cv["rkf_mae_mean"],
        internal_cv["rkf_mae_std"],
        internal_cv["rkf_r2_mean"],
        internal_cv["rkf_r2_std"],
        internal_cv["n_evals"],
    )

    return {
        "estimator": estimator,
        "complete_params": complete_params,
        "selection_mae": development_cv_mae,
        "development_cv_mae": development_cv_mae,
        "stability": stability,
        "stability_mae_mean": stability["mae_mean"],
        "stability_mae_std": stability["mae_std"],
        "internal_cv": internal_cv,
    }
