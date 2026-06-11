import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor
import logging

from src.evaluation import repeated_kfold_evaluate  # Unified evaluation center
from src.gplearn_wrapper import GPLearnRegressor
from src.fixed_params import get_fixed_params  # Single source of truth — used in objective() and train_and_evaluate()
from sklearn.base import clone  # Used to isolate model state in the 100-split stability loop

logger = logging.getLogger(__name__)


def train_and_evaluate(model_class, X, y, random_state=42, n_trials=100, n_jobs=-1):
    """
    Train and evaluate a regression model using Optuna for hyperparameter tuning.

    Parameters:
    model_class (class): The regression model class to be used (e.g., LinearRegression, Ridge).
    X (DataFrame): The feature matrix.
    y (Series): The target vector.
    random_state (int): The random seed for reproducibility (default is 42).
    n_trials (int): Number of trials for Optuna optimization (default is 100).
    n_jobs (int): Number of CPU cores to use (-1 for all cores).

    Returns:
    tuple: A tuple containing the best mean absolute error (MAE) from Optuna, the average MAE from test sets, and the best hyperparameters.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    logger.info(f"n_jobs: {n_jobs}")

    def objective(trial):
        """
        Define the objective function for Optuna to optimize.

        Parameters:
        trial (Trial): A trial object for hyperparameter sampling.

        Returns:
        float: The mean absolute error (MAE) for the current trial.
        """
        # Pull the shared fixed-params base from the Single Source of Truth
        # (imported at module level — see top of file).
        # This guarantees Optuna tunes on the EXACT same configuration that
        # will be used in the deployed model — no silent config drift.
        base_params = get_fixed_params(model_class, n_jobs).copy()

        if model_class == LinearRegression:
            params = {**base_params}
        elif model_class == Ridge:
            # RidgeCV auto-selects alpha internally — no Optuna tuning needed
            params = {**base_params}
        elif model_class == Lasso:
            # LassoCV auto-selects alpha; only tol is tuned by Optuna
            params = {**base_params, "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True)}
        elif model_class == ElasticNet:
            params = {**base_params,
                "alpha": trial.suggest_float("alpha", 1e-1, 100.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
                "tol": trial.suggest_float("tol", 1e-3, 1e-1, log=True),
            }
        elif model_class == SVR:
            params = {**base_params,
                "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 0.5, log=True),
                "gamma": trial.suggest_float("gamma", 0.1, 10, log=True),
            }
        elif model_class == DecisionTreeRegressor:
            params = {**base_params,
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),
            }
        elif model_class == RandomForestRegressor:
            params = {**base_params,
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        elif model_class == GradientBoostingRegressor:
            params = {**base_params,
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "min_samples_split": trial.suggest_int("min_samples_split", 3, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "max_features": trial.suggest_float("max_features", 0.5, 0.9),
            }
        elif model_class == XGBRegressor:
            params = {**base_params,
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
        elif model_class == LGBMRegressor:
            params = {**base_params,
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
        elif model_class == CatBoostRegressor:
            # CatBoost: gradient boosting with ordered target encoding (reduces overfitting).
            # verbose=0 is set in FIXED_PARAMS to suppress per-iteration output.
            params = {**base_params,
                "iterations": trial.suggest_int("iterations", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "depth": trial.suggest_int("depth", 3, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }
        elif model_class == MLPRegressor:
            # String keys avoid Optuna UserWarning about tuple types in categorical distributions
            hidden_layer_choices = ["20", "50", "20_10", "50_25"]
            choice_str = trial.suggest_categorical("hidden_layer_sizes", hidden_layer_choices)
            if choice_str == "20_10":
                parsed = (20, 10)
            elif choice_str == "50_25":
                parsed = (50, 25)
            else:
                parsed = (int(choice_str),)
            params = {**base_params,
                "hidden_layer_sizes": parsed,
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-3, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-2, 1e-1, log=True),
            }
        elif model_class == AdaBoostRegressor:
            params = {**base_params,
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
            }
        elif model_class == GPLearnRegressor:
            # Genetic programming: p_crossover is fixed (0.7) because gplearn's
            # p_subtree_mutation (0.1) + p_hoist_mutation (0.05) + p_point_mutation (0.1)
            # already sum to 0.25.  If Optuna picked p_crossover > 0.75, total > 1.0
            # → ValueError crash.  Only tune pop_size, generations, parsimony_coeff.
            params = {**base_params,
                "population_size": trial.suggest_int("population_size", 1000, 5000, step=500),
                "generations": trial.suggest_int("generations", 8, 25),
                "parsimony_coefficient": trial.suggest_float("parsimony_coefficient", 1e-4, 1e-2, log=True),
            }
        elif model_class == KNeighborsRegressor:
            params = {**base_params,
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 10),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"]),
                "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"]),
                "p": trial.suggest_int("p", 1, 3) if trial.params.get("metric", "") == "minkowski" else 2,
                "leaf_size": trial.suggest_int("leaf_size", 20, 60),
            }
        elif model_class == GaussianProcessRegressor:
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
            kernel_choice = trial.suggest_categorical("kernel", ["RBF", "Matern", "RBF+White"])
            # Widened length_scale from 0.1~10 to 1e-3~1e3 to reduce ConvergenceWarning
            l_scale = trial.suggest_float("length_scale", 1e-3, 1e3, log=True)
            if kernel_choice == "RBF":
                kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale)
            elif kernel_choice == "Matern":
                kernel = ConstantKernel(1.0) * Matern(length_scale=l_scale)
            else:
                n_level = trial.suggest_float("noise_level", 1e-5, 1.0, log=True)
                kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale) + WhiteKernel(noise_level=n_level)
            params = {**base_params,
                "kernel": kernel,
                "alpha": trial.suggest_float("alpha", 1e-10, 1e-1, log=True),
            }
        elif model_class == KernelRidge:
            params = {**base_params,
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "polynomial"]),
            }
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        # --- RidgeCV / LassoCV: auto-select alpha via internal CV ---
        # Pipeline wrapper ensures per-fold scaling inside CV (no data leakage).
        from sklearn.pipeline import Pipeline

        # --- RidgeCV: embedded CV to find alpha, then manual 5-fold for real-unit MAE ---
        if model_class == Ridge:
            sX = MinMaxScaler()
            sY = MinMaxScaler(feature_range=(0, 100))
            X_tr_s = sX.fit_transform(X_train)
            y_tr_s = sY.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            ridge_cv = RidgeCV(
                alphas=np.logspace(-3, 3, 50),
                cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
                scoring='neg_mean_absolute_error'
            )
            ridge_cv.fit(X_tr_s, y_tr_s)
            trial.set_user_attr("best_alpha", ridge_cv.alpha_)

            # Manual 5-fold with inverse-transform to return MAE in real kcal/mol
            best_alpha = ridge_cv.alpha_
            mae_list = []
            sX_cv = MinMaxScaler()
            sY_cv = MinMaxScaler(feature_range=(0, 100))
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            for train_idx, test_idx in kf.split(X_train):
                X_tr = X_train.iloc[train_idx]
                X_te = X_train.iloc[test_idx]
                y_tr = y_train.iloc[train_idx]
                y_te_cv = y_train.iloc[test_idx]
                X_tr_s_cv = sX_cv.fit_transform(X_tr)
                X_te_s_cv = sX_cv.transform(X_te)
                y_tr_s_cv = sY_cv.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
                m = Ridge(alpha=best_alpha)
                m.fit(X_tr_s_cv, y_tr_s_cv)
                y_pred_s = m.predict(X_te_s_cv)
                y_pred_orig = sY_cv.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                mae_list.append(mean_absolute_error(y_te_cv, y_pred_orig))
            return np.mean(mae_list)

        # --- LassoCV: embedded CV to find alpha, then manual 5-fold for real-unit MAE ---
        elif model_class == Lasso:
            sX = MinMaxScaler()
            sY = MinMaxScaler(feature_range=(0, 100))
            X_tr_s = sX.fit_transform(X_train)
            y_tr_s = sY.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            lasso_cv = LassoCV(
                alphas=np.logspace(-4, 1, 50),
                cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
                max_iter=params.get("max_iter", 10000),
                tol=params.get("tol", 1e-4),
                selection=params.get("selection", "cyclic"),
                random_state=params.get("random_state", 42),
                n_jobs=1
            )
            lasso_cv.fit(X_tr_s, y_tr_s)
            trial.set_user_attr("best_alpha", lasso_cv.alpha_)

            # Manual 5-fold with inverse-transform for fair comparison
            best_alpha = lasso_cv.alpha_
            mae_list = []
            sX_cv = MinMaxScaler()
            sY_cv = MinMaxScaler(feature_range=(0, 100))
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            for train_idx, test_idx in kf.split(X_train):
                X_tr = X_train.iloc[train_idx]
                X_te = X_train.iloc[test_idx]
                y_tr = y_train.iloc[train_idx]
                y_te_cv = y_train.iloc[test_idx]
                X_tr_s_cv = sX_cv.fit_transform(X_tr)
                X_te_s_cv = sX_cv.transform(X_te)
                y_tr_s_cv = sY_cv.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
                best_lasso = Lasso(
                    alpha=best_alpha,
                    max_iter=params.get("max_iter", 10000),
                    tol=params.get("tol", 1e-4),
                    selection=params.get("selection", "cyclic"),
                    random_state=params.get("random_state", 42)
                )
                best_lasso.fit(X_tr_s_cv, y_tr_s_cv)
                y_pred_s = best_lasso.predict(X_te_s_cv)
                y_pred_orig = sY_cv.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                mae_list.append(mean_absolute_error(y_te_cv, y_pred_orig))
            return np.mean(mae_list)

        # --- Standard models: 5-fold CV (replaces LOO for efficiency) ---
        model = model_class(**params)
        mae_list = []
        sX = MinMaxScaler()
        sY = MinMaxScaler(feature_range=(0, 100))
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(X_train):
            X_tr = X_train.iloc[train_idx]
            X_te = X_train.iloc[test_idx]
            y_tr = y_train.iloc[train_idx]
            y_te = y_train.iloc[test_idx]

            X_tr_s = sX.fit_transform(X_tr)
            X_te_s = sX.transform(X_te)
            y_tr_s = sY.fit_transform(y_tr.values.reshape(-1, 1)).ravel()

            model.fit(X_tr_s, y_tr_s)

            y_pred_s = model.predict(X_te_s)
            y_pred = sY.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            mae_list.append(mean_absolute_error(y_te, y_pred))

        return np.mean(mae_list)

    # Pull fixed params from the single source of truth (imported at module level)
    FIXED_PARAMS = get_fixed_params(model_class, n_jobs)

    # Ridge has no tunable params (RidgeCV handles alpha internally) — skip Optuna entirely
    if model_class == Ridge:
        # Step 1: Use RidgeCV's built-in CV solely to find the optimal alpha
        sX = MinMaxScaler()
        sY = MinMaxScaler(feature_range=(0, 100))
        X_tr_s = sX.fit_transform(X_train)
        y_tr_s = sY.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        ridge_cv = RidgeCV(
            alphas=np.logspace(-3, 3, 50),
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='neg_mean_absolute_error'
        )
        ridge_cv.fit(X_tr_s, y_tr_s)
        best_alpha = ridge_cv.alpha_
        best_params = {"alpha": best_alpha}

        # Step 2 (CRITICAL): Manual 5-fold CV with inverse-transform to kcal/mol.
        # ridge_cv.best_score_ is in MinMaxScaler(0,100) space (~0.5-5),
        # which is NOT comparable to other models' MAE in real kcal/mol (~15-20).
        # We must re-evaluate in original units for fair comparison.
        mae_list = []
        sX_cv = MinMaxScaler()
        sY_cv = MinMaxScaler(feature_range=(0, 100))
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(X_train):
            X_tr = X_train.iloc[train_idx]
            X_te = X_train.iloc[test_idx]
            y_tr = y_train.iloc[train_idx]
            y_te = y_train.iloc[test_idx]

            X_tr_s_cv = sX_cv.fit_transform(X_tr)
            X_te_s_cv = sX_cv.transform(X_te)
            y_tr_s_cv = sY_cv.fit_transform(y_tr.values.reshape(-1, 1)).ravel()

            m = Ridge(alpha=best_alpha)
            m.fit(X_tr_s_cv, y_tr_s_cv)
            y_pred_s = m.predict(X_te_s_cv)
            y_pred_orig = sY_cv.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            mae_list.append(mean_absolute_error(y_te, y_pred_orig))

        best_value = float(np.mean(mae_list))
        logger.info("RidgeCV selected alpha=%.4f | 5-Fold MAE (real kcal/mol)=%.4f", best_alpha, best_value)
    else:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5, n_ei_candidates=24, seed=42
        )
        study = optuna.create_study(
            sampler=sampler,
            direction="minimize",
        )  # In-memory storage: avoids SQLite lock contention under n_jobs > 1
        study.optimize(objective, n_jobs=n_jobs, n_trials=n_trials)
        best_value = study.best_value
        best_params = study.best_params

        # LassoCV: the best alpha was selected internally by LassoCV and stored
        # as a user attribute on the best trial.  study.best_params only contains
        # Optuna-suggested params (tol), NOT the alpha — so we must retrieve it.
        if model_class == Lasso:
            best_alpha = study.best_trial.user_attrs.get("best_alpha", None)
            if best_alpha is not None:
                best_params = dict(best_params)
                best_params["alpha"] = best_alpha
                logger.info("LassoCV best_alpha retrieved from trial: %.6f", best_alpha)
            else:
                logger.warning("LassoCV best_alpha not found in trial user_attrs — "
                               "falling back to default alpha=1.0")

        # GPR: kernel-building params ('kernel', 'length_scale', 'noise_level')
        # were consumed inside objective() to construct the kernel object, but
        # Optuna's study.best_params leaks them back.  We must strip them and
        # explicitly reconstruct the tuned kernel, otherwise GPR falls back to
        # the default 1.0 * RBF(1.0) — silently discarding all tuning work.
        elif model_class == GaussianProcessRegressor:
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
            best_params = dict(best_params)

            # Strip kernel-building params (NOT valid GPR constructor arguments)
            k_choice = best_params.pop("kernel", "RBF")
            l_scale = best_params.pop("length_scale", 1.0)
            n_level = best_params.pop("noise_level", 1e-5)

            # Reconstruct the exact tuned kernel object
            if k_choice == "RBF":
                final_kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale)
            elif k_choice == "Matern":
                final_kernel = ConstantKernel(1.0) * Matern(length_scale=l_scale)
            else:  # "RBF+White"
                final_kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale) + WhiteKernel(noise_level=n_level)

            best_params["kernel"] = final_kernel
            logger.info("GPR kernel reconstructed: %s (length_scale=%.4f)", k_choice, l_scale)

        # MLP: hidden_layer_sizes is stored by Optuna as a categorical string
        # (e.g. '20_10') to avoid UserWarning about tuple types.  We must parse
        # it back to a real tuple before passing it to MLPRegressor(), otherwise
        # sklearn >= 1.2 raises InvalidParameterError.
        elif model_class == MLPRegressor:
            best_params = dict(best_params)
            raw = best_params.get("hidden_layer_sizes", "20")
            if isinstance(raw, str):
                if "_" in raw:
                    parts = raw.split("_")
                    best_params["hidden_layer_sizes"] = (int(parts[0]), int(parts[1]))
                else:
                    best_params["hidden_layer_sizes"] = (int(raw),)
            # else: already a tuple/int from a previous fix, leave as-is

    # Merge fixed (non-Optuna) params with Optuna-suggested params to ensure
    # the deployed model is IDENTICAL to what was evaluated during tuning.
    # study.best_params only contains trial.suggest_* values.
    final_params = {**FIXED_PARAMS, **best_params}
    best_model = model_class(**final_params)

    # Unified y-scaler: MinMaxScaler(0,100) — consistent with Optuna objective.
    # Using the same scaler type everywhere prevents regularization mismatch.
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 100))

    # Single train/test split for legacy reporting only (NOT model selection)
    # Use the caller's random_state instead of hardcoded 42 for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    best_model.fit(X_train_scaled, y_train_scaled)

    y_pred_train_scaled = best_model.predict(X_train_scaled)
    y_pred_test_scaled = best_model.predict(X_test_scaled)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    logger.info(f"random_state = {random_state}")
    logger.info(f"Best hyperparameters: {final_params}")
    logger.info(f"Train R^2: {r2_train:.4f}")
    logger.info(f"Test R^2: {r2_test:.4f}")
    logger.info(f"Test RMSE: {rmse_test:.4f}")
    logger.info(f"Test MAE: {mae_test:.4f}")

    # 100-split stability check (mean ± std only — NO best_random_state selection).
    # Each split uses a FRESH clone of best_model so the original model state
    # (trained on the unified split, random_state=random_state) is preserved
    # for the subsequent 5×5 RepeatedKFold evaluation below.
    mae_test_list = []
    for i in range(100):
        X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
            X, y, test_size=0.2, random_state=i
        )
        X_tr_s = scaler_X.fit_transform(X_tr_i)
        X_te_s = scaler_X.transform(X_te_i)
        y_tr_s = scaler_y.fit_transform(y_tr_i.values.reshape(-1, 1)).ravel()
        fold_model = clone(best_model)
        fold_model.fit(X_tr_s, y_tr_s)
        y_pred_s = fold_model.predict(X_te_s)
        y_pred_i = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        mae_test_list.append(mean_absolute_error(y_te_i, y_pred_i))
    mae_test_mean = np.mean(mae_test_list)
    mae_test_std = np.std(mae_test_list)
    logger.info(f"100-split stability MAE: {mae_test_mean:.4f} ± {mae_test_std:.4f}")

    # Print current feature combination
    logger.info("Current feature combination:")
    logger.info(X.columns)

    # --- 5×5 RepeatedKFold evaluation (PRIMARY metric) ---
    rkf_results = repeated_kfold_evaluate(best_model, X, y)
    logger.info(
        "5×5 RepeatedKFold: MAE = %.4f ± %.4f | R² = %.4f ± %.4f (%d evaluations)",
        rkf_results['rkf_mae_mean'], rkf_results['rkf_mae_std'],
        rkf_results['rkf_r2_mean'], rkf_results['rkf_r2_std'],
        rkf_results['n_evals']
    )

    return best_value, mae_test_mean, best_params, rkf_results
