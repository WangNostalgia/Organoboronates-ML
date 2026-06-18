"""
Single Source of Truth for fixed (non-Optuna-tuned) model hyperparameters.

All modules that need to instantiate a fully-configured model MUST call
get_fixed_params() and merge the result with Optuna-selected best_params:

    from src.fixed_params import get_fixed_params
    final_params = {**get_fixed_params(model_class, n_jobs), **best_params}
    model = model_class(**final_params)

This guarantees that the deployed model uses the EXACT same configuration
that was evaluated during hyperparameter tuning 鈥?no missing constants.
"""

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - only hit when optional dependency is absent
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - only hit when optional dependency is absent
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - only hit when optional dependency is absent
    CatBoostRegressor = None

try:
    from src.gplearn_wrapper import GPLearnRegressor
except ImportError:  # pragma: no cover - only hit when optional dependency is absent
    GPLearnRegressor = None


def get_fixed_params(model_class, n_jobs=-1):
    """
    Return the fixed (non-tunable) hyperparameters for a given model class.

    These constants are NOT searched by Optuna but are essential for correct
    model behavior (e.g. max_iter, solver, early_stopping, random_state).
    Merge them with Optuna's best_params before instantiating the final model.

    Parameters
    ----------
    model_class : sklearn estimator class
    n_jobs : int, number of parallel jobs (used by tree/ensemble models)

    Returns
    -------
    dict : fixed parameter names 鈫?values (empty dict if model not in map)
    """
    FIXED_PARAMS_MAP = {
        LinearRegression: {},
        Ridge: {},
        Lasso: {"max_iter": 10000, "selection": "cyclic", "random_state": 42},
        ElasticNet: {
            "max_iter": 50000, "selection": "cyclic", "random_state": 42,
            "fit_intercept": True, "precompute": True, "warm_start": True, "copy_X": True,
        },
        SVR: {"kernel": "rbf", "tol": 1e-3, "max_iter": 10000, "cache_size": 1000},
        DecisionTreeRegressor: {"splitter": "best", "max_features": None, "random_state": 42},
        RandomForestRegressor: {"n_jobs": n_jobs, "random_state": 42},
        GradientBoostingRegressor: {
            "loss": "squared_error", "random_state": 42, "n_iter_no_change": 10, "tol": 1e-4,
        },
        MLPRegressor: {
            "learning_rate": "adaptive", "max_iter": 5000, "early_stopping": True,
            "validation_fraction": 0.2, "n_iter_no_change": 20, "tol": 1e-3,
            "random_state": 42, "solver": "adam", "batch_size": "auto",
        },
        AdaBoostRegressor: {"random_state": 42},
        KNeighborsRegressor: {"n_jobs": n_jobs},
        GaussianProcessRegressor: {"random_state": 42, "n_restarts_optimizer": 5},
        KernelRidge: {},
    }

    if XGBRegressor is not None:
        FIXED_PARAMS_MAP[XGBRegressor] = {
            "n_jobs": n_jobs, "random_state": 42, "tree_method": "hist",
            "grow_policy": "depthwise", "base_score": 0.5,
        }
    if LGBMRegressor is not None:
        FIXED_PARAMS_MAP[LGBMRegressor] = {"n_jobs": n_jobs, "random_state": 42, "verbose": -1}
    if CatBoostRegressor is not None:
        FIXED_PARAMS_MAP[CatBoostRegressor] = {"random_state": 42, "verbose": 0}
    if GPLearnRegressor is not None:
        FIXED_PARAMS_MAP[GPLearnRegressor] = {
            "n_jobs": 1,
            "random_state": 42,
            "p_crossover": 0.7,
            "p_subtree_mutation": 0.1,
            "p_hoist_mutation": 0.05,
            "p_point_mutation": 0.1,
            "function_set": ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
        }

    return FIXED_PARAMS_MAP.get(model_class, {})
