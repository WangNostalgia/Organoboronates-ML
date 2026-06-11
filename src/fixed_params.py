"""
Single Source of Truth for fixed (non-Optuna-tuned) model hyperparameters.

All modules that need to instantiate a fully-configured model MUST call
get_fixed_params() and merge the result with Optuna-selected best_params:

    from src.fixed_params import get_fixed_params
    final_params = {**get_fixed_params(model_class, n_jobs), **best_params}
    model = model_class(**final_params)

This guarantees that the deployed model uses the EXACT same configuration
that was evaluated during hyperparameter tuning — no missing constants.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor
from src.gplearn_wrapper import GPLearnRegressor


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
    dict : fixed parameter names → values (empty dict if model not in map)
    """
    # NOTE: This dict is defined inside the function (not at module level)
    # because several entries reference the runtime `n_jobs` parameter.
    # The dict is small (<20 entries) so the recreation cost is negligible.
    FIXED_PARAMS_MAP = {
        LinearRegression: {},
        Ridge: {},  # RidgeCV handles alpha internally
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
        XGBRegressor: {
            "n_jobs": n_jobs, "random_state": 42, "tree_method": "hist",
            "grow_policy": "depthwise", "base_score": 0.5,
        },
        LGBMRegressor: {"n_jobs": n_jobs, "random_state": 42, "verbose": -1},  # -1 suppresses per-tree split warnings
        MLPRegressor: {
            "learning_rate": "adaptive", "max_iter": 5000, "early_stopping": True,
            "validation_fraction": 0.2, "n_iter_no_change": 20, "tol": 1e-3,
            "random_state": 42, "solver": "adam", "batch_size": "auto",
            # max_iter raised from 2000 to 5000: sklearn default (200) was
            # occasionally reached during Optuna trials even with 2000,
            # triggering "ConvergenceWarning: Maximum iterations (200)
            # reached".  5000 gives ample headroom for all trials.
        },
        AdaBoostRegressor: {"random_state": 42},
        KNeighborsRegressor: {"n_jobs": n_jobs},
        GaussianProcessRegressor: {"random_state": 42, "n_restarts_optimizer": 5},  # multiple restarts help escape local optima in marginal likelihood
        KernelRidge: {},
        CatBoostRegressor: {"random_state": 42, "verbose": 0},
        GPLearnRegressor: {
            "n_jobs": 1,  # single-process to avoid nested parallelism with Optuna
            "random_state": 42,  # lock GP random seed — essential for reproducible formula discovery
            "p_crossover": 0.7,  # explicit crossover probability (must sum ≤1.0 with mutation probs)
            "p_subtree_mutation": 0.1,  # subtree mutation probability
            "p_hoist_mutation": 0.05,  # hoist mutation probability
            "p_point_mutation": 0.1,  # point mutation probability
            "function_set": ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
        },
    }
    return FIXED_PARAMS_MAP.get(model_class, {})
