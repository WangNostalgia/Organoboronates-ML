# find and evaluate the best model and features
import argparse


SHOW_PROGRESS_BAR = True
DEFAULT_MODEL_NAMES = (
    "LinearRegression",
    "Ridge",
    "Lasso",
    "SVR",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "KRR",
    "MLP",
    "AdaBoost",
    "ElasticNet",
    "KNR",
    "LightGBM",
    "CatBoost",
    "GPlearn",
)


def configure_runtime():
    import logging
    import os
    import warnings

    import optuna

    from src.logger_config import setup_logger

    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="BaseEstimator._validate_data",
        category=FutureWarning,
    )

    os.makedirs("models", exist_ok=True)
    logger = setup_logger("models")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not SHOW_PROGRESS_BAR:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

    return logger


def build_default_models():
    """Build the default model registry after CLI parsing."""
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    missing_dependencies = []

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None
        missing_dependencies.append("xgboost")

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None
        missing_dependencies.append("lightgbm")

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None
        missing_dependencies.append("catboost")

    try:
        from src.gplearn_wrapper import GPLearnRegressor
    except ImportError:
        GPLearnRegressor = None
        missing_dependencies.append("gplearn")

    if missing_dependencies:
        missing = ", ".join(missing_dependencies)
        raise ImportError(
            "The default model registry requires additional packages that are "
            f"not installed: {missing}. Install the full environment with "
            "'uv sync' or 'pip install -r requirements.txt' before training."
        )

    return {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "SVR": SVR,
        "DecisionTree": DecisionTreeRegressor,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": XGBRegressor,
        # "GPR": GaussianProcessRegressor,  # Temporarily disabled — extreme sensitivity to geometric features.
        "KRR": KernelRidge,
        "MLP": MLPRegressor,
        "AdaBoost": AdaBoostRegressor,
        "ElasticNet": ElasticNet,
        "KNR": KNeighborsRegressor,
        "LightGBM": LGBMRegressor,
        "CatBoost": CatBoostRegressor,
        "GPlearn": GPLearnRegressor,
    }


def evaluate_baseline(baseline, X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from src.visualization import plot_scatter

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline.fit(scaler.fit_transform(X_train), y_train)
    y_pred_train = baseline.predict(scaler.transform(X_train))
    y_pred_test = baseline.predict(scaler.transform(X_test))

    plot_scatter(
        y_train,
        y_pred_train,
        y_test,
        y_pred_test,
        model_name="Baseline",
        mae_mean=None,
        output_dir="models/Baseline/",
        output_name="baseline_scatter.png",
        X_train=X_train,
        X_test=X_test,
    )


def build_argument_parser():
    """Build the command-line parser for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Machine learning model training and evaluation"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use (-1 means using all cores)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for Optuna optimization",
    )
    parser.add_argument(
        "--keep_versions",
        type=int,
        default=2,
        help="Number of versions to keep for each model",
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=5,
        help="Minimum number of features (floor for SHAP-RFECV auto-selection; "
        "set low to let auto-selection explore the full path)",
    )
    parser.add_argument(
        "--force_n_features",
        type=int,
        default=None,
        help="Evaluate the SHAP-RFECV path through --min_features, then select the "
        "entry with this exact feature count instead of auto-selection. main.py "
        "rejects this for the default GPlearn-containing registry.",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.force_n_features is not None and "GPlearn" in DEFAULT_MODEL_NAMES:
        parser.error(
            "--force_n_features is not available from main.py while the default "
            "registry still includes GPlearn. Use the default auto-selection "
            "path, or call iterative_optimization() with a custom registry that "
            "excludes GPlearn."
        )

    configure_runtime()

    import numpy as np
    import pandas as pd

    from src.iterative_optimization import iterative_optimization

    data = pd.read_csv("example/B_dataset.csv")
    data = data.dropna(axis=1, how="all")
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features].drop("activation_energy", axis=1)
    y = data["activation_energy"]

    # Evaluate baseline model
    # baseline = Baseline()
    # evaluate_baseline(baseline, X, y)

    # Per-model minimum feature limits (optional).
    # Set to None to use the unified `--min_features` CLI argument for all models.
    # Uncomment and modify the dict below to override per model.
    #
    # NOTE: All feature removal is now driven by SHAP-RFECV. When features < 10,
    # CV metrics are recorded along the removal path, and after the loop the
    # optimal feature count is AUTO-SELECTED (minimum RKfold MAE on the path).
    # `--min_features` / `custom_min_features` sets the FLOOR — the loop stops
    # when this many features remain. Set it low (2–3) to let auto-selection
    # explore the full path, or higher (7–8) to constrain the search.
    #
    # Ridge and Lasso use explicit fold-local alpha search on development data.
    # SVR epsilon is searched in 0.001–0.5, gamma in 0.1–10 (RBF kernel).
    custom_model_min_features = None
    # custom_model_min_features = {
    #     "LinearRegression": 2,
    #     "Lasso": 2,
    #     "Ridge": 2,
    #     "SVR": 2,
    #     "GPR": 2,
    #     "KRR": 2,
    #     "RandomForest": 2,
    #     "AdaBoost": 2,
    #     "GradientBoosting": 2,
    #     "KNR": 2,
    # }

    iterative_optimization(
        build_default_models(),
        X,
        y,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        keep_versions=args.keep_versions,
        min_features=args.min_features,
        custom_min_features=custom_model_min_features,
        force_n_features=args.force_n_features,
    )


if __name__ == "__main__":
    main()
