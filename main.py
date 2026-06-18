# find and evaluate the best model and features
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from src.gplearn_wrapper import GPLearnRegressor
import optuna
from src.visualization import plot_scatter

# ── Suppress noisy but harmless third-party warnings ───────────────────
# These warnings are produced by external libraries (LightGBM, CatBoost) and
# do NOT affect model training or prediction accuracy.  We suppress them
# targetedly using substring matching on the warning message so that all
# other warnings remain visible.
#
# (1) LightGBM internally converts DataFrames to numpy arrays, which causes
#     sklearn's validation layer to report a feature-names mismatch on every
#     predict() call.  The actual computation uses numeric arrays and is
#     unaffected.  ~9,000 instances per full run.
warnings.filterwarnings(
    'ignore',
    message='X does not have valid feature names',
    category=UserWarning,
)
# (2) sklearn 1.6 deprecated BaseEstimator._validate_data; CatBoost still
#     calls it internally.  This is a library compatibility issue — CatBoost
#     works correctly regardless.  ~2,600 instances per full run.
warnings.filterwarnings(
    'ignore',
    message='BaseEstimator._validate_data',
    category=FutureWarning,
)

# ── Progress bar configuration ──
# tqdm progress bars appear during Optuna trial evaluation (e.g., 0%| | 0/33 [00:00<?, ?it/s]).
# These show the number of CV folds or feature subsets being processed per iteration.
# Set to False to suppress them for cleaner log output in headless/CI environments.
SHOW_PROGRESS_BAR = True

# Set optuna logging level
from src.logger_config import setup_logger
import os
os.makedirs('models', exist_ok=True)
logger = setup_logger("models")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress tqdm progress bars if disabled
if not SHOW_PROGRESS_BAR:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    import logging as _logging
    _logging.getLogger('optuna').setLevel(_logging.ERROR)

# Import iterative optimization
from src.iterative_optimization import iterative_optimization
import argparse

models = {
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'SVR': SVR,
    'DecisionTree': DecisionTreeRegressor,
    'RandomForest': RandomForestRegressor,
    'GradientBoosting': GradientBoostingRegressor,
    'XGBoost': XGBRegressor,
    # 'GPR': GaussianProcessRegressor,  # Temporarily disabled — extreme sensitivity to geometric features (see model_analysis_report_2.md §4.4)
    'KRR': KernelRidge,
    'MLP': MLPRegressor,
    'AdaBoost': AdaBoostRegressor,
    'ElasticNet': ElasticNet,
    'KNR': KNeighborsRegressor,
    'LightGBM': LGBMRegressor,
    'CatBoost': CatBoostRegressor,
    'GPlearn': GPLearnRegressor,
}

def evaluate_baseline(baseline, X, y):
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training and prediction
    baseline.fit(scaler.fit_transform(X_train), y_train)
    y_pred_train = baseline.predict(scaler.transform(X_train))
    y_pred_test = baseline.predict(scaler.transform(X_test))
    
    # Calculate evaluation metrics
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test, 
                 model_name='Baseline', mae_mean=None,
                 output_dir='models/Baseline/',
                 output_name='baseline_scatter.png',
                 X_train=X_train,
                 X_test=X_test)

def build_argument_parser():
    """Build the command-line parser for the training pipeline."""
    parser = argparse.ArgumentParser(description='Machine learning model training and evaluation')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of CPU cores to use (-1 means using all cores)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna optimization')
    parser.add_argument('--keep_versions', type=int, default=2, help='Number of versions to keep for each model')
    parser.add_argument('--min_features', type=int, default=5,
                        help='Minimum number of features (floor for SHAP-RFECV auto-selection; '
                             'set low to let auto-selection explore the full path)')
    parser.add_argument('--force_n_features', type=int, default=None,
                        help='Evaluate the SHAP-RFECV path through --min_features, then select the '
                             'entry with this exact feature count instead of auto-selection.')
    return parser


def main():
    # Parse command line arguments
    args = build_argument_parser().parse_args()
    
    # Read data
    data = pd.read_csv('example/B_dataset.csv')
    data = data.dropna(axis=1, how='all')
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features].drop('activation_energy', axis=1)
    # X = X.drop(columns=['sub_H']) # Delete sub_H column used as sequence
    y = data['activation_energy']
    # print(X.head())
    # print(y.head())
     
    # Evaluate baseline model
    # baseline = Baseline()
    # evaluate_baseline(baseline, X, y)
    
    # ── Per-model minimum feature limits (optional) ──
    # Set to None to use the unified `--min_features` CLI argument for all models.
    # Uncomment and modify the dict below to override per model.
    #
    # NOTE: All feature removal is now driven by SHAP-RFECV.  When features < 10,
    # CV metrics are recorded along the removal path, and after the loop the
    # optimal feature count is AUTO-SELECTED (minimum RKfold MAE on the path).
    # `--min_features` / `custom_min_features` sets the FLOOR — the loop stops
    # when this many features remain.  Set it low (2–3) to let auto-selection
    # explore the full path, or higher (7–8) to constrain the search.
    #
    # Ridge uses RidgeCV (auto alpha, skips Optuna); Lasso uses LassoCV.
    # SVR epsilon is searched in 0.001–0.5, gamma in 0.1–10 (RBF kernel).
    custom_model_min_features = None
    # custom_model_min_features = {
    #     'LinearRegression': 2,
    #     'Lasso': 2,
    #     'Ridge': 2,
    #     'SVR': 2,
    #     'GPR': 2,
    #     'KRR': 2,
    #     'RandomForest': 2,
    #     'AdaBoost': 2,
    #     'GradientBoosting': 2,
    #     'KNR': 2
    # }
    
    # Iterative optimization
    results, best_models = iterative_optimization(
        models, 
        X, 
        y, 
        n_trials=args.n_trials, 
        n_jobs=args.n_jobs,
        keep_versions=args.keep_versions,
        min_features=args.min_features,
        custom_min_features=custom_model_min_features,
        force_n_features=args.force_n_features
    )

if __name__ == '__main__':
    main()
