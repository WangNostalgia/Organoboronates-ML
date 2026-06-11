# ==============================================================================
# DEPRECATED: This module has been archived. It contains known bugs including
# data-label misalignment and force-clipping of predictions. Its functionality
# is fully superseded by the SHAP-RFECV pipeline in iterative_optimization.py.
# DO NOT import or call from active workflows.
# ==============================================================================
raise DeprecationWarning("This legacy module has been archived. Use the SHAP-RFECV pipeline in src/iterative_optimization.py instead.")


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.feature_filter import feature_filter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
os.makedirs('output', exist_ok=True)
models = {
    # 'SVR': SVR,
    'RandomForest': RandomForestRegressor,
    'GradientBoosting': GradientBoostingRegressor,
    'XGBoost': XGBRegressor,
    'Lasso': Lasso,
    'GPR': GaussianProcessRegressor,
    'KRR': KernelRidge
}

data = pd.read_csv('example/B_dataset.csv')
data = data.dropna(axis=1, how='all')

# scaler = StandardScaler()
features = data.select_dtypes(include=[np.number]).columns # 选择数值列
X = data[features].drop('activation_energy', axis=1)  # 假设目标列名为 'activation_energy'
# X = X.drop(columns=['sub_H']) # 删除作为序列的sub_H列
y = data['activation_energy']
results, char_change = feature_filter(models, X, y, n_trials=100,max_features=3,mae_threshold=2.0)