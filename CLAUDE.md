# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning tool for predicting organoboronate reaction activation energies (kcal/mol). Supplementary material for the Nature Communications paper "Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond" (DOI: 10.1038/s41467-025-60674-9).

See [Pipeline.md](Pipeline.md) for a detailed conceptual explanation of the six-stage ML workflow, and [user_manual.md](user_manual.md) for step-by-step execution instructions.

## Commands

```bash
# Install dependencies (uv is preferred, Python 3.12+)
uv sync

# Run the full training pipeline
python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5

# Run a quick test (fewer trials)
python main.py --n_trials 20 --mae_threshold 2.0 --min_features 5

# Run Jupyter notebooks for evaluation/prediction
jupyter notebook example/example_pic.ipynb
jupyter notebook example/prediction_round2.ipynb

# Config reference: single source of truth for fixed model params
python -c "from src.fixed_params import get_fixed_params; print(get_fixed_params(SVR))"
```
```

### main.py arguments

| Argument | Default | Description |
|---|---|---|
| `--n_trials` | 100 | Optuna optimization trials per model |
| `--mae_threshold` | 2.0 | MAE threshold for "good" model classification |
| `--min_features` | 5 | Minimum features to retain |
| `--n_jobs` | -1 | CPU cores (-1 = all) |
| `--keep_versions` | 2 | Number of model versions to retain on disk |

## Architecture

### Entry point and orchestration

`main.py` reads `example/B_dataset.csv`, selects numeric columns, drops the `activation_energy` column as the target, and calls `iterative_optimization()` with a dict of model classes. Only SVR, RandomForest, and KNeighborsRegressor are active by default; several others are commented out.

`src/iterative_optimization.py` is the core orchestration engine. Its main loop:

1. Calls `hyperparameter_optimization_and_training()` to train the current model with Optuna-tuned hyperparameters
2. Scales features (MinMaxScaler) and target (MinMaxScaler 0-100), retrains on scaled data
3. Runs `feature_importance_analysis()` (SHAP) to get per-feature importance percentages
4. Runs `feature_correlation_analysis()` to get the correlation matrix
5. Runs `shap_rfecv_select_worst_feature()` (SHAP-RFECV with multi-fold CV consensus) to decide which feature to remove
6. Removes selected features and iterates until `min_features` is reached or no more features qualify for removal
7. After iteration: LOO validation, performance history plot/CSV, final scatter plot, saves `*_final_*.joblib`

Each iteration saves `{ModelName}_iteration_{N}_{timestamp}.joblib` containing the model, scalers, feature list, hyperparameters, and metrics.

### Hyperparameter optimization (`src/train_and_evaluate.py`)

The `train_and_evaluate()` function:
- Splits data 80/20 (random_state=42)
- Defines an Optuna `objective()` that performs Leave-One-Out CV on the training split, returning mean MAE
- Uses `TPESampler` with `n_startup_trials=5`
- Each model class has its own `trial.suggest_*` parameter space
- After optimization, evaluates on 100 random train/test splits and reports average MAE
- Optuna studies are persisted as SQLite databases: `optuna_optimization_{ModelClass}.db`

`src/hyperparameter_optimization_and_training.py` is a thin wrapper: calls `train_and_evaluate()`, then does another 80/20 split and returns the untrained model instance, scalers, splits, MAE, and best params.

### Feature selection (`src/feature_selection.py`)

SHAP-RFECV with two-tier strategy:
- **Coarse filtering** (many features): single-fit SHAP (`cv_folds=0`) for speed
- **Fine-grained selection** (few features, ≤ max(10, min_features+3)): multi-fold CV consensus SHAP (`cv_folds=5`) for robustness

Two-phase removal within each SHAP evaluation:
1. **High correlation**: finds feature pairs with |r| > 0.8, picks the less SHAP-important one
2. **Low importance**: if no high-correlation pairs, picks the globally least important feature

All scaling is done internally with per-fold `MinMaxScaler` instances — no data leakage. The deprecated legacy `feature_selection()` function (threshold-based, non-SHAP) has been removed.


### Feature importance (`src/feature_importance_analysis.py`)

Uses SHAP with model-specific explainers:
- Tree-based models (RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting, DecisionTree): `shap.TreeExplainer`
- Linear models (LinearRegression, Ridge, Lasso, ElasticNet): `shap.LinearExplainer`
- GPlearn, SVR, MLP, KNR, GPR, KRR: `shap.KernelExplainer` with K-means background summarization

### Data format

Input CSV must contain numeric features plus an `activation_energy` column (the target). The example data at `example/B_dataset.csv` also has `sub_H` and `sub_B` as compound identifiers (non-numeric, excluded automatically).

### Output structure

```
models/
├── <ModelName>/
│   ├── <ModelName>_iteration_N_<timestamp>.joblib   # Per-iteration checkpoint
│   ├── <ModelName>_iteration_N_<timestamp>_metrics.txt
│   ├── <ModelName>_final_<timestamp>.joblib          # Final model (after all iterations)
│   ├── <ModelName>_final_<timestamp>_metrics.txt
│   ├── final_scatter_<timestamp>.png                 # Final scatter plot
│   ├── final_scatter_<timestamp>_outliers.csv        # Outlier data (deviation >= 5)
│   ├── performance_history_<timestamp>.csv           # Iteration-by-iteration metrics
│   └── performance_history_<timestamp>.png           # Performance history plot
├── optimization_<timestamp>.log                      # Full run log
├── optuna_optimization_<ModelClass>.db               # Optuna study storage
```

### Additional scripts

- `src/validation_process.py` — Generates a complete combinatorial validation dataset from `sub_H`/`sub_B` pairs, filling in feature values from known data.
- `src/visualization.py` — `plot_scatter` with outlier detection (deviation ≥ 5.0 kcal/mol) and automatic CSV export. `plot_scatter_standard` for publication-quality output.
- `src/fixed_params.py` — **Single source of truth** for all model fixed hyperparameters. All modules import from here.
- `src/evaluation.py` — **Unified evaluation center** for 5×5 RepeatedKFold. `train_and_evaluate.py` and `y_randomization.py` both import from here.
- `archive/` — Deprecated modules (`feature_filter.py`, `feature_importance_analysis.py`, `feature_correlation_analysis.py`, `model_feature_filter.py`, `evaluate_and_plot.py`) guarded by `raise DeprecationWarning`.

### Key implementation notes

- All random states are fixed to 42 for reproducibility
- Target values are never clipped (activation energy has no physical upper bound)
- Matplotlib backend is set to `Agg` in `iterative_optimization.py` before any pyplot imports
- `src/fixed_params.py` is the **single source of truth** for all model fixed hyperparameters — every module imports from here
- `src/evaluation.py` is the **unified evaluation center** for 5×5 RepeatedKFold — `train_and_evaluate.py` and `y_randomization.py` both import from it
- Every fold/repeat independently fits its own `MinMaxScaler(feature_range=(0, 100))` on training data only — no data leakage
- MAE is always computed after `inverse_transform` back to real kcal/mol
- Deprecated modules (`feature_filter.py`, `feature_importance_analysis.py`, `feature_correlation_analysis.py`, `model_feature_filter.py`, `evaluate_and_plot.py`) live in `archive/` with `raise DeprecationWarning` guards
- GPlearn mutation probabilities are explicitly set in `fixed_params.py` and passed through to `SymbolicRegressor`: p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1 (sum=0.95 ≤ 1.0)
- GPlearn formula replacement uses `re.sub` with `\b` word boundaries to prevent X0 from matching X10, X11, etc.
- CatBoost is classified as a tree model for SHAP (`TreeExplainer`) alongside RandomForest, XGBoost, LightGBM, etc.
