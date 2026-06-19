# AGENTS.md

This file provides guidance for agents working in this repository.

## Project overview

Machine-learning workflow for predicting organoboronate reaction activation energies (kcal/mol). This repository accompanies the Nature Communications paper "Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond" (DOI: 10.1038/s41467-025-60674-9).

Read:

- [pipeline.md](pipeline.md) for the conceptual workflow
- [user_manual.md](user_manual.md) for execution steps

## Core commands

```bash
uv sync
python main.py --n_trials 100 --min_features 5
python main.py --n_trials 20 --min_features 5
python main.py --help
jupyter notebook example/example_pic.ipynb
jupyter notebook example/prediction_round2.ipynb
python example/standalone_y_randomization.py
python -c "from sklearn.svm import SVR; from src.fixed_params import get_fixed_params; print(get_fixed_params(SVR))"
```

## CLI semantics

`main.py` currently exposes:

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain |
| `--min_features` | `5` | SHAP-RFECV stopping floor |
| `--force_n_features` | `None` | Select an exact evaluated feature count from the SHAP-RFECV path |

`--force_n_features` still evaluates the development feature path first; it does not bypass SHAP-RFECV.

## Default model registry

The active default registry in `main.py` is:

- LinearRegression
- Ridge
- Lasso
- SVR
- DecisionTree
- RandomForest
- GradientBoosting
- XGBoost
- KRR
- MLP
- AdaBoost
- ElasticNet
- KNR
- LightGBM
- CatBoost
- GPlearn

`GaussianProcessRegressor` exists in source but is commented out by default.

## Architecture and protocol

### Entry point

`main.py` reads `example/B_dataset.csv`, keeps numeric columns, removes `activation_energy` from `X`, and calls `iterative_optimization()` with the default model registry.

### Fixed boundary between development and final test

`src/iterative_optimization.py` creates one 80/20 split with `random_state=40` before entering the model loop.

- Development set: tuning, SHAP feature elimination, feature-count path evaluation, internal 5×5 RepeatedKFold, LOOCV, and 100-split stability analysis
- Final test set: scored exactly once after the feature set and hyperparameters are locked

All default models share the same split.

### Hyperparameter tuning

`src/train_and_evaluate.py` performs development-only parameter selection.

- Optuna objective: internal 5-fold MAE
- Optuna storage: in-memory study
- Ridge/Lasso: explicit fold-local alpha loops
- All folds fit their own `MinMaxScaler` instances on training data only
- MAE is computed after inverse-transform back to kcal/mol

### Feature elimination

Non-GPlearn models iterate through SHAP-driven feature removal:

1. train/tune on the current development feature subset
2. save an iteration checkpoint
3. use SHAP-RFECV logic to remove one feature
4. stop at `min_features` or when no feature qualifies for removal

GPlearn is a single-pass exception because genetic programming performs inherent feature selection.

### Metric roles

Primary final metrics:

- `test_mae`
- `test_r2`

Secondary development-only metrics:

- `internal_cv.rkf_mae_mean/std`
- `internal_cv.rkf_r2_mean/std`
- `stability.mae_mean/std`
- `loo.mae`
- `loo.r2`

Legacy aliases may still be stored for compatibility, but documentation and reviews should treat them according to the current primary/secondary split.

### y-randomization

The automatic full-pipeline y-randomization block remains disabled because of runtime cost. Supported validation is the standalone script `example/standalone_y_randomization.py`, which reuses the same precomputed 5×5 RepeatedKFold splits for original and permuted targets and reports the corrected finite-permutation p-value.

### Checkpoint loading

`src.external_validation.load_model()`:

- searches both final and iteration checkpoints
- requires an exact feature count by default
- prefers exact final over exact iteration checkpoints
- uses filename timestamps for deterministic tie-breaking
- only allows nearest-match fallback with `allow_closest=True`

### Applicability domain

`src.applicability_domain.py` calibrates residual thresholds from training-set 5-fold OOF residuals using MAD-based scaling with finite fallbacks. Prediction-only mode is leverage-only; there is no external `sqrt(1-h)` correction term.

## Data format

Input CSVs must contain:

- numeric feature columns
- one target column named `activation_energy`

Optional identifier columns such as `sub_H` and `sub_B` are ignored automatically by the main training entry point.

## Output structure

```text
models/
├── <ModelName>/
│   ├── <ModelName>_iteration_<N>_<timestamp>.joblib
│   ├── <ModelName>_iteration_<N>_<timestamp>_metrics.txt
│   ├── <ModelName>_final_<timestamp>.joblib
│   ├── <ModelName>_final_<timestamp>_metrics.txt
│   ├── final_scatter_<timestamp>.png
│   ├── final_scatter_<timestamp>_outliers.csv
│   ├── performance_history_<timestamp>.csv
│   └── performance_history_<timestamp>.png
└── optimization_<timestamp>.log
```

Iteration checkpoints contain development-only metrics. Final checkpoints additionally include the final one-time test result, full split metadata, fitted scalers, complete merged hyperparameters, and selected feature order.

## Single sources of truth

- `src/fixed_params.py` — fixed model parameters
- `src/evaluation.py` — 5×5 RepeatedKFold evaluation
- `src/model_utils.py` — final model construction helper
