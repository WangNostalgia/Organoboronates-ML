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
python example/standalone_y_randomization.py
jupyter notebook example/example_pic.ipynb
jupyter notebook example/prediction_round2.ipynb
python -c "from sklearn.svm import SVR; from src.fixed_params import get_fixed_params; print(get_fixed_params(SVR))"
```

## CLI semantics

`main.py` currently exposes:

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain |
| `--min_features` | `5` | SHAP-RFECV feature floor |
| `--force_n_features` | `None` | Exact evaluated feature count; rejected by `main.py` for the default GPlearn-containing registry |

`--force_n_features` is only meaningful for custom registries that exclude GPlearn. `main.py` aborts before runtime setup if that flag is requested against the default registry.

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

`main.py` reads `example/B_dataset.csv`, keeps numeric columns, removes `activation_energy` from `X`, and calls `iterative_optimization()` with the default registry.

### Fixed boundary between development and final test

`src/iterative_optimization.py` creates one 80/20 split with `random_state=40` before entering the model loop.

- Development set: tuning, SHAP feature elimination, feature-count path evaluation, internal 5×5 RepeatedKFold, LOOCV, and 100-split stability
- Final test set: scored exactly once after feature count and hyperparameters are locked

All default models share the same split.

### Hyperparameter tuning

`src/train_and_evaluate.py` performs development-only parameter selection.

- Optuna objective: internal 5-fold MAE
- Optuna storage: in-memory study
- Ridge/Lasso: explicit fold-local alpha loops
- all folds fit their own `MinMaxScaler` instances on training data only
- MAE/R² are computed after inverse-transform back to kcal/mol

### Feature elimination

Non-GPlearn models iterate through SHAP-driven feature removal:

1. train and tune on the current development feature subset
2. save an iteration checkpoint
3. remove exactly one feature

Removal priority is: weaker member of a high-correlation pair first, otherwise the globally least important feature. The path continues until the configured `min_features` floor is reached.

GPlearn remains the single-pass exception because genetic programming performs its own embedded selection.

### Metric schema

Final checkpoints use:

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`
- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

Iteration checkpoints use:

- `metrics.internal_cv.*`
- `metrics.stability.*`
- `metrics.loo.*`

Legacy aliases may still be stored for compatibility, but agent-facing docs and reviews should treat the current schema as authoritative.

### y-randomization

The automatic full-pipeline y-randomization block remains disabled because of runtime cost. Supported validation is the standalone script `example/standalone_y_randomization.py`, which reuses the same precomputed 5×5 RepeatedKFold splits for original and permuted targets, reports the corrected finite-permutation p-value, and writes plots to `models/y_randomization_<ModelName>.png`.

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

Iteration checkpoints contain development-path metrics. Final checkpoints additionally include the final one-time test result, split metadata, fitted scalers, merged hyperparameters, and selected feature order.

## Single sources of truth

- `src/fixed_params.py` — fixed model parameters
- `src/evaluation.py` — 5×5 RepeatedKFold evaluation
- `src/model_utils.py` — final model construction helper
