# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project overview

Machine-learning workflow for predicting organoboronate reaction activation energies (kcal/mol). This repository accompanies the Nature Communications paper "Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond" (DOI: 10.1038/s41467-025-60674-9).

Read:

- [pipeline.md](pipeline.md) for the conceptual workflow
- [user_manual.md](user_manual.md) for execution steps
- [AGENTS.md](AGENTS.md) for repository-specific agent notes

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

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain |
| `--min_features` | `5` | SHAP-RFECV feature floor |
| `--force_n_features` | `None` | Exact evaluated feature count; rejected by `main.py` for the default GPlearn-containing registry |

`main.py` aborts before runtime setup if `--force_n_features` is requested while the default registry still includes GPlearn.

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

`GaussianProcessRegressor` is present in source but commented out by default.

## Architecture and current protocol

### Entry point

`main.py` reads `example/B_dataset.csv`, keeps numeric columns, removes `activation_energy` from `X`, and passes the default model registry into `iterative_optimization()`.

### Development/final-test split

`src/iterative_optimization.py` creates one shared 80/20 split with `random_state=40`.

- Development set: tuning, SHAP feature elimination, feature-count path evaluation, internal 5×5 RepeatedKFold, LOOCV, and 100-split stability
- Final test set: evaluated exactly once after feature count and hyperparameters are locked

The final test set is not part of model selection.

### Hyperparameter tuning

`src/train_and_evaluate.py` performs development-only parameter selection.

- Optuna objective: internal 5-fold MAE
- Optuna storage: in-memory study
- Ridge/Lasso: explicit fold-local alpha loops
- each fold scales training data only, then inverse-transforms predictions before computing MAE/R²

### Feature elimination

For non-GPlearn models, each iteration removes exactly one feature:

1. tune on the current development feature subset
2. save an iteration checkpoint
3. remove the weaker member of a high-correlation pair, or otherwise the globally least important feature

The path continues until the `min_features` floor is reached.

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

Legacy aliases may still exist for compatibility, but the current schema above is authoritative.

### y-randomization

The automatic full-pipeline y-randomization path remains disabled. The supported route is `example/standalone_y_randomization.py`, which reuses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets and reports the corrected finite-permutation p-value.

### Checkpoint loading

`src.external_validation.load_model()`:

- searches both final and iteration checkpoints
- requires exact feature counts by default
- prefers exact final over exact iteration
- uses filename timestamps for deterministic tie-breaking
- only allows nearest-match fallback with `allow_closest=True`

### Applicability domain

`src.applicability_domain.py` uses training-set 5-fold OOF residuals with MAD-based scaling and descriptor-space leverage. Prediction-only mode is leverage-only, with no external `sqrt(1-h)` correction.

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

Iteration checkpoints contain development-path metrics. Final checkpoints add the one-time final-test metrics, split metadata, fitted scalers, merged hyperparameters, and selected feature order.
