# Machine Learning Tool for Organoboronate Activation-Energy Prediction

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]() [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--60674--9-blue)](https://doi.org/10.1038/s41467-025-60674-9)

[English](README.md) | [中文](README_CN.md)

This repository contains the supplementary machine-learning workflow for the Nature Communications paper ["Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond"](https://doi.org/10.1038/s41467-025-60674-9).

Active references:

- Conceptual workflow: [pipeline.md](pipeline.md)
- Step-by-step execution guide: [user_manual.md](user_manual.md)

## Requirements

- Python >= 3.12
- Recommended installer: `uv sync`
- Alternative installer: `pip install -r requirements.txt`

The default runtime includes the scikit-learn stack plus XGBoost, LightGBM, CatBoost, and `gplearn==0.4.2`.

## Quick start

```bash
uv sync
python main.py --n_trials 100 --min_features 5
python main.py --n_trials 20 --min_features 5
python main.py --help
python example/standalone_y_randomization.py
```

## Active default model registry

`main.py` currently enables these model families by default:

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

`GaussianProcessRegressor` remains commented out and is not part of the default run.

## Current training and evaluation protocol

### 1. One shared development/final-test split

`iterative_optimization()` creates one 80/20 split at the entry point with `random_state=40`, and all default models share that same split.

- Development set: tuning, SHAP feature elimination, feature-count path evaluation, internal 5×5 RepeatedKFold, LOOCV, and 100-split stability
- Final test set: scored exactly once after the feature count and hyperparameters are locked

The final test split is never used for model selection.

### 2. Development-only tuning

`src/train_and_evaluate.py` tunes on development data only.

- Optuna uses an internal 5-fold MAE objective
- the Optuna study is in memory for the run
- Ridge and Lasso use explicit fold-local alpha loops
- every fold fits its own scalers on training data only, then inverse-transforms predictions before MAE/R²

### 3. SHAP-driven feature elimination

For non-GPlearn models, each iteration removes exactly one feature:

- remove the weaker member of a high-correlation pair first
- otherwise remove the globally least important feature

That path continues until the configured floor is reached. `--min_features` defaults to `5`.

`--force_n_features` still means “evaluate the path, then choose an exact feature count”, but `main.py` now rejects that option immediately for the default registry because the default run still includes GPlearn. Exact feature-count forcing is only valid for custom registries that exclude GPlearn.

### 4. Metric roles and checkpoint schema

Final checkpoints use nested metrics:

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`
- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

Iteration checkpoints keep the development-path metrics in flat form:

- `metrics.internal_cv.*`
- `metrics.stability.*`
- `metrics.loo.*`

Legacy aliases may still appear for compatibility, but current readers should prefer the current keys first and only fall back to the aliases when needed.

### 5. Standalone y-randomization

Automatic full-pipeline y-randomization remains disabled in the main workflow. The supported path is `example/standalone_y_randomization.py`, which:

- loads selected checkpoints
- reuses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets
- reports the corrected finite-permutation p-value `(b + 1) / (m + 1)`
- writes each histogram to `models/y_randomization_<ModelName>.png`

### 6. Checkpoint loading

`src.external_validation.load_model()`:

- searches both final and iteration checkpoints
- requires an exact feature count by default
- prefers exact final checkpoints over exact iteration checkpoints
- breaks ties by the filename timestamp
- only allows nearest-match fallback with `allow_closest=True` / `--allow-closest`

`example/load_checkpoint_guide.py` follows that same selection logic and safely formats both current and legacy metric layouts.

### 7. Applicability domain

Applicability-domain analysis uses:

- training-set 5-fold OOF residuals
- MAD-based residual scaling with finite fallbacks
- leverage in descriptor space

Prediction-only mode is leverage-only when external labels are absent; there is no external `sqrt(1-h)` correction term.

## CLI arguments for `main.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain |
| `--min_features` | `5` | SHAP-RFECV feature-floor for path evaluation |
| `--force_n_features` | `None` | Exact evaluated feature count; rejected by `main.py` for the default GPlearn-containing registry |

## Output layout

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

Iteration checkpoints store development-path results. Final checkpoints add the one-time final-test result, split metadata, fitted scalers, merged hyperparameters, and final feature order.

## Reproducibility notes

- fixed development/final split seed: `40`
- fixed development-side evaluation seeds: `42`
- target values are not clipped
- Matplotlib uses the `Agg` backend in the optimization workflow
- `src/fixed_params.py` is the single source of truth for fixed model parameters
- `src/evaluation.py` is the single source of truth for 5×5 RepeatedKFold evaluation
