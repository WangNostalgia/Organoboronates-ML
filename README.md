# Machine Learning Tool for Organoboronate Activation-Energy Prediction

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]() [![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41467--025--60674--9-blue)](https://doi.org/10.1038/s41467-025-60674-9)

[English](README.md) | [中文](README_CN.md)

This repository contains the supplementary machine-learning workflow for the Nature Communications paper ["Organometallic-type reactivity of stable organoboronates for selective (hetero)arene C−H/C-halogen borylation and beyond"](https://doi.org/10.1038/s41467-025-60674-9). The code predicts reaction activation energies in kcal/mol and ships with training, checkpointing, external validation, applicability-domain analysis, and standalone y-randomization utilities.

More detailed references:

- Conceptual workflow: [Pipeline.md](Pipeline.md)
- Step-by-step execution guide: [user_manual.md](user_manual.md)
- Repository-specific agent guidance: [AGENTS.md](AGENTS.md)

## Requirements

- Python >= 3.12
- Recommended installer: `uv sync`
- Alternative installer: `pip install -r requirements.txt`

The supported full installation includes:

- scikit-learn
- optuna
- shap
- xgboost
- lightgbm
- catboost
- gplearn
- matplotlib / seaborn / pandas / numpy / joblib

## Quick start

```bash
uv sync
python main.py --n_trials 100 --min_features 5
```

Useful variants:

- Quick smoke run: `python main.py --n_trials 20 --min_features 5`
- Show CLI help: `python main.py --help`
- Evaluation notebook: `jupyter notebook example/example_pic.ipynb`
- Prediction notebook: `jupyter notebook example/prediction_round2.ipynb`
- Standalone y-randomization: `python example/standalone_y_randomization.py`

## Active default model registry

`main.py` currently enables these models by default:

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

`GaussianProcessRegressor` is still present in the file but commented out, so it is not part of the default run.

## Current training and evaluation protocol

### 1. Fixed boundary between development and final test

`iterative_optimization()` creates one 80/20 split at the entry point with `random_state=40`, and every active model family shares that exact split.

- Development set (80%): hyperparameter tuning, SHAP-driven feature elimination, feature-count path evaluation, internal 5×5 RepeatedKFold, LOOCV, and 100-split stability analysis
- Final test set (20%): scored exactly once after feature count and hyperparameters are locked

The test set is not used for model selection.

### 2. Hyperparameter tuning

`src/train_and_evaluate.py` tunes on development data only.

- Optuna uses an internal 5-fold MAE objective
- Studies are in-memory for each run
- Ridge and Lasso use explicit fold-local alpha loops with fold-local scaling
- Every fold scales `X` and `y` on training data only, then inverse-transforms predictions before computing MAE

### 3. Feature elimination and feature-count choice

For non-GPlearn models, the feature-removal loop is SHAP-driven:

- coarse single-fit SHAP when the feature count is still high
- 5-fold consensus SHAP-RFECV when the feature count is small enough
- remove the weaker member of any highly correlated pair first, otherwise remove the globally least important feature

`--min_features` is the stopping floor and defaults to `5`.

`--force_n_features` does not skip path evaluation. It evaluates the SHAP-RFECV path down to `--min_features`, then selects the exact requested feature count from that evaluated path.

### 4. Metric roles

Primary final metrics:

- `test_mae`
- `test_r2`

Secondary development-only metrics:

- `internal_cv.rkf_mae_mean/std`
- `internal_cv.rkf_r2_mean/std`
- `stability.mae_mean/std`
- `loo.mae`
- `loo.r2`

Legacy aliases are still written for compatibility, but they are not the primary model-selection outputs anymore.

### 5. y-randomization

The automatic full-pipeline y-randomization block remains disabled because of runtime cost. The supported path is the standalone script:

- `example/standalone_y_randomization.py`

That script:

- loads already selected checkpoints
- reuses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets
- reports the corrected finite-permutation p-value `(b + 1) / (m + 1)`

### 6. Checkpoint loading and external validation

Checkpoint selection follows `src.external_validation.load_model()`:

- search both final and iteration checkpoints
- exact feature count required by default
- exact final checkpoints preferred over exact iteration checkpoints
- newest filename timestamp wins inside the chosen checkpoint class
- nearest-match fallback is allowed only with `allow_closest=True` / `--allow-closest`

`example/load_checkpoint_guide.py` shows how to inspect these checkpoints safely.

### 7. Applicability domain

Applicability-domain analysis uses:

- training-set 5-fold OOF residuals
- MAD-based residual scaling with finite fallbacks
- descriptor-space leverage

If external ground truth is absent, the Williams plot switches to a leverage-only prediction view instead of inventing residual thresholds from missing labels.

## CLI arguments for `main.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent final-model runs/checkpoint families to retain |
| `--min_features` | `5` | Minimum feature floor for SHAP-RFECV path evaluation |
| `--force_n_features` | `None` | Select an exact evaluated feature count instead of auto-selecting from the path |

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

Iteration checkpoints store development-only metrics. Final checkpoints add the one-time final test result, split metadata, scalers, complete merged hyperparameters, and the chosen feature order.

## Notes on reproducibility

- Fixed development/final split seed: `40`
- Fixed evaluation seeds inside development utilities: `42`
- Target values are not clipped
- Matplotlib uses the `Agg` backend in the optimization workflow
- `src/fixed_params.py` is the single source of truth for fixed model parameters
- `src/evaluation.py` is the single source of truth for 5×5 RepeatedKFold evaluation
