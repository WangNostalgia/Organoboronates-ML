# Machine Learning Tool for Organoboronate Activation-Energy Prediction

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)]() [![License](https://img.shields.io/badge/License-MIT-green)]()

[English](README.md) | [中文](README_CN.md)

This repository provides a machine-learning workflow for organoboronate activation-energy prediction. The current task format is single-sample regression: one compound/conformer/record maps to one `activation_energy` value.

Active references:

- Conceptual workflow: [pipeline.md](pipeline.md)
- Step-by-step execution guide: [user_manual.md](user_manual.md)

## Requirements

- Python >= 3.12
- Primary reproducible installer: `uv sync --locked`
- Pip fallback: `requirements.txt` is a compatibility snapshot generated from the lockfile. Do not edit it by hand or treat it as a second dependency source.

The default runtime includes the scikit-learn stack plus XGBoost, LightGBM, CatBoost, and `gplearn==0.4.2`.

## Input data format

Training data should look like:

```text
ID, SMILES, filename, activation_energy, descriptor_1, descriptor_2, ...
```

- `activation_energy` is the target for training and labelled evaluation.
- `ID`, `SMILES`, and `filename` are recommended metadata columns and are not used as model features.
- Model features are numeric descriptor columns.
- External prediction CSVs may omit `activation_energy`; this triggers prediction-only mode.

## Quick start

```bash
uv sync --locked
python main.py --n_trials 100 --min_features 5
python main.py --n_trials 20 --min_features 5
python main.py --help
python example/standalone_y_randomization.py
```

Useful post-training commands:

```bash
python example/manual_selection_and_plot.py
python src/external_validation.py --list-models
python src/external_validation.py --model SVR --n_features 5 --data external.csv
python src/applicability_domain.py --model SVR --n_features 5 --training example/B_dataset.csv --external external.csv
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

The final test split is never used for model selection. Split indices are saved in checkpoint `evaluation_protocol` metadata and reused by post-processing scripts.

### 2. Development-only tuning

`src/train_and_evaluate.py` tunes on development data only.

- Optuna uses an internal 5-fold MAE objective
- Optuna trial parallelism is controlled by `--optuna_jobs`, default `1`
- estimator-internal parallelism is controlled by `--model_jobs`, default `-1`
- Ridge and Lasso use explicit fold-local alpha loops
- every fold fits its own scalers on training data only, then inverse-transforms predictions before MAE/R²

### 3. SHAP-driven feature elimination

For non-GPlearn models, each iteration removes exactly one feature:

- remove the weaker member of a high-correlation pair first
- otherwise remove the globally least important feature

Multi-fold SHAP consensus uses training-fold background/maskers for non-tree explainers. That path continues until the configured floor is reached. `--min_features` defaults to `5`.

`--force_n_features` still means “evaluate the path, then choose an exact feature count”, but `main.py` now rejects that option immediately for the default registry because the default run still includes GPlearn. Exact feature-count forcing is only valid for custom registries that exclude GPlearn.

### 4. Automatic final and manual-final artifacts

Automatic final checkpoints are created after the feature count and hyperparameters are selected using development-only evidence. The final test is then evaluated once.

Manual-final checkpoints are created by `example/manual_selection_and_plot.py` from `example/manual_feature_selection.csv`. This script:

- exactly loads the requested iteration checkpoint
- refuses closest-feature fallback
- reuses checkpoint development/final-test indices
- refits selected features and complete params on development rows
- evaluates final test once
- saves a manual-final checkpoint, metrics txt, and scatter plot labelled as manual feature-count selection

Manual feature counts should be chosen before looking at final-test plots.

### 5. Metric roles and checkpoint schema

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

### 6. Standalone y-randomization

Automatic full-pipeline y-randomization remains disabled in the main workflow. The supported path is `example/standalone_y_randomization.py`, which:

- loads selected checkpoints
- runs on checkpoint development indices by default
- reuses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets
- reports the corrected finite-permutation p-value `(b + 1) / (m + 1)`
- writes each histogram to `models/y_randomization_<ModelName>.png`

### 7. Checkpoint loading

`src.external_validation.load_model()`:

- searches both final and iteration checkpoints
- requires an exact feature count by default
- prefers exact final checkpoints over exact iteration checkpoints
- breaks ties by the filename timestamp
- only allows nearest-match fallback with `allow_closest=True` / `--allow-closest`

`example/load_checkpoint_guide.py` follows that same selection logic and safely formats both current and legacy metric layouts.

### 8. External validation and metadata preservation

`src/external_validation.py` supports single-model and ensemble external validation.

Prediction outputs preserve:

- prioritized metadata columns such as `ID`, `SMILES`, and `filename`
- all other non-feature, non-target metadata columns by default
- model feature columns
- predicted activation energy
- ground-truth and error columns when `activation_energy` is present

Use `--id-cols` to prioritize custom metadata columns, and `--only-id-cols` if you do not want to preserve all metadata.

### 9. Applicability domain

Applicability-domain analysis uses:

- checkpoint development indices for calibration by default
- training-set 5-fold OOF residuals
- MAD-based residual scaling with finite fallbacks
- leverage in descriptor space

Use `--allow-full-training-csv-for-ad` only if you intentionally accept full-CSV AD calibration. Prediction-only mode is leverage-only when external labels are absent; there is no external `sqrt(1-h)` correction term.

## CLI arguments for `main.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--model_jobs` | `-1` | Estimator-internal CPU parallelism (`-1` = all available where supported) |
| `--optuna_jobs` | `1` | Optuna trial parallelism; default serial for reproducibility and resource stability |
| `--n_jobs` | `None` | Deprecated alias for `--model_jobs` |
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
│   ├── <ModelName>_manual_final_<N>feat_<timestamp>.joblib
│   ├── <ModelName>_manual_final_<N>feat_<timestamp>_metrics.txt
│   ├── final_scatter_<timestamp>.png
│   ├── final_scatter_<timestamp>_outliers.csv
│   ├── performance_history_<timestamp>.csv
│   └── performance_history_<timestamp>.png
└── optimization_<timestamp>.log
```

Iteration checkpoints store development-path results. Final checkpoints add the one-time final-test result, split metadata, fitted scalers, merged hyperparameters, and final feature order. Manual-final checkpoints follow the same final-test protocol but use the feature count specified in `example/manual_feature_selection.csv`.

## Reproducibility notes

- fixed development/final split seed: `40`
- fixed development-side evaluation seeds: `42`
- target values are not clipped
- Matplotlib uses the `Agg` backend in the optimization workflow
- `src/fixed_params.py` is the single source of truth for fixed model parameters
- `src/evaluation.py` is the single source of truth for 5×5 RepeatedKFold evaluation
- `uv.lock` is the exact dependency-resolution source of truth; keep `requirements.txt` generated from the lockfile when pip compatibility is needed
