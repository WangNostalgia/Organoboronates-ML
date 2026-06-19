# User Manual

This manual describes the repository as it exists in code today. Read [pipeline.md](pipeline.md) first for the conceptual six-stage overview.

## 1. Environment setup

Required runtime:

- Python >= 3.12

Recommended install:

```bash
uv sync
```

Alternative install:

```bash
pip install -r requirements.txt
```

The supported full environment includes XGBoost, LightGBM, CatBoost, and `gplearn==0.4.2` in addition to the scikit-learn stack.

## 2. Input data format

The main training entry point expects a CSV like [example/B_dataset.csv](example/B_dataset.csv).

Required structure:

- numeric descriptor columns
- one target column named `activation_energy`
- optional identifier columns such as `sub_H` and `sub_B`

`main.py` automatically reads `example/B_dataset.csv`, drops all-NaN columns, keeps numeric columns, removes `activation_energy` from `X`, and uses it as `y`.

## 3. Main commands

Full run:

```bash
python main.py --n_trials 100 --min_features 5
```

Quick run:

```bash
python main.py --n_trials 20 --min_features 5
```

Show CLI:

```bash
python main.py --help
```

Standalone y-randomization:

```bash
python example/standalone_y_randomization.py
```

## 4. CLI arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain per model |
| `--min_features` | `5` | SHAP-RFECV feature floor |
| `--force_n_features` | `None` | Exact evaluated feature count; rejected by `main.py` for the default GPlearn-containing registry |

`--force_n_features` is not a shortcut around feature elimination. It still means “evaluate the path, then choose the exact point”, but with the default registry the CLI exits immediately because GPlearn is enabled there. If you need exact feature-count forcing, call `iterative_optimization()` with a custom registry that excludes GPlearn.

## 5. What `main.py` actually does

### Stage A: fixed split at the top level

`src.iterative_optimization.iterative_optimization()` creates one 80/20 split with `random_state=40`.

- Development set: all model-selection work
- Final test set: held out until the very end

All default models share the same final-test rows.

### Stage B: default model registry

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

`GaussianProcessRegressor` is present in source but commented out.

### Stage C: development-only hyperparameter selection

`src/train_and_evaluate.py` handles parameter selection on development data only.

Current behavior:

- Optuna objective: internal 5-fold MAE on development data
- Optuna storage: in-memory study per run
- Ridge: explicit fold-local alpha loop
- Lasso: explicit fold-local alpha loop plus additional Optuna-managed parameters such as `tol`
- every fold fits feature and target scalers on training data only
- MAE and R² are computed after inverse-transform back to kcal/mol

### Stage D: SHAP-driven feature elimination

For non-GPlearn models, each iteration:

1. tunes on the current development feature subset
2. records development-path metrics
3. saves an iteration checkpoint
4. removes one feature with SHAP-RFECV logic

Removal strategy:

- if a high-correlation pair exists, remove the less important member
- otherwise remove the globally weakest feature

The path therefore continues until the configured floor is reached. `--min_features` defaults to `5`.

### Stage E: final model build and one-time final test

After path selection is finished:

1. the selected feature set is locked
2. the chosen estimator is refit on all development rows for those features
3. the untouched final test split is scored exactly once

Primary final metrics live in:

- `metrics.primary.final_test.test_mae`
- `metrics.primary.final_test.test_r2`

Secondary development-only metrics live in:

- `metrics.secondary.internal_cv.*`
- `metrics.secondary.stability.*`
- `metrics.secondary.loo.*`

Iteration checkpoints use the flat development-path schema:

- `metrics.internal_cv.*`
- `metrics.stability.*`
- `metrics.loo.*`

The final test set is never used for tuning, SHAP analysis, feature elimination, feature-count selection, LOOCV, or 100-split stability analysis.

### Stage F: saved artifacts

```text
models/<ModelName>/
├── <ModelName>_iteration_<N>_<timestamp>.joblib
├── <ModelName>_iteration_<N>_<timestamp>_metrics.txt
├── <ModelName>_final_<timestamp>.joblib
├── <ModelName>_final_<timestamp>_metrics.txt
├── final_scatter_<timestamp>.png
├── final_scatter_<timestamp>_outliers.csv
├── performance_history_<timestamp>.csv
└── performance_history_<timestamp>.png
```

The final checkpoint includes the fitted estimator, fitted scalers, final feature order, merged hyperparameters, nested primary/secondary metrics, split metadata, and removed-feature history.

## 6. Checkpoint loading

Use [example/load_checkpoint_guide.py](example/load_checkpoint_guide.py) for examples, or call `src.external_validation.load_model()` directly.

Selection rules:

- search both final and iteration checkpoints
- exact feature count required by default
- exact final preferred over exact iteration
- newest filename timestamp wins inside the preferred class
- closest-match fallback only if `allow_closest=True`

This keeps “load 5 features” deterministic unless you explicitly opt into nearest-match fallback.

## 7. External validation

The external-validation CLI lives in `src/external_validation.py`.

Examples:

```bash
python src/external_validation.py --list-models
python src/external_validation.py --model SVR --data external.csv
python src/external_validation.py --model SVR --n_features 5 --data external.csv
python src/external_validation.py --model SVR --n_features 5 --data external.csv --allow-closest
python src/external_validation.py --model SVR --data new_compounds.csv --predict-only
```

Notes:

- default behavior is exact feature-count loading
- `--allow-closest` opts into nearest-match fallback
- ensemble validation aligns members on the common original row index

## 8. Applicability domain

The applicability-domain CLI lives in `src/applicability_domain.py`.

Current behavior:

- residual calibration comes from training-set 5-fold OOF residuals
- residual scale uses MAD with finite fallbacks
- no external `sqrt(1-h)` correction is applied
- prediction-only mode becomes leverage-only if no external ground truth is present

## 9. Standalone y-randomization

The repository supports y-randomization as a standalone validation step, not as an automatic step inside `main.py`.

`example/standalone_y_randomization.py`:

- reads the configured dataset/checkpoint settings
- loads locked checkpoints
- uses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets
- reports the corrected finite-permutation p-value `(b + 1) / (m + 1)`

## 10. How to read metrics

Use the metrics like this:

- `metrics.primary.final_test.test_mae` / `metrics.primary.final_test.test_r2`: final generalization result
- `metrics.secondary.internal_cv.*`: development-set selection evidence
- `metrics.secondary.stability.*`: repeated-split robustness inside development data
- `metrics.secondary.loo.*`: sensitivity / literature-reference numbers
- `metrics.internal_cv.*`: the flat iteration-checkpoint counterpart

If an older checkpoint still exposes compatibility aliases such as `mae_test_avg`, `r2_test_avg`, or `mae_mean`, treat them as fallbacks rather than the primary schema.
