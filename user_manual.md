# User Manual

This manual explains how to run the current repository as it exists in code today. For the conceptual six-stage overview, read [Pipeline.md](Pipeline.md) first.

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

The supported full installation includes the default model stack: XGBoost, LightGBM, CatBoost, and GPlearn in addition to the scikit-learn models.

## 2. Input data format

The main training entry point expects a CSV like [example/B_dataset.csv](example/B_dataset.csv).

Required structure:

- numeric descriptor columns
- one target column named `activation_energy`
- optional identifier columns such as `sub_H` and `sub_B`

`main.py` automatically:

- reads `example/B_dataset.csv`
- drops all-NaN columns
- keeps numeric columns
- removes `activation_energy` from `X`
- uses `activation_energy` as `y`

## 3. Main training commands

Full run:

```bash
python main.py --n_trials 100 --min_features 5
```

Quick run:

```bash
python main.py --n_trials 20 --min_features 5
```

Show current CLI:

```bash
python main.py --help
```

Notebook entry points:

```bash
jupyter notebook example/example_pic.ipynb
jupyter notebook example/prediction_round2.ipynb
```

Standalone y-randomization:

```bash
python example/standalone_y_randomization.py
```

## 4. CLI arguments

| Argument | Default | What it controls |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--n_jobs` | `-1` | CPU cores (`-1` = all available) |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain per model |
| `--min_features` | `5` | Stopping floor for SHAP-RFECV path evaluation |
| `--force_n_features` | `None` | Select an exact evaluated feature count after the path has been evaluated |

`--force_n_features` is not a shortcut around feature elimination. It still evaluates the development-set SHAP-RFECV path and then selects the requested exact point from that path.

## 5. What `main.py` actually does

### Stage A: fixed split at the top level

`src.iterative_optimization.iterative_optimization()` creates one 80/20 split with `random_state=40`.

- Development set (80%): all model selection activity
- Final test set (20%): held out until the very end

All models share that same split, so their final test metrics are paired on the same samples.

### Stage B: model registry

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

`GaussianProcessRegressor` is present in the source but commented out.

### Stage C: development-only hyperparameter selection

`src/train_and_evaluate.py` handles parameter selection on development data only.

Current behavior:

- Optuna objective: internal 5-fold MAE on the development subset
- Optuna storage: in-memory study per run
- Ridge: explicit inner-fold alpha loop
- Lasso: explicit inner-fold alpha loop plus Optuna tuning for `tol`
- Every fold fits feature and target scalers on training data only
- MAE is computed after inverse-transform back to kcal/mol

### Stage D: SHAP-driven feature elimination

For non-GPlearn models, each iteration:

1. tunes on the current development feature subset
2. records development-only metrics
3. saves an iteration checkpoint
4. removes one feature using SHAP-RFECV logic

Removal strategy:

- if a high-correlation pair is present, remove the less important member
- otherwise remove the globally weakest feature

Mode switching:

- many features: coarse single-fit SHAP
- few features: 5-fold consensus SHAP-RFECV

The loop stops when either:

- remaining features reach `min_features`
- no more features qualify for removal

### Stage E: final model build and one-time final test

After path selection is finished:

1. the selected feature set is locked
2. the chosen estimator is refit on all development rows for those features
3. the untouched final test split is scored exactly once

Primary final metrics:

- `test_mae`
- `test_r2`

Secondary development-only metrics:

- `internal_cv.rkf_mae_mean/std`
- `internal_cv.rkf_r2_mean/std`
- `stability.mae_mean/std`
- `loo.mae`
- `loo.r2`

The final test set is not used for:

- tuning
- SHAP analysis
- feature elimination
- feature-count selection
- LOOCV
- 100-split stability analysis

### Stage F: saved artifacts

Per-model artifacts:

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

The final checkpoint includes:

- fitted estimator
- fitted `scaler_X`
- fitted `scaler_y`
- final feature order
- complete merged hyperparameters
- primary and secondary metrics
- split metadata
- removed-feature history

## 6. Checkpoint loading

Use [example/load_checkpoint_guide.py](example/load_checkpoint_guide.py) for interactive examples, or call `src.external_validation.load_model()` directly.

Selection rules:

- search both final and iteration checkpoints
- exact feature count required by default
- exact final preferred over exact iteration
- newest filename timestamp wins inside the preferred class
- closest-match fallback only if `allow_closest=True`

This means “load 5 features” is deterministic and will not silently pick a nearby checkpoint unless you explicitly allow that.

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
- ensemble validation aligns members on the common original row index, not by positional truncation

## 8. Applicability domain

The applicability-domain CLI lives in `src/applicability_domain.py`.

Example:

```bash
python src/applicability_domain.py --model SVR --n_features 5 --training example/B_dataset.csv --external external.csv
```

Current behavior:

- residual calibration comes from training-set 5-fold OOF residuals
- residual scale uses MAD with finite fallbacks
- no external `sqrt(1-h)` correction is applied
- prediction-only mode becomes leverage-only if no external ground truth is present

## 9. Standalone y-randomization

The repository currently supports y-randomization as a standalone validation step, not as an automatic step inside `main.py`.

`example/standalone_y_randomization.py`:

- reads `example/manual_feature_selection.csv`
- loads locked checkpoints
- uses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets
- reports the corrected finite-permutation p-value `(b + 1) / (m + 1)`

Before running it, edit the configuration block near the top of the script if you need a different CSV, model directory, dataset path, or permutation count.

## 10. Practical reading of metrics

Use the metrics like this:

- `test_mae` / `test_r2`: the main final generalization result
- internal CV metrics: development-set selection evidence
- stability MAE: repeated split robustness inside development data
- LOOCV metrics: sensitivity / literature-reference numbers only

If a report or old notebook still shows compatibility aliases such as `mae_mean`, `mae_test_avg`, or `r2_test_avg`, treat them according to the metric-role notes in the current saved checkpoint, not according to older workflow descriptions.
