# User Manual

This manual describes the repository as it exists in code today. Read [pipeline.md](pipeline.md) first for the conceptual workflow and evaluation protocol.

## 1. Environment setup

Required runtime:

- Python >= 3.12

Primary reproducible install:

```bash
uv sync --locked
```

`pyproject.toml` declares dependency ranges, while `uv.lock` is the source of truth for the exact resolved environment. `requirements.txt` is only a pip compatibility snapshot exported from the lockfile; do not edit it by hand or treat it as a second dependency source.

If you must use pip, regenerate `requirements.txt` from the lockfile for the target platform and then install it:

```bash
uv export --locked --format requirements-txt --output-file requirements.txt
pip install -r requirements.txt
```

The supported full environment includes XGBoost, LightGBM, CatBoost, and `gplearn==0.4.2` in addition to the scikit-learn stack.

## 2. Input data format

The main training entry point expects a CSV like [example/B_dataset.csv](example/B_dataset.csv).

Recommended structure:

```text
ID, SMILES, filename, activation_energy, descriptor_1, descriptor_2, ...
```

Required for training and labelled evaluation:

- one target column named `activation_energy`
- numeric descriptor columns used as model features

Recommended metadata columns:

- `ID`
- `SMILES`
- `filename`

Metadata columns are not used as model features. `main.py` reads `example/B_dataset.csv`, drops all-NaN columns, keeps numeric descriptor columns, explicitly excludes `activation_energy` and known metadata columns from `X`, and uses `activation_energy` as `y`.

External prediction CSV files may omit `activation_energy`. In that case, external validation runs in prediction-only mode and writes predictions without MAE/R²/RMSE.

## 3. Main commands

Full run:

```bash
python main.py --n_trials 100 --min_features 5
```

Quick run:

```bash
python main.py --n_trials 20 --min_features 5
```

Stable default parallelism:

```bash
python main.py --n_trials 100 --min_features 5 --optuna_jobs 1 --model_jobs -1
```

Show CLI:

```bash
python main.py --help
```

Standalone y-randomization:

```bash
python example/standalone_y_randomization.py
```

Manual feature-count finalization:

```bash
python example/manual_selection_and_plot.py
```

## 4. CLI arguments for `main.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--n_trials` | `100` | Optuna trials per model on the development set |
| `--model_jobs` | `-1` | Estimator-internal CPU parallelism; `-1` means all available cores where supported |
| `--optuna_jobs` | `1` | Optuna trial parallelism; default serial for reproducibility and to avoid nested parallelism overload |
| `--n_jobs` | `None` | Deprecated alias for `--model_jobs` |
| `--keep_versions` | `2` | Number of recent checkpoint families to retain per model |
| `--min_features` | `5` | SHAP-RFECV feature floor |
| `--force_n_features` | `None` | Exact evaluated feature count; rejected by `main.py` for the default GPlearn-containing registry |

`--force_n_features` is not a shortcut around feature elimination. It still means “evaluate the path, then choose the exact point”. With the default registry, the CLI exits immediately because GPlearn is enabled there. If you need exact feature-count forcing, call `iterative_optimization()` with a custom registry that excludes GPlearn, or use the manual-final workflow on already generated iteration checkpoints.

## 5. What `main.py` actually does

### Stage A: fixed split at the top level

`src.iterative_optimization.iterative_optimization()` creates one 80/20 split with `random_state=40`.

- Development set: all model-selection work
- Final test set: held out until the very end

All default models share the same final-test rows. The split indices are saved in checkpoint `evaluation_protocol` metadata:

- `evaluation_protocol["development_indices"]`
- `evaluation_protocol["final_test_indices"]`

Downstream scripts reuse these indices instead of silently creating a new split.

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
- Optuna trial parallelism: controlled by `--optuna_jobs`, default `1`
- model internal parallelism: controlled by `--model_jobs`, default `-1`
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

For multi-fold SHAP consensus, non-tree explainers now use the training fold as background or masker and explain the held-out fold. This keeps the held-out feature distribution out of the explanation baseline.

The path continues until the configured floor is reached. `--min_features` defaults to `5`.

### Stage E: automatic final model build and one-time final test

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

The final test set is never used for tuning, SHAP analysis, feature elimination, feature-count selection, LOOCV, y-randomization, AD calibration, or 100-split stability analysis.

### Stage F: saved artifacts

```text
models/<ModelName>/
├── <ModelName>_iteration_<N>_<timestamp>.joblib
├── <ModelName>_iteration_<N>_<timestamp>_metrics.txt
├── <ModelName>_final_<timestamp>.joblib
├── <ModelName>_final_<timestamp>_metrics.txt
├── <ModelName>_manual_final_<N>feat_<timestamp>.joblib
├── <ModelName>_manual_final_<N>feat_<timestamp>_metrics.txt
├── final_scatter_<timestamp>.png
├── final_scatter_<timestamp>_outliers.csv
├── performance_history_<timestamp>.csv
└── performance_history_<timestamp>.png
```

The automatic final checkpoint includes the fitted estimator, fitted scalers, final feature order, merged hyperparameters, nested primary/secondary metrics, split metadata, and removed-feature history.

## 6. Manual feature-count finalization

Use `example/manual_selection_and_plot.py` when you want to manually choose each model's feature count based on development-only evidence and chemistry interpretability.

Create or edit:

```text
example/manual_feature_selection.csv
```

Format:

```csv
model_name,n_features
SVR,5
RandomForest,7
XGBoost,4
```

Run:

```bash
python example/manual_selection_and_plot.py
```

The script now performs strict manual-final generation:

1. reads `manual_feature_selection.csv`
2. exactly loads the requested iteration checkpoint for each model and feature count
3. fails if no exact checkpoint exists; it does not use closest fallback
4. reads checkpoint `development_indices` and `final_test_indices`
5. refits selected features and complete parameters on development rows
6. evaluates final test once
7. saves a manual-final checkpoint, metrics txt, and scatter plot
8. labels the artifact as `manual_feature_count_selection`

This makes manual-final artifacts comparable with automatic final artifacts. Do not choose or change manual feature counts after inspecting final-test plots.

## 7. Checkpoint loading

Use [example/load_checkpoint_guide.py](example/load_checkpoint_guide.py) for examples, or call `src.external_validation.load_model()` directly.

Selection rules:

- search both final and iteration checkpoints
- exact feature count required by default
- exact final preferred over exact iteration
- newest filename timestamp wins inside the preferred class
- closest-match fallback only if `allow_closest=True` or CLI `--allow-closest`

This keeps “load 5 features” deterministic unless you explicitly opt into nearest-match fallback.

## 8. External validation and prediction

The external-validation CLI lives in `src/external_validation.py`.

Examples:

```bash
python src/external_validation.py --list-models
python src/external_validation.py --model SVR --data external.csv
python src/external_validation.py --model SVR --n_features 5 --data external.csv
python src/external_validation.py --model SVR --n_features 5 --data external.csv --allow-closest
python src/external_validation.py --model SVR --data new_compounds.csv --predict-only
```

Ensemble example:

```bash
python src/external_validation.py --ensemble ensemble_spec.csv --data external.csv
```

Metadata behavior:

- default behavior preserves all non-feature, non-target metadata columns
- `ID`, `SMILES`, and `filename` are prioritized first when present
- prediction outputs include model features, predicted activation energy, and ground-truth/error columns when available
- `--id-cols ID,SMILES,filename,conformer_id` prioritizes a custom metadata order
- `--only-id-cols` limits output metadata to the prioritized ID columns

This makes external prediction CSVs traceable back to specific compounds, conformers, source files, or batches.

## 9. Applicability domain

The applicability-domain CLI lives in `src/applicability_domain.py`.

Example:

```bash
python src/applicability_domain.py --model SVR --n_features 5 --training example/B_dataset.csv --external external.csv
```

Current behavior:

- AD calibration defaults to checkpoint `development_indices`
- if the checkpoint lacks development indices, the CLI refuses to use the full training CSV by default
- `--allow-full-training-csv-for-ad` is required for explicitly acknowledged full-CSV calibration
- residual calibration comes from development-row 5-fold OOF residuals
- residual scale uses MAD with finite fallbacks
- descriptor-space leverage is reported
- no external `sqrt(1-h)` correction is applied
- prediction-only mode becomes leverage-only if no external ground truth is present

## 10. Standalone y-randomization

The repository supports y-randomization as a standalone validation step, not as an automatic step inside `main.py`.

`example/standalone_y_randomization.py`:

- reads the configured dataset/checkpoint settings
- loads locked checkpoints
- uses checkpoint `evaluation_protocol["development_indices"]` by default
- refuses to use the full dataset unless legacy fallback is explicitly enabled in the script
- uses the same precomputed 5×5 RepeatedKFold splits for observed and permuted targets
- reports the corrected finite-permutation p-value `(b + 1) / (m + 1)`
- saves plots at `models/y_randomization_<ModelName>.png`

## 11. How to read metrics

Use the metrics like this:

- `metrics.primary.final_test.test_mae` / `metrics.primary.final_test.test_r2`: final generalization result
- `metrics.secondary.internal_cv.*`: development-set selection evidence
- `metrics.secondary.stability.*`: repeated-split robustness inside development data
- `metrics.secondary.loo.*`: sensitivity / literature-reference numbers
- `metrics.internal_cv.*`: the flat iteration-checkpoint counterpart

If an older checkpoint still exposes compatibility aliases such as `mae_test_avg`, `r2_test_avg`, or `mae_mean`, treat them as fallbacks rather than the primary schema.
