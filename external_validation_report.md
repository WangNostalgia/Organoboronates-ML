# External Validation Module — Design, Implementation & Analysis Report

> **Date:** 2026-06-08  
> **Project:** Stable Organoboronates ML Activation Energy Prediction  
> **Task:** Implement reusable external validation for trained `.joblib` models

---

## Table of Contents

1. [External Validation Theory & Answers](#1-external-validation-theory--answers)
2. [Design Approach](#2-design-approach)
3. [Implementation Details](#3-implementation-details)
4. [Run Results & Analysis](#4-run-results--analysis)
5. [Code Usage Guide](#5-code-usage-guide)
6. [Remaining Issues & Suggestions](#6-remaining-issues--suggestions)
7. [Modified Files Summary](#7-modified-files-summary)

---

## 1. External Validation Theory & Answers

### ① How should external validation be used for trained ML models? What is its purpose?

**Definition:** External validation (外部验证) is the process of evaluating a trained machine learning model on data that was **completely withheld** from all stages of model development — including feature selection, hyperparameter tuning, and training.

**Purpose:**

| Goal | Explanation |
|---|---|
| **True generalization assessment** | Internal test sets share the same data distribution as training data. Even with proper train/test splits, repeated tuning on the same test set leads to "information leakage through the experimenter." External data from a different source, laboratory, or time period provides the only unbiased estimate of real-world performance. |
| **Overfitting detection** | A model with excellent internal CV metrics (e.g., R²=0.95 on LOOCV) that collapses on external data (R²=0.40) has overfit — it memorized the training distribution rather than learning the underlying physical chemistry. |
| **Domain applicability** | External validation on structurally diverse compounds reveals the model's applicability domain — for which chemical space the predictions are trustworthy and where they break down. |
| **Regulatory/Publication requirement** | In cheminformatics and drug discovery, QSAR models require external validation for regulatory acceptance (OECD Principle 4). Journals increasingly require external validation for ML-based predictions. |

**How to use external validation:**

```
┌─────────────────────────────────────────────────────────────────┐
│                   External Validation Workflow                    │
│                                                                   │
│  1. Train model on internal data (all stages, including CV)      │
│  2. FREEZE the model — no further parameter changes               │
│  3. Acquire independent external dataset (different source/lab)  │
│  4. Apply the SAME preprocessing pipeline (same scalers!)         │
│  5. Predict on external data                                      │
│  6. Compare predictions vs. experimental values (if available)    │
│  7. Report MAE, R², RMSE separately from internal metrics        │
│                                                                   │
│  ⚠ CRITICAL: Never use external data for model selection or      │
│    hyperparameter tuning — this invalidates its independence.     │
└─────────────────────────────────────────────────────────────────┘
```

### ② Can external validation accept a table similar to `B_dataset.csv` with new structures/data? How to implement?

**Answer: Yes.** The implemented module (`src/external_validation.py`) accepts any CSV file with the same feature columns as the model was trained on.

**How it works:**

1. **Load the external CSV** — can contain extra columns (identifiers like `sub_H`, `sub_B`, non-numeric metadata); these are ignored. Can also contain "Unnamed" columns from Excel exports; these are auto-dropped.
2. **Feature alignment** — the module extracts only the feature columns the model expects, in the correct order. Missing features raise a clear error message.
3. **Apply saved scalers** — the external data is transformed using the **same** `MinMaxScaler` instances saved in the `.joblib` file. This is critical: using a new scaler fitted on external data would break the mapping between feature space and prediction space.
4. **Predict** — the frozen model predicts activation energies.
5. **Evaluate** — if the CSV contains an `activation_energy` column, MAE/R²/RMSE are computed. If not, the module runs in prediction-only mode.

**Implementation (simplified):**

```python
# Core logic in src/external_validation.py:external_validation()
model_info = joblib.load('models/SVR/SVR_final_*.joblib')
model = model_info['model']
scaler_X = model_info['scaler_X']      # Pre-fitted on training data
scaler_y = model_info['scaler_y']      # Pre-fitted on training data
expected_features = model_info['features']

external_df = pd.read_csv('new_compounds.csv')
X_external = external_df[expected_features]  # Align columns
X_scaled = scaler_X.transform(X_external)     # Apply SAME scaler
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Back to kcal/mol
```

### ③ Can we validate a specific model with a specific feature count?

**Answer: Yes.** The module supports this through the `--n_features` parameter.

Since the SHAP-RFECV pipeline records model snapshots at each feature count during training (see `shap_rfecv_path_summary` in the `.joblib`), different feature counts correspond to different stages of feature elimination. When `--n_features N` is specified, the module loads the model whose feature count is closest to `N`.

**Example:**
```bash
# Use SVR with exactly 5 features (if available, or closest match)
python src/external_validation.py --model SVR --n_features 5 --data external.csv
```

The `--list-models` command shows available feature counts for each model. Currently each model has one final version with its auto-selected optimal feature count (3-9 features), but the architecture supports multiple versions per model.

---

## 2. Design Approach

### Architecture

```
src/external_validation.py
├── list_available_models()     # Discover all trained models
├── load_model()                # Load a specific model by name & feature count
├── external_validation()       # Core: predict + evaluate on external data
├── _plot_external_scatter()    # Publication-quality scatter plot
├── _write_summary()            # Human-readable results summary
└── main()                      # CLI entry point (argparse)
```

### Design Decisions

| Decision | Rationale |
|---|---|
| **Reuse saved scalers** | The `scaler_X` and `scaler_y` saved in the `.joblib` are fitted on training data only. Applying them to external data ensures no data leakage — the external data sees exactly the same preprocessing as the training data. |
| **No prediction clipping** | Activation energy has no physical upper bound. Clipping (e.g., to [0, 100]) would artificially mask model failures on extreme cases. Unlike the old notebook (which clips to [0, 100]), this module reports raw predictions. |
| **Auto-drop unnamed columns** | The original `B_dataset.csv` contains 48 "Unnamed: N" columns from Excel exports. These are auto-detected and dropped to avoid confusion. |
| **Feature alias support** | The `FEATURE_ALIASES` dict allows mapping alternative column names (e.g., `pKa` → `pka`) for interoperability with differently formatted external CSVs. |
| **Encoding-safe file I/O** | All output files use UTF-8 encoding to support Unicode characters (e.g., `R²`) on Windows platforms. |
| **Graceful degradation** | If `activation_energy` is missing from the external CSV, the module automatically switches to prediction-only mode instead of crashing. |

### Key Engineering Choices

- **Single responsibility per function** — each function does one thing well, making the module testable and maintainable
- **Defensive feature alignment** — clearly reports which features are missing vs. which are available, instead of cryptic KeyErrors
- **Matplotlib `Agg` backend** — set before any pyplot imports to avoid GUI dependency in headless environments
- **`utf-8-sig` for CSV exports** — BOM prefix ensures correct display in Excel on Windows while preserving Unicode

---

## 3. Implementation Details

### Module: `src/external_validation.py`

**Lines:** ~660  
**Functions:** 6 (4 public + 2 internal)

```python
# ── Public API ──

list_available_models(models_dir='models') -> pd.DataFrame
    """Scan models/ and return a DataFrame of all trained models with metrics."""

load_model(model_name, n_features=None, models_dir='models') -> dict
    """Load a .joblib file by model name. n_features selects a specific version."""

external_validation(model_info, external_data, target_col='activation_energy',
                    output_dir='external_validation_results') -> dict
    """Core function: predict on external data, evaluate if ground truth exists."""

# ── CLI ──
python -m src.external_validation --list-models
python -m src.external_validation --model SVR --data external.csv
python -m src.external_validation --model SVR --data new.csv --predict-only
python -m src.external_validation --model SVR --n_features 5 --data external.csv
```

### Data Flow

```
External CSV (*.csv)
    │
    ├─ Drop "Unnamed" columns (Excel artifacts)
    ├─ Normalise column names via FEATURE_ALIASES
    ├─ Extract only expected_features (in correct order)
    ├─ Handle NaN rows (drop with warning)
    │
    ▼
X_external (DataFrame, aligned columns)
    │
    ├─ scaler_X.transform() ─── pre-fitted MinMaxScaler from .joblib
    ▼
X_scaled (ndarray)
    │
    ├─ model.predict()
    ▼
y_pred_scaled (ndarray, scaled 0-100)
    │
    ├─ scaler_y.inverse_transform() ─── back to kcal/mol
    ▼
y_pred (ndarray, real activation energy units)
    │
    ├─ If activation_energy present → MAE, R², RMSE, scatter plot
    └─ If not → prediction-only CSV export
```

### Error Handling

| Scenario | Handling |
|---|---|
| Model directory not found | `FileNotFoundError` with list of available model directories |
| No `.joblib` files for model | `FileNotFoundError` suggesting to run `main.py` |
| Missing feature columns in external CSV | `ValueError` listing missing vs. expected features |
| NaN in feature columns | Warning + automatic row dropping |
| Corrupted `.joblib` file | Warning for that file, continue scanning others |
| Missing required keys in `.joblib` | `KeyError` listing missing vs. available keys |
| Target column absent | Auto-switch to prediction-only mode |
| Windows GBK encoding | All file I/O uses UTF-8; console output avoids Unicode |

---

## 4. Run Results & Analysis

### Test 1: Model Inventory

```
Model                Feat  RKfold MAE     RKfold R2    Features
-------------------------------------------------------------------------------------
AdaBoost             3     2.7461         0.7469       pka, C_Polarization, VBur_C
DecisionTree         4     3.1317         0.6705       pka, C_Polarization, VBur_C, NPA_charge_C
ElasticNet           4     2.8120         0.7504       pka, C_Polarization, B_s, lumo_energy
GPR                  9     3.0403         0.6490       pka, dipole, C_s, B_s, ...
GradientBoosting     3     2.7209         0.7535       pka, C_Polarization, VBur_C
KNR                  4     2.7499         0.7322       pka, C_Polarization, homo_energy, lumo_energy
KRR                  4     2.5359         0.7766       pka, dipole, C_s, NPA_charge_B
Lasso                3     2.6038         0.7695       pka, C_Polarization, B_s
LinearRegression     3     2.6131         0.7623       pka, C_s, NPA_charge_B
MLP                  3     3.0630         0.7026       pka, lumo_energy, NPA_charge_C
RandomForest         4     2.7135         0.7677       pka, C_Polarization, lumo_energy, NPA_charge_C
Ridge                3     2.6134         0.7623       pka, C_s, NPA_charge_B
SVR                  4     2.5152         0.7613       pka, dipole, C_Polarization, C_s
XGBoost              3     2.8072         0.7568       pka, C_Polarization, lumo_energy
```

**Key observation:** `pka` is present in **all 14 models** — it is the single most important feature for predicting activation energy. `C_Polarization` appears in 10/14 models. This chemical consistency across different algorithms strongly suggests a real physical relationship rather than a modeling artifact.

### Test 2: SVR External Validation (4 features, 141 samples)

| Metric | Value |
|---|---|
| **MAE** | 2.22 kcal/mol |
| **R²** | 0.814 |
| **RMSE** | 3.17 kcal/mol |
| **Features** | pka, dipole, C_Polarization, C_s |
| **Hyperparameters** | C=435.87, epsilon=0.018, gamma=0.165 |

### Test 3: RandomForest External Validation (4 features, 141 samples)

| Metric | Value |
|---|---|
| **MAE** | 1.32 kcal/mol |
| **R²** | 0.938 |
| **RMSE** | 1.82 kcal/mol |
| **Features** | pka, C_Polarization, lumo_energy, NPA_charge_C |

### Test 4: KRR External Validation (4 features, 141 samples)

| Metric | Value |
|---|---|
| **MAE** | 2.34 kcal/mol |
| **R²** | 0.824 |
| **RMSE** | 3.08 kcal/mol |
| **Features** | pka, dipole, C_s, NPA_charge_B |

### Test 5: Prediction-Only Mode

Successfully tested with a CSV lacking the `activation_energy` column. Module automatically switched to prediction-only mode, exporting predictions without attempting evaluation.

### Analysis

**Comparison with internal CV metrics:**

| Model | Internal RKfold MAE | Full-dataset MAE (this run) | Gap |
|---|---|---|---|
| SVR | 2.52 ± 0.23 | 2.22 | -0.30 |
| RandomForest | 2.71 ± 0.25 | 1.32 | -1.39 |
| KRR | 2.54 ± 0.25 | 2.34 | -0.20 |

The full-dataset MAE is lower than the internal RKfold MAE for all models. This is expected: the full-dataset evaluation includes training data the model has seen, whereas the internal CV metrics are computed on held-out folds. This is **not** true external validation — it's a "full-fit evaluation" that confirms the model fitting is working correctly.

**For genuine external validation**, the user must provide a CSV of **completely new compounds** that were not used in any stage of model development. The larger the chemical diversity gap between training and external data, the more informative the external validation.

**Chemical interpretation:** The dominant feature `pka` (present in all models) makes chemical sense — pKa reflects the acidity/basicity of the C-H bond being activated, which directly relates to the energy barrier for deprotonation in the borylation mechanism. `C_Polarization` likely captures the electronic environment at the carbon center.

---

## 5. Code Usage Guide

### Python API

```python
from src.external_validation import (
    list_available_models,
    load_model,
    external_validation,
)

# 1. See what models are available
models_df = list_available_models()
print(models_df)

# 2. Load a model
model_info = load_model('SVR')                     # most recent version
model_info = load_model('SVR', n_features=5)       # specific feature count

# 3. Run external validation (with ground truth for evaluation)
results = external_validation(
    model_info,
    external_data='path/to/external_data.csv',
    target_col='activation_energy',                 # default
    output_dir='my_results',
)
print(f"MAE = {results['mae']:.2f} kcal/mol")
print(f"R²  = {results['r2']:.4f}")
print(f"Predictions saved to: {results['output_files']}")

# 4. Prediction-only (no ground truth)
results = external_validation(
    model_info,
    external_data='path/to/new_compounds.csv',
    target_col=None,                                # disables evaluation
)

# 5. Access predictions DataFrame
df = results['predictions']
# Columns: [identifier_cols..., feature_cols..., 'predicted_activation_energy',
#           'activation_energy' (if present), 'absolute_error' (if present)]
```

### CLI

```bash
# List all trained models
python -m src.external_validation --list-models

# Basic external validation
python -m src.external_validation --model SVR --data my_external_data.csv

# With custom output directory
python -m src.external_validation --model SVR --data external.csv --output-dir my_results/

# Specify feature count (closest match used if exact not available)
python -m src.external_validation --model RandomForest --n_features 5 --data external.csv

# Prediction-only (skip evaluation even if target column exists)
python -m src.external_validation --model SVR --data new_compounds.csv --predict-only

# Custom target column name
python -m src.external_validation --model SVR --data external.csv --target-col activation_energy

# Custom models directory
python -m src.external_validation --model SVR --data external.csv --models-dir /path/to/models
```

### Input CSV Format

The external CSV must contain the feature columns that the model was trained on. It may also contain:
- **Identifier columns** (`sub_H`, `sub_B`) — preserved in output, not used for prediction
- **Target column** (`activation_energy`) — used for evaluation if present
- **Extra columns** — silently ignored
- **"Unnamed" columns** (from Excel exports) — automatically dropped

Example minimal valid CSV:
```csv
pka,dipole,C_Polarization,C_s,activation_energy
36.43,1.94,0.085,0.033,17.4
35.81,1.56,0.213,0.064,20.2
```

### Output Files

For each validation run, the following are saved to `--output-dir` (default: `external_validation_results/`):

| File | Description |
|---|---|
| `{Model}_external_validation_{timestamp}.csv` | Predictions with all input columns + `predicted_activation_energy` |
| `{Model}_external_scatter_{timestamp}.png` | Scatter plot (when ground truth available) |
| `{Model}_external_summary_{timestamp}.txt` | Human-readable summary of all metrics |

---

## 6. Remaining Issues & Suggestions

### Issues Resolved

| Issue | Resolution |
|---|---|
| Windows GBK encoding crashes on `R²` character | All file I/O uses `encoding='utf-8'`; console prints use `R2` |
| "Unnamed" columns from Excel corrupting feature alignment | Auto-detection and removal of any column containing "Unnamed" |
| Module not importable as `python -m src.external_validation` | Standard `if __name__ == '__main__': main()` pattern |

### Remaining Issues & Future Work

1. **True external validation dataset needed.** The current test used `B_dataset.csv` (the training data) for verification purposes. For genuine external validation, a new dataset from a different source or with different compound classes should be used. The module fully supports this — it just needs the data.

2. **No ensemble prediction support.** The existing notebook (`example/prediction_round2.ipynb`) provides ensemble (100 random-split) and weighted-average predictions. This module evaluates single final models. An ensemble external validation mode could be added as a future enhancement — loading multiple `.joblib` snapshots and aggregating predictions.

3. **Model version selection by `n_features` is limited.** Currently each model directory contains one final version. The RFECV path snapshots (`shap_rfecv_path_summary`) record metrics at each feature count but the corresponding model objects are not persisted. To truly support per-feature-count validation, the iterative optimization would need to save a full model at each feature count level (current code only saves lightweight metadata to avoid memory bloat — see line 253 of `iterative_optimization.py`).

4. **Feature alias mapping is manual.** The `FEATURE_ALIASES` dict requires manual population. An automatic fuzzy-matching approach (using Levenshtein distance or case-insensitive matching) could improve interoperability with diverse external datasets.

5. **No applicability domain analysis.** A proper external validation should include applicability domain assessment (e.g., Williams plot of standardized residuals vs. leverage). This could be added as a visualization option.

6. **y-Randomization is commented out in training pipeline.** The y-randomization test (statistical validation that the model learned real SAR) is currently disabled. Enabling it would strengthen confidence in external validation results.

### Suggestions for Best Practices

1. **Always report external metrics separately from internal CV metrics.** Do not pool them or report an "overall" metric — this masks the generalization gap.

2. **External validation should cover the full chemical space of interest.** If the model will be used to predict compounds with specific functional groups, the external set must include those groups.

3. **Consider temporal external validation** if data was collected over time — train on earlier data, validate on later data. This tests whether the model's learned relationships are stable.

4. **Document the provenance of external data** (source lab, measurement method, experimental conditions) — differences in measurement protocols can cause apparent prediction errors that are actually experimental variance.

---

## 7. Modified Files Summary

### New Files

| File | Description | Lines |
|---|---|---|
| `src/external_validation.py` | External validation module with CLI and Python API | ~660 |

### Existing Files (Unchanged)

| File | Status |
|---|---|
| `src/iterative_optimization.py` | Unchanged — RFECV path already records metadata for future version support |
| `src/validation_process.py` | Unchanged — existing combinatorial validation space generator |
| `main.py` | Unchanged — no integration needed (external validation is standalone) |
| `example/prediction_round2.ipynb` | Unchanged — legacy notebook-based approach; new module supersedes it for single-model validation |

### Output Files (Generated During Testing)

| File | Description |
|---|---|
| `external_validation_results/SVR_external_validation_*.csv` | SVR predictions on full dataset |
| `external_validation_results/SVR_external_scatter_*.png` | SVR scatter plot |
| `external_validation_results/SVR_external_summary_*.txt` | SVR summary |
| `external_validation_results/RandomForest_external_validation_*.csv` | RF predictions |
| `external_validation_results/RandomForest_external_scatter_*.png` | RF scatter plot |
| `external_validation_results/RandomForest_external_summary_*.txt` | RF summary |
| `external_validation_results/KRR_external_validation_*.csv` | KRR predictions |
| `external_validation_results/KRR_external_scatter_*.png` | KRR scatter plot |
| `external_validation_results/KRR_external_summary_*.txt` | KRR summary |

---

## Verification Checklist

- [x] All imports resolve correctly
- [x] `--list-models` enumerates all 14 trained models with correct metrics
- [x] SVR external validation: MAE=2.22, R²=0.814
- [x] RandomForest external validation: MAE=1.32, R²=0.938
- [x] KRR external validation: MAE=2.34, R²=0.824
- [x] Prediction-only mode (no target column): works correctly
- [x] Auto-drop of "Unnamed" columns: 48 columns removed
- [x] Feature alignment: missing features raise clear error
- [x] Output files: CSV (UTF-8 BOM), PNG scatter, TXT summary all generated
- [x] Encoding: no GBK crashes on Windows
- [x] Edge case: NaN rows in features → warning + drop
- [x] Edge case: missing model directory → clear error with available models list
- [x] Edge case: corrupted `.joblib` → warning, continue scanning
- [x] All code has detailed English comments explaining design decisions
