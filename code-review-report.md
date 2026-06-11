# Code Review Report — Comprehensive Audit & Remediation

**Date**: 2026-06-11
**Scope**: Full project codebase (14 files in `src/` + 6 in `example/` + `main.py`)
**Methodology**: 9-angle systematic review (line-by-line diff scan, removed-behavior auditor, cross-file tracer, language-pitfall specialist, wrapper/proxy correctness, reuse analysis, simplification, efficiency, altitude check) + adversarial verification

---

## 1. Issues Discovered & Fixes Applied

### BUG-1: Dead Code — `extra` Variable Identical to `missing`

**File**: `example/manual_feature_selection.py` (line 142) — **ARCHIVED**

**Problem**: The variable `extra` was computed as `set(features) - set(data.columns)`, which is identical to `missing` computed two lines earlier. If `missing` was non-empty, a `ValueError` was raised before reaching `extra`. If `missing` was empty, `extra` was also empty. The `extra` branch was unreachable dead code. The intended logic was likely `set(data.columns) - set(features)` (CSV columns not needed by the model).

**Resolution**: The entire file was **archived to `archive/manual_feature_selection.py`** with a `DeprecationWarning` guard. Its functionality is fully superseded by `example/manual_selection_and_plot.py`.

---

### LOGIC-1: Williams Plot — `y_train` Parameter Accepted but Never Used

**File**: `src/applicability_domain.py` (lines 356–362)

**Problem**: The `_compute_williams()` function accepted a `y_train` parameter, entered an `if y_train is not None:` block, created a variable `X_tr_scaled_for_pred` by incorrectly applying a MinMaxScaler to StandardScaler-transformed data (which would produce garbage if executed), and then did nothing (`pass`). The residual standard deviation for the Williams plot was estimated **solely from external residuals** via `_estimate_residual_std(residuals_ext)`, which is less reliable than calibrating against training-set residuals.

**Fix**: Removed the dead code block entirely. Replaced it with a detailed comment explaining:
- Why training residuals are NOT computed here (the function receives StandardScaler-transformed features, but the model requires MinMaxScaler-transformed features — computing training residuals would need the original unscaled data)
- The conservative nature of the current approach (external residuals tend to be larger → wider ±3σ threshold → fewer false positives)
- How callers can obtain training residuals if needed (pass unscaled `X_train` and `y_train`, use the model's own `scaler_X`/`scaler_y`)

---

### LOGIC-2: 100-Split Loop Mutates `best_model` Before RKfold Evaluation

**File**: `src/train_and_evaluate.py` (lines 472–497)

**Problem**: The 100-split stability loop called `best_model.fit(...)` directly, overwriting the model's trained weights 100 times. After the loop, `best_model` contained weights from split 99, not split 42 (the intended training split). The subsequent `repeated_kfold_evaluate(best_model, X, y)` on line 493 received this mutated model.

**Impact assessment**: The bug's practical impact was **limited** because `repeated_kfold_evaluate()` uses `sklearn.base.clone(model)` internally, which creates a fresh unfitted estimator with the same **hyperparameters** (cloning copies constructor params, not learned weights). The incorrect weights from split 99 were discarded by `clone()`. However:
- For models with `warm_start=True` or persistent internal state across `fit()` calls, this could cause incorrect behavior
- It was a maintenance hazard — future code changes relying on `best_model` being in a known state could break silently

**Fix**: Inside the 100-split loop, replaced `best_model.fit(...)` with:
```python
fold_model = clone(best_model)
fold_model.fit(X_tr_s, y_tr_s)
y_pred_s = fold_model.predict(X_te_s)
```
This preserves `best_model`'s original state (trained on the unified split with `random_state=random_state`) for the RKfold evaluation. Added `from sklearn.base import clone` to module-level imports.

---

### QUAL-1: Duplicate Import

**File**: `src/iterative_optimization.py` (lines 18 & 22)

**Problem**: `from src.leave_one_out_validation import leave_one_out_validation` appeared twice.

**Fix**: Removed the duplicate (line 22). Python deduplicates imports automatically, so this was purely a code cleanliness issue.

---

### QUAL-2: Import Inside `objective()` Function (Called N Times)

**File**: `src/train_and_evaluate.py` (line 60, also line 320)

**Problem**: `from src.fixed_params import get_fixed_params` was imported:
1. Inside the Optuna `objective()` function (called once per trial — 100+ times per model)
2. Inside `train_and_evaluate()` function body (called once per iteration)

Python caches imports, so the performance overhead was negligible, but it's a code smell that suggests potential circular import issues.

**Fix**: Moved `from src.fixed_params import get_fixed_params` to the **module-level imports** (top of file). Also added `from sklearn.base import clone` at module level (needed by LOGIC-2 fix). Removed both internal imports.

---

### QUAL-3: `FIXED_PARAMS_MAP` Inside Function Scope

**File**: `src/fixed_params.py` (line 46)

**Problem**: The `FIXED_PARAMS_MAP` dictionary was defined inside `get_fixed_params()` and recreated on every call. The dict is small (~20 entries), so the cost is negligible, but it violates the principle of defining constants at module level.

**Fix**: Added a comment explaining the design choice — the dict is intentionally inside the function because several entries reference the runtime `n_jobs` parameter. Moving it to module level would require a two-stage initialization pattern, adding complexity for no measurable benefit.

---

### ARCH-1: Duplicate Functionality — `manual_feature_selection.py` vs `manual_selection_and_plot.py`

**Problem**: `example/manual_feature_selection.py` and `example/manual_selection_and_plot.py` both provided `load_by_feature_count`/`find_checkpoint` and `predict_external` functionality. The latter (`manual_selection_and_plot.py`) is more complete, adding integrated scatter-plot generation with RKfold metrics and CSV-driven batch processing.

**Fix**: Archived `example/manual_feature_selection.py` → `archive/manual_feature_selection.py` with `DeprecationWarning` guard. Updated all source code comments that referenced the archived file:
- `src/iterative_optimization.py`: `manual_feature_selection.py` → `manual_selection_and_plot.py`
- `src/external_validation.py`: `manual_feature_selection.py` → `manual_selection_and_plot.py`

---

## 2. Post-Fix Usage Guide

### No CLI or API changes

All fixes are internal — no command-line arguments, configuration file fields, or public API signatures changed. The pipeline runs identically:
```bash
python main.py --n_trials 100 --mae_threshold 2.0 --min_features 5
```

### Archived file migration

If you previously used `example/manual_feature_selection.py`, switch to:
```bash
python example/manual_selection_and_plot.py
```
It reads the same `example/manual_feature_selection.csv` format and provides all the same functionality plus scatter plots.

---

## 3. Remaining Concerns & Recommendations

### 3.1 Williams Plot residual estimation ✅ RESOLVED (2026-06-11)

~~The Williams plot in `applicability_domain.py` estimates residual standard deviation from external residuals only, not training residuals.~~

**Fix**: `_compute_williams()` now accepts an `X_train_unscaled` parameter. When provided (alongside `y_train`), the function uses the model's own `scaler_X` (MinMaxScaler) to transform the unscaled training features, computes training-set predictions via `model.predict()`, inverse-transforms with `scaler_y`, and derives training residuals. These training residuals drive the `_estimate_residual_std()` sigma estimation, with external residuals as fallback. This makes the Williams Plot ±3σ threshold more accurate, based on the model's typical error distribution rather than external data alone.

### 3.2 SHAP reproducibility — `shap.kmeans` non-determinism ✅ RESOLVED (2026-06-11)

~~`feature_selection.py` calls `shap.kmeans(X_te_s, n_clusters)` without a `random_state` parameter.~~ shap 0.47.2's `kmeans()` does not accept `random_state` (passing it causes TypeError — confirmed in revision_8).

**Fix**: Both `shap.kmeans()` call sites (single-fit path line 80 and multi-fold consensus path line 124) now wrap the call with numpy random state save/restore:
```python
_stashed = np.random.get_state()
np.random.seed(42)
background = shap.kmeans(X_scaled, n_clusters)
np.random.set_state(_stashed)
```
This pins the global numpy seed during KMeans initialisation without polluting other modules (Optuna, train/test splits). Affects ~7 of 16 model types that use KernelExplainer (SVR, KNR, MLP, GPR, KRR, GPlearn, ElasticNet with specific configurations).

### 3.3 CatBoost version sensitivity

CatBoost is classified as a tree model for SHAP (`TreeExplainer`). This works with CatBoost ≥1.0. If using older CatBoost versions, `TreeExplainer` may not correctly extract tree structure. The current `pyproject.toml` should specify a minimum CatBoost version.

### 3.4 Future runs will produce correct checkpoints

The off-by-one checkpoint save bug (fixed in the previous session) means **existing checkpoint files in `models/` are still affected**. Only checkpoints produced by a **fresh run** of `main.py` will have correct feature-to-model alignment. When analyzing results from the new run, the metrics in manual scatter plots will match `metrics.txt`.

---

## 4. Modified Files Summary

### Round 1 (initial code review)

| File | Change | Category |
|------|--------|----------|
| `archive/manual_feature_selection.py` | Moved from `example/`, added `DeprecationWarning` guard | Archival |
| `src/applicability_domain.py` | Removed dead `y_train` code block, added explanatory comment | Bug fix |
| `src/train_and_evaluate.py` | Consolidated imports to module level; `clone()` in 100-split loop | Bug fix + Quality |
| `src/iterative_optimization.py` | Removed duplicate `leave_one_out_validation` import; updated comment reference | Quality |
| `src/fixed_params.py` | Added comment explaining in-function dict design | Quality |
| `src/external_validation.py` | Updated comment reference to archived file | Quality |

### Round 2 (remaining concerns 3.1 & 3.2)

| File | Change | Category |
|------|--------|----------|
| `src/applicability_domain.py` | Added `X_train_unscaled` param; training residuals drive Williams sigma estimation | Enhancement |
| `src/feature_selection.py` | numpy seed save/restore around both `shap.kmeans()` calls for reproducibility | Enhancement |
| `code-review-report.md` | This report | Documentation |

### Verification Results (Round 2)

```
SYNTAX OK: src/feature_selection.py
SYNTAX OK: src/applicability_domain.py
PASS: feature_selection.py — both shap.kmeans sites have seed save/restore
PASS: applicability_domain.py — training residual pipeline complete
PASS: feature_selection.py import
PASS: applicability_domain.py import
All verification checks passed.
```
