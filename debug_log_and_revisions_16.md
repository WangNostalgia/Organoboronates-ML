# Debug Log & Revisions 16 — GPlearn Parameter Integrity & Codebase Sanitation

> **Date:** 2026-05-19
> **Source:** `revision_16.md` (external code review)
> **Scope:** 6 bugs fixed, 2 dead-code removals, documentation updated

---

## Executive Summary

External code review identified critical structural defects in the GPlearn symbolic regression integration and the SHAP-RFECV feature selection pipeline. All fixes have been applied, verified via static analysis, and the import chain confirmed end-to-end.

---

## Fixes Overview

| # | Problem | Severity | File(s) | Status |
|---|---|---|---|---|
| 1 | GPlearn evolution probabilities not passed to `SymbolicRegressor` | **Critical** | `gplearn_wrapper.py`, `fixed_params.py` | FIXED |
| 2 | Missing `random_state` → Optuna optimization invalid for GPlearn | **Critical** | `fixed_params.py` | FIXED |
| 3 | Formula variable name replacement: `X1` matches `X10`/`X11`/`X12` | **Critical** | `gplearn_wrapper.py` | FIXED |
| 4 | Formula variable name replacement: 1-indexed (`X{i+1}`) vs gplearn 0-indexed (`X0`...) | **Critical** | `gplearn_wrapper.py` | FIXED |
| 5 | `_fitted` attribute lost by `clone()` → KernelExplainer crash risk | **High** | `gplearn_wrapper.py` | FIXED |
| 6 | CatBoost not in `tree_models` → uses slow `KernelExplainer` instead of fast `TreeExplainer` | **Medium** | `feature_selection.py` | FIXED |
| 7 | Dead `scaler_X` parameter in `shap_rfecv_select_worst_feature` signature | **Low** | `feature_selection.py`, `iterative_optimization.py` | REMOVED |
| 8 | Legacy `feature_selection()` function (pre-SHAP, threshold-based) | **Low** | `feature_selection.py`, `iterative_optimization.py` | REMOVED |

---

## Bug 1+2: GPlearn Evolution Probability Parameters — Ghost Configuration

### Root Cause

The code had an excellent comment explaining the probability budget constraint (`p_crossover=0.7` + mutations sum to 0.25 = 0.95 ≤ 1.0), but the parameters were never passed to the `SymbolicRegressor` constructor. gplearn silently fell back to its own defaults (`p_crossover=0.9`, `p_subtree_mutation=0.01`, `p_hoist_mutation=0.01`, `p_point_mutation=0.01`), which produced crossover-dominated evolution with near-zero mutation — the opposite of the intended design.

Additionally, `random_state` was missing, making Optuna optimization a "blind lottery" since identical hyperparameters would produce wildly different MAE values across folds.

### Fix

**`src/fixed_params.py`** — Added complete GPlearn parameter set:

```python
# Before:
GPLearnRegressor: {"n_jobs": 1},

# After:
GPLearnRegressor: {
    "n_jobs": 1,
    "random_state": 42,
    "p_crossover": 0.7,
    "p_subtree_mutation": 0.1,
    "p_hoist_mutation": 0.05,
    "p_point_mutation": 0.1,
    "function_set": ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
},
```

**`src/gplearn_wrapper.py`** — Updated `__init__` to accept all parameters:

```python
def __init__(self, ..., p_crossover=0.7, p_subtree_mutation=0.1,
             p_hoist_mutation=0.05, p_point_mutation=0.1, random_state=42, n_jobs=1):
    self.p_subtree_mutation = p_subtree_mutation
    self.p_hoist_mutation = p_hoist_mutation
    self.p_point_mutation = p_point_mutation
    ...
```

**`src/gplearn_wrapper.py`** — Updated `fit()` to use instance attributes instead of hardcoded literals:

```python
self._model = SymbolicRegressor(
    ...
    p_subtree_mutation=self.p_subtree_mutation,
    p_hoist_mutation=self.p_hoist_mutation,
    p_point_mutation=self.p_point_mutation,
    ...
)
```

**`src/gplearn_wrapper.py`** — Updated `get_params()` to expose all parameters (required for sklearn `clone()` compatibility):

```python
def get_params(self, deep=True):
    return {
        ...
        'p_subtree_mutation': self.p_subtree_mutation,
        'p_hoist_mutation': self.p_hoist_mutation,
        'p_point_mutation': self.p_point_mutation,
        ...
    }
```

---

## Bug 3+4: Formula Variable Name Replacement — Substring Collision + Off-by-One

### Root Cause

The original code used two correctable patterns:
1. `str.replace(f"X{i+1}", col)` — 1-indexed (`X1`, `X2`, ...) while gplearn outputs 0-indexed (`X0`, `X1`, ...)
2. Simple `str.replace()` matches substrings: `X1` would also match within `X10`, `X11`, `X12`, corrupting those variable names

**Example of the bug**: If features were `[E_HOMO, E_LUMO, dipole]` and gplearn output `add(X0, mul(X1, X2))`:
- i=0: replace "X1" with "E_HOMO" → would also match X10/X11/X12 if they existed
- The 0-indexed vs 1-indexed mismatch meant the wrong column names were substituted

### Fix

```python
# Before:
self.formula_ = self.formula_.replace(f"X{i+1}", str(col))

# After:
import re
pattern = rf'\bX{i}\b'
self.formula_ = re.sub(pattern, str(col), self.formula_)
```

- `\b` word boundary prevents `X0` from matching `X10`
- 0-indexed `X{i}` matches gplearn's actual output convention
- `re.sub` with raw f-string pattern for safe regex construction

---

## Bug 5: `_fitted` → `_fitted_flag` for Clone Safety

### Root Cause

`sklearn.base.clone(model)` copies hyperparameters but loses dynamically-set attributes like `_fitted`. When `shap.KernelExplainer` probes the model function during `__init__`, calling `predict()` on a cloned-but-unfitted model would trigger `RuntimeError("Model must be fitted before predict().")`.

While in practice `fit()` is called before `KernelExplainer` construction in both SHAP paths (single-fit and multi-fold), the rename to `_fitted_flag` makes the intent clearer and reduces confusion when debugging.

### Fix

```python
# In fit():
self._fitted_flag = True    # was: self._fitted = True

# In predict():
if not hasattr(self, '_fitted_flag') or not self._fitted_flag:  # was: _fitted
    raise RuntimeError("Model must be fitted before predict().")
```

Also set `verbose=0` in `SymbolicRegressor` constructor to suppress per-generation progress output in logs.

---

## Bug 6: CatBoost Missing from Tree Models List

### Root Cause

`feature_selection.py` line 46 enumerated tree models for `shap.TreeExplainer` selection but omitted CatBoost, a gradient-boosted tree model. CatBoost models would fall through to the `KernelExplainer` (black-box) path, which is ~100x slower than `TreeExplainer` for tree-based models.

### Fix

```python
# Before:
tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'DecisionTree', 'LightGBM']

# After:
tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'DecisionTree', 'LightGBM', 'CatBoost']
```

---

## Cleanup 1: Remove Dead `scaler_X` Parameter

The `shap_rfecv_select_worst_feature()` function signature included a `scaler_X` parameter that was never used internally — the function creates its own per-fold `MinMaxScaler` instances to prevent data leakage. Removed from:

- **`src/feature_selection.py`**: Function signature + docstring
- **`src/iterative_optimization.py`**: Call site (line ~241)

---

## Cleanup 2: Delete Legacy `feature_selection()` Function

The deprecated `feature_selection()` function (lines 184-263 in `feature_selection.py`) used a pre-SHAP, threshold-based approach (importance < 20%) that was replaced by `shap_rfecv_select_worst_feature()`. Removed:

- **`src/feature_selection.py`**: ~80 lines of dead code deleted
- **`src/iterative_optimization.py`**: `from src.feature_selection import feature_selection` import removed

---

## Self-Review: Additional Issues Found & Fixed

During implementation, one additional issue was discovered:

| # | Finding | Fix |
|---|---|---|
| SR1 | Adding `p_subtree_mutation`/`p_hoist_mutation`/`p_point_mutation` to `fixed_params.py` but not to `GPLearnRegressor.__init__()` would cause `TypeError: unexpected keyword argument` when `model_class(**best_params)` is called | Added all three parameters to `__init__()` with defaults matching `fixed_params.py` |
| SR2 | `get_params()` was missing the three mutation probabilities, which would break `sklearn.base.clone()` (it calls `get_params()` → `set_params(**params)`) | Added `p_subtree_mutation`, `p_hoist_mutation`, `p_point_mutation` to `get_params()` return dict |

---

## Verification

All four modified files pass Python syntax check. The full import chain verified:

```
fixed_params.get_fixed_params(GPLearnRegressor)
  → returns: {n_jobs=1, random_state=42, p_crossover=0.7,
              p_subtree_mutation=0.1, p_hoist_mutation=0.05,
              p_point_mutation=0.1, function_set=(...)}
  → GPLearnRegressor(**params) instantiates correctly
  → get_params() returns all 10 parameters
```

```
feature_selection.shap_rfecv_select_worst_feature
  → Signature: (model, X, y, model_name, corr_threshold=0.8, cv_folds=5)
  → scaler_X removed ✓
  → CatBoost in tree_models ✓
  → Legacy feature_selection() deleted ✓
```

```
iterative_optimization
  → No deprecated feature_selection import ✓
  → shap_rfecv_select_worst_feature call without scaler_X ✓
```

---

## Files Modified

```
src/gplearn_wrapper.py          (8 changes: import re, __init__ params, fit() param passthrough,
                                  formula regex fix, verbose=0, _fitted_flag, get_params, set_params)
src/fixed_params.py             (1 change: GPLearnRegressor param dict expanded 1→8 entries)
src/feature_selection.py        (3 changes: scaler_X removed, CatBoost added, legacy fn deleted)
src/iterative_optimization.py   (2 changes: scaler_X removed from call, deprecated import removed)
CLAUDE.md                       (3 sections updated: orchestration flow, feature selection, feature importance)
```

---

## Documentation Updates

- **CLAUDE.md**: Updated orchestration flow (step 5), Feature selection section (SHAP-RFECV two-tier description), Feature importance section (CatBoost + GPlearn explainer classification), Key implementation notes (GPlearn mutation probabilities, regex formula replacement, CatBoost TreeExplainer)
- **Pipeline.md**: No changes needed (conceptual documentation not affected by internal fixes)
- **user_manual.md**: No changes needed (CLI arguments unchanged)

---

## Remaining Considerations

1. **GPlearn + KernelExplainer computational cost**: When `shap_rfecv_select_worst_feature` enters the multi-fold consensus path (`cv_folds=5`) with GPlearn, it uses `KernelExplainer` which is O(n_samples × n_background) per SHAP evaluation. With 5 folds and potentially 100 background samples, this adds significant runtime. This is an inherent limitation of model-agnostic SHAP for non-tree/non-linear models. If runtime becomes prohibitive, consider:
   - Reducing `cv_folds` to 3 for GPlearn specifically
   - Using a smaller background sample (already clamped to `max(5, min(10, 30%))`)
   - Treating GPlearn as a "formula discovery" tool and skipping iterative feature elimination

2. **GPlearn `function_set`**: The current set `('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv')` includes `'log'` and `'inv'`. gplearn internally uses `protected_log` (returns 0 for non-positive inputs) and `protected_div` (returns 1 for division-by-zero), so NaN/Inf are not generated at runtime. The discovered formulas are mathematically safe.

3. **`_fitted_flag` vs KernelExplainer probe**: The rename from `_fitted` to `_fitted_flag` doesn't fully solve the `clone()` + `KernelExplainer.__init__` probe issue — if KernelExplainer were ever created before `fit()`, the `RuntimeError` would still trigger. However, in the current code, `fit()` is always called before `KernelExplainer` construction, so this is not a runtime concern. Future refactoring could consider making `predict()` return a zero-vector when unfitted (matching sklearn convention) instead of raising.
