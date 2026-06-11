# Revision Implementation Plan

> **For agentic workers:** Execute tasks with superpowers:executing-plans or inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SHAP-RFECV, dual CV (5×5 RepeatedKFold + LOOCV), SVR/Ridge/Lasso hyperparameter fixes, and y-randomization per revision.md.

**Architecture:** Modifications span 3 core source files. `train_and_evaluate.py` gets CV upgrade + hyperparameter fixes. `feature_selection.py` gets SHAP-RFECV hybrid logic. `iterative_optimization.py` gets y-randomization hook + updated output. New utility `y_randomization.py` for the randomization test.

**Tech Stack:** Python 3.12, scikit-learn, Optuna, SHAP, numpy, pandas

---

### Task 1: Modify SVR epsilon range in train_and_evaluate.py

**Files:** Modify `src/train_and_evaluate.py`

- [ ] Change SVR epsilon from `trial.suggest_float("epsilon", 1e-2, 10, log=True)` to `trial.suggest_float("epsilon", 1e-3, 0.5, log=True)`
- [ ] Verify syntax: `python -c "import py_compile; py_compile.compile('src/train_and_evaluate.py', doraise=True)"`

### Task 2: Add RidgeCV/LassoCV to train_and_evaluate.py

**Files:** Modify `src/train_and_evaluate.py`
- Modify the Ridge and Lasso sections in `objective()` to use `RidgeCV`/`LassoCV` instead of Optuna-suggested alpha
- Remove `trial.suggest_float("alpha", ...)` for Ridge/Lasso, replace with CV-based selection

### Task 3: Add 5×5 RepeatedKFold evaluation function

**Files:** Modify `src/train_and_evaluate.py`
- Add `repeated_kfold_evaluate()` function that runs 5×5 RepeatedKFold and returns R², MAE with std
- Integrate into the evaluation flow after Optuna optimization

### Task 4: Modify feature_selection.py — hybrid SHAP-RFECV

**Files:** Modify `src/feature_selection.py`
- Add `shap_rfecv_selection()` function
- Modify `feature_selection()` to call SHAP-RFECV when `len(features) < 10`
- Keep original logic for `len(features) >= 10`

### Task 5: Add y-randomization module

**Files:** Create `src/y_randomization.py`
- `y_randomization_test()` function

### Task 6: Integrate y-randomization into iterative_optimization.py

**Files:** Modify `src/iterative_optimization.py`
- After best model found, call y_randomization
- Update console/log output format to show dual CV metrics

### Task 7: Update output format in iterative_optimization.py

**Files:** Modify `src/iterative_optimization.py`
- Each iteration: output 5×5 RepeatedKFold R²/MAE + LOOCV R²/MAE in table format
- Performance history CSV: add RepeatedKFold columns

### Task 8: Create modifications_explained.md

**Files:** Create `modifications_explained.md`

### Task 9: Update README.md, README_CN.md, pipeline.md

**Files:** Modify `README.md`, `README_CN.md`, `pipeline.md`

### Task 10: Final verification

- Syntax check all modified files
- Import test
