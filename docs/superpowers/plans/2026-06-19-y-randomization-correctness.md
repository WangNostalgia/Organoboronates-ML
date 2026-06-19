# y-Randomization Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make y-randomization statistically consistent, estimator-faithful, and standalone-checkpoint loading exact by default.

**Architecture:** Keep evaluation logic centralized in `src.evaluation`, update `src.y_randomization` to reuse precomputed repeated-k-fold splits and either clone a stored estimator or build one via `src.model_utils.build_model`, and switch the standalone script to Task 4 checkpoint loading so it passes the stored estimator directly.

**Tech Stack:** Python, scikit-learn, pandas, pytest/unittest, joblib

## Global Constraints

- `y_randomization_test(..., model=None, model_class=None, best_params=None, n_jobs=-1, ...)` must accept either a full estimator or a build path, with clear conflict/missing-input errors.
- Observed and permuted evaluations must reuse one `make_repeated_kfold_splits(..., random_state=42)` result.
- All fold-local scaling and MAE-in-original-units logic must stay inside `src.evaluation.repeated_kfold_evaluate`.
- `n_permutations` must be a positive integer.
- Corrected MAE p-value must use `(better_or_equal + 1) / (n_permutations + 1)`.
- Standalone loading must default to exact matching and pass `info['model']` to `y_randomization_test`.
- Do not re-enable expensive automatic y-randomization in the training pipeline.

---

### Task 1: Lock the intended behavior with tests

**Files:**
- Modify: `tests/test_y_randomization.py`
- Test: `tests/test_y_randomization.py`

**Interfaces:**
- Consumes: `src.y_randomization.y_randomization_test`, `example.standalone_y_randomization.main`
- Produces: failing tests for estimator cloning, split reuse, corrected p-values, input validation, and standalone exact loading

- [x] **Step 1: Write the failing tests**

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_y_randomization.py -v`
Expected: FAIL on missing split reuse hook, legacy signature, missing validation, and standalone exact-loading behavior.

### Task 2: Make `src.y_randomization` statistically consistent

**Files:**
- Modify: `src/y_randomization.py`
- Test: `tests/test_y_randomization.py`

**Interfaces:**
- Consumes: `src.evaluation.make_repeated_kfold_splits`, `src.evaluation.repeated_kfold_evaluate`, `src.model_utils.build_model`
- Produces: `y_randomization_test(model_class=None, best_params=None, X=None, y=None, selected_features=None, n_permutations=100, random_state=42, model=None, n_jobs=-1) -> dict`

- [ ] **Step 1: Keep the failing test in place**

- [ ] **Step 2: Implement argument validation, estimator construction/cloning, shared splits, and corrected p-value**

- [ ] **Step 3: Run targeted tests to verify GREEN**

Run: `pytest tests/test_y_randomization.py -v`
Expected: PASS

### Task 3: Switch standalone loading to exact checkpoint selection

**Files:**
- Modify: `example/standalone_y_randomization.py`
- Test: `tests/test_y_randomization.py`

**Interfaces:**
- Consumes: `src.external_validation.load_model`, `src.y_randomization.y_randomization_test`
- Produces: exact-by-default standalone flow using `MODELS_DIR`, `ALLOW_CLOSEST`, and `model=info['model']`

- [ ] **Step 1: Replace legacy checkpoint finder path with Task 4 load_model usage**

- [ ] **Step 2: Re-run targeted tests**

Run: `pytest tests/test_y_randomization.py -v`
Expected: PASS

### Task 4: Verify the full change set

**Files:**
- Modify: `src/y_randomization.py`, `example/standalone_y_randomization.py`, `tests/test_y_randomization.py`

**Interfaces:**
- Consumes: repository test/verification commands
- Produces: verified, commit-ready change set

- [ ] **Step 1: Run the full test suite**

Run: `pytest`
Expected: PASS

- [ ] **Step 2: Run compile verification**

Run: `python -m compileall src example tests`
Expected: PASS

- [ ] **Step 3: Run LinearRegression/Ridge smoke checks**

Run: targeted Python smoke command invoking `y_randomization_test` with small `n_permutations`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/y_randomization.py example/standalone_y_randomization.py tests/test_y_randomization.py docs/superpowers/plans/2026-06-19-y-randomization-correctness.md
git commit -m "fix: make y randomization statistically consistent"
```
