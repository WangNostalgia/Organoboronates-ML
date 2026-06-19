# Final Review Edge Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every Important and Minor item from the final branch review without reverting the existing remediation work.

**Architecture:** Preserve the current public APIs and add narrowly scoped validation/parsing helpers. A model execution receives one timestamp family; checkpoint discovery validates canonical filenames and feature metadata; ensemble processing uses unique member and row identities while keeping display labels and original sample data separate.

**Tech Stack:** Python 3.12+, unittest/pytest, pandas, NumPy, scikit-learn, joblib.

## Global Constraints

- Use TDD for every behavior change and record the observed RED and GREEN result.
- Do not download tools or run the multi-day training workflow.
- Preserve existing correct changes and Task 7 formatter/canonical fixes.
- Final verification must use only commands actually executed in this worktree.

---

### Task 1: Run-family persistence

**Files:**
- Modify: `src/iterative_optimization.py`
- Test: `tests/test_iterative_boundary.py`

**Interfaces:**
- Consumes: existing checkpoint filenames `<Model>_iteration_<N>_<timestamp>.joblib` and `<Model>_final_<timestamp>.joblib`
- Produces: one `run_timestamp` per model execution and family-aware retention in `clean_old_versions`

- [ ] Add a regression assertion that a 3→2 path gives every iteration and final checkpoint the same timestamp.
- [ ] Run the focused test and record the expected timestamp mismatch (RED).
- [ ] Generate `run_timestamp` once per model and reuse it for checkpoints, history, scatter, and final artifacts.
- [ ] Add retention coverage proving `keep_versions=2` preserves both complete newest families.
- [ ] Run focused tests and record GREEN.

### Task 2: Canonical checkpoint discovery

**Files:**
- Modify: `src/external_validation.py`
- Test: `tests/test_external_validation.py`

**Interfaces:**
- Consumes: model directory basename as canonical model name
- Produces: anchored classification for final/iteration filenames and `len(features)` as the actual feature count

- [ ] Add tests for inconsistent `optimal_n_features`, model names containing `_final_`, and one-time discovery.
- [ ] Run each focused test to record RED.
- [ ] Implement escaped anchored regex parsing and metadata consistency validation.
- [ ] Run focused tests and record GREEN.

### Task 3: Ensemble identity, weighting, and row alignment

**Files:**
- Modify: `src/external_validation.py`
- Test: `tests/test_external_validation.py`

**Interfaces:**
- Consumes: `_loaded_from`, current/legacy metric schemas, external DataFrame rows
- Produces: unique `member_id`, separate display labels, finite positive inverse-square weights, and unique internal row IDs

- [ ] Add tests for duplicate resolved checkpoints and `allow_closest` convergence.
- [ ] Add tests for current final/iteration metric precedence and invalid MAE rejection.
- [ ] Add tests for duplicate input indexes, zero valid member rows, and disjoint member intersections.
- [ ] Run tests and record RED failures.
- [ ] Implement member identity checks, metric extraction, row-id alignment, and domain errors.
- [ ] Run focused tests and record GREEN.

### Task 4: Minor review items

**Files:**
- Modify: `src/applicability_domain.py`
- Modify: `tests/test_applicability_domain.py`
- Modify: `tests/test_examples_and_cli.py`
- Modify: `docs/superpowers/specs/2026-06-18-code-review-remediation-design.md`
- Modify documentation mentioning standalone y-randomization output
- Conditionally modify: `uv.lock`

**Interfaces:**
- Produces: complete AD summary and prediction-only interpretation, exact CLI exit-code assertion, canonical pipeline documentation, accurate y-randomization path, and warning-free focused AD tests

- [ ] Add AD summary regression tests and record RED.
- [ ] Implement total/flagged/percentage/interpretation output and record GREEN.
- [ ] Change the atomic CLI assertion to exact exit code 2.
- [ ] Correct canonical pipeline and y-randomization documentation.
- [ ] Set the relevant AD estimator/test path to `n_jobs=1` and confirm the loky warning is gone.
- [ ] Compare the three greenlet s390x lock entries with the base revision; restore only if lock validation remains truthful without downloading tools.

### Task 5: Verification, report, and commits

**Files:**
- Create or modify: `code-review-report-new.md`
- Modify: only files proven necessary by Tasks 1–4

**Interfaces:**
- Produces: evidence-backed Chinese review report and one or two logical commits

- [ ] Run focused iterative/external/AD/examples tests.
- [ ] Run full pytest and unittest discovery when available.
- [ ] Run compileall, core imports, `main.py --help`, static `rg`, and synthetic smoke checks.
- [ ] Update the report with actual counts, warnings, commands, behavior changes, risks, and complete file list.
- [ ] Inspect the full diff and `git status`.
- [ ] Commit implementation and report with clear messages.
