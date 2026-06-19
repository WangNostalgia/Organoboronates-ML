# Code Review Remediation Design

Date: 2026-06-18

## Objective

Implement the fixes authorized in `decision.md` while preserving unrelated
working-tree changes. The repaired workflow must use one fixed development/test
boundary, keep all model selection inside the development set, save estimators
with the exact configuration that was evaluated, and make prediction and
applicability-domain outputs sample-aligned and statistically defensible.

## Scope

Included:

- CR-1, CR-2, CR-3
- H-1, H-2, H-3, H-4
- M-1, M-2, M-3, M-4
- L-1
- Focused regression tests needed to prove these fixes
- Second-round review and update of `code-review-report-new.md`

Excluded:

- Broad test-suite or CI infrastructure work under M-5
- Deleting, moving, or archiving files
- Changes inside the `sissopp` third-party submodule
- Unrelated cleanup or style-only refactors
- Reproducing the full multi-day training run

## Evaluation Architecture

### Fixed data boundary

`iterative_optimization()` creates one 80/20 split for each run using
`random_state=40`.

- Development set (80%): all hyperparameter tuning, SHAP feature elimination,
  feature-count selection, internal 5-fold CV, LOOCV sensitivity analysis, and
  100-split stability analysis.
- Final test set (20%): never passed to tuning, SHAP, feature selection,
  internal CV, LOOCV, or stability analysis. It is evaluated once after the
  feature set and hyperparameters are locked.

The split is shared across all model families so their final test results are
paired on the same samples.

### Metric roles

- `test_mae` and `test_r2`: primary final generalization metrics, computed once
  on the untouched 20% test set.
- `internal_cv_mae_mean/std` and `internal_cv_r2_mean/std`: development-set
  selection metrics from 5×5 RepeatedKFold.
- `loo_mae` and `loo_r2`: development-set sensitivity/literature-reference
  metrics only.
- `stability_mae_mean/std`: secondary development-set robustness analysis from
  repeated 80/20 splits within the development set.

Existing RKF-oriented fields may be retained as compatibility aliases where
needed, but saved reports and documentation must state that they are internal
development metrics, not final performance estimates.

### Training flow

1. Split the full dataset once into development and final test sets.
2. For each model and feature subset, tune parameters using only the
   development set.
3. Evaluate that locked parameter set with internal 5×5 RKF on the development
   set.
4. Run SHAP feature elimination using only development data.
5. Select the feature count from the development-set internal CV path.
6. Build the final estimator with complete fixed and tuned parameters.
7. Fit feature and target scalers on the complete development set.
8. Fit the final estimator on the complete scaled development set.
9. Evaluate once on the untouched final test set.
10. Save the fitted estimator, fitted development scalers, exact feature order,
    complete hyperparameters, metric roles, and split metadata.

## Model Construction and Hyperparameter Selection

A shared model-construction helper will merge:

```python
{**get_fixed_params(model_class, n_jobs), **best_params}
```

All tuning, final fitting, y-randomization, and reconstruction paths must use
this helper or clone an already complete estimator. Saved `hyperparameters`
must contain the complete merged parameter set.

Ridge and Lasso alpha selection will use explicit nested fold loops:

- Each inner fold fits feature and target scalers only on that fold's training
  portion.
- Candidate alphas are evaluated in original kcal/mol after inverse transform.
- The selected alpha is then evaluated by the normal development-set internal
  CV path.

This avoids fitting preprocessing on inner validation folds.

## External Validation and Model Loading

### Exact feature-count loading

`load_model()` will search both final and iteration checkpoints.

- If `n_features` is supplied, an exact match is required by default.
- Exact final checkpoints have priority over exact iteration checkpoints.
- Within the selected checkpoint class, choose the candidate with the latest
  filename timestamp (`YYYYMMDD_HHMMSS`).
- If more than one highest-priority candidate has the same latest timestamp and
  feature count, raise an ambiguity error instead of selecting by filesystem
  order or modification time.
- Missing exact matches raise a clear error listing available counts.
- A closest match is allowed only with an explicit `allow_closest=True`. Closest
  candidates are ordered by absolute feature-count distance, then checkpoint
  class (final before iteration), then newest filename timestamp. A remaining
  tie raises an ambiguity error.
- Returned metadata records requested count, actual count, checkpoint type, and
  source path.

### Ensemble alignment

Each member prediction is stored as a `pandas.Series` indexed by the original
external-data index. Ensemble validation uses the intersection of valid indexes
across all successful members, as authorized in `decision.md`.

- IDs, targets, individual predictions, and aggregate predictions are joined by
  index.
- No positional truncation is allowed.
- The output states how many rows were excluded because they were not complete
  for every member.
- Labels and summaries use the actual loaded feature count.

## Applicability Domain

### Residual calibration

Williams-style residual thresholds will be calibrated from training-set
out-of-fold predictions, not in-sample fitted residuals or the external set.
The CLI will extract `activation_energy` from the supplied training CSV and pass
it to the analysis.

Generate one prediction for every training sample with shuffled 5-fold CV
(`random_state=42`). Every fold clones the complete fitted-estimator
configuration, fits fresh feature and target scalers on that fold's training
portion, predicts its held-out portion, and inverse-transforms predictions to
kcal/mol. Let:

```text
e_i = y_i - y_i,OOF
m = median(e_1, ..., e_n)
training_oof_residual_scale = 1.4826 * median(|e_i - m|)
```

If the median absolute deviation is zero, use the sample standard deviation
`std(e, ddof=1)`. If that is also zero or non-finite, use
`numpy.finfo(float).eps` to keep standardized residuals finite without changing
their zero-error interpretation.

For external observations, standardized residuals will be:

```text
(observed - predicted) / training_oof_residual_scale
```

No `sqrt(1 - h_external)` factor will be used. Descriptor-space leverage remains
an approximation for nonlinear estimators and will be labelled accordingly.

If no external ground truth exists:

- Residual values are unavailable.
- Williams warnings are based on leverage only.
- The plot renders a leverage-only view instead of applying NaN axis limits.

Training data with fewer than two usable samples, invalid neighbour counts,
empty external data, and non-finite feature values produce explicit errors.

## y-Randomization

- Accept a complete estimator or construct one through the shared model helper.
- Use the same precomputed 5×5 RKF splits for the observed and every permuted
  target, isolating target randomization from split variation.
- Fit scalers independently inside every fold.
- Compute the finite-permutation p-value as `(b + 1) / (m + 1)`.
- Validate that `n_permutations` is positive.

The automatic full-pipeline y-randomization block remains disabled because of
its large runtime; documentation will describe the standalone command as the
supported validation path.

## Dependencies and Model Registry

The supported Python version will be Python 3.12 or newer, matching
`pyproject.toml` and `uv.lock`.

- Synchronize required model dependencies across `pyproject.toml` and
  `requirements.txt`.
- Add `gplearn` because GPlearn is currently in the default model registry.
- Keep CatBoost, LightGBM, and XGBoost declared because they are also enabled by
  default.
- Avoid optional-import redesign unless installation compatibility requires it;
  the default registry is treated as the supported full installation.

## CLI Semantics

- `--min_features` default is 5 everywhere.
- Remove `--mae_threshold` from the CLI and all internal function signatures,
  result dictionaries, logs, documentation, and examples.
- `--force_n_features` still evaluates the development feature path and then
  selects the requested exact count; its help text will describe this behavior
  accurately.
- Other argument names and invocation style remain unchanged.

## Documentation

The implementation becomes authoritative. Update:

- `README.md`
- `README_CN.md`
- `user_manual.md`
- the canonical `pipeline.md` workflow document
- Relevant examples and `AGENTS.md` only where their operational descriptions
  contradict the repaired workflow

Documentation must distinguish final test metrics from development metrics,
describe the fixed split, list the actual default models and dependencies, and
state that Optuna uses internal 5-fold CV with in-memory studies.

## Verification Strategy

Focused regression tests will cover authorized bugs without creating broad CI
infrastructure:

- Fixed test indexes never reach tuning or feature selection.
- Complete fixed parameters survive final reconstruction and persistence.
- Ensemble predictions align on the common original index.
- Exact feature-count loading searches final and iteration files and does not
  silently fall back.
- Ridge/Lasso scaling is fit within inner folds.
- y-randomization uses complete parameters, common splits, and corrected p
  values.
- Prediction-only AD completes and produces a leverage-only plot.
- AD residual scale comes from training OOF residuals.
- Checkpoint examples format current and legacy metric dictionaries safely.
- CLI defaults and removal of `mae_threshold` match the approved semantics.

Final verification includes:

- Focused tests
- Full available test suite
- `compileall`
- Core imports in the project environment
- Static searches for obsolete metric labels and unsafe reconstruction patterns
- Small synthetic end-to-end experiments
- Second-round code review of every modified file

## Compatibility and Expected Behavior Changes

- Historical RKF values remain readable but are relabelled as internal
  development metrics.
- Final model metrics will change because the test set is no longer involved in
  model selection.
- Final saved estimators may change because fixed parameters are now preserved.
- Ensemble outputs may contain fewer rows because only common complete samples
  are retained.
- AD warnings will change because residual calibration is corrected.
- Requests for unavailable feature counts now fail unless closest-match behavior
  is explicitly enabled.
- Commands that still pass `--mae_threshold` will fail argument parsing and must
  remove that obsolete option.

## Deliverables

- Corrected source and example files
- Synchronized dependency metadata
- Updated user and pipeline documentation
- Focused regression tests for the authorized fixes
- Updated `code-review-report-new.md` with actual changes, verification evidence,
  remaining risks, and the complete modified-file list
