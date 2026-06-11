import logging
import os
import joblib
from datetime import datetime
import glob
import pandas as pd
import numpy as np

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import r2_score, mean_absolute_error
from src.hyperparameter_optimization_and_training import hyperparameter_optimization_and_training

from src.leave_one_out_validation import leave_one_out_validation
from src.logger_config import setup_logger
from sklearn.preprocessing import MinMaxScaler
from src.visualization import plot_scatter
from src.y_randomization import y_randomization_test
from src.gplearn_wrapper import GPLearnRegressor

# Set default font
plt.rcParams['font.family'] = 'DejaVu Sans'


def clean_old_versions(model_dir, keep_versions=2):
    """
    Clean old versions, keep only the latest N versions
    
    Parameters:
    model_dir (str): Model directory path
    keep_versions (int): Number of versions to keep
    """
    logger = logging.getLogger(__name__)
    # Get all version timestamps using regex for robustness
    import re
    version_timestamps = {}
    ts_pattern = re.compile(r'_final_(\d{8}_\d{6})\.joblib$')
    for filepath in glob.glob(os.path.join(model_dir, "*_final_*.joblib")):
        m = ts_pattern.search(filepath)
        if not m:
            continue
        timestamp = m.group(1)
        version_timestamps[filepath] = timestamp
        metrics_file = filepath.replace(".joblib", "_metrics.txt")
        if os.path.exists(metrics_file):
            version_timestamps[metrics_file] = timestamp
    
    # If version count exceeds limit, delete oldest versions
    if len(set(version_timestamps.values())) > keep_versions:
        # Sort by timestamp
        sorted_files = sorted(version_timestamps.items(), key=lambda x: x[1])
        # Calculate versions to delete
        delete_count = len(set(version_timestamps.values())) - keep_versions
        # Delete oldest files
        for filepath, _ in sorted_files[:delete_count*2]:  # *2 because each version has .joblib and _metrics.txt
            try:
                os.remove(filepath)
                logger.info(f"Deleted old version file: {os.path.basename(filepath)}")
            except Exception as e:
                logger.info(f"Failed to delete file {filepath}: {str(e)}")

    # Clean iteration files — preserve checkpoints for ensemble prediction
    # and per-feature-count external validation.
    # Strategy: group iteration files by run timestamp, keep only the latest
    # `keep_versions` runs (same policy as final models). This ensures iteration
    # snapshots remain available for manual_selection_and_plot.py and the
    # ensemble external validation module.
    iter_ts_pattern = re.compile(r'_iteration_\d+_(\d{8}_\d{6})\.joblib$')
    iter_by_run = {}  # timestamp → list of (filepath, metrics_path)
    for filepath in glob.glob(os.path.join(model_dir, "*_iteration_*.joblib")):
        m = iter_ts_pattern.search(filepath)
        if not m:
            continue
        ts = m.group(1)
        iter_by_run.setdefault(ts, []).append(filepath)

    if len(iter_by_run) > keep_versions:
        # Keep only the most recent runs; delete the rest
        sorted_runs = sorted(iter_by_run.keys(), reverse=True)
        runs_to_keep = set(sorted_runs[:keep_versions])
        for ts, files in iter_by_run.items():
            if ts in runs_to_keep:
                continue
            for filepath in files:
                try:
                    os.remove(filepath)
                    metrics_file = filepath.replace(".joblib", "_metrics.txt")
                    if os.path.exists(metrics_file):
                        os.remove(metrics_file)
                    logger.info(f"Deleted old iteration file: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.info(f"Failed to delete iteration file {filepath}: {str(e)}")


def iterative_optimization(models, X, y, n_trials=100, mae_threshold=20, n_jobs=-1, keep_versions=2, min_features=5, custom_min_features=None, force_n_features=None):
    """
    Perform iterative optimization by removing features based on importance and correlation.

    Parameters:
    models (dict): A dictionary of model classes.
    X (DataFrame): The feature data.
    y (Series): The target data.
    n_trials (int): Number of trials for hyperparameter optimization.
    mae_threshold (float): Threshold for MAE to determine good models.
    n_jobs (int): Number of CPU cores to use (-1 for all cores).
    keep_versions (int): Number of versions to keep for each model.
    min_features (int): Minimum number of features to keep (floor for SHAP-RFECV auto-selection).
    force_n_features (int or None): If set, skip auto-selection and force this exact feature count from the RFECV path.

    Returns:
    results (dict): The results of the optimization.
    best_models (dict): Dictionary containing the best models for each algorithm.
    """
    results = {}
    best_models = {}
    
    # Create root directory for model saving
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(models_dir)

    for model_name, model_class in models.items():
        logger.info(f"Starting to train model: {model_name}")
        
        # Determine the effective minimum features for this specific model
        eff_min_features = min_features
        if custom_min_features is not None and model_name in custom_min_features:
            eff_min_features = custom_min_features[model_name]
            logger.info(f"Using custom minimum feature limit ({eff_min_features}) for {model_name}")

        
        # Create separate directory for each model
        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Clean old versions
        clean_old_versions(model_dir, keep_versions)
        
        X_model = X.copy()
        removed_features = []
        final_features = []
        best_model = None
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler(feature_range=(0, 100))
        
        # Record performance metrics for each iteration (including dual CV)
        performance_history = {
            'iteration': [],
            'mae': [],            # 100-split average MAE (legacy)
            'r2': [],             # Test set R² (single split)
            'rkf_mae_mean': [],   # 5×5 RepeatedKFold MAE (PRIMARY metric)
            'rkf_mae_std': [],
            'rkf_r2_mean': [],    # 5×5 RepeatedKFold R² (PRIMARY metric)
            'rkf_r2_std': [],
            'loo_r2': [],         # LOOCV R² (auxiliary reference)
            'loo_mae': [],        # LOOCV MAE (auxiliary reference)
            'removed_feature': [],
            'remaining_features': [],
        }
        
        # SHAP-RFECV path: records model snapshots when features < 10.
        # After the loop, the best feature count is chosen automatically
        # based on minimum RKfold MAE along this path.
        shap_rfecv_path = []

        # ── GPlearn shortcut: train once with all features ──────────────
        # Genetic programming inherently performs feature selection through
        # tournament selection and parsimony pressure.  The discovered formula
        # uses only the features that improve fitness — irrelevant features are
        # automatically ignored.  Running GPlearn through the iterative
        # SHAP-RFECV loop is therefore redundant and wastes 30+ hours of
        # compute (population=1000-5000 × generations=8-25 × 100 Optuna trials
        # × 12 iterations × 5-fold SHAP ≈ millions of GP fits).
        #
        # Strategy: run ONE Optuna hyperparameter optimisation on all features,
        # fit the final model, evaluate, save, and move on.
        if model_class == GPLearnRegressor:
            logger.info("GPlearn detected — using single-pass training "
                        "(genetic programming performs inherent feature selection).")
            logger.info("Starting hyperparameter optimisation with all %d features.",
                        len(X_model.columns))

            # ── Phase 1: hyperparameter optimisation ─────────────────
            (
                best_model,
                _,
                X_train,
                X_test,
                y_train,
                y_test,
                current_mae,
                best_params,
                rkf_results,
            ) = hyperparameter_optimization_and_training(
                model_class, X_model, y, n_trials=n_trials, n_jobs=n_jobs,
                random_state=40,
            )

            # ── Phase 2: fit final model on all features ──────────────
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(
                y_train.values.reshape(-1, 1)
            ).ravel()
            best_model.fit(X_train_scaled, y_train_scaled)

            y_pred_test_scaled = best_model.predict(X_test_scaled)
            y_pred_test = scaler_y.inverse_transform(
                y_pred_test_scaled.reshape(-1, 1)
            ).ravel()
            current_r2 = r2_score(y_test, y_pred_test)
            final_mae_test = mean_absolute_error(y_test, y_pred_test)

            # ── Phase 3: LOO validation ──────────────────────────────
            try:
                loo_r2_val, loo_mae_val = leave_one_out_validation(
                    best_model, X_model, y
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning("LOOCV failed for GPlearn: %s", e, exc_info=True)
                loo_r2_val = float('nan')
                loo_mae_val = float('nan')

            # ── Phase 4: assemble results ─────────────────────────────
            optimal_features = list(X_model.columns)
            optimal_n_features = len(optimal_features)
            final_features = optimal_features
            removed_features = []

            result = {
                "mae_mean": current_mae,
                "r2_test_avg": current_r2,
                "mae_test_avg": final_mae_test,
                "best_params_avg": best_params,
                "r2_loo_avg": loo_r2_val,
                "mae_loo_avg": loo_mae_val,
                "rkf_mae_opt_mean": rkf_results['rkf_mae_mean'],
                "rkf_r2_opt_mean": rkf_results['rkf_r2_mean'],
                "final_features": final_features,
                "optimal_n_features": optimal_n_features,
                "optimal_features": optimal_features,
                "shap_rfecv_path": [],
            }
            results[model_name] = result
            best_models[model_name] = best_model

            # Log GPlearn-discovered formula prominently
            if hasattr(best_model, 'formula_') and best_model.formula_:
                logger.info("GPlearn final formula (all %d features):\n  %s",
                            optimal_n_features, best_model.formula_)

            logger.info(
                f"\n{'='*80}\n"
                f"  GPlearn Final (all {optimal_n_features} features)\n"
                f"  {'Metric':<22} {'Value':>18}\n"
                f"  {'-'*40}\n"
                f"  {'5×5 RKfold MAE':<22} {rkf_results['rkf_mae_mean']:>12.4f} ± {rkf_results['rkf_mae_std']:.4f}  ← PRIMARY\n"
                f"  {'5×5 RKfold R²':<22}  {rkf_results['rkf_r2_mean']:>12.4f} ± {rkf_results['rkf_r2_std']:.4f}  ← PRIMARY\n"
                f"  {'LOOCV R²':<22}       {loo_r2_val:>12.4f}        ← auxiliary\n"
                f"  {'LOOCV MAE':<22}      {loo_mae_val:>12.4f}        ← auxiliary\n"
                f"  {'100-split MAE':<22}  {current_mae:>12.4f}        ← legacy\n"
                f"  {'Test R² (single)':<22} {current_r2:>12.4f}        ← legacy\n"
                f"  {'Test MAE (single)':<22} {final_mae_test:>12.4f}        ← legacy\n"
                f"{'='*80}"
            )

            # ── Phase 5: save outputs ─────────────────────────────────
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Single-entry performance history
            perf_hist = {
                'iteration': [1],
                'mae': [current_mae],
                'r2': [current_r2],
                'rkf_mae_mean': [rkf_results['rkf_mae_mean']],
                'rkf_mae_std': [rkf_results['rkf_mae_std']],
                'rkf_r2_mean': [rkf_results['rkf_r2_mean']],
                'rkf_r2_std': [rkf_results['rkf_r2_std']],
                'loo_r2': [loo_r2_val],
                'loo_mae': [loo_mae_val],
                'removed_feature': ['None (single pass)'],
                'remaining_features': [optimal_n_features],
            }
            plot_performance_history(
                perf_hist, model_name,
                os.path.join(model_dir, f"performance_history_{timestamp}.png")
            )
            save_performance_history(
                perf_hist,
                os.path.join(model_dir, f"performance_history_{timestamp}.csv")
            )

            # Final model file
            final_model_info = {
                'model': best_model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'features': optimal_features,
                'hyperparameters': best_params,
                'metrics': result,
                'optimal_n_features': optimal_n_features,
                'force_n_features': None,
                'removed_features': [],
                'shap_rfecv_path_summary': [],
            }
            final_model_path = os.path.join(
                model_dir, f"{model_name}_final_{timestamp}"
            )
            joblib.dump(final_model_info, f"{final_model_path}.joblib")

            with open(f"{final_model_path}_metrics.txt", 'w') as f:
                f.write(f"Model: {model_name} (Final — single-pass, all features)\n")
                f.write(f"Note: GPlearn performs inherent feature selection via "
                        f"genetic programming.\n")
                f.write(f"      Iterative SHAP-RFECV was skipped.\n")
                f.write(f"Feature count: {optimal_n_features} (all features)\n")
                f.write(f"Features: {', '.join(optimal_features)}\n")
                f.write(f"\n--- PRIMARY (5x5 RepeatedKFold) ---\n")
                f.write(f"RKfold MAE: {rkf_results['rkf_mae_mean']:.4f} +/- {rkf_results['rkf_mae_std']:.4f}\n")
                f.write(f"RKfold R²:  {rkf_results['rkf_r2_mean']:.4f} +/- {rkf_results['rkf_r2_std']:.4f}\n")
                f.write(f"\n--- AUXILIARY (LOOCV) ---\n")
                f.write(f"LOO R²:  {loo_r2_val:.4f}\n")
                f.write(f"LOO MAE: {loo_mae_val:.4f}\n")
                f.write(f"\n--- LEGACY ---\n")
                f.write(f"100-split MAE: {current_mae:.4f}\n")
                f.write(f"Test R²:       {current_r2:.4f}\n")
                f.write(f"Test MAE:      {final_mae_test:.4f}\n")
                f.write(f"\nBest Parameters: {best_params}\n")
                if hasattr(best_model, 'formula_') and best_model.formula_:
                    f.write(f"\nGPlearn Formula:\n  {best_model.formula_}\n")

            # Final scatter plot
            y_pred_train_scaled = best_model.predict(X_train_scaled)
            y_pred_train = scaler_y.inverse_transform(
                y_pred_train_scaled.reshape(-1, 1)
            ).ravel()
            plot_scatter(
                y_train=y_train, y_pred_train=y_pred_train,
                y_test=y_test, y_pred_test=y_pred_test,
                model_name=f"{model_name} (all {optimal_n_features} features, single-pass)",
                mae_mean=current_mae,
                output_dir=model_dir + '/',
                output_name=f'final_scatter_{timestamp}.png',
                X_train=X_train, X_test=X_test,
                rkf_mae=rkf_results['rkf_mae_mean'],
                rkf_r2=rkf_results['rkf_r2_mean'],
            )

            logger.info("GPlearn single-pass training complete.")
            continue  # skip the iterative loop, proceed to next model

        # ── Normal iterative SHAP-RFECV path (all other models) ────────
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")
            
            # Train model with current feature set
            (
                best_model,
                _,
                X_train,
                X_test,
                y_train,
                y_test,
                current_mae,
                best_params,
                rkf_results,
            ) = hyperparameter_optimization_and_training(
                model_class, X_model, y, n_trials=n_trials, n_jobs=n_jobs,random_state=40
            )
            
            # Scale features and target
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            # Train model with scaled data
            best_model.fit(X_train_scaled, y_train_scaled)
            
            # Calculate current performance metrics
            y_pred_test_scaled = best_model.predict(X_test_scaled)
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            # No clipping — activation_energy has no physical upper bound at 100
            current_r2 = r2_score(y_test, y_pred_test)
            
            # Record current iteration performance (dual CV + legacy metrics)
            performance_history['iteration'].append(iteration)
            performance_history['mae'].append(current_mae)
            performance_history['r2'].append(current_r2)
            performance_history['rkf_mae_mean'].append(rkf_results['rkf_mae_mean'])
            performance_history['rkf_mae_std'].append(rkf_results['rkf_mae_std'])
            performance_history['rkf_r2_mean'].append(rkf_results['rkf_r2_mean'])
            performance_history['rkf_r2_std'].append(rkf_results['rkf_r2_std'])
            # Compute LOOCV for auxiliary reference (returns both R² and MAE)
            try:
                loo_r2_val, loo_mae_val = leave_one_out_validation(best_model, X_model, y)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"LOOCV failed at iteration {iteration}: {e}", exc_info=True)
                loo_r2_val = float('nan')
                loo_mae_val = float('nan')
            performance_history['loo_r2'].append(loo_r2_val)
            performance_history['loo_mae'].append(loo_mae_val)
            performance_history['remaining_features'].append(len(X_model.columns))
            performance_history['removed_feature'].append(
                removed_features[-1] if removed_features else "Initial"
            )

            # Unified Dual CV Comparison table (all metrics, single output per iteration)
            logger.info(
                f"\n{'='*80}\n"
                f"  Iteration {iteration} — Dual CV Comparison\n"
                f"  {'Metric':<22} {'Value':>18}\n"
                f"  {'-'*40}\n"
                f"  {'5×5 RKfold MAE':<22} {rkf_results['rkf_mae_mean']:>12.4f} ± {rkf_results['rkf_mae_std']:.4f}  ← PRIMARY\n"
                f"  {'5×5 RKfold R²':<22}  {rkf_results['rkf_r2_mean']:>12.4f} ± {rkf_results['rkf_r2_std']:.4f}  ← PRIMARY\n"
                f"  {'LOOCV R²':<22}       {loo_r2_val:>12.4f}        ← auxiliary\n"
                f"  {'LOOCV MAE':<22}      {loo_mae_val:>12.4f}        ← auxiliary\n"
                f"  {'100-split MAE':<22}  {current_mae:>12.4f}        ← legacy\n"
                f"  {'Test R² (single)':<22} {current_r2:>12.4f}        ← legacy\n"
                f"  {'Test MAE (single)':<22} {mean_absolute_error(y_test, y_pred_test):>12.4f}        ← legacy\n"
                f"{'='*80}"
            )
            
            # --- Feature selection: SHAP-RFECV (always used) ---
            # All iterations use SHAP importance to select the least important feature.
            # Two-tier strategy based on feature count:
            #   Above threshold → simple single-fit SHAP (fast, sufficient for coarse filtering)
            #   At/below threshold → multi-fold CV consensus SHAP (robust, for fine-grained decisions)
            # CV metrics are recorded at/below threshold for auto-selection.
            current_n_features = len(X_model.columns)
            RECORD_RFECV_PATH = (current_n_features <= max(10, eff_min_features + 3))
            USE_CONSENSUS_SHAP = RECORD_RFECV_PATH  # use multi-fold only when recording

            shim_mode = "Multi-fold consensus" if USE_CONSENSUS_SHAP else "Single-fit"
            logger.info(f"=== SHAP-RFECV (Iteration {iteration}, {current_n_features} features, {shim_mode})"
                        f"{' — recording CV path' if RECORD_RFECV_PATH else ''} ===")
            from src.feature_selection import shap_rfecv_select_worst_feature

            # cv_folds=0 triggers the simple single-fit path (no CV looping)
            shim_cv_folds = 5 if USE_CONSENSUS_SHAP else 0
            worst_feat, shap_ranking, removal_reason = shap_rfecv_select_worst_feature(
                best_model, X_train, y_train, model_name, cv_folds=shim_cv_folds
            )

            logger.info(f"  SHAP Importance Ranking:")
            for rank, (feat_name, feat_imp) in enumerate(shap_ranking, 1):
                logger.info(f"    [{rank}] {feat_name}: {feat_imp:.2f}%")
            logger.info(f"  → Selected for removal: '{worst_feat}' "
                        f"(reason: {removal_reason})")
            logger.info("==========================================================")

            # Record RFECV path data when features < 10 (for later auto-selection).
            # ONLY lightweight primitives — NO model/scaler objects to avoid memory bloat
            # and object-reference pollution across iterations.
            if RECORD_RFECV_PATH:
                rfecv_model_info = {
                    'features': X_model.columns.tolist(),
                    'hyperparameters': best_params.copy(),
                    'metrics': {
                        'mae_mean': float(current_mae),
                        'r2_test': float(current_r2),
                        'mae_test': float(mean_absolute_error(y_test, y_pred_test)),
                        'rkf_mae_mean': float(rkf_results['rkf_mae_mean']),
                        'rkf_mae_std': float(rkf_results['rkf_mae_std']),
                        'rkf_r2_mean': float(rkf_results['rkf_r2_mean']),
                        'rkf_r2_std': float(rkf_results['rkf_r2_std']),
                        'loo_r2': float(loo_r2_val) if not np.isnan(loo_r2_val) else None,
                        'loo_mae': float(loo_mae_val) if not np.isnan(loo_mae_val) else None,
                    },
                    'removed_features': removed_features.copy(),
                    'iteration': iteration,
                    'n_features': current_n_features,
                }
                shap_rfecv_path.append(rfecv_model_info)
                logger.info(
                    f"  ✓ RFECV path recorded [{current_n_features} features] — "
                    f"RKfold MAE={rkf_results['rkf_mae_mean']:.4f} (see table above for full metrics)"
                )

            remove_feature = [worst_feat]

            # Check if minimum feature count limit is reached
            if len(X_model.columns) <= eff_min_features:
                logger.info(f"Minimum feature count limit ({eff_min_features}) reached")
                final_features = X_model.columns.tolist()
                break

            if not remove_feature:
                logger.info("No more features to remove")
                final_features = X_model.columns.tolist()
                break

            # ── Save iteration checkpoint BEFORE feature removal ──
            # The checkpoint must capture the model + features at the moment of
            # training, NOT after the feature is dropped.  Otherwise downstream
            # consumers (manual_selection_and_plot.py) will train on a different
            # feature set than the one the stored metrics describe.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_iteration_{iteration}_{timestamp}"
            model_path = os.path.join(model_dir, model_filename)

            model_info = {
                'model': best_model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'features': X_model.columns.tolist(),  # PRE-removal → matches training data
                'hyperparameters': best_params,
                'metrics': {
                    'mae_mean': current_mae,            # 100-split average MAE
                    'r2_test': current_r2,
                    'mae_test': mean_absolute_error(y_test, y_pred_test),
                    'rkf_mae_mean': rkf_results['rkf_mae_mean'],  # 5×5 RepeatedKFold MAE
                    'rkf_r2_mean': rkf_results['rkf_r2_mean'],    # 5×5 RepeatedKFold R²
                },
                'removed_features': removed_features.copy()
            }
            joblib.dump(model_info, f"{model_path}.joblib")

            # Save per-iteration metrics text file
            with open(f"{model_path}_metrics.txt", 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Features ({len(X_model.columns)}): {', '.join(X_model.columns)}\n")
                f.write(f"Removed Features: {', '.join(removed_features)}\n")
                f.write(f"\n--- PRIMARY (5x5 RepeatedKFold) ---\n")
                f.write(f"RKfold MAE: {rkf_results['rkf_mae_mean']:.4f} +/- {rkf_results['rkf_mae_std']:.4f}\n")
                f.write(f"RKfold R²:  {rkf_results['rkf_r2_mean']:.4f} +/- {rkf_results['rkf_r2_std']:.4f}\n")
                f.write(f"\n--- AUXILIARY (LOOCV) ---\n")
                f.write(f"LOO R²:  {loo_r2_val:.4f}\n")
                f.write(f"LOO MAE: {loo_mae_val:.4f}\n")
                f.write(f"\n--- LEGACY ---\n")
                f.write(f"100-split MAE: {current_mae:.4f}\n")
                f.write(f"Test R²:       {current_r2:.4f}\n")
                f.write(f"Test MAE:      {mean_absolute_error(y_test, y_pred_test):.4f}\n")
                f.write(f"\nBest Parameters: {best_params}\n")
                if hasattr(best_model, 'formula_') and best_model.formula_:
                    f.write(f"\nGPlearn Formula:\n  {best_model.formula_}\n")

            # Remove selected features (AFTER saving checkpoint)
            for feature in remove_feature:
                logger.info(f"\nRemoving feature: {feature}")
                removed_features.append(feature)
                X_model = X_model.drop(columns=[feature])
                X_train = X_train.drop(columns=[feature])
                X_test = X_test.drop(columns=[feature])

        # ── SHAP-RFECV auto-selection: find optimal feature count ──
        X_train_opt = X_test_opt = y_train_opt = y_test_opt = None
        y_pred_train_opt = y_pred_test_opt = None

        if len(shap_rfecv_path) > 0:
            logger.info(f"\n{'='*110}")
            logger.info(f"  SHAP-RFECV Path Summary for {model_name}")
            logger.info(f"  {'Feat':<5} {'RKfold MAE ± std':<22} {'RKfold R² ± std':<21} {'LOOCV R²':<10} {'LOOCV MAE':<10} {'100-spl MAE':<12} {'Test R²':<9} {'Test MAE':<10}")
            logger.info(f"  {'-'*106}")
            for entry in shap_rfecv_path:
                m = entry['metrics']
                loo_r2_str = f"{m['loo_r2']:.4f}" if m['loo_r2'] is not None else "N/A"
                loo_mae_str = f"{m['loo_mae']:.4f}" if m['loo_mae'] is not None else "N/A"
                logger.info(
                    f"  {entry['n_features']:<5} "
                    f"{m['rkf_mae_mean']:.4f} ± {m.get('rkf_mae_std',0):.4f}     "
                    f"{m['rkf_r2_mean']:.4f} ± {m.get('rkf_r2_std',0):.4f}    "
                    f"{loo_r2_str:<10} "
                    f"{loo_mae_str:<10} "
                    f"{m['mae_mean']:<12.4f} "
                    f"{m['r2_test']:<9.4f} "
                    f"{m['mae_test']:<10.4f}"
                )

            # Determine optimal feature count
            if force_n_features is not None:
                # ── Manual override ──
                matching = [e for e in shap_rfecv_path if e['n_features'] == force_n_features]
                if matching:
                    best_entry = matching[0]
                    logger.info(f"  ★ MANUAL OVERRIDE (--force_n_features={force_n_features})")
                else:
                    best_entry = min(shap_rfecv_path,
                                     key=lambda e: abs(e['n_features'] - force_n_features))
                    logger.info(f"  ⚠ --force_n_features={force_n_features} not in path, using closest: {best_entry['n_features']}")
            else:
                # ── Fractional 1-SE Rule ──
                # Standard 1-SE (threshold = min_mae + 1.0 * min_std) is too aggressive
                # on small-N datasets: RepeatedKFold fold-level std is inflated by the
                # high overlap between training splits, pushing the threshold far beyond
                # what the data can support — the model collapses to min_features
                # (underfitting).  PENALTY_ALPHA = 0.25 limits the tolerance to a
                # fraction of one standard error, which empirically preserves the
                # "sweet spot" where MAE is near-minimal AND the feature count is
                # parsimonious without discarding structurally critical SAR features.
                PENALTY_ALPHA = 0.25  # fraction of 1 std to tolerate (0.2–0.5 typical)

                min_entry = min(shap_rfecv_path, key=lambda e: e['metrics']['rkf_mae_mean'])
                min_mae = min_entry['metrics']['rkf_mae_mean']
                min_std = min_entry['metrics'].get('rkf_mae_std', 0.0)
                threshold_mae = min_mae + (PENALTY_ALPHA * max(min_std, 1e-8))

                candidates = [e for e in shap_rfecv_path
                              if e['metrics']['rkf_mae_mean'] <= threshold_mae]
                best_entry = min(candidates, key=lambda e: e['n_features'])

                if best_entry['n_features'] < min_entry['n_features']:
                    logger.info(f"  ★ Fractional 1-SE (α={PENALTY_ALPHA}): "
                                f"absolute min at {min_entry['n_features']} feat "
                                f"(MAE={min_mae:.4f} ± {min_std:.4f}), "
                                f"threshold={threshold_mae:.4f}, "
                                f"selecting {best_entry['n_features']} feat "
                                f"(MAE={best_entry['metrics']['rkf_mae_mean']:.4f})")
                else:
                    logger.info(f"  ★ Fractional 1-SE: absolute minimum is already "
                                f"the most parsimonious ({best_entry['n_features']} feat, "
                                f"MAE={min_mae:.4f})")

            optimal_n_features = best_entry['n_features']
            optimal_features = best_entry['features']
            best_params = best_entry['hyperparameters']

            logger.info(f"  ★ RKfold MAE={best_entry['metrics']['rkf_mae_mean']:.4f}, "
                        f"RKfold R²={best_entry['metrics']['rkf_r2_mean']:.4f}, "
                        f"LOOCV R²={best_entry['metrics']['loo_r2']}, "
                        f"LOOCV MAE={best_entry['metrics']['loo_mae']}")
            logger.info(f"  ★ Optimal features: {optimal_features}")
            logger.info(f"{'='*110}")

            # ── Reconstruct model cleanly on optimal features (no data leakage) ──
            from sklearn.model_selection import train_test_split as tts
            X_opt = X[optimal_features]
            X_train_opt, X_test_opt, y_train_opt, y_test_opt = tts(
                X_opt, y, test_size=0.2, random_state=40
            )

            # Scalers fitted ONLY on X_train_opt — test set never leaks into preprocessing
            scaler_X_opt = MinMaxScaler()
            scaler_y_opt = MinMaxScaler(feature_range=(0, 100))
            X_train_scaled = scaler_X_opt.fit_transform(X_train_opt)
            X_test_scaled = scaler_X_opt.transform(X_test_opt)
            y_train_scaled = scaler_y_opt.fit_transform(y_train_opt.values.reshape(-1, 1)).ravel()

            best_model = model_class(**best_params)
            best_model.fit(X_train_scaled, y_train_scaled)

            y_pred_train_scaled = best_model.predict(X_train_scaled)
            y_pred_test_scaled = best_model.predict(X_test_scaled)
            y_pred_train = scaler_y_opt.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            y_pred_test_opt = scaler_y_opt.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

            # Compute metrics on the PAIRED optimal split (fixes y_test mismatch bug)
            current_mae = best_entry['metrics']['mae_mean']
            current_r2 = r2_score(y_test_opt, y_pred_test_opt)
            r2_loo = best_entry['metrics']['loo_r2'] if best_entry['metrics']['loo_r2'] is not None else float('nan')
            mae_loo = best_entry['metrics']['loo_mae'] if best_entry['metrics']['loo_mae'] is not None else float('nan')
        else:
            optimal_n_features = len(final_features)
            optimal_features = final_features
            r2_loo, mae_loo = leave_one_out_validation(best_model, X_model, y)
            logger.info(f"No RFECV path (<10 features never reached). "
                        f"Using final model with {optimal_n_features} features.")

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            best_model.fit(X_train_scaled, y_train_scaled)
            y_pred_train_scaled = best_model.predict(X_train_scaled)
            y_pred_test_scaled = best_model.predict(X_test_scaled)
            y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            y_pred_test_opt = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            y_train_opt, y_test_opt = y_train, y_test
            X_train_opt, X_test_opt = X_train, X_test
            scaler_X_opt, scaler_y_opt = scaler_X, scaler_y

        # ── Assemble result dict with correct keys ──
        final_mae_test = mean_absolute_error(y_test_opt, y_pred_test_opt)
        final_r2_test = r2_score(y_test_opt, y_pred_test_opt)

        result = {
            "mae_mean": current_mae,
            "r2_test_avg": final_r2_test,
            "mae_test_avg": final_mae_test,
            "best_params_avg": best_params,
            "r2_loo_avg": r2_loo,
            "mae_loo_avg": mae_loo,
            "rkf_mae_opt_mean": best_entry['metrics']['rkf_mae_mean'] if len(shap_rfecv_path) > 0 else float('nan'),
            "rkf_r2_opt_mean": best_entry['metrics']['rkf_r2_mean'] if len(shap_rfecv_path) > 0 else float('nan'),
            "final_features": final_features,
            "optimal_n_features": optimal_n_features,
            "optimal_features": optimal_features,
            "shap_rfecv_path": shap_rfecv_path,
        }
        results[model_name] = result
        best_models[model_name] = best_model

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plot_performance_history(
            performance_history, model_name,
            os.path.join(model_dir, f"performance_history_{timestamp}.png")
        )
        save_performance_history(
            performance_history,
            os.path.join(model_dir, f"performance_history_{timestamp}.csv")
        )

        # Save final model — scalers fitted ONLY on training subset (no leakage)
        final_model_info = {
            'model': best_model,
            'scaler_X': scaler_X_opt,
            'scaler_y': scaler_y_opt,
            'features': optimal_features,
            'hyperparameters': best_params,
            'metrics': result,
            'optimal_n_features': optimal_n_features,
            'force_n_features': force_n_features,
            'removed_features': removed_features,
            'shap_rfecv_path_summary': [
                {'n_features': e['n_features'],
                 'rkf_mae_mean': e['metrics']['rkf_mae_mean'],
                 'rkf_mae_std': e['metrics'].get('rkf_mae_std'),
                 'rkf_r2_mean': e['metrics']['rkf_r2_mean'],
                 'rkf_r2_std': e['metrics'].get('rkf_r2_std'),
                 'loo_r2': e['metrics']['loo_r2'],
                 'loo_mae': e['metrics']['loo_mae'],
                 'mae_mean': e['metrics']['mae_mean'],
                 'r2_test': e['metrics']['r2_test'],
                 'mae_test': e['metrics']['mae_test']}
                for e in shap_rfecv_path
            ],
        }
        final_model_path = os.path.join(model_dir, f"{model_name}_final_{timestamp}")
        joblib.dump(final_model_info, f"{final_model_path}.joblib")

        with open(f"{final_model_path}_metrics.txt", 'w') as f:
            selection_mode = "manually forced" if force_n_features is not None else "auto-selected"
            f.write(f"Model: {model_name} (Final — SHAP-RFECV {selection_mode})\n")
            f.write(f"Optimal feature count: {optimal_n_features}")
            if force_n_features is not None:
                f.write(f"  (requested: {force_n_features})")
            f.write("\n")
            f.write(f"Features: {', '.join(optimal_features)}\n")
            f.write(f"Removed Features: {', '.join(removed_features)}\n")
            f.write(f"\n--- PRIMARY (5x5 RepeatedKFold) ---\n")
            f.write(f"RKfold MAE: {result.get('rkf_mae_opt_mean', 'N/A')}\n")
            f.write(f"RKfold R²:  {result.get('rkf_r2_opt_mean', 'N/A')}\n")
            f.write(f"\n--- AUXILIARY (LOOCV) ---\n")
            f.write(f"LOO R²:  {r2_loo:.4f}\n")
            f.write(f"LOO MAE: {mae_loo:.4f}\n")
            f.write(f"\n--- LEGACY ---\n")
            f.write(f"100-split MAE: {current_mae:.4f}\n")
            f.write(f"Test R²:       {final_r2_test:.4f}\n")
            f.write(f"Test MAE:      {final_mae_test:.4f}\n")
            f.write(f"\nBest Parameters: {best_params}\n")
            if hasattr(best_model, 'formula_') and best_model.formula_:
                f.write(f"\nGPlearn Formula:\n  {best_model.formula_}\n")
            if len(shap_rfecv_path) > 0:
                f.write(f"\nSHAP-RFECV Path (all metrics with ± std):\n")
                f.write(f"{'Feat':<5} {'RKfold MAE ± std':<23} {'RKfold R² ± std':<22} "
                        f"{'LOOCV R²':<10} {'LOOCV MAE':<10} "
                        f"{'100-spl MAE':<12} {'Test R²':<9} {'Test MAE':<10}\n")
                for entry in shap_rfecv_path:
                    m = entry['metrics']
                    loo_r2_s = f"{m['loo_r2']:.4f}" if m['loo_r2'] is not None else "N/A"
                    loo_mae_s = f"{m['loo_mae']:.4f}" if m['loo_mae'] is not None else "N/A"
                    f.write(f"{entry['n_features']:<5} "
                            f"{m['rkf_mae_mean']:.4f} ± {m.get('rkf_mae_std',0):.4f}     "
                            f"{m['rkf_r2_mean']:.4f} ± {m.get('rkf_r2_std',0):.4f}    "
                            f"{loo_r2_s:<10} "
                            f"{loo_mae_s:<10} "
                            f"{m['mae_mean']:<12.4f} "
                            f"{m['r2_test']:<9.4f} "
                            f"{m['mae_test']:<10.4f}\n")

        # ── y-Randomization (commented out — uncomment to re-enable) ──
        # y-Randomization runs 100×5×5=2500 model fits per model and can take
        # hours for tree ensembles (RandomForest ~17min, XGBoost ~8h).
        # Uncomment the block below when statistical validation is needed.
        #
        # logger.info(f"\n{'='*60}\n  y-Randomization Test for {model_name}"
        #             f" (optimal: {optimal_n_features} features)\n{'='*60}")
        # try:
        #     yr_results = y_randomization_test(
        #         model_class=model_class,
        #         best_params=best_params,
        #         X=X, y=y,
        #         selected_features=optimal_features,
        #         n_permutations=100, random_state=42
        #     )
        #     result['y_randomization'] = {
        #         'p_value': yr_results['p_value_mae'],
        #         'passed': yr_results['passed'],
        #         'original_mae_mean': yr_results['original_mae_mean'],
        #         'random_mae_mean': yr_results['random_mae_mean'],
        #     }
        #     logger.info("y-Randomization: %s (p=%.4f)",
        #                 "PASSED" if yr_results['passed'] else "FAILED",
        #                 yr_results['p_value_mae'])
        # except Exception as e:
        #     logger.warning(f"y-Randomization failed: {e}. Skipping.", exc_info=True)
        #     result['y_randomization'] = {'error': str(e)}

        # ── Final scatter plot (optimal model, paired data) ──
        suffix = "[forced]" if force_n_features is not None else "[auto]"
        scatter_title = f"{model_name} ({optimal_n_features} features, {suffix})"
        rkf_mae_opt = result.get('rkf_mae_opt_mean')
        rkf_r2_opt = result.get('rkf_r2_opt_mean')
        plot_scatter(
            y_train=y_train_opt, y_pred_train=y_pred_train,
            y_test=y_test_opt, y_pred_test=y_pred_test_opt,
            model_name=scatter_title,
            mae_mean=current_mae,
            output_dir=model_dir + '/',
            output_name=f'final_scatter_{timestamp}.png',
            X_train=X_train_opt, X_test=X_test_opt,
            rkf_mae=rkf_mae_opt,
            rkf_r2=rkf_r2_opt,
        )

    for model_name, result in results.items():
        logger.info(f"Model: {model_name}")
        logger.info(f'Average MAE: {result["mae_mean"]}')
        logger.info(f'R^2 (Test): {result["r2_test_avg"]}')
        logger.info(f'MAE (Test): {result["mae_test_avg"]}')
        logger.info(f'Best Params: {result["best_params_avg"]}')
        logger.info(f'R^2 (LOO): {result["r2_loo_avg"]}')
        logger.info(f'Final Features: {result["final_features"]}')
        logger.info("-" * 40)

    return results, best_models


def plot_performance_history(history, model_name, output_path):
    """
    Plot performance history with dual CV metrics, grouped by metric TYPE.

    Top panel (MAE):  5×5 RKfold MAE ± std (PRIMARY, bold) + LOOCV MAE (auxiliary)
                       + 100-split avg MAE (legacy, dashed)
    Bottom panel (R²): 5×5 RKfold R² ± std (PRIMARY, bold) + LOOCV R² (auxiliary)
                       + Test R² single split (legacy, dashed)

    Parameters:
    history (dict): Dictionary containing performance metrics history
    model_name (str): Model name
    output_path (str): Output file path
    """
    # Check if new dual-CV columns exist; fall back to legacy if not
    has_rkf = 'rkf_mae_mean' in history and any(
        v is not None and not (isinstance(v, float) and np.isnan(v))
        for v in history.get('rkf_mae_mean', [])
    )

    if has_rkf:
        # --- Dual-panel layout: MAE on top, R² on bottom ---
        # Grouping by metric TYPE keeps comparisons meaningful.
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        iterations = np.array(history['iteration'])
        rkf_mae = np.array(history['rkf_mae_mean'], dtype=float)
        rkf_mae_std = np.array(history['rkf_mae_std'], dtype=float)
        rkf_r2 = np.array(history['rkf_r2_mean'], dtype=float)
        rkf_r2_std = np.array(history['rkf_r2_std'], dtype=float)
        loo_r2_vals = np.array(history.get('loo_r2', [np.nan]*len(iterations)), dtype=float)
        loo_mae_vals = np.array(history.get('loo_mae', [np.nan]*len(iterations)), dtype=float)

        # ═══ Top panel: ALL MAE metrics together ═══
        line_rkf_mae = ax_top.plot(iterations, rkf_mae, 'b-o', linewidth=2.5, markersize=8,
                                    label='5×5 RKfold MAE (PRIMARY)')
        ax_top.fill_between(iterations, rkf_mae - rkf_mae_std, rkf_mae + rkf_mae_std,
                             alpha=0.15, color='blue')
        line_loo_mae = ax_top.plot(iterations, loo_mae_vals, 'd-', color='green',
                                    linewidth=1.5, markersize=6, label='LOOCV MAE (auxiliary)')
        line_100mae = ax_top.plot(iterations, history['mae'], 's--', color='gray',
                                   alpha=0.5, markersize=5, label='100-split avg MAE (legacy)')

        ax_top.set_ylabel('MAE (kcal/mol)\nlower is better', color='b', fontsize=12)
        ax_top.set_ylim(ax_top.get_ylim()[::-1])  # Reverse: lower MAE = better
        ax_top.set_title(f'{model_name} — Dual-CV Performance History', fontsize=14, fontweight='bold')

        top_lines = line_rkf_mae + line_loo_mae + line_100mae
        top_labels = [l.get_label() for l in top_lines]
        ax_top.legend(top_lines, top_labels, loc='upper left', fontsize=9)

        # ═══ Bottom panel: ALL R² metrics together ═══
        line_rkf_r2 = ax_bottom.plot(iterations, rkf_r2, 'r-o', linewidth=2.5, markersize=8,
                                      label='5×5 RKfold R² (PRIMARY)')
        ax_bottom.fill_between(iterations, rkf_r2 - rkf_r2_std, rkf_r2 + rkf_r2_std,
                                alpha=0.15, color='red')
        line_loo_r2 = ax_bottom.plot(iterations, loo_r2_vals, 'd-', color='orange',
                                      linewidth=1.5, markersize=6, label='LOOCV R² (auxiliary)')
        line_testr2 = ax_bottom.plot(iterations, history['r2'], 's--', color='gray',
                                      alpha=0.5, markersize=5, label='Test R² single split (legacy)')

        ax_bottom.set_xlabel('Iteration', fontsize=12)
        ax_bottom.set_ylabel('R²\nhigher is better', color='r', fontsize=12)

        bottom_lines = line_rkf_r2 + line_loo_r2 + line_testr2
        bottom_labels = [l.get_label() for l in bottom_lines]
        ax_bottom.legend(bottom_lines, bottom_labels, loc='upper left', fontsize=9)

        # Integer ticks + grid
        for ax in [ax_top, ax_bottom]:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

        # Annotate removed features on top (MAE) panel
        for i, (feature, remaining) in enumerate(zip(history['removed_feature'],
                                                      history['remaining_features'])):
            if i > 0 and not np.isnan(rkf_mae[i]):
                ax_top.annotate(
                    f"{feature}\n({remaining} left)",
                    (iterations[i], rkf_mae[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, rotation=45, ha='left', alpha=0.8
                )

    else:
        # --- Legacy single-panel layout (backward compatible) ---
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        line1 = ax1.plot(history['iteration'], history['mae'], 'b-o', label='MAE')
        line2 = ax2.plot(history['iteration'], history['r2'], 'r-o', label='R²')

        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MAE (lower is better)', color='b')
        ax2.set_ylabel('R² (higher is better)', color='r')
        plt.title(f'{model_name} Performance History')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        ax1.grid(True, alpha=0.3)

        for i, (feature, remaining) in enumerate(zip(history['removed_feature'],
                                                      history['remaining_features'])):
            if i > 0:
                plt.annotate(
                    f"{feature}\n({remaining} features left)",
                    (history['iteration'][i], history['mae'][i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, rotation=45, ha='left'
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicit figure reference prevents memory leak


def save_performance_history(history, output_path):
    """
    Save performance history to CSV file.
    
    Parameters:
    history (dict): Dictionary containing performance metrics history
    output_path (str): Output file path
    """
    logger = logging.getLogger(__name__)
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    logger.info(f"Performance history saved to: {output_path}")
