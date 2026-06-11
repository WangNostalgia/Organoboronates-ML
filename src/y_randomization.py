"""
y-Randomization (target shuffling) test for validating ML model predictions.

y-Randomization destroys the relationship between features X and target y by randomly
shuffling y, then re-trains the model with identical hyperparameters and features.
If the original model significantly outperforms the randomized models (p < 0.05),
the model has learned genuine structure-activity relationships rather than chance patterns.

Reference: Rücker, Rücker, & Meringer (2007), J. Chem. Inf. Model. 47(6), 2345-2357.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler
from src.evaluation import repeated_kfold_evaluate  # Unified evaluation center

logger = logging.getLogger(__name__)


def y_randomization_test(model_class, best_params, X, y, selected_features,
                          n_permutations=100, random_state=42):
    """
    Perform y-randomization test to verify that the model has learned real
    structure-activity relationships rather than chance correlations.

    Procedure:
      1. Compute original model performance via 5×5 RepeatedKFold.
      2. For n_permutations iterations: randomly shuffle y (keeping X fixed),
         re-train the model with the same hyperparameters and features,
         and evaluate with 5×5 RepeatedKFold.
      3. Compare original performance against the distribution of randomized
         model performances. Compute p-value.

    Parameters
    ----------
    model_class : sklearn model class (e.g., SVR)
    best_params : dict, best hyperparameters (same as used for final model)
    X : DataFrame, full feature matrix (will be subset to selected_features)
    y : Series, target vector
    selected_features : list, feature names to use
    n_permutations : int, number of y-randomization iterations (default 100)
    random_state : int, base random seed for reproducibility

    Returns
    -------
    results : dict with keys:
        'original_mae_mean', 'original_mae_std', 'original_r2_mean', 'original_r2_std',
        'random_mae_mean', 'random_mae_std', 'random_r2_mean', 'random_r2_std',
        'all_random_mae', 'all_random_r2',
        'p_value_mae' (proportion of random models with MAE ≤ original MAE),
        'passed' (bool, True if p_value_mae < 0.05)
    """
    X_subset = X[selected_features]
    rng = np.random.RandomState(random_state)
    y_values = y.values.copy()

    logger.info("=" * 60)
    logger.info("  y-Randomization Test")
    logger.info(f"  Model: {model_class.__name__}")
    logger.info(f"  Features: {selected_features}")
    logger.info(f"  Permutations: {n_permutations}")
    logger.info("=" * 60)

    # ── Step 1: Original model performance (5×5 RepeatedKFold) ──
    logger.info("Step 1: Computing original model performance (5×5 RepeatedKFold)...")
    # Build complete model instance (best_params already includes fixed params from pipeline)
    base_model = model_class(**best_params)
    original_results = repeated_kfold_evaluate(
        base_model, X_subset, y, random_state=random_state
    )
    logger.info(
        "  Original: MAE = %.4f ± %.4f | R² = %.4f ± %.4f",
        original_results['mae_mean'], original_results['mae_std'],
        original_results['r2_mean'], original_results['r2_std']
    )

    # ── Step 2: N permutations of y ──
    logger.info(f"Step 2: Running {n_permutations} y-randomization iterations...")
    all_random_mae = []
    all_random_r2 = []

    for i in range(n_permutations):
        # Shuffle y (destroy X→y relationship)
        y_shuffled = rng.permutation(y_values)

        # Evaluate randomized model with 5×5 RepeatedKFold
        rand_results = repeated_kfold_evaluate(
            base_model, X_subset, y_shuffled,  # shuffled target
            random_state=random_state + i  # different seed per iteration
        )

        all_random_mae.append(rand_results['mae_mean'])
        all_random_r2.append(rand_results['r2_mean'])

        if (i + 1) % 20 == 0:
            logger.info(f"  Completed {i+1}/{n_permutations}...")

    all_random_mae = np.array(all_random_mae)
    all_random_r2 = np.array(all_random_r2)

    # ── Step 3: Statistical comparison ──
    # p-value: proportion of random models that perform AS GOOD OR BETTER
    # than the original model (i.e., MAE ≤ original MAE)
    p_value_mae = np.mean(all_random_mae <= original_results['mae_mean'])

    logger.info("Step 3: Statistical comparison")
    logger.info("  Random MAE: %.4f ± %.4f (range: [%.4f, %.4f])",
                all_random_mae.mean(), all_random_mae.std(),
                all_random_mae.min(), all_random_mae.max())
    logger.info("  Random R²:  %.4f ± %.4f (range: [%.4f, %.4f])",
                all_random_r2.mean(), all_random_r2.std(),
                all_random_r2.min(), all_random_r2.max())
    logger.info("  p-value (MAE): %.4f", p_value_mae)

    if p_value_mae < 0.05:
        logger.info("  ✓ PASSED — model learned genuine SAR (p=%.4f < 0.05)", p_value_mae)
    elif p_value_mae < 0.10:
        logger.info("  ⚠ MARGINAL (p=%.4f) — consider more data or stricter validation", p_value_mae)
    else:
        logger.info("  ✗ FAILED (p=%.4f ≥ 0.05) — model may rely on chance correlations", p_value_mae)

    # ── Generate diagnostic plot ──
    _plot_y_randomization(
        original_results['mae_mean'], all_random_mae, p_value_mae,
        model_class.__name__
    )

    return {
        'original_mae_mean': original_results['mae_mean'],
        'original_mae_std': original_results['mae_std'],
        'original_r2_mean': original_results['r2_mean'],
        'original_r2_std': original_results['r2_std'],
        'random_mae_mean': float(all_random_mae.mean()),
        'random_mae_std': float(all_random_mae.std()),
        'random_r2_mean': float(all_random_r2.mean()),
        'random_r2_std': float(all_random_r2.std()),
        'all_random_mae': all_random_mae,
        'all_random_r2': all_random_r2,
        'n_permutations': n_permutations,
        'p_value_mae': float(p_value_mae),
        'passed': p_value_mae < 0.05,
    }


def _plot_y_randomization(original_mae, random_mae_array, p_value, model_name):
    """Generate and save y-randomization histogram."""
    import os
    os.makedirs('models', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_mae_array, bins=min(30, len(random_mae_array) // 3),
            edgecolor='k', alpha=0.7, color='steelblue',
            label=f'y-randomized (n={len(random_mae_array)})')
    ax.axvline(x=original_mae, color='red', linewidth=2.5, linestyle='--',
               label=f'Original Model MAE = {original_mae:.3f}')
    ax.axvline(x=np.mean(random_mae_array), color='blue', linewidth=1.5, linestyle='-',
               label=f'Random Mean MAE = {np.mean(random_mae_array):.3f}')
    ax.set_xlabel('5×5 RepeatedKFold MAE (kcal/mol)')
    ax.set_ylabel('Frequency')
    status = 'PASS' if p_value < 0.05 else 'FAIL'
    ax.set_title(f'y-Randomization Test: {model_name}\np-value = {p_value:.4f} ({status})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'models/y_randomization_{model_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  y-randomization plot saved to %s", save_path)
