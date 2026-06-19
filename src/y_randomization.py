"""
y-Randomization (target shuffling) test for validating ML model predictions.

y-Randomization destroys the relationship between features X and target y by randomly
shuffling y, then re-trains the model with identical hyperparameters and features.
If the original model significantly outperforms the randomized models (p < 0.05),
the model has learned genuine structure-activity relationships rather than chance patterns.

Reference: Rücker, Rücker, & Meringer (2007), J. Chem. Inf. Model. 47(6), 2345-2357.
"""

import logging
from numbers import Integral

import matplotlib
import numpy as np
from sklearn.base import clone

from src.evaluation import make_repeated_kfold_splits, repeated_kfold_evaluate
from src.model_utils import build_model

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def y_randomization_test(model_class=None, best_params=None, X=None, y=None,
                         selected_features=None, n_permutations=100,
                         random_state=42, model=None, n_jobs=-1):
    """
    Perform y-randomization test to verify that the model has learned real
    structure-activity relationships rather than chance correlations.

    Procedure:
      1. Compute original model performance via fixed 5×5 RepeatedKFold splits.
      2. For n_permutations iterations: randomly shuffle y (keeping X fixed),
         re-train the model with the same estimator configuration,
         and evaluate on the exact same 5×5 RepeatedKFold splits.
      3. Compare original performance against the distribution of randomized
         model performances. Compute the corrected MAE p-value.

    Parameters
    ----------
    model_class : sklearn model class (e.g., SVR), optional
    best_params : dict or None
        Tuned hyperparameters used when rebuilding an estimator through
        src.model_utils.build_model().
    X : DataFrame, full feature matrix (will be subset to selected_features)
    y : Series or array, target vector
    selected_features : list, feature names to use
    n_permutations : int, number of y-randomization iterations (default 100)
    random_state : int, base random seed for target shuffling reproducibility
    model : sklearn estimator instance, optional
        Fully constructed estimator to clone for each evaluation. Mutually
        exclusive with model_class / best_params.
    n_jobs : int
        Passed to src.model_utils.build_model() when rebuilding an estimator.

    Returns
    -------
    results : dict with keys:
        'original_mae_mean', 'original_mae_std', 'original_r2_mean', 'original_r2_std',
        'random_mae_mean', 'random_mae_std', 'random_r2_mean', 'random_r2_std',
        'all_random_mae', 'all_random_r2',
        'p_value_mae' (corrected MAE p-value),
        'p_value' (alias of corrected MAE p-value for backward compatibility),
        'passed' (bool, True if p_value_mae < 0.05)
    """
    if model is not None and (model_class is not None or best_params is not None):
        raise ValueError(
            "Provide exactly one of model or model_class/best_params; got both."
        )
    if model is None and model_class is None:
        raise ValueError(
            "Provide exactly one of model or model_class/best_params; got neither."
        )
    if X is None or y is None:
        raise ValueError("X and y must both be provided.")
    if selected_features is None:
        raise ValueError("selected_features must be provided.")
    if not isinstance(n_permutations, Integral) or isinstance(n_permutations, bool) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer.")

    X_subset = X[selected_features]
    y_values = y.values.copy() if hasattr(y, 'values') else np.asarray(y).copy()
    rng = np.random.RandomState(random_state)

    if model is not None:
        base_model = clone(model)
        model_name = type(model).__name__
    else:
        base_model, _ = build_model(model_class, best_params=best_params, n_jobs=n_jobs)
        model_name = model_class.__name__

    shared_splits = make_repeated_kfold_splits(
        X_subset,
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    logger.info("=" * 60)
    logger.info("  y-Randomization Test")
    logger.info("  Model: %s", model_name)
    logger.info("  Features: %s", selected_features)
    logger.info("  Permutations: %d", n_permutations)
    logger.info("=" * 60)

    logger.info("Step 1: Computing original model performance (5×5 RepeatedKFold)...")
    original_results = repeated_kfold_evaluate(
        clone(base_model),
        X_subset,
        y,
        splits=shared_splits,
    )
    logger.info(
        "  Original: MAE = %.4f ± %.4f | R² = %.4f ± %.4f",
        original_results['mae_mean'], original_results['mae_std'],
        original_results['r2_mean'], original_results['r2_std'],
    )

    logger.info("Step 2: Running %d y-randomization iterations...", n_permutations)
    all_random_mae = []
    all_random_r2 = []

    for i in range(n_permutations):
        y_shuffled = rng.permutation(y_values)
        rand_results = repeated_kfold_evaluate(
            clone(base_model),
            X_subset,
            y_shuffled,
            splits=shared_splits,
        )
        all_random_mae.append(rand_results['mae_mean'])
        all_random_r2.append(rand_results['r2_mean'])

        if (i + 1) % 20 == 0 or (i + 1) == n_permutations:
            logger.info("  Completed %d/%d...", i + 1, n_permutations)

    all_random_mae = np.array(all_random_mae, dtype=float)
    all_random_r2 = np.array(all_random_r2, dtype=float)

    better_or_equal = int(np.sum(all_random_mae <= original_results['mae_mean']))
    p_value_mae = (better_or_equal + 1) / (n_permutations + 1)

    logger.info("Step 3: Statistical comparison")
    logger.info(
        "  Random MAE: %.4f ± %.4f (range: [%.4f, %.4f])",
        all_random_mae.mean(), all_random_mae.std(),
        all_random_mae.min(), all_random_mae.max(),
    )
    logger.info(
        "  Random R²:  %.4f ± %.4f (range: [%.4f, %.4f])",
        all_random_r2.mean(), all_random_r2.std(),
        all_random_r2.min(), all_random_r2.max(),
    )
    logger.info(
        "  Corrected p-value (MAE): %.4f [%d random models <= original]",
        p_value_mae,
        better_or_equal,
    )

    if p_value_mae < 0.05:
        logger.info("  PASSED — model learned genuine SAR (p=%.4f < 0.05)", p_value_mae)
    elif p_value_mae < 0.10:
        logger.info("  MARGINAL (p=%.4f) — consider more data or stricter validation", p_value_mae)
    else:
        logger.info("  FAILED (p=%.4f ≥ 0.05) — model may rely on chance correlations", p_value_mae)

    _plot_y_randomization(
        original_results['mae_mean'],
        all_random_mae,
        p_value_mae,
        model_name,
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
        'n_permutations': int(n_permutations),
        'better_or_equal': better_or_equal,
        'p_value_mae': float(p_value_mae),
        'p_value': float(p_value_mae),  # alias: corrected MAE p-value
        'passed': p_value_mae < 0.05,
    }


def _plot_y_randomization(original_mae, random_mae_array, p_value, model_name):
    """Generate and save y-randomization histogram."""
    import os

    os.makedirs('models', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        random_mae_array,
        bins=max(1, min(30, len(random_mae_array) // 3)),
        edgecolor='k',
        alpha=0.7,
        color='steelblue',
        label=f'y-randomized (n={len(random_mae_array)})',
    )
    ax.axvline(
        x=original_mae,
        color='red',
        linewidth=2.5,
        linestyle='--',
        label=f'Original Model MAE = {original_mae:.3f}',
    )
    ax.axvline(
        x=np.mean(random_mae_array),
        color='blue',
        linewidth=1.5,
        linestyle='-',
        label=f'Random Mean MAE = {np.mean(random_mae_array):.3f}',
    )
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
