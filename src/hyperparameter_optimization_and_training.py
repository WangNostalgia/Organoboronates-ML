import logging

from src.train_and_evaluate import train_and_evaluate

logger = logging.getLogger(__name__)


def hyperparameter_optimization_and_training(
    model_class, X, y, n_trials=100, random_state=42, n_jobs=-1
):
    """
    Run development-data-only model selection and return unfitted artifacts.

    Returns
    -------
    dict
        {
            "estimator": unfitted sklearn-compatible estimator,
            "complete_params": merged fixed+tuned params,
            "selection_mae": development-data CV MAE,
            "development_cv_mae": alias of selection_mae,
            "stability": {"mae_mean", "mae_std", "n_splits"},
            "stability_mae_mean": float alias,
            "stability_mae_std": float alias,
            "internal_cv": repeated_kfold_evaluate(...) payload,
        }
    """
    artifacts = train_and_evaluate(
        model_class,
        X,
        y,
        random_state=random_state,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    logger.info(
        "Development-data selection complete for %s with CV MAE %.4f",
        model_class.__name__,
        artifacts["selection_mae"],
    )
    return artifacts
