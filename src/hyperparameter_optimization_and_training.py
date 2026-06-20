import logging

from src.train_and_evaluate import train_and_evaluate

logger = logging.getLogger(__name__)


def hyperparameter_optimization_and_training(
    model_class,
    X,
    y,
    n_trials=100,
    random_state=42,
    n_jobs=-1,
    optuna_jobs=1,
):
    """
    Run development-data-only model selection and return unfitted artifacts.

    n_jobs controls estimator-internal parallelism. optuna_jobs controls Optuna
    trial parallelism and defaults to 1 for deterministic, resource-stable TPE.
    """
    artifacts = train_and_evaluate(
        model_class,
        X,
        y,
        random_state=random_state,
        n_trials=n_trials,
        n_jobs=n_jobs,
        optuna_jobs=optuna_jobs,
    )
    logger.info(
        "Development-data selection complete for %s with CV MAE %.4f",
        model_class.__name__,
        artifacts["selection_mae"],
    )
    return artifacts
