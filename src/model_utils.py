"""Shared estimator construction utilities."""

from src.fixed_params import get_fixed_params


def build_model(model_class, best_params=None, n_jobs=-1):
    """
    Build a model from fixed defaults plus tuned parameters.

    Parameters
    ----------
    model_class : sklearn estimator class
    best_params : dict or None
    n_jobs : int

    Returns
    -------
    tuple
        (model instance, complete_params dict)
    """
    complete_params = {
        **get_fixed_params(model_class, n_jobs),
        **(best_params or {}),
    }
    return model_class(**complete_params), complete_params
