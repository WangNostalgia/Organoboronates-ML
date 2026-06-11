from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.train_and_evaluate import train_and_evaluate
from src.fixed_params import get_fixed_params
import logging

logger = logging.getLogger(__name__)


def hyperparameter_optimization_and_training(
    model_class, X, y, n_trials=100, random_state=42, n_jobs=-1
):
    """
    Perform hyperparameter optimization and train the model.

    Pulls fixed params from the single source of truth (src/fixed_params),
    passes through the caller's random_state, and returns a MinMaxScaler
    aligned with the global scaling convention.

    Parameters
    ----------
    model_class : sklearn estimator class
    X : DataFrame, feature matrix
    y : Series, target vector
    n_trials : int, Optuna trials
    random_state : int, passed through to train_and_evaluate
    n_jobs : int, CPU cores

    Returns
    -------
    best_model, scaler, X_train, X_test, y_train, y_test, mae_mean, best_params, rkf_results
    """
    cv_mae_best, mae_mean, best_params, rkf_results = train_and_evaluate(
        model_class, X, y, random_state=random_state, n_trials=n_trials, n_jobs=n_jobs
    )

    # Merge fixed constants from the shared configuration center
    final_params = {**get_fixed_params(model_class, n_jobs), **best_params}
    best_model = model_class(**final_params)

    # Global convention: MinMaxScaler (not StandardScaler)
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return best_model, scaler, X_train, X_test, y_train, y_test, mae_mean, best_params, rkf_results
