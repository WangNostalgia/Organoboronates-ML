"""
Sklearn-compatible wrapper for gplearn (Genetic Programming symbolic regression).

Symbolic regression discovers explicit mathematical formulas relating features
to the target.  Unlike black-box models (SVR, RF, XGBoost), it outputs a
human-readable equation — ideal for interpretable QSAR/QSPR.

Backend: gplearn (pure Python, pip-installable)
Reference: Koza, J. R. (1992). Genetic Programming. MIT Press.
"""

import logging
import re
import numpy as np

logger = logging.getLogger(__name__)


class GPLearnRegressor:
    """
    Sklearn-compatible symbolic regression regressor using genetic programming.

    After fitting, the discovered formula is stored in self.formula_ and
    printed to the log.

    Parameters
    ----------
    population_size : int, GP population size (default 3000). Larger values
        explore more thoroughly but increase runtime linearly.
    generations : int, number of generations to evolve (default 15).
        10-20 generations typically suffice for convergence.
    parsimony_coefficient : float, penalty for formula complexity (default 0.001).
        Larger values produce simpler (but potentially less accurate) formulas.
    function_set : tuple, allowed mathematical operators.  Default includes
        ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv').
    p_crossover : float, crossover probability (default 0.7)
    random_state : int, random seed for reproducibility (default 42)
    n_jobs : int, parallel jobs for fitness evaluation (default 1)

    Attributes
    ----------
    formula_ : str, the human-readable mathematical formula discovered
    """

    def __init__(
        self,
        population_size=3000,
        generations=15,
        parsimony_coefficient=0.001,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        random_state=42,
        n_jobs=1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self.function_set = function_set
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Fit the symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        from gplearn.genetic import SymbolicRegressor

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()

        logger.info(
            "GPlearn: population=%d, generations=%d, features=%d",
            self.population_size, self.generations, X_arr.shape[1],
        )

        self._model = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            stopping_criteria=0.01,
            p_crossover=self.p_crossover,
            p_subtree_mutation=self.p_subtree_mutation,
            p_hoist_mutation=self.p_hoist_mutation,
            p_point_mutation=self.p_point_mutation,
            parsimony_coefficient=self.parsimony_coefficient,
            function_set=self.function_set,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,  # suppress per-generation progress output
        )
        self._model.fit(X_arr, y_arr)

        # Extract the formula and map generic X0..Xn to actual column names.
        # gplearn uses 0-indexed feature placeholders (X0, X1, ...).
        # re.sub with \b word boundaries prevents X0 from matching X10, X11, etc.
        self.formula_ = str(self._model._program)
        if hasattr(X, 'columns'):
            for i, col in enumerate(X.columns):
                pattern = rf'\bX{i}\b'
                self.formula_ = re.sub(pattern, str(col), self.formula_)

        self._fitted_flag = True

        logger.info("=" * 60)
        logger.info("  GPlearn discovered formula:")
        logger.info("  %s", self.formula_)
        logger.info("=" * 60)

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if not hasattr(self, '_fitted_flag') or not self._fitted_flag:
            raise RuntimeError("Model must be fitted before predict().")
        return self._model.predict(np.asarray(X, dtype=float))

    def get_params(self, deep=True):
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'parsimony_coefficient': self.parsimony_coefficient,
            'function_set': self.function_set,
            'p_crossover': self.p_crossover,
            'p_subtree_mutation': self.p_subtree_mutation,
            'p_hoist_mutation': self.p_hoist_mutation,
            'p_point_mutation': self.p_point_mutation,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return (
            f"GPLearnRegressor(pop={self.population_size}, gen={self.generations})"
        )
