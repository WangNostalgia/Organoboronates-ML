import unittest

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.model_utils import build_model


class BuildModelTests(unittest.TestCase):
    def test_build_model_merges_fixed_and_tuned_params_for_svr(self):
        model, complete_params = build_model(
            SVR,
            best_params={"C": 3.5, "epsilon": 0.2, "gamma": 0.5},
            n_jobs=7,
        )

        self.assertIsInstance(model, SVR)
        self.assertEqual(complete_params["kernel"], "rbf")
        self.assertEqual(complete_params["max_iter"], 10000)
        self.assertEqual(complete_params["cache_size"], 1000)
        self.assertEqual(complete_params["C"], 3.5)
        self.assertEqual(complete_params["epsilon"], 0.2)
        self.assertEqual(complete_params["gamma"], 0.5)

        model_params = model.get_params(deep=False)
        self.assertEqual(model_params["kernel"], "rbf")
        self.assertEqual(model_params["C"], 3.5)
        self.assertEqual(model_params["epsilon"], 0.2)
        self.assertEqual(model_params["gamma"], 0.5)

    def test_build_model_applies_n_jobs_to_parallel_estimators(self):
        model, complete_params = build_model(
            RandomForestRegressor,
            best_params={"n_estimators": 25},
            n_jobs=3,
        )

        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(complete_params["n_jobs"], 3)
        self.assertEqual(complete_params["random_state"], 42)
        self.assertEqual(complete_params["n_estimators"], 25)
        self.assertEqual(model.get_params(deep=False)["n_jobs"], 3)
        self.assertEqual(model.get_params(deep=False)["random_state"], 42)


if __name__ == "__main__":
    unittest.main()
