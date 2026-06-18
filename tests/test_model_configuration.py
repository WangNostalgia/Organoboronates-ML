import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.model_utils import build_model


class BuildModelTests(unittest.TestCase):
    def test_fixed_params_source_preserves_original_unicode_text(self):
        source = Path("src/fixed_params.py").read_text(encoding="utf-8")

        self.assertIn("that was evaluated during hyperparameter tuning — no missing constants.", source)
        self.assertIn("dict : fixed parameter names → values (empty dict if model not in map)", source)
        self.assertIn('"random_state": 42,  # lock GP random seed — essential for reproducible formula discovery', source)
        self.assertIn('"p_crossover": 0.7,  # explicit crossover probability (must sum ≤1.0 with mutation probs)', source)

    def test_fixed_params_imports_without_optional_gradient_boosting_packages(self):
        with tempfile.TemporaryDirectory() as blocker_dir:
            blocker_path = Path(blocker_dir)
            for module_name in ("xgboost", "lightgbm", "catboost"):
                (blocker_path / f"{module_name}.py").write_text(
                    'raise ImportError("blocked for optional dependency test")\n',
                    encoding="utf-8",
                )

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(blocker_path), os.getcwd(), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json\n"
                        "from sklearn.ensemble import RandomForestRegressor\n"
                        "from sklearn.svm import SVR\n"
                        "from src.fixed_params import get_fixed_params\n"
                        "print(json.dumps({"
                        "'rf': get_fixed_params(RandomForestRegressor, n_jobs=3), "
                        "'svr': get_fixed_params(SVR)"
                        "}, sort_keys=True))\n"
                    ),
                ],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                env=env,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"subprocess failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["rf"]["n_jobs"], 3)
        self.assertEqual(payload["rf"]["random_state"], 42)
        self.assertEqual(payload["svr"]["kernel"], "rbf")
        self.assertEqual(payload["svr"]["max_iter"], 10000)

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
