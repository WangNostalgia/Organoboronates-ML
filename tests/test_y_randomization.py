import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from example import standalone_y_randomization
from src.y_randomization import y_randomization_test


class YRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            }
        )
        self.y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="activation_energy")

    def test_y_randomization_with_model_clones_estimator_and_reuses_fixed_splits(self):
        stored_model = Ridge(alpha=2.5, fit_intercept=False, solver="lsqr", tol=0.123)
        fixed_splits = [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([2, 3, 4, 5]), np.array([0, 1])),
        ]
        captured_models = []
        captured_splits = []

        def fake_evaluate(model, X, y, splits=None, **kwargs):
            captured_models.append(model)
            captured_splits.append(splits)
            mae = 0.4 if len(captured_models) == 1 else 1.4
            return {
                "mae_mean": mae,
                "mae_std": 0.0,
                "r2_mean": 0.0,
                "r2_std": 0.0,
            }

        with (
            patch("src.y_randomization.make_repeated_kfold_splits", return_value=fixed_splits),
            patch("src.y_randomization.repeated_kfold_evaluate", side_effect=fake_evaluate),
            patch("src.y_randomization._plot_y_randomization", return_value=None),
        ):
            results = y_randomization_test(
                X=self.X,
                y=self.y,
                selected_features=["f1", "f2"],
                model=stored_model,
                n_permutations=2,
                random_state=99,
            )

        self.assertEqual(len(captured_models), 3)
        for cloned_model in captured_models:
            self.assertIsNot(cloned_model, stored_model)
            self.assertEqual(cloned_model.get_params(deep=False), stored_model.get_params(deep=False))

        self.assertEqual(len(captured_splits), 3)
        for splits in captured_splits:
            self.assertIs(splits, fixed_splits)
            for (train_idx, test_idx), (expected_train, expected_test) in zip(splits, fixed_splits):
                np.testing.assert_array_equal(train_idx, expected_train)
                np.testing.assert_array_equal(test_idx, expected_test)

        self.assertAlmostEqual(results["p_value_mae"], 1.0 / 3.0)

    def test_y_randomization_corrected_p_value_never_zero(self):
        mae_sequence = iter([0.25, 0.8, 0.9, 1.0])

        def fake_evaluate(model, X, y, splits=None, **kwargs):
            return {
                "mae_mean": next(mae_sequence),
                "mae_std": 0.0,
                "r2_mean": 0.0,
                "r2_std": 0.0,
            }

        with (
            patch("src.y_randomization.make_repeated_kfold_splits", return_value=[]),
            patch("src.y_randomization.repeated_kfold_evaluate", side_effect=fake_evaluate),
            patch("src.y_randomization._plot_y_randomization", return_value=None),
        ):
            results = y_randomization_test(
                model_class=Ridge,
                best_params={"alpha": 1.5, "solver": "svd"},
                X=self.X,
                y=self.y,
                selected_features=["f1", "f2"],
                n_permutations=3,
            )

        self.assertEqual(results["n_permutations"], 3)
        self.assertAlmostEqual(results["p_value_mae"], 1.0 / 4.0)
        self.assertAlmostEqual(results["p_value"], 1.0 / 4.0)
        self.assertFalse(results["passed"])

    def test_y_randomization_rejects_nonpositive_permutations(self):
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    y_randomization_test(
                        model_class=Ridge,
                        best_params={"alpha": 1.0},
                        X=self.X,
                        y=self.y,
                        selected_features=["f1", "f2"],
                        n_permutations=invalid,
                    )

    def test_y_randomization_requires_exactly_one_model_source(self):
        with self.assertRaisesRegex(ValueError, "Provide exactly one of model or model_class"):
            y_randomization_test(
                X=self.X,
                y=self.y,
                selected_features=["f1", "f2"],
                n_permutations=1,
            )

        with self.assertRaisesRegex(ValueError, "Provide exactly one of model or model_class"):
            y_randomization_test(
                model=Ridge(alpha=1.0),
                model_class=Ridge,
                best_params={"alpha": 5.0},
                X=self.X,
                y=self.y,
                selected_features=["f1", "f2"],
                n_permutations=1,
            )


class StandaloneYRandomizationTests(unittest.TestCase):
    def test_standalone_documents_and_logs_root_level_plot_path(self):
        source = Path("example/standalone_y_randomization.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("models/y_randomization_<ModelName>.png", source)
        self.assertNotIn("models/<ModelName>/y_randomization_<ModelName>.png", source)
        self.assertIn("Check models/ for y_randomization_*.png", source)

    def test_main_uses_exact_checkpoint_loading_and_stored_estimator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "manual_feature_selection.csv"
            data_path = root / "dataset.csv"
            models_dir = root / "models"
            (models_dir / "SVR").mkdir(parents=True)

            pd.DataFrame([{"model_name": "SVR", "n_features": 2}]).to_csv(csv_path, index=False)
            pd.DataFrame(
                {
                    "f1": [0.0, 1.0, 2.0, 3.0, 4.0],
                    "f2": [5.0, 6.0, 7.0, 8.0, 9.0],
                    "activation_energy": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            ).to_csv(data_path, index=False)

            stored_model = Ridge(alpha=9.0, solver="lsqr")
            load_calls = []
            yr_calls = []

            def fake_load_model(model_name, n_features=None, models_dir=None, allow_closest=False):
                load_calls.append(
                    {
                        "model_name": model_name,
                        "n_features": n_features,
                        "models_dir": models_dir,
                        "allow_closest": allow_closest,
                    }
                )
                return {
                    "model": stored_model,
                    "features": ["f1", "f2"],
                    "hyperparameters": {"alpha": 999.0},
                }

            def fake_y_randomization_test(**kwargs):
                yr_calls.append(kwargs)
                return {
                    "original_mae_mean": 0.1,
                    "original_mae_std": 0.0,
                    "original_r2_mean": 0.9,
                    "original_r2_std": 0.0,
                    "random_mae_mean": 1.0,
                    "random_mae_std": 0.0,
                    "random_r2_mean": 0.0,
                    "random_r2_std": 0.0,
                    "p_value_mae": 0.1,
                    "passed": False,
                }

            with (
                patch.object(standalone_y_randomization, "SELECTION_CSV", str(csv_path)),
                patch.object(standalone_y_randomization, "DATA_PATH", str(data_path)),
                patch.object(standalone_y_randomization, "MODELS_DIR", str(models_dir)),
                patch.object(standalone_y_randomization, "N_PERMS", 2),
                patch.object(standalone_y_randomization, "load_model", side_effect=fake_load_model),
                patch.object(standalone_y_randomization, "y_randomization_test", side_effect=fake_y_randomization_test),
            ):
                standalone_y_randomization.main()

        self.assertEqual(
            load_calls,
            [
                {
                    "model_name": "SVR",
                    "n_features": 2,
                    "models_dir": str(models_dir),
                    "allow_closest": False,
                }
            ],
        )
        self.assertEqual(len(yr_calls), 1)
        self.assertIs(yr_calls[0]["model"], stored_model)
        self.assertEqual(yr_calls[0]["selected_features"], ["f1", "f2"])
        self.assertEqual(yr_calls[0]["n_permutations"], 2)
        self.assertNotIn("model_class", yr_calls[0])
        self.assertNotIn("best_params", yr_calls[0])


if __name__ == "__main__":
    unittest.main()
