import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.external_validation import (
    _discover_model_checkpoints,
    list_available_models,
    load_model,
)


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float)


class ConstantPredictionModel:
    def __init__(self, prediction):
        self.prediction = float(prediction)

    def predict(self, X):
        return np.full(len(X), self.prediction, dtype=float)


def write_checkpoint(root, model_name, filename, features, prediction):
    model_dir = Path(root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    joblib.dump(
        {
            "model": ConstantPredictionModel(prediction),
            "scaler_X": IdentityScaler(),
            "scaler_y": IdentityScaler(),
            "features": list(features),
            "optimal_n_features": len(features),
            "metrics": {"secondary": {"internal_cv": {"rkf_mae_mean": 1.0}}},
        },
        path,
    )
    return path


class ManualFinalCheckpointLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmpdir.name) / "models"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_manual_final_checkpoint_is_discovered_listed_and_loadable(self):
        manual_path = write_checkpoint(
            self.models_dir,
            "SVR",
            "SVR_manual_final_2feat_20260620_120000.joblib",
            ["f1", "f2"],
            prediction=7.0,
        )

        checkpoints = _discover_model_checkpoints(str(self.models_dir / "SVR"))
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["checkpoint_type"], "manual_final")
        self.assertEqual(checkpoints[0]["actual_n_features"], 2)

        available = list_available_models(str(self.models_dir))
        self.assertEqual(available["model_name"].tolist(), ["SVR"])
        self.assertEqual(available["checkpoint_type"].tolist(), ["manual_final"])
        self.assertEqual(available["n_features"].tolist(), [2])

        model_info = load_model("SVR", n_features=2, models_dir=str(self.models_dir))
        self.assertEqual(Path(model_info["_loaded_from"]), manual_path)
        self.assertEqual(model_info["_checkpoint_type"], "manual_final")
        self.assertEqual(model_info["_actual_n_features"], 2)
        self.assertEqual(model_info["model"].prediction, 7.0)

    def test_automatic_final_is_preferred_over_manual_final_for_same_feature_count(self):
        final_path = write_checkpoint(
            self.models_dir,
            "SVR",
            "SVR_final_20260620_110000.joblib",
            ["f1", "f2"],
            prediction=5.0,
        )
        write_checkpoint(
            self.models_dir,
            "SVR",
            "SVR_manual_final_2feat_20260620_120000.joblib",
            ["f1", "f2"],
            prediction=7.0,
        )

        model_info = load_model("SVR", n_features=2, models_dir=str(self.models_dir))
        self.assertEqual(Path(model_info["_loaded_from"]), final_path)
        self.assertEqual(model_info["_checkpoint_type"], "final")
        self.assertEqual(model_info["model"].prediction, 5.0)

    def test_manual_final_filename_feature_count_must_match_feature_schema(self):
        write_checkpoint(
            self.models_dir,
            "SVR",
            "SVR_manual_final_3feat_20260620_120000.joblib",
            ["f1", "f2"],
            prediction=7.0,
        )

        with self.assertRaisesRegex(ValueError, "filename feature count"):
            load_model("SVR", n_features=2, models_dir=str(self.models_dir))


if __name__ == "__main__":
    unittest.main()
