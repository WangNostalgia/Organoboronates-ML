import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.external_validation import ensemble_validation, load_model


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


def write_checkpoint(
    root,
    model_name,
    checkpoint_type,
    timestamp,
    features,
    prediction=1.0,
    filename_prefix=None,
    metrics_mae=1.0,
):
    model_dir = Path(root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = filename_prefix or model_name
    if checkpoint_type == "final":
        filename = f"{prefix}_final_{timestamp}.joblib"
    elif checkpoint_type == "iteration":
        filename = f"{prefix}_iteration_1_{timestamp}.joblib"
    else:
        raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}")

    path = model_dir / filename
    joblib.dump(
        {
            "model": ConstantPredictionModel(prediction),
            "scaler_X": IdentityScaler(),
            "scaler_y": IdentityScaler(),
            "features": list(features),
            "optimal_n_features": len(features),
            "metrics": {"rkf_mae_opt_mean": metrics_mae},
        },
        path,
    )
    return path


class ExternalValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmpdir.name) / "models"
        self.output_dir = Path(self.tmpdir.name) / "outputs"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_model_exact_final_preferred_over_iteration(self):
        write_checkpoint(
            self.models_dir,
            "SVR",
            "iteration",
            "20240102_120000",
            ["f1", "f2", "f3"],
            prediction=9.0,
        )
        final_path = write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240101_120000",
            ["f1", "f2", "f3"],
            prediction=5.0,
        )

        model_info = load_model("SVR", n_features=3, models_dir=str(self.models_dir))

        self.assertEqual(Path(model_info["_loaded_from"]), final_path)
        self.assertEqual(model_info["_checkpoint_type"], "final")
        self.assertEqual(model_info["_requested_n_features"], 3)
        self.assertEqual(model_info["_actual_n_features"], 3)

    def test_load_model_uses_latest_filename_timestamp_not_mtime(self):
        older_timestamp = write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240101_120000",
            ["f1", "f2", "f3"],
            prediction=1.0,
            filename_prefix="older",
        )
        newer_timestamp = write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240103_120000",
            ["f1", "f2", "f3"],
            prediction=2.0,
            filename_prefix="newer",
        )

        os.utime(older_timestamp, (2_000_000_000, 2_000_000_000))
        os.utime(newer_timestamp, (1_000_000_000, 1_000_000_000))

        model_info = load_model("SVR", n_features=3, models_dir=str(self.models_dir))

        self.assertEqual(Path(model_info["_loaded_from"]), newer_timestamp)
        self.assertEqual(model_info["model"].prediction, 2.0)

    def test_load_model_raises_for_ambiguous_same_timestamp_candidates(self):
        write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240103_120000",
            ["f1", "f2", "f3"],
            filename_prefix="candidate_a",
        )
        write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240103_120000",
            ["f1", "f2", "f3"],
            filename_prefix="candidate_b",
        )

        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_model("SVR", n_features=3, models_dir=str(self.models_dir))

    def test_load_model_without_exact_match_fails_by_default_and_lists_available_counts(self):
        write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240101_120000",
            ["f1", "f2", "f3", "f4"],
        )
        write_checkpoint(
            self.models_dir,
            "SVR",
            "iteration",
            "20240102_120000",
            ["f1", "f2", "f3", "f4", "f5", "f6"],
        )

        with self.assertRaisesRegex(ValueError, r"Available feature counts: \[4, 6\]"):
            load_model("SVR", n_features=5, models_dir=str(self.models_dir))

    def test_load_model_allow_closest_uses_priority_order_and_reports_actual_metadata(self):
        expected_path = write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240101_120000",
            ["f1", "f2", "f3", "f4"],
            prediction=4.0,
        )
        write_checkpoint(
            self.models_dir,
            "SVR",
            "iteration",
            "20240105_120000",
            ["f1", "f2", "f3", "f4", "f5", "f6"],
            prediction=6.0,
        )

        model_info = load_model(
            "SVR",
            n_features=5,
            models_dir=str(self.models_dir),
            allow_closest=True,
        )

        self.assertEqual(Path(model_info["_loaded_from"]), expected_path)
        self.assertEqual(model_info["_checkpoint_type"], "final")
        self.assertEqual(model_info["_requested_n_features"], 5)
        self.assertEqual(model_info["_actual_n_features"], 4)

    def test_ensemble_uses_common_original_index_for_predictions_ids_and_targets(self):
        write_checkpoint(
            self.models_dir,
            "ModelA",
            "final",
            "20240103_120000",
            ["f1", "f2"],
            prediction=1.0,
            metrics_mae=1.0,
        )
        write_checkpoint(
            self.models_dir,
            "ModelB",
            "final",
            "20240104_120000",
            ["f2", "f3"],
            prediction=3.0,
            metrics_mae=1.0,
        )

        ensemble_spec = Path(self.tmpdir.name) / "ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "ModelA", "n_features": 2},
                {"model_name": "ModelB", "n_features": 2},
            ]
        ).to_csv(ensemble_spec, index=False)

        external_df = pd.DataFrame(
            {
                "sub_H": ["H0", "H1", "H2", "H3"],
                "sub_B": ["B0", "B1", "B2", "B3"],
                "activation_energy": [10.0, 20.0, 30.0, 40.0],
                "f1": [0.1, np.nan, 0.3, 0.4],
                "f2": [1.0, 1.1, 1.2, 1.3],
                "f3": [2.0, 2.1, np.nan, 2.3],
            },
            index=[100, 101, 102, 103],
        )

        with patch("src.external_validation._plot_external_scatter", return_value=None):
            results = ensemble_validation(
                ensemble_csv=str(ensemble_spec),
                external_data=external_df,
                output_dir=str(self.output_dir),
                models_dir=str(self.models_dir),
            )

        predictions = results["predictions"]

        self.assertEqual(predictions["sub_H"].tolist(), ["H0", "H3"])
        self.assertEqual(predictions["sub_B"].tolist(), ["B0", "B3"])
        self.assertEqual(predictions["activation_energy"].tolist(), [10.0, 40.0])
        self.assertEqual(predictions["predicted_mean"].tolist(), [2.0, 2.0])
        self.assertEqual(predictions["predicted_weighted"].tolist(), [2.0, 2.0])
        self.assertEqual(predictions["pred_ModelA_2_feat"].tolist(), [1.0, 1.0])
        self.assertEqual(predictions["pred_ModelB_2_feat"].tolist(), [3.0, 3.0])
        self.assertEqual(results["n_samples"], 2)
        self.assertEqual(results["excluded_rows"], 2)
        self.assertEqual(results["ensemble_members"], [("ModelA", 2), ("ModelB", 2)])
        self.assertEqual(
            [item["label"] for item in results["individual_results"]],
            ["ModelA (2 feat)", "ModelB (2 feat)"],
        )
        self.assertTrue(any(Path(path).name.startswith("ensemble_summary_") for path in results["output_files"]))
        summary_path = next(path for path in results["output_files"] if path.endswith(".txt"))
        summary_text = Path(summary_path).read_text(encoding="utf-8")
        self.assertIn("Excluded rows: 2", summary_text)


if __name__ == "__main__":
    unittest.main()
