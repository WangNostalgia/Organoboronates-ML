import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.external_validation import (
    _discover_model_checkpoints,
    _loaded_feature_count,
    ensemble_validation,
    list_available_models,
    load_model,
)


class IdentityScaler:
    def transform(self, X):
        if len(X) == 0:
            raise AssertionError("scaler.transform must not receive zero rows")
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
    iteration_number=1,
    optimal_n_features=None,
    metrics=None,
):
    model_dir = Path(root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    prefix = filename_prefix or model_name
    if checkpoint_type == "final":
        filename = f"{prefix}_final_{timestamp}.joblib"
    elif checkpoint_type == "iteration":
        filename = f"{prefix}_iteration_{iteration_number}_{timestamp}.joblib"
    else:
        raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}")

    path = model_dir / filename
    joblib.dump(
        {
            "model": ConstantPredictionModel(prediction),
            "scaler_X": IdentityScaler(),
            "scaler_y": IdentityScaler(),
            "features": list(features),
            "optimal_n_features": (
                len(features)
                if optimal_n_features is None
                else optimal_n_features
            ),
            "metrics": (
                {"rkf_mae_opt_mean": metrics_mae}
                if metrics is None
                else metrics
            ),
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

    def test_source_has_single_definitions_for_key_entrypoints_and_updated_docstring(self):
        source = Path("src/external_validation.py").read_text(encoding="utf-8")

        self.assertEqual(source.count("def load_model("), 1)
        self.assertEqual(source.count("def ensemble_validation("), 1)
        self.assertEqual(source.count("def _write_ensemble_summary("), 1)
        self.assertEqual(source.count("def main("), 1)
        self.assertNotIn("Load any final model by name", source)
        self.assertIn("Load final or iteration checkpoints", source)

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
        )
        newer_timestamp = write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240103_120000",
            ["f1", "f2", "f3"],
            prediction=2.0,
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
            "iteration",
            "20240103_120000",
            ["f1", "f2", "f3"],
            iteration_number=1,
        )
        write_checkpoint(
            self.models_dir,
            "SVR",
            "iteration",
            "20240103_120000",
            ["f1", "f2", "f3"],
            iteration_number=2,
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

    def test_load_model_rejects_inconsistent_optimal_feature_metadata(self):
        write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240101_120000",
            ["f1", "f2"],
            optimal_n_features=3,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"optimal_n_features.*3.*len\(features\).*2",
        ):
            load_model("SVR", n_features=3, models_dir=str(self.models_dir))

    def test_loaded_feature_count_rejects_non_integral_metadata_with_context(self):
        filepath = str(self.models_dir / "SVR" / "SVR_final_20240101_120000.joblib")
        invalid_cases = (
            (2.9, ["f1", "f2"]),
            ("2", ["f1", "f2"]),
            (True, ["f1"]),
        )

        for metadata, features in invalid_cases:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError) as context:
                    _loaded_feature_count(
                        {
                            "features": features,
                            "optimal_n_features": metadata,
                        },
                        filepath=filepath,
                    )

                message = str(context.exception)
                self.assertIn(filepath, message)
                self.assertIn(repr(metadata), message)
                self.assertIn(f"len(features)={len(features)}", message)

    def test_loaded_feature_count_accepts_python_and_numpy_integers(self):
        for metadata in (2, np.int64(2)):
            with self.subTest(metadata=metadata):
                self.assertEqual(
                    _loaded_feature_count(
                        {
                            "features": ["f1", "f2"],
                            "optimal_n_features": metadata,
                        },
                        filepath="checkpoint.joblib",
                    ),
                    2,
                )

    def test_checkpoint_discovery_parses_final_token_in_model_name_once(self):
        model_name = "Catalyst_final_variant"
        model_dir = self.models_dir / model_name
        write_checkpoint(
            self.models_dir,
            model_name,
            "final",
            "20240101_120000",
            ["f1", "f2"],
        )
        write_checkpoint(
            self.models_dir,
            model_name,
            "iteration",
            "20240101_120000",
            ["f1", "f2", "f3"],
            iteration_number=7,
        )

        checkpoints = _discover_model_checkpoints(str(model_dir))

        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(
            sorted(checkpoint["checkpoint_type"] for checkpoint in checkpoints),
            ["final", "iteration"],
        )
        self.assertEqual(
            {Path(checkpoint["path"]).name for checkpoint in checkpoints},
            {
                f"{model_name}_final_20240101_120000.joblib",
                f"{model_name}_iteration_7_20240101_120000.joblib",
            },
        )

    def test_list_available_models_prefers_current_nested_internal_cv_metrics(self):
        write_checkpoint(
            self.models_dir,
            "SVR",
            "final",
            "20240101_120000",
            ["f1", "f2"],
            metrics={
                "secondary": {
                    "internal_cv": {
                        "rkf_mae_mean": 1.1,
                        "rkf_r2_mean": 0.81,
                    },
                },
                "internal_cv": {
                    "rkf_mae_mean": 2.2,
                    "rkf_r2_mean": 0.72,
                },
                "rkf_mae_mean": 3.3,
                "rkf_r2_mean": 0.63,
                "rkf_mae_opt_mean": 4.4,
                "rkf_r2_opt_mean": 0.54,
            },
        )

        row = list_available_models(str(self.models_dir)).iloc[0]

        self.assertEqual(row["rkf_mae"], 1.1)
        self.assertEqual(row["rkf_r2"], 0.81)

    def test_list_available_models_falls_back_through_supported_metric_schemas(self):
        schemas = {
            "DirectModel": (
                {
                    "internal_cv": {
                        "rkf_mae_mean": 2.1,
                        "rkf_r2_mean": 0.71,
                    },
                },
                2.1,
                0.71,
            ),
            "FlatCurrentModel": (
                {
                    "rkf_mae_mean": 3.1,
                    "rkf_r2_mean": 0.61,
                },
                3.1,
                0.61,
            ),
            "LegacyModel": (
                {
                    "rkf_mae_opt_mean": 4.1,
                    "rkf_r2_opt_mean": 0.51,
                },
                4.1,
                0.51,
            ),
        }
        for model_name, (metrics, _, _) in schemas.items():
            write_checkpoint(
                self.models_dir,
                model_name,
                "final",
                "20240101_120000",
                ["f1"],
                metrics=metrics,
            )

        records = list_available_models(str(self.models_dir)).set_index(
            "model_name"
        )

        for model_name, (_, expected_mae, expected_r2) in schemas.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(records.loc[model_name, "rkf_mae"], expected_mae)
                self.assertEqual(records.loc[model_name, "rkf_r2"], expected_r2)

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

    def test_ensemble_skips_duplicate_specs_resolving_to_same_checkpoint(self):
        checkpoint = write_checkpoint(
            self.models_dir,
            "ModelA",
            "final",
            "20240103_120000",
            ["f1", "f2"],
            prediction=2.0,
        )
        ensemble_spec = Path(self.tmpdir.name) / "duplicate_ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "ModelA", "n_features": 2},
                {"model_name": "ModelA", "n_features": 2},
            ]
        ).to_csv(ensemble_spec, index=False)

        results = ensemble_validation(
            ensemble_csv=str(ensemble_spec),
            external_data=pd.DataFrame({"f1": [1.0], "f2": [2.0]}),
            target_col=None,
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir),
        )

        self.assertEqual(len(results["individual_results"]), 1)
        self.assertEqual(len(results["ensemble_errors"]), 1)
        self.assertIn("same checkpoint", results["ensemble_errors"][0][2])
        self.assertIn(checkpoint.name, results["ensemble_errors"][0][2])

    def test_ensemble_allow_closest_convergence_is_reported_as_duplicate_checkpoint(self):
        write_checkpoint(
            self.models_dir,
            "ModelA",
            "final",
            "20240103_120000",
            ["f1", "f2", "f3", "f4"],
            prediction=2.0,
        )
        ensemble_spec = Path(self.tmpdir.name) / "closest_duplicate_ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "ModelA", "n_features": 3},
                {"model_name": "ModelA", "n_features": 5},
            ]
        ).to_csv(ensemble_spec, index=False)

        results = ensemble_validation(
            ensemble_csv=str(ensemble_spec),
            external_data=pd.DataFrame(
                {"f1": [1.0], "f2": [2.0], "f3": [3.0], "f4": [4.0]}
            ),
            target_col=None,
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir),
            allow_closest=True,
        )

        self.assertEqual(len(results["individual_results"]), 1)
        self.assertEqual(len(results["ensemble_errors"]), 1)
        self.assertIn("same checkpoint", results["ensemble_errors"][0][2])

    def test_ensemble_weight_mae_prefers_current_internal_cv_and_ignores_stability(self):
        write_checkpoint(
            self.models_dir,
            "FinalModel",
            "final",
            "20240103_120000",
            ["f1"],
            prediction=1.0,
            metrics={
                "secondary": {
                    "internal_cv": {"rkf_mae_mean": 1.0},
                    "stability": {"mae_mean": 0.01},
                },
                "rkf_mae_mean": 8.0,
                "rkf_mae_opt_mean": 9.0,
            },
        )
        write_checkpoint(
            self.models_dir,
            "IterationModel",
            "iteration",
            "20240104_120000",
            ["f2"],
            prediction=3.0,
            metrics={
                "internal_cv": {"rkf_mae_mean": 2.0},
                "mae_mean": 0.001,
                "rkf_mae_mean": 7.0,
                "rkf_mae_opt_mean": 6.0,
            },
        )
        ensemble_spec = Path(self.tmpdir.name) / "weighted_ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "FinalModel", "n_features": 1},
                {"model_name": "IterationModel", "n_features": 1},
            ]
        ).to_csv(ensemble_spec, index=False)

        results = ensemble_validation(
            ensemble_csv=str(ensemble_spec),
            external_data=pd.DataFrame({"f1": [1.0], "f2": [2.0]}),
            target_col=None,
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir),
        )

        self.assertAlmostEqual(
            results["predictions"]["predicted_weighted"].iloc[0],
            1.4,
        )

    def test_ensemble_invalid_weight_metric_is_recorded_without_nan_output(self):
        write_checkpoint(
            self.models_dir,
            "ValidModel",
            "final",
            "20240103_120000",
            ["f1"],
            prediction=2.0,
            metrics={
                "secondary": {"internal_cv": {"rkf_mae_mean": 1.0}},
            },
        )
        write_checkpoint(
            self.models_dir,
            "InvalidModel",
            "final",
            "20240104_120000",
            ["f2"],
            prediction=9.0,
            metrics={
                "secondary": {"internal_cv": {"rkf_mae_mean": float("nan")}},
                "mae_mean": 0.5,
            },
        )
        ensemble_spec = Path(self.tmpdir.name) / "invalid_weight_ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "ValidModel", "n_features": 1},
                {"model_name": "InvalidModel", "n_features": 1},
            ]
        ).to_csv(ensemble_spec, index=False)

        results = ensemble_validation(
            ensemble_csv=str(ensemble_spec),
            external_data=pd.DataFrame({"f1": [1.0], "f2": [2.0]}),
            target_col=None,
            output_dir=str(self.output_dir),
            models_dir=str(self.models_dir),
        )

        self.assertEqual(results["predictions"]["predicted_weighted"].tolist(), [2.0])
        self.assertTrue(np.isfinite(results["predictions"]["predicted_weighted"]).all())
        self.assertEqual(len(results["ensemble_errors"]), 1)
        self.assertIn("finite and > 0", results["ensemble_errors"][0][2])

    def test_ensemble_duplicate_input_index_preserves_each_original_row(self):
        write_checkpoint(
            self.models_dir,
            "ModelA",
            "final",
            "20240103_120000",
            ["f1"],
            prediction=1.0,
        )
        write_checkpoint(
            self.models_dir,
            "ModelB",
            "final",
            "20240104_120000",
            ["f2"],
            prediction=3.0,
        )
        ensemble_spec = Path(self.tmpdir.name) / "duplicate_index_ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "ModelA", "n_features": 1},
                {"model_name": "ModelB", "n_features": 1},
            ]
        ).to_csv(ensemble_spec, index=False)
        external_df = pd.DataFrame(
            {
                "sub_H": ["H0", "H1", "H2"],
                "activation_energy": [10.0, 20.0, 30.0],
                "f1": [0.1, 0.2, 0.3],
                "f2": [1.1, 1.2, 1.3],
            },
            index=pd.Index([7, 7, 8], name="sample_index"),
        )

        with patch("src.external_validation._plot_external_scatter", return_value=None):
            results = ensemble_validation(
                ensemble_csv=str(ensemble_spec),
                external_data=external_df,
                output_dir=str(self.output_dir),
                models_dir=str(self.models_dir),
            )

        predictions = results["predictions"]
        self.assertEqual(predictions.index.tolist(), [7, 7, 8])
        self.assertEqual(predictions["original_index"].tolist(), [7, 7, 8])
        self.assertEqual(predictions["sub_H"].tolist(), ["H0", "H1", "H2"])
        self.assertEqual(predictions["activation_energy"].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(predictions["predicted_weighted"].tolist(), [2.0, 2.0, 2.0])

    def test_ensemble_member_with_no_complete_rows_fails_before_scaler_transform(self):
        write_checkpoint(
            self.models_dir,
            "ModelA",
            "final",
            "20240103_120000",
            ["f1"],
        )
        ensemble_spec = Path(self.tmpdir.name) / "all_nan_ensemble.csv"
        pd.DataFrame(
            [{"model_name": "ModelA", "n_features": 1}]
        ).to_csv(ensemble_spec, index=False)

        with self.assertRaisesRegex(RuntimeError, r"ModelA.*no complete rows"):
            ensemble_validation(
                ensemble_csv=str(ensemble_spec),
                external_data=pd.DataFrame({"f1": [np.nan, np.nan]}),
                target_col=None,
                output_dir=str(self.output_dir),
                models_dir=str(self.models_dir),
            )

    def test_ensemble_disjoint_member_rows_raise_clear_domain_error(self):
        write_checkpoint(
            self.models_dir,
            "ModelA",
            "final",
            "20240103_120000",
            ["f1"],
        )
        write_checkpoint(
            self.models_dir,
            "ModelB",
            "final",
            "20240104_120000",
            ["f2"],
        )
        ensemble_spec = Path(self.tmpdir.name) / "disjoint_ensemble.csv"
        pd.DataFrame(
            [
                {"model_name": "ModelA", "n_features": 1},
                {"model_name": "ModelB", "n_features": 1},
            ]
        ).to_csv(ensemble_spec, index=False)

        with self.assertRaisesRegex(
            RuntimeError,
            r"No common complete rows.*member prediction intersection",
        ):
            ensemble_validation(
                ensemble_csv=str(ensemble_spec),
                external_data=pd.DataFrame(
                    {"f1": [1.0, np.nan], "f2": [np.nan, 2.0]}
                ),
                target_col=None,
                output_dir=str(self.output_dir),
                models_dir=str(self.models_dir),
            )


if __name__ == "__main__":
    unittest.main()
