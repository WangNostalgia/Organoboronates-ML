import ast
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


WORKTREE_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = WORKTREE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_main_source():
    main_path = WORKTREE_ROOT / "main.py"
    return ast.parse(main_path.read_text(encoding="utf-8"))


def collect_cli_arguments():
    tree = parse_main_source()
    arguments = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue

        option = node.args[0].value
        argument_info = {"keywords": {}}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                argument_info["keywords"][keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                continue
        arguments[option] = argument_info

    return arguments


class CheckpointGuideFormatterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "load_checkpoint_guide",
            "example/load_checkpoint_guide.py",
        )

    def test_metric_value_prefers_current_keys_then_nested_then_legacy_aliases(self):
        metrics = {
            "test_mae": 1.2345,
            "primary": {"final_test": {"test_mae": 2.3456}},
            "mae_test_avg": 3.4567,
        }

        value = self.module._metric_value(
            metrics,
            "test_mae",
            ("primary", "final_test", "test_mae"),
            "mae_test_avg",
        )

        self.assertEqual(value, 1.2345)

    def test_metric_value_uses_nested_current_metric_before_legacy_alias(self):
        metrics = {
            "primary": {"final_test": {"test_r2": 0.81234}},
            "r2_test_avg": 0.45678,
        }

        value = self.module._metric_value(
            metrics,
            "test_r2",
            ("primary", "final_test", "test_r2"),
            "r2_test_avg",
        )

        self.assertEqual(value, 0.81234)

    def test_metric_value_skips_none_and_nan_and_falls_through_to_later_candidates(self):
        metrics = {
            "test_mae": None,
            "primary": {"final_test": {"test_mae": float("nan")}},
            "mae_test_avg": 3.4567,
        }

        value = self.module._metric_value(
            metrics,
            "test_mae",
            ("primary", "final_test", "test_mae"),
            "mae_test_avg",
        )

        self.assertEqual(value, 3.4567)

    def test_format_metric_returns_na_for_missing_and_does_not_numeric_format_strings_or_none(self):
        self.assertEqual(
            self.module._format_metric({}, "test_mae", ("primary", "final_test", "test_mae")),
            "N/A",
        )
        self.assertEqual(
            self.module._format_metric({"test_mae": None}, "test_mae"),
            "N/A",
        )
        self.assertEqual(
            self.module._format_metric({"test_mae": "already formatted"}, "test_mae"),
            "already formatted",
        )

    def test_compare_checkpoints_does_not_render_direction_arrow_when_any_side_is_na(self):
        info_a = {"features": ["f1"], "metrics": {"test_mae": None}}
        info_b = {"features": ["f1"], "metrics": {"test_mae": 1.2345}}

        with patch.object(
            self.module,
            "_load_checkpoint_info",
            side_effect=[info_a, info_b],
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.module.compare_checkpoints("models/SVR", 1, 1)

        output = buffer.getvalue()
        self.assertIn("N/A", output)
        self.assertNotIn("<-", output)
        self.assertNotIn("->", output)


class MainCliDefinitionTests(unittest.TestCase):
    def test_main_parser_has_no_mae_threshold_argument(self):
        arguments = collect_cli_arguments()

        self.assertNotIn("--mae_threshold", arguments)

    def test_main_parser_min_features_default_is_five(self):
        arguments = collect_cli_arguments()

        self.assertEqual(arguments["--min_features"]["keywords"]["default"], 5)

    def test_main_help_remains_available_when_optional_model_packages_are_missing(self):
        with tempfile.TemporaryDirectory() as blocker_dir:
            blocker_path = Path(blocker_dir)
            for module_name in ("xgboost", "lightgbm", "catboost", "gplearn"):
                (blocker_path / f"{module_name}.py").write_text(
                    'raise ImportError("blocked for help smoke test")\n',
                    encoding="utf-8",
                )

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(blocker_path), str(WORKTREE_ROOT), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

            completed = subprocess.run(
                [sys.executable, "main.py", "--help"],
                capture_output=True,
                text=True,
                cwd=WORKTREE_ROOT,
                env=env,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"main.py --help failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("--force_n_features", completed.stdout)

    def test_main_force_n_features_fails_atomically_before_runtime_setup_for_default_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(WORKTREE_ROOT / "main.py"),
                    "--force_n_features",
                    "3",
                ],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("GPlearn", completed.stderr)
            self.assertFalse((Path(tmpdir) / "models").exists())
            self.assertEqual(list(Path(tmpdir).glob("optimization_*.log")), [])


class DependencyMetadataTests(unittest.TestCase):
    def test_dependency_files_include_default_model_packages(self):
        pyproject_data = tomllib.loads(
            (WORKTREE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        pyproject_dependencies = pyproject_data["project"]["dependencies"]
        requirements_lines = (
            WORKTREE_ROOT / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()

        expected_pyproject = ("catboost", "gplearn", "lightgbm", "xgboost")
        expected_requirements = ("catboost", "gplearn", "lightgbm", "xgboost")

        for package_name in expected_pyproject:
            self.assertTrue(
                any(dep.lower().startswith(package_name) for dep in pyproject_dependencies),
                msg=f"{package_name} missing from pyproject.toml dependencies",
            )

        for package_name in expected_requirements:
            self.assertTrue(
                any(line.lower().startswith(package_name) for line in requirements_lines),
                msg=f"{package_name} missing from requirements.txt",
            )

    def test_pyproject_declares_exact_gplearn_pin_for_supported_installations(self):
        pyproject_text = (WORKTREE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertRegex(pyproject_text, r'"gplearn==0\.4\.2"')


class ActiveDocsAndExamplesTests(unittest.TestCase):
    def test_active_docs_and_examples_do_not_reference_mae_threshold(self):
        active_paths = [
            "README.md",
            "README_CN.md",
            "user_manual.md",
            "pipeline.md",
            "AGENTS.md",
            "CLAUDE.md",
            "example/load_checkpoint_guide.py",
        ]

        offenders = []
        for relative_path in active_paths:
            text = (WORKTREE_ROOT / relative_path).read_text(encoding="utf-8")
            if re.search(r"\bmae_threshold\b", text):
                offenders.append(relative_path)

        self.assertEqual(offenders, [])

    def test_git_tracks_only_canonical_pipeline_markdown_and_docs_link_to_lowercase_name(self):
        completed = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
            check=True,
        )
        tracked_files = completed.stdout.splitlines()

        self.assertIn("pipeline.md", tracked_files)
        self.assertNotIn("Pipeline.md", tracked_files)

        for relative_path in [
            "README.md",
            "README_CN.md",
            "user_manual.md",
            "AGENTS.md",
            "CLAUDE.md",
        ]:
            text = (WORKTREE_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("](Pipeline.md)", text, msg=relative_path)
            self.assertIn("](pipeline.md)", text, msg=relative_path)

    def test_active_docs_describe_current_metric_schema(self):
        for relative_path in [
            "README.md",
            "README_CN.md",
            "user_manual.md",
            "AGENTS.md",
            "CLAUDE.md",
            "pipeline.md",
        ]:
            text = (WORKTREE_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("metrics.primary.final_test", text, msg=relative_path)
            self.assertIn("metrics.secondary.internal_cv", text, msg=relative_path)
            self.assertIn("metrics.secondary.stability", text, msg=relative_path)
            self.assertIn("metrics.secondary.loo", text, msg=relative_path)
            self.assertIn("metrics.internal_cv", text, msg=relative_path)

    def test_active_docs_do_not_describe_feature_removal_as_stopping_when_no_candidate_qualifies(self):
        for relative_path in [
            "README.md",
            "README_CN.md",
            "user_manual.md",
            "AGENTS.md",
            "CLAUDE.md",
            "pipeline.md",
        ]:
            text = (WORKTREE_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("no more features qualify for removal", text, msg=relative_path)
            self.assertNotIn("no more features to remove", text, msg=relative_path)

    def test_historical_examples_are_labeled_non_authoritative(self):
        diagnose_text = (WORKTREE_ROOT / "example/diagnose_lasso.py").read_text(encoding="utf-8")
        improvement_text = (WORKTREE_ROOT / "example/improvement_code_examples.py").read_text(encoding="utf-8")

        self.assertRegex(diagnose_text[:400], r"historical|legacy")
        self.assertRegex(diagnose_text[:400], r"not current|non-current|non-authoritative")
        self.assertRegex(improvement_text[:400], r"historical|conceptual")
        self.assertRegex(improvement_text[:400], r"not authoritative|non-authoritative")


if __name__ == "__main__":
    unittest.main()
