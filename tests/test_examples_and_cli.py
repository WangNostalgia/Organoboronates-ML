import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


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

    def test_pyproject_declares_gplearn_floor_for_supported_installations(self):
        pyproject_text = (WORKTREE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertRegex(pyproject_text, r'"gplearn>=0\.4\.2"')


class ActiveDocsAndExamplesTests(unittest.TestCase):
    def test_active_docs_and_examples_do_not_reference_mae_threshold(self):
        active_paths = [
            "README.md",
            "README_CN.md",
            "user_manual.md",
            "Pipeline.md",
            "pipeline.md",
            "example/load_checkpoint_guide.py",
        ]

        offenders = []
        for relative_path in active_paths:
            text = (WORKTREE_ROOT / relative_path).read_text(encoding="utf-8")
            if re.search(r"\bmae_threshold\b", text):
                offenders.append(relative_path)

        self.assertEqual(offenders, [])


class PipelineMirrorTests(unittest.TestCase):
    def test_pipeline_documents_remain_identical(self):
        uppercase = (WORKTREE_ROOT / "Pipeline.md").read_text(encoding="utf-8")
        lowercase = (WORKTREE_ROOT / "pipeline.md").read_text(encoding="utf-8")

        self.assertEqual(uppercase, lowercase)


if __name__ == "__main__":
    unittest.main()
