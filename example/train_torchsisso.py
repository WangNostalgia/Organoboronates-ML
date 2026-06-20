#!/usr/bin/env python
"""Standalone TorchSISSO baseline for organoboronate activation-energy data.

This script deliberately lives outside the default Optuna/SHAP-RFECV pipeline.
TorchSISSO is a symbolic-regression baseline with a different fitting API, so we
keep it isolated while preserving the repository's key evaluation rule:

    one fixed 80/20 development/final-test split with random_state=40.

Model selection is performed only inside the development set. The final test set
is scored exactly once after the TorchSISSO hyperparameter grid has been chosen.

Example
-------
python example/train_torchsisso.py \
    --data example/B_dataset.csv \
    --target activation_energy \
    --output_dir models/TorchSISSO \
    --n_expansion 1 2 \
    --n_term 1 2 \
    --k 20 50 \
    --operators + - "*" / "pow(2)" ln \
    --initial_screening spearman 0.95

Notes
-----
TorchSISSO returns compact symbolic equations. In this script the equations are
learned on median-imputed and standardized descriptor values, not raw physical
units. The saved feature mapping and scaler are therefore part of the model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_OPERATORS = ["+", "-", "*", "/", "pow(2)", "ln"]
DEFAULT_DROP_COLUMNS = ("ID", "SMILES", "filename", "stoichiometry")


@dataclass(frozen=True)
class TorchSISSOParams:
    """Serializable TorchSISSO search parameters."""

    n_expansion: int
    n_term: int
    k: int
    operators: tuple[str, ...]
    initial_screening_method: str | None = None
    initial_screening_quantile: float | None = None
    use_gpu: bool = False


@dataclass
class FittedTorchSISSO:
    """Small wrapper used for saving final metadata.

    The TorchSISSO model object itself may not be stable across package versions,
    so the main reusable artifact is the fitted equation plus preprocessing.
    """

    equation: str
    feature_names: list[str]
    original_feature_names: list[str]
    feature_name_mapping: dict[str, str]
    preprocessing: Pipeline
    params: dict[str, Any]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def bounded_quantile(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a standalone TorchSISSO symbolic-regression baseline with a "
            "development-only model-selection grid and one final-test score."
        )
    )
    parser.add_argument("--data", default="example/B_dataset.csv", help="Input CSV path.")
    parser.add_argument(
        "--target",
        default="activation_energy",
        help="Regression target column. Default: activation_energy.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/TorchSISSO",
        help="Directory for metrics, predictions, and fitted metadata.",
    )
    parser.add_argument(
        "--drop_columns",
        nargs="*",
        default=list(DEFAULT_DROP_COLUMNS),
        help="Columns to drop before numeric feature selection.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Final-test fraction. Keep at 0.2 to match the main pipeline.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=40,
        help="Seed for the shared development/final-test split.",
    )
    parser.add_argument(
        "--cv_splits",
        type=positive_int,
        default=5,
        help="RepeatedKFold split count inside development data.",
    )
    parser.add_argument(
        "--cv_repeats",
        type=positive_int,
        default=5,
        help="RepeatedKFold repeat count inside development data.",
    )
    parser.add_argument(
        "--max_grid_fits",
        type=positive_int,
        default=None,
        help="Optional cap on parameter combinations for smoke tests.",
    )
    parser.add_argument(
        "--n_expansion",
        type=positive_int,
        nargs="+",
        default=[1, 2],
        help="TorchSISSO feature-expansion depths to search.",
    )
    parser.add_argument(
        "--n_term",
        type=positive_int,
        nargs="+",
        default=[1, 2, 3],
        help="Number of terms in the final equation.",
    )
    parser.add_argument(
        "--k",
        type=positive_int,
        nargs="+",
        default=[20, 50],
        help="Number of SIS-screened features passed to L0 regularization.",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=DEFAULT_OPERATORS,
        help=(
            "Operators used for feature construction. Conservative default: "
            "+ - * / pow(2) ln. Quote shell-sensitive operators such as '*'."
        ),
    )
    parser.add_argument(
        "--initial_screening",
        nargs=2,
        metavar=("METHOD", "QUANTILE"),
        default=None,
        help=(
            "Optional TorchSISSO initial screening, e.g. 'spearman 0.95' or "
            "'mi 0.95'. The second value is parsed as a quantile in (0, 1]."
        ),
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Pass use_gpu=True to TorchSISSO. Requires a CUDA-enabled torch install.",
    )
    parser.add_argument(
        "--no_cv",
        action="store_true",
        help=(
            "Smoke-test mode: skip development CV and fit only the first parameter "
            "combination on the development set before final-test scoring."
        ),
    )
    return parser.parse_args()


def make_safe_feature_name(name: str, index: int) -> str:
    """Create a stable SymPy/Python-safe feature name."""

    cleaned = re.sub(r"\W+", "_", str(name)).strip("_")
    if not cleaned:
        cleaned = f"feature_{index}"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return f"x{index:03d}_{cleaned}"


def load_numeric_dataset(
    csv_path: str | Path,
    target: str,
    drop_columns: Iterable[str],
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Load CSV data, keep numeric descriptors, and sanitize feature names."""

    data = pd.read_csv(csv_path)
    data = data.dropna(axis=1, how="all")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {csv_path}.")

    rows_before = len(data)
    data = data.dropna(subset=[target]).copy()
    if len(data) != rows_before:
        print(f"Dropped {rows_before - len(data)} rows with missing target values.")

    y = data[target].astype(float)
    candidate = data.drop(columns=[target], errors="ignore")
    candidate = candidate.drop(columns=list(drop_columns), errors="ignore")
    X_numeric = candidate.select_dtypes(include=[np.number]).copy()
    X_numeric = X_numeric.dropna(axis=1, how="all")

    if X_numeric.empty:
        raise ValueError("No numeric feature columns remain after preprocessing.")

    mapping: dict[str, str] = {}
    safe_columns: list[str] = []
    used: set[str] = set()

    for idx, column in enumerate(X_numeric.columns):
        safe = make_safe_feature_name(str(column), idx)
        while safe in used:
            safe = f"{safe}_{len(used)}"
        used.add(safe)
        mapping[safe] = str(column)
        safe_columns.append(safe)

    X_numeric.columns = safe_columns
    return X_numeric, y, mapping


def build_preprocessor() -> Pipeline:
    """Median-impute and standardize features using training data only."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def transform_to_dataframe(
    preprocessor: Pipeline,
    X: pd.DataFrame,
    feature_names: list[str],
    fit: bool,
) -> pd.DataFrame:
    values = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    return pd.DataFrame(values, columns=feature_names, index=X.index)


def make_torchsisso_dataframe(X_scaled: pd.DataFrame, y: pd.Series, target: str) -> pd.DataFrame:
    """TorchSISSO expects the target as the first dataframe column."""

    aligned_y = y.loc[X_scaled.index]
    return pd.concat(
        [aligned_y.rename(target).reset_index(drop=True), X_scaled.reset_index(drop=True)],
        axis=1,
    )


def build_param_grid(args: argparse.Namespace) -> list[TorchSISSOParams]:
    if args.initial_screening is None:
        method = None
        quantile = None
    else:
        method = args.initial_screening[0]
        quantile = bounded_quantile(args.initial_screening[1])

    params: list[TorchSISSOParams] = []
    for n_expansion, n_term, k in itertools.product(args.n_expansion, args.n_term, args.k):
        params.append(
            TorchSISSOParams(
                n_expansion=n_expansion,
                n_term=n_term,
                k=k,
                operators=tuple(args.operators),
                initial_screening_method=method,
                initial_screening_quantile=quantile,
                use_gpu=args.use_gpu,
            )
        )

    if args.max_grid_fits is not None:
        params = params[: args.max_grid_fits]
    return params


def import_torchsisso():
    try:
        from TorchSisso import SissoModel
    except ImportError as exc:
        raise ImportError(
            "TorchSisso is not installed. Install the optional environment with:\n"
            "    python -m pip install -r requirements-torchsisso.txt\n"
            "or install directly with:\n"
            "    python -m pip install TorchSisso torch sympy"
        ) from exc
    return SissoModel


def fit_torchsisso(
    train_df: pd.DataFrame,
    params: TorchSISSOParams,
) -> tuple[Any, float | None, str, float | None, Any]:
    """Fit TorchSISSO and return model plus its native fit outputs."""

    SissoModel = import_torchsisso()

    kwargs: dict[str, Any] = {
        "df": train_df,
        "operators": list(params.operators),
        "n_expansion": params.n_expansion,
        "n_term": params.n_term,
        "k": params.k,
        "use_gpu": params.use_gpu,
    }
    if params.initial_screening_method is not None:
        kwargs["initial_screening"] = [
            params.initial_screening_method,
            params.initial_screening_quantile,
        ]

    model = SissoModel(**kwargs)
    fit_result = model.fit()

    rmse: float | None = None
    equation = ""
    r2: float | None = None
    extra = None

    if isinstance(fit_result, tuple):
        if len(fit_result) > 0:
            rmse = _safe_float_or_none(fit_result[0])
        if len(fit_result) > 1:
            equation = str(fit_result[1])
        if len(fit_result) > 2:
            r2 = _safe_float_or_none(fit_result[2])
        if len(fit_result) > 3:
            extra = fit_result[3]
    else:
        equation = str(fit_result)

    return model, rmse, equation, r2, extra


def _safe_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def predict_with_torchsisso(
    model: Any,
    equation: str,
    X_scaled: pd.DataFrame,
) -> np.ndarray:
    """Predict using TorchSISSO's predict method, falling back to equation parsing."""

    predict = getattr(model, "predict", None)
    if callable(predict):
        prediction_attempts = (
            lambda: predict(X_scaled),
            lambda: predict(X_scaled.to_numpy()),
            lambda: predict(pd.concat([pd.Series(np.nan, index=X_scaled.index, name="target"), X_scaled], axis=1)),
        )
        for attempt in prediction_attempts:
            try:
                y_pred = np.asarray(attempt(), dtype=float).reshape(-1)
                if len(y_pred) == len(X_scaled) and np.all(np.isfinite(y_pred)):
                    return y_pred
            except Exception:
                continue

    return evaluate_equation(equation, X_scaled)


def normalise_equation_string(equation: str) -> str:
    """Convert common TorchSISSO equation text into a SymPy-friendly expression."""

    text = str(equation).strip()
    text = text.replace("^", "**")
    text = text.replace("ln(", "log(")

    # Strip simple left-hand sides such as "y = ..." or "activation_energy: ...".
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    elif ":" in text and text.split(":", 1)[0].strip().lower() in {"y", "target", "equation"}:
        text = text.split(":", 1)[1].strip()

    # Some reprs wrap the expression in a list-like one-element container.
    text = text.strip("[]")
    return text


def evaluate_equation(equation: str, X_scaled: pd.DataFrame) -> np.ndarray:
    """Evaluate a returned symbolic equation against a scaled feature dataframe."""

    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "sympy is required to evaluate TorchSISSO equations when the model "
            "does not expose a working predict() method."
        ) from exc

    expression_text = normalise_equation_string(equation)
    symbols = {name: sp.Symbol(name) for name in X_scaled.columns}
    local_dict = {
        **symbols,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "Abs": sp.Abs,
        "abs": sp.Abs,
    }

    try:
        expression = sp.sympify(expression_text, locals=local_dict)
    except Exception as exc:
        raise RuntimeError(
            "Could not parse the TorchSISSO equation for out-of-sample "
            f"prediction: {equation!r}. If the installed TorchSISSO version "
            "stores predictions under a different API, adapt "
            "predict_with_torchsisso()."
        ) from exc

    ordered_symbols = [symbols[name] for name in X_scaled.columns if symbols[name] in expression.free_symbols]
    if not ordered_symbols:
        constant = float(expression)
        return np.full(len(X_scaled), constant, dtype=float)

    func = sp.lambdify(ordered_symbols, expression, modules="numpy")
    values = [X_scaled[str(sym)].to_numpy(dtype=float) for sym in ordered_symbols]
    y_pred = np.asarray(func(*values), dtype=float)
    if y_pred.shape == ():
        y_pred = np.full(len(X_scaled), float(y_pred), dtype=float)
    return y_pred.reshape(-1)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=float).reshape(-1)
    pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_array) != len(pred_array):
        raise ValueError(f"Prediction length mismatch: {len(y_array)} != {len(pred_array)}")

    return {
        "mae": float(mean_absolute_error(y_array, pred_array)),
        "rmse": float(mean_squared_error(y_array, pred_array, squared=False)),
        "r2": float(r2_score(y_array, pred_array)),
    }


def evaluate_params_cv(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    target: str,
    params: TorchSISSOParams,
    cv_splits: int,
    cv_repeats: int,
    random_state: int,
) -> dict[str, Any]:
    """Run fold-local preprocessing and TorchSISSO fitting inside development data."""

    splitter = RepeatedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )

    fold_rows: list[dict[str, Any]] = []
    feature_names = list(X_dev.columns)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_dev), start=1):
        X_train = X_dev.iloc[train_idx].copy()
        X_val = X_dev.iloc[val_idx].copy()
        y_train = y_dev.iloc[train_idx].copy()
        y_val = y_dev.iloc[val_idx].copy()

        preprocessor = build_preprocessor()
        X_train_scaled = transform_to_dataframe(preprocessor, X_train, feature_names, fit=True)
        X_val_scaled = transform_to_dataframe(preprocessor, X_val, feature_names, fit=False)
        train_df = make_torchsisso_dataframe(X_train_scaled, y_train, target)

        started = time.time()
        model, native_rmse, equation, native_r2, _ = fit_torchsisso(train_df, params)
        y_val_pred = predict_with_torchsisso(model, equation, X_val_scaled)
        metrics = regression_metrics(y_val, y_val_pred)

        fold_rows.append(
            {
                "fold": fold_idx,
                "fit_seconds": time.time() - started,
                "equation": equation,
                "native_train_rmse": native_rmse,
                "native_train_r2": native_r2,
                **{f"val_{key}": value for key, value in metrics.items()},
            }
        )

    summary: dict[str, Any] = {
        **asdict(params),
        "folds": fold_rows,
        "cv_mae_mean": float(np.mean([row["val_mae"] for row in fold_rows])),
        "cv_mae_std": float(np.std([row["val_mae"] for row in fold_rows], ddof=1)),
        "cv_rmse_mean": float(np.mean([row["val_rmse"] for row in fold_rows])),
        "cv_r2_mean": float(np.mean([row["val_r2"] for row in fold_rows])),
    }
    return summary


def fit_and_score_final(
    X_dev: pd.DataFrame,
    X_test: pd.DataFrame,
    y_dev: pd.Series,
    y_test: pd.Series,
    target: str,
    params: TorchSISSOParams,
    mapping: dict[str, str],
) -> tuple[FittedTorchSISSO, pd.DataFrame, dict[str, Any]]:
    """Refit on full development data and score once on the final test set."""

    feature_names = list(X_dev.columns)
    preprocessor = build_preprocessor()
    X_dev_scaled = transform_to_dataframe(preprocessor, X_dev, feature_names, fit=True)
    X_test_scaled = transform_to_dataframe(preprocessor, X_test, feature_names, fit=False)

    train_df = make_torchsisso_dataframe(X_dev_scaled, y_dev, target)
    model, native_rmse, equation, native_r2, extra = fit_torchsisso(train_df, params)

    dev_pred = predict_with_torchsisso(model, equation, X_dev_scaled)
    test_pred = predict_with_torchsisso(model, equation, X_test_scaled)

    dev_metrics = regression_metrics(y_dev, dev_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "development",
                    "row_index": X_dev.index,
                    "y_true": y_dev.to_numpy(dtype=float),
                    "y_pred": dev_pred,
                }
            ),
            pd.DataFrame(
                {
                    "split": "final_test",
                    "row_index": X_test.index,
                    "y_true": y_test.to_numpy(dtype=float),
                    "y_pred": test_pred,
                }
            ),
        ],
        ignore_index=True,
    )

    fitted = FittedTorchSISSO(
        equation=equation,
        feature_names=feature_names,
        original_feature_names=[mapping[name] for name in feature_names],
        feature_name_mapping=mapping,
        preprocessing=preprocessor,
        params=asdict(params),
    )
    metrics = {
        "native_train_rmse": native_rmse,
        "native_train_r2": native_r2,
        "native_extra": str(extra) if extra is not None else None,
        "development": dev_metrics,
        "final_test": test_metrics,
    }
    return fitted, predictions, metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, mapping = load_numeric_dataset(args.data, args.target, args.drop_columns)
    print(f"Loaded {len(X)} rows and {X.shape[1]} numeric features from {args.data}.")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(
        f"Split data into development={len(X_dev)} and final_test={len(X_test)} "
        f"with random_state={args.random_state}."
    )

    param_grid = build_param_grid(args)
    if not param_grid:
        raise ValueError("No TorchSISSO parameter combinations were generated.")

    cv_results: list[dict[str, Any]] = []
    if args.no_cv:
        best_params = param_grid[0]
        print(f"--no_cv enabled; using first parameter set: {best_params}")
    else:
        print(f"Evaluating {len(param_grid)} TorchSISSO parameter combinations inside development data.")
        for idx, params in enumerate(param_grid, start=1):
            print(f"[{idx}/{len(param_grid)}] {params}")
            try:
                summary = evaluate_params_cv(
                    X_dev=X_dev,
                    y_dev=y_dev,
                    target=args.target,
                    params=params,
                    cv_splits=args.cv_splits,
                    cv_repeats=args.cv_repeats,
                    random_state=42,
                )
                summary["status"] = "ok"
                print(
                    "    cv_mae_mean="
                    f"{summary['cv_mae_mean']:.4f}, cv_r2_mean={summary['cv_r2_mean']:.4f}"
                )
            except Exception as exc:
                summary = {
                    **asdict(params),
                    "status": "failed",
                    "error": repr(exc),
                    "cv_mae_mean": np.inf,
                    "cv_mae_std": np.nan,
                    "cv_rmse_mean": np.inf,
                    "cv_r2_mean": np.nan,
                    "folds": [],
                }
                print(f"    failed: {exc}", file=sys.stderr)
            cv_results.append(summary)

        successful = [row for row in cv_results if row["status"] == "ok"]
        if not successful:
            raise RuntimeError("All TorchSISSO parameter combinations failed.")

        best_summary = min(successful, key=lambda row: row["cv_mae_mean"])
        best_params = TorchSISSOParams(
            n_expansion=int(best_summary["n_expansion"]),
            n_term=int(best_summary["n_term"]),
            k=int(best_summary["k"]),
            operators=tuple(best_summary["operators"]),
            initial_screening_method=best_summary["initial_screening_method"],
            initial_screening_quantile=best_summary["initial_screening_quantile"],
            use_gpu=bool(best_summary["use_gpu"]),
        )

    fitted, predictions, final_metrics = fit_and_score_final(
        X_dev=X_dev,
        X_test=X_test,
        y_dev=y_dev,
        y_test=y_test,
        target=args.target,
        params=best_params,
        mapping=mapping,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = output_dir / f"torchsisso_{timestamp}"

    if cv_results:
        flat_cv = []
        for row in cv_results:
            flat_row = {key: value for key, value in row.items() if key != "folds"}
            flat_cv.append(flat_row)
        pd.DataFrame(flat_cv).to_csv(f"{run_prefix}_cv_results.csv", index=False)
        write_json(Path(f"{run_prefix}_cv_results_full.json"), {"results": cv_results})

    predictions.to_csv(f"{run_prefix}_predictions.csv", index=False)
    joblib.dump(fitted, f"{run_prefix}_fitted_metadata.joblib")

    report = {
        "data": args.data,
        "target": args.target,
        "split": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "development_rows": int(len(X_dev)),
            "final_test_rows": int(len(X_test)),
        },
        "cv": None
        if args.no_cv
        else {
            "cv_splits": args.cv_splits,
            "cv_repeats": args.cv_repeats,
            "selection_metric": "cv_mae_mean",
            "selection_random_state": 42,
        },
        "best_params": asdict(best_params),
        "equation": fitted.equation,
        "feature_name_mapping": fitted.feature_name_mapping,
        "metrics": final_metrics,
        "outputs": {
            "predictions": f"{run_prefix}_predictions.csv",
            "fitted_metadata": f"{run_prefix}_fitted_metadata.joblib",
            "cv_results": None if args.no_cv else f"{run_prefix}_cv_results.csv",
            "cv_results_full": None if args.no_cv else f"{run_prefix}_cv_results_full.json",
        },
    }
    write_json(Path(f"{run_prefix}_metrics.json"), report)
    Path(f"{run_prefix}_equation.txt").write_text(fitted.equation + "\n", encoding="utf-8")

    print("\nBest TorchSISSO equation:")
    print(fitted.equation)
    print("\nFinal-test metrics:")
    print(json.dumps(final_metrics["final_test"], indent=2))
    print(f"\nSaved outputs with prefix: {run_prefix}")


if __name__ == "__main__":
    main()
