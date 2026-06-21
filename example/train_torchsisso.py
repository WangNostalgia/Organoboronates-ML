#!/usr/bin/env python
"""Standalone TorchSISSO baseline for organoboronate activation-energy data.

The script keeps TorchSISSO outside the default Optuna/SHAP-RFECV registry while
preserving the repository's evaluation rule: one fixed 80/20 development/final-
test split with random_state=40, development-only model selection, and one final-
test score after the symbolic equation is locked.
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
    n_expansion: int
    n_term: int
    k: int
    operators: tuple[str, ...]
    initial_screening_method: str | None = None
    initial_screening_quantile: float | None = None
    use_gpu: bool = False


@dataclass
class FittedTorchSISSO:
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
    parser = argparse.ArgumentParser(description="Train a standalone TorchSISSO baseline.")
    parser.add_argument("--data", default="example/B_dataset.csv", help="Input CSV path.")
    parser.add_argument("--target", default="activation_energy", help="Regression target column.")
    parser.add_argument("--output_dir", default="models/TorchSISSO", help="Output directory.")
    parser.add_argument("--drop_columns", nargs="*", default=list(DEFAULT_DROP_COLUMNS))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=40)
    parser.add_argument("--cv_splits", type=positive_int, default=5)
    parser.add_argument("--cv_repeats", type=positive_int, default=5)
    parser.add_argument("--max_grid_fits", type=positive_int, default=None)
    parser.add_argument("--n_expansion", type=positive_int, nargs="+", default=[1, 2])
    parser.add_argument("--n_term", type=positive_int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--k", type=positive_int, nargs="+", default=[20, 50])
    parser.add_argument("--operators", nargs="+", default=DEFAULT_OPERATORS)
    parser.add_argument("--initial_screening", nargs=2, metavar=("METHOD", "QUANTILE"), default=None)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--no_cv", action="store_true", help="Smoke-test mode without development CV.")
    return parser.parse_args()


def safe_feature_name(name: str, index: int) -> str:
    cleaned = re.sub(r"\W+", "_", str(name)).strip("_") or f"feature_{index}"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return f"x{index:03d}_{cleaned}"


def load_numeric_dataset(
    csv_path: str | Path,
    target: str,
    drop_columns: Iterable[str],
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    data = pd.read_csv(csv_path).dropna(axis=1, how="all")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {csv_path}.")

    rows_before = len(data)
    data = data.dropna(subset=[target]).copy()
    if len(data) != rows_before:
        print(f"Dropped {rows_before - len(data)} rows with missing target values.")

    y = data[target].astype(float)
    X = data.drop(columns=[target], errors="ignore")
    X = X.drop(columns=list(drop_columns), errors="ignore")
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all").copy()
    if X.empty:
        raise ValueError("No numeric feature columns remain after preprocessing.")

    mapping: dict[str, str] = {}
    safe_columns: list[str] = []
    used: set[str] = set()
    for index, column in enumerate(X.columns):
        safe = safe_feature_name(str(column), index)
        while safe in used:
            safe = f"{safe}_{len(used)}"
        used.add(safe)
        mapping[safe] = str(column)
        safe_columns.append(safe)
    X.columns = safe_columns
    return X, y, mapping


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def transform(preprocessor: Pipeline, X: pd.DataFrame, feature_names: list[str], fit: bool) -> pd.DataFrame:
    values = preprocessor.fit_transform(X) if fit else preprocessor.transform(X)
    return pd.DataFrame(values, columns=feature_names, index=X.index)


def make_sisso_df(X_scaled: pd.DataFrame, y: pd.Series, target: str) -> pd.DataFrame:
    return pd.concat(
        [y.loc[X_scaled.index].rename(target).reset_index(drop=True), X_scaled.reset_index(drop=True)],
        axis=1,
    )


def build_param_grid(args: argparse.Namespace) -> list[TorchSISSOParams]:
    if args.initial_screening is None:
        method = None
        quantile = None
    else:
        method = args.initial_screening[0]
        quantile = bounded_quantile(args.initial_screening[1])

    grid = [
        TorchSISSOParams(
            n_expansion=n_expansion,
            n_term=n_term,
            k=k,
            operators=tuple(args.operators),
            initial_screening_method=method,
            initial_screening_quantile=quantile,
            use_gpu=args.use_gpu,
        )
        for n_expansion, n_term, k in itertools.product(args.n_expansion, args.n_term, args.k)
    ]
    return grid[: args.max_grid_fits] if args.max_grid_fits is not None else grid


def import_torchsisso():
    try:
        from TorchSisso import SissoModel
    except ImportError as exc:
        raise ImportError(
            "TorchSisso is not installed. Run: python -m pip install -r requirements-torchsisso.txt"
        ) from exc
    return SissoModel


def fit_torchsisso(train_df: pd.DataFrame, params: TorchSISSOParams) -> tuple[Any, float | None, str, float | None, Any]:
    kwargs: dict[str, Any] = {
        "data": train_df,
        "operators": list(params.operators),
        "n_expansion": params.n_expansion,
        "n_term": params.n_term,
        "k": params.k,
        "use_gpu": params.use_gpu,
    }
    if params.initial_screening_method is not None:
        kwargs["initial_screening"] = [params.initial_screening_method, params.initial_screening_quantile]

    model = import_torchsisso()(**kwargs)
    result = model.fit()
    if not isinstance(result, tuple):
        return model, None, str(result), None, None

    native_rmse = _safe_float(result[0]) if len(result) > 0 else None
    equation = str(result[1]) if len(result) > 1 else ""
    native_r2 = _safe_float(result[2]) if len(result) > 2 else None
    extra = result[3] if len(result) > 3 else None
    return model, native_rmse, equation, native_r2, extra


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalise_equation(equation: str) -> str:
    text = str(equation).strip().replace("^", "**").replace("ln(", "log(")
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    elif ":" in text and text.split(":", 1)[0].strip().lower() in {"y", "target", "equation"}:
        text = text.split(":", 1)[1].strip()
    return text.strip("[]")


def evaluate_equation(equation: str, X_scaled: pd.DataFrame) -> np.ndarray:
    import sympy as sp

    expression_text = normalise_equation(equation)
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
        raise RuntimeError(f"Could not parse TorchSISSO equation: {equation!r}") from exc

    ordered_symbols = [symbols[name] for name in X_scaled.columns if symbols[name] in expression.free_symbols]
    if not ordered_symbols:
        return np.full(len(X_scaled), float(expression), dtype=float)

    func = sp.lambdify(ordered_symbols, expression, modules="numpy")
    values = [X_scaled[str(symbol)].to_numpy(dtype=float) for symbol in ordered_symbols]
    prediction = np.asarray(func(*values), dtype=float)
    if prediction.shape == ():
        prediction = np.full(len(X_scaled), float(prediction), dtype=float)
    return prediction.reshape(-1)


def predict_with_torchsisso(model: Any, equation: str, X_scaled: pd.DataFrame) -> np.ndarray:
    predict = getattr(model, "predict", None)
    if callable(predict):
        attempts = (
            lambda: predict(X_scaled),
            lambda: predict(X_scaled.to_numpy()),
            lambda: predict(pd.concat([pd.Series(np.nan, index=X_scaled.index, name="target"), X_scaled], axis=1)),
        )
        for attempt in attempts:
            try:
                y_pred = np.asarray(attempt(), dtype=float).reshape(-1)
                if len(y_pred) == len(X_scaled) and np.all(np.isfinite(y_pred)):
                    return y_pred
            except Exception:
                continue
    return evaluate_equation(equation, X_scaled)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=float).reshape(-1)
    pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_array) != len(pred_array):
        raise ValueError(f"Prediction length mismatch: {len(y_array)} != {len(pred_array)}")

    # Compatible with both older and newer scikit-learn versions; avoid squared=False.
    mse = float(mean_squared_error(y_array, pred_array))
    return {
        "mae": float(mean_absolute_error(y_array, pred_array)),
        "rmse": float(np.sqrt(mse)),
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
    splitter = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)
    feature_names = list(X_dev.columns)
    folds: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_dev), start=1):
        X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        preprocessor = build_preprocessor()
        X_train_scaled = transform(preprocessor, X_train, feature_names, fit=True)
        X_val_scaled = transform(preprocessor, X_val, feature_names, fit=False)
        train_df = make_sisso_df(X_train_scaled, y_train, target)

        started = time.time()
        model, native_rmse, equation, native_r2, _ = fit_torchsisso(train_df, params)
        y_val_pred = predict_with_torchsisso(model, equation, X_val_scaled)
        metrics = regression_metrics(y_val, y_val_pred)
        folds.append(
            {
                "fold": fold,
                "fit_seconds": time.time() - started,
                "equation": equation,
                "native_train_rmse": native_rmse,
                "native_train_r2": native_r2,
                **{f"val_{key}": value for key, value in metrics.items()},
            }
        )

    return {
        **asdict(params),
        "folds": folds,
        "cv_mae_mean": float(np.mean([row["val_mae"] for row in folds])),
        "cv_mae_std": float(np.std([row["val_mae"] for row in folds], ddof=1)),
        "cv_rmse_mean": float(np.mean([row["val_rmse"] for row in folds])),
        "cv_r2_mean": float(np.mean([row["val_r2"] for row in folds])),
    }


def fit_and_score_final(
    X_dev: pd.DataFrame,
    X_test: pd.DataFrame,
    y_dev: pd.Series,
    y_test: pd.Series,
    target: str,
    params: TorchSISSOParams,
    mapping: dict[str, str],
) -> tuple[FittedTorchSISSO, pd.DataFrame, dict[str, Any]]:
    feature_names = list(X_dev.columns)
    preprocessor = build_preprocessor()
    X_dev_scaled = transform(preprocessor, X_dev, feature_names, fit=True)
    X_test_scaled = transform(preprocessor, X_test, feature_names, fit=False)
    train_df = make_sisso_df(X_dev_scaled, y_dev, target)

    model, native_rmse, equation, native_r2, extra = fit_torchsisso(train_df, params)
    dev_pred = predict_with_torchsisso(model, equation, X_dev_scaled)
    test_pred = predict_with_torchsisso(model, equation, X_test_scaled)

    predictions = pd.concat(
        [
            pd.DataFrame({"split": "development", "row_index": X_dev.index, "y_true": y_dev.to_numpy(float), "y_pred": dev_pred}),
            pd.DataFrame({"split": "final_test", "row_index": X_test.index, "y_true": y_test.to_numpy(float), "y_pred": test_pred}),
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
        "development": regression_metrics(y_dev, dev_pred),
        "final_test": regression_metrics(y_test, test_pred),
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
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Split data into development={len(X_dev)} and final_test={len(X_test)} with random_state={args.random_state}.")

    param_grid = build_param_grid(args)
    if not param_grid:
        raise ValueError("No TorchSISSO parameter combinations were generated.")

    cv_results: list[dict[str, Any]] = []
    if args.no_cv:
        best_params = param_grid[0]
        print(f"--no_cv enabled; using first parameter set: {best_params}")
    else:
        print(f"Evaluating {len(param_grid)} TorchSISSO parameter combinations inside development data.")
        for index, params in enumerate(param_grid, start=1):
            print(f"[{index}/{len(param_grid)}] {params}")
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
                print(f"    cv_mae_mean={summary['cv_mae_mean']:.4f}, cv_r2_mean={summary['cv_r2_mean']:.4f}")
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
        pd.DataFrame([{key: value for key, value in row.items() if key != "folds"} for row in cv_results]).to_csv(
            f"{run_prefix}_cv_results.csv", index=False
        )
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
