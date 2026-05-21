"""Train preoperative measurement-only POD1 vault regression baselines.

This is a preoperative measurement-only baseline. Postoperative 2DAnalysis
measurements must not be used as input features. The script reads an existing
preop measurement-only manifest, uses the manifest-provided patient-level
train/val/test split, and does not read images or modify training code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [
    "cct_mean_um",
    "acd_epi_mean_mm",
    "acd_endo_mean_mm",
    "clr_mean_um",
    "ata_mean_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train preop measurement-only POD1 vault baselines.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_batch_01_ready.csv",
    )
    parser.add_argument("--run_name", type=str, default="preop_measurement_ready")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def output_root(run_name: str) -> Path:
    run = run_name.strip() or "preop_measurement_ready"
    return PROJECT_ROOT / "artifacts/reports/preop_measurement_baseline_batch_01" / run


def validate_manifest(df: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "patient_id",
        "patient_uid",
        "eye_side",
        "split",
        "vault_label",
        "measurement_ready_status",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique.")
    split_counts = df.groupby("patient_uid")["split"].nunique()
    leaked = split_counts[split_counts > 1].index.tolist()
    if leaked:
        raise ValueError(f"Patient split leakage found: {leaked}")
    for column in FEATURE_COLUMNS + ["vault_label"]:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{column} contains missing/non-numeric values.")
    if (pd.to_numeric(df["vault_label"], errors="coerce") <= 0).any():
        raise ValueError("vault_label must be > 0.")


def split_xy(df: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    subset = df[df["split"] == split].copy()
    x = subset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(subset["vault_label"], errors="coerce").to_numpy(dtype=float)
    return x, y, subset


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def prediction_df(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": meta["sample_id"].to_numpy(),
            "patient_id": meta["patient_id"].to_numpy(),
            "eye_side": meta["eye_side"].to_numpy(),
            "split": meta["split"].to_numpy(),
            "vault_label_um": y_true,
            "pred_vault_um": y_pred,
            "abs_error_um": np.abs(y_pred - y_true),
            "signed_error_um": y_pred - y_true,
            "measurement_ready_status": meta["measurement_ready_status"].to_numpy(),
        }
    )


def candidate_models(seed: int) -> Dict[str, List[Tuple[Dict[str, Any], Any]]]:
    return {
        "linear_regression": [
            (
                {},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LinearRegression()),
                    ]
                ),
            )
        ],
        "ridge_regression": [
            (
                {"alpha": alpha},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=alpha, random_state=seed)),
                    ]
                ),
            )
            for alpha in [0.1, 1.0, 10.0, 100.0]
        ],
        "random_forest": [
            (
                {"n_estimators": 500, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf},
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=1,
                ),
            )
            for max_depth in [2, 3, 5, None]
            for min_samples_leaf in [1, 3, 5]
        ],
        "mlp_regressor": [
            (
                {"hidden_layer_sizes": hidden, "alpha": alpha},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=hidden,
                                alpha=alpha,
                                max_iter=2000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
            for hidden in [(16,), (32,), (32, 16)]
            for alpha in [0.0001, 0.001, 0.01]
        ],
    }


def select_and_evaluate_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for model_name, candidates in candidate_models(seed).items():
        best: Dict[str, Any] | None = None
        for params, estimator in candidates:
            estimator.fit(x_train, y_train)
            val_pred = estimator.predict(x_val)
            val_metrics = metrics(y_val, val_pred)
            if best is None or val_metrics["mae"] < best["val_metrics"]["mae"]:
                best = {
                    "params": params,
                    "estimator": estimator,
                    "val_metrics": val_metrics,
                    "val_pred": val_pred,
                }
        assert best is not None
        test_pred = best["estimator"].predict(x_test)
        best["test_metrics"] = metrics(y_test, test_pred)
        best["test_pred"] = test_pred
        results[model_name] = best
    return results


def write_predictions(
    results: Dict[str, Dict[str, Any]],
    val_meta: pd.DataFrame,
    y_val: np.ndarray,
    test_meta: pd.DataFrame,
    y_test: np.ndarray,
    predictions_dir: Path,
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for model_name, result in results.items():
        prediction_df(val_meta, y_val, result["val_pred"]).to_csv(
            predictions_dir / f"{model_name}_val_predictions.csv",
            index=False,
            encoding="utf-8",
        )
        prediction_df(test_meta, y_test, result["test_pred"]).to_csv(
            predictions_dir / f"{model_name}_test_predictions.csv",
            index=False,
            encoding="utf-8",
        )


def build_summary(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, result in results.items():
        rows.append(
            {
                "model_name": model_name,
                "best_params": json.dumps(result["params"], sort_keys=True),
                "val_mae_um": result["val_metrics"]["mae"],
                "val_rmse_um": result["val_metrics"]["rmse"],
                "val_r2": result["val_metrics"]["r2"],
                "test_mae_um": result["test_metrics"]["mae"],
                "test_rmse_um": result["test_metrics"]["rmse"],
                "test_r2": result["test_metrics"]["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values("val_mae_um", kind="stable")


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in ["val_mae_um", "val_rmse_um", "test_mae_um", "test_rmse_um"]:
        display[column] = display[column].map(lambda value: f"{float(value):.2f}")
    for column in ["val_r2", "test_r2"]:
        display[column] = display[column].map(lambda value: f"{float(value):.3f}")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines) + "\n"


def feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FEATURE_COLUMNS + ["vault_label"]:
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def get_linear_coefficients(estimator: Any, model_name: str) -> pd.DataFrame:
    model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
    coef = getattr(model, "coef_", np.zeros(len(FEATURE_COLUMNS)))
    intercept = float(getattr(model, "intercept_", 0.0))
    return pd.DataFrame(
        {
            "model_name": model_name,
            "feature": FEATURE_COLUMNS,
            "coefficient": np.asarray(coef).reshape(-1),
            "intercept": intercept,
        }
    )


def write_model_artifacts(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    linear = results["linear_regression"]["estimator"]
    ridge = results["ridge_regression"]["estimator"]
    rf = results["random_forest"]["estimator"]
    get_linear_coefficients(linear, "linear_regression").to_csv(
        out_dir / "linear_coefficients.csv",
        index=False,
        encoding="utf-8",
    )
    get_linear_coefficients(ridge, "ridge_regression").to_csv(
        out_dir / "ridge_coefficients.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        out_dir / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_mae(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    df = summary_df.sort_values("val_mae_um", kind="stable")
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, df["val_mae_um"], width=width, label="Val MAE", color="#4C78A8")
    ax.bar(x + width / 2, df["test_mae_um"], width=width, label="Test MAE", color="#F58518")
    ax.set_title("Preop Measurement Baseline MAE")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (um)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "model_val_test_mae_bar.png")


def plot_pred_vs_gt(best_model: str, test_pred_df: pd.DataFrame, figures_dir: Path) -> None:
    y = test_pred_df["vault_label_um"]
    p = test_pred_df["pred_vault_um"]
    lower = min(y.min(), p.min()) - 40
    upper = max(y.max(), p.max()) + 40
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.plot([lower, upper], [lower, upper], color="#D62728", linestyle="--", linewidth=1.2)
    ax.set_title(f"Predicted vs Ground Truth ({best_model})")
    ax.set_xlabel("Ground truth POD1 vault (um)")
    ax.set_ylabel("Predicted POD1 vault (um)")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "pred_vs_gt_best_model_test.png")


def plot_abs_error(best_model: str, test_pred_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(test_pred_df["abs_error_um"], bins=8, color="#4C78A8", alpha=0.85)
    ax.set_title(f"Test Absolute Error Distribution ({best_model})")
    ax.set_xlabel("Absolute error (um)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "abs_error_distribution_best_model_test.png")


def plot_feature_correlation(df: pd.DataFrame, figures_dir: Path) -> None:
    rows = []
    label = pd.to_numeric(df["vault_label"], errors="coerce")
    for column in FEATURE_COLUMNS:
        feature = pd.to_numeric(df[column], errors="coerce")
        rows.append({"feature": column, "correlation": float(feature.corr(label))})
    corr_df = pd.DataFrame(rows).sort_values("correlation")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(corr_df["feature"], corr_df["correlation"], color="#4C78A8")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Feature Correlation with POD1 Vault")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures_dir / "feature_correlation_with_vault.png")


def write_figures(summary_df: pd.DataFrame, results: Dict[str, Dict[str, Any]], predictions_dir: Path, figures_dir: Path, manifest_df: pd.DataFrame) -> None:
    plot_model_mae(summary_df, figures_dir)
    best_model = str(summary_df.sort_values("val_mae_um", kind="stable").iloc[0]["model_name"])
    best_test_pred = pd.read_csv(predictions_dir / f"{best_model}_test_predictions.csv")
    plot_pred_vs_gt(best_model, best_test_pred, figures_dir)
    plot_abs_error(best_model, best_test_pred, figures_dir)
    plot_feature_correlation(manifest_df, figures_dir)


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    out_dir = output_root(args.run_name)
    predictions_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures"

    df = pd.read_csv(manifest_path)
    validate_manifest(df)
    x_train, y_train, train_meta = split_xy(df, "train")
    x_val, y_val, val_meta = split_xy(df, "val")
    x_test, y_test, test_meta = split_xy(df, "test")

    results = select_and_evaluate_models(x_train, y_train, x_val, y_val, x_test, y_test, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(results, val_meta, y_val, test_meta, y_test, predictions_dir)
    summary_df = build_summary(results)
    summary_path = out_dir / "summary.csv"
    summary_md_path = out_dir / "summary.md"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    summary_md_path.write_text(markdown_table(summary_df), encoding="utf-8")
    feature_stats(df).to_csv(out_dir / "feature_stats.csv", index=False, encoding="utf-8")
    write_model_artifacts(results, out_dir)
    write_figures(summary_df, results, predictions_dir, figures_dir, df)

    best_val = summary_df.iloc[0]
    best_test = summary_df.sort_values("test_mae_um", kind="stable").iloc[0]
    print(f"Manifest path: {relative_path(manifest_path)}")
    print(f"Train/val/test samples: {len(train_meta)} / {len(val_meta)} / {len(test_meta)}")
    print(f"Feature columns: {', '.join(FEATURE_COLUMNS)}")
    for row in summary_df.to_dict(orient="records"):
        print(
            f"{row['model_name']}: "
            f"best val MAE={row['val_mae_um']:.2f} um, test MAE={row['test_mae_um']:.2f} um"
        )
    print(f"Best val model: {best_val['model_name']} ({best_val['val_mae_um']:.2f} um)")
    print(f"Best test model: {best_test['model_name']} ({best_test['test_mae_um']:.2f} um)")
    print(f"Summary CSV: {relative_path(summary_path)}")
    print(f"Summary Markdown: {relative_path(summary_md_path)}")
    print(f"Predictions directory: {relative_path(predictions_dir)}")
    print(f"Figures directory: {relative_path(figures_dir)}")
    print(f"Feature stats: {relative_path(out_dir / 'feature_stats.csv')}")


if __name__ == "__main__":
    main()
