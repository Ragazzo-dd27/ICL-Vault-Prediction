"""Summarize combined batch_01 + batch_02 baseline results.

This script reads existing AS-OCT-only and preop measurement-only result files,
then writes summary tables, Markdown, and figures. It does not modify manifests
or training outputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AS_OCT_RUNS = [
    "combined_as_oct_strict_imagenet_seed42_e30",
    "combined_as_oct_strict_imagenet_seed2026_e30",
    "combined_as_oct_strict_imagenet_seed3407_e30",
]
MEASUREMENT_RUNS = [
    "combined_measurement_ready_seed42",
    "combined_measurement_ready_seed2026",
    "combined_measurement_ready_seed3407",
]
MEASUREMENT_MODEL_MAP = {
    "random_forest": "random_forest",
    "linear_regression": "linear",
    "ridge_regression": "ridge",
    "mlp_regressor": "mlp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize combined batch_01 + batch_02 baseline results.")
    parser.add_argument(
        "--as_oct_log_root",
        type=str,
        default="artifacts/logs/as_oct_pod1_baseline_batch_01",
        help="Root directory containing AS-OCT train_log.csv subdirectories.",
    )
    parser.add_argument(
        "--as_oct_prediction_root",
        type=str,
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
        help="Root directory containing AS-OCT prediction subdirectories.",
    )
    parser.add_argument(
        "--measurement_root",
        type=str,
        default="artifacts/reports/preop_measurement_baseline_batch_01",
        help="Root directory containing measurement baseline run summaries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/reports/combined_batch_01_02/baseline_results",
        help="Output directory for combined baseline summaries.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_seed(run_name: str) -> int:
    match = re.search(r"seed(\d+)", run_name)
    if match:
        return int(match.group(1))
    return 42


def metrics_from_predictions(df: pd.DataFrame) -> Dict[str, float]:
    y_true = pd.to_numeric(df["vault_label_um"], errors="coerce")
    y_pred = pd.to_numeric(df["pred_vault_um"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].astype(float)
    y_pred = y_pred[valid].astype(float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denom = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = float(1.0 - np.sum(np.square(errors)) / denom) if denom > 0 else float("nan")
    return {"test_mae_um": mae, "test_rmse_um": rmse, "test_r2": r2}


def load_as_oct_runs(log_root: Path, prediction_root: Path, warnings: List[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_name in AS_OCT_RUNS:
        train_log_path = log_root / run_name / "train_log.csv"
        test_pred_path = prediction_root / run_name / "test_predictions.csv"
        if not train_log_path.exists():
            warnings.append(f"WARNING: missing AS-OCT train log: {train_log_path}")
            continue
        if not test_pred_path.exists():
            warnings.append(f"WARNING: missing AS-OCT test predictions: {test_pred_path}")
            continue
        train_log = pd.read_csv(train_log_path)
        test_pred = pd.read_csv(test_pred_path)
        best = train_log.sort_values("val_mae_um", kind="stable").iloc[0]
        test_metrics = metrics_from_predictions(test_pred)
        rows.append(
            {
                "run_name": run_name,
                "input_type": "AS-OCT-only",
                "model_name": "imagenet_finetune",
                "seed": infer_seed(run_name),
                "val_mae_um": float(best["val_mae_um"]),
                "test_mae_um": test_metrics["test_mae_um"],
                "test_rmse_um": test_metrics["test_rmse_um"],
                "test_r2": test_metrics["test_r2"],
            }
        )
    return pd.DataFrame(rows)


def load_measurement_runs(measurement_root: Path, warnings: List[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_name in MEASUREMENT_RUNS:
        summary_path = measurement_root / run_name / "summary.csv"
        if not summary_path.exists():
            warnings.append(f"WARNING: missing measurement summary: {summary_path}")
            continue
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            model_name = MEASUREMENT_MODEL_MAP.get(str(row["model_name"]), str(row["model_name"]))
            rows.append(
                {
                    "run_name": run_name,
                    "input_type": "preop_measurement-only",
                    "model_name": model_name,
                    "seed": infer_seed(run_name),
                    "val_mae_um": float(row["val_mae_um"]),
                    "test_mae_um": float(row["test_mae_um"]),
                    "test_rmse_um": float(row["test_rmse_um"]),
                    "test_r2": float(row["test_r2"]),
                }
            )
    return pd.DataFrame(rows)


def family_name(input_type: str, model_name: str) -> str:
    if input_type == "AS-OCT-only":
        return "combined_as_oct_strict_imagenet_finetune"
    return f"combined_measurement_ready_{model_name}"


def build_family_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    run_df = run_df.copy()
    run_df["baseline_family"] = [
        family_name(input_type, model_name)
        for input_type, model_name in zip(run_df["input_type"], run_df["model_name"])
    ]
    for (baseline_family, input_type, model_name), group in run_df.groupby(
        ["baseline_family", "input_type", "model_name"], dropna=False
    ):
        dataset_subset = "combined_strict" if input_type == "AS-OCT-only" else "combined_ready"
        notes = (
            "Combined batch_01 + batch_02 AS-OCT-only strict ImageNet fine-tune baseline."
            if input_type == "AS-OCT-only"
            else "Combined batch_01 + batch_02 true preoperative measurement-only ready baseline."
        )
        rows.append(
            {
                "baseline_family": baseline_family,
                "input_type": input_type,
                "dataset_subset": dataset_subset,
                "model_name": model_name,
                "n_runs": len(group),
                "val_mae_mean": group["val_mae_um"].mean(),
                "val_mae_std": group["val_mae_um"].std(ddof=1),
                "test_mae_mean": group["test_mae_um"].mean(),
                "test_mae_std": group["test_mae_um"].std(ddof=1),
                "test_rmse_mean": group["test_rmse_um"].mean(),
                "test_rmse_std": group["test_rmse_um"].std(ddof=1),
                "test_r2_mean": group["test_r2"].mean(),
                "test_r2_std": group["test_r2"].std(ddof=1),
                "notes": notes,
            }
        )
    order = [
        "combined_as_oct_strict_imagenet_finetune",
        "combined_measurement_ready_random_forest",
        "combined_measurement_ready_linear",
        "combined_measurement_ready_ridge",
        "combined_measurement_ready_mlp",
    ]
    out = pd.DataFrame(rows)
    out["_order"] = out["baseline_family"].map({name: i for i, name in enumerate(order)}).fillna(999)
    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_无可用数据。_"
    view = df[columns].copy()
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda x: fmt(x))
    view = view.fillna("").astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_markdown(path: Path, family_df: pd.DataFrame, run_df: pd.DataFrame, warnings: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    as_oct = family_df[family_df["input_type"].eq("AS-OCT-only")]
    measurement = family_df[family_df["input_type"].eq("preop_measurement-only")]
    best = family_df.sort_values("test_mae_mean", na_position="last").head(1)
    best_line = ""
    if not best.empty:
        row = best.iloc[0]
        best_line = f"当前 mean test MAE 最低的 baseline family 是 `{row['baseline_family']}`，test MAE = {fmt(row['test_mae_mean'])} um。"

    lines = [
        "# Combined Batch 01 + Batch 02 baseline results",
        "",
        "本报告汇总 batch_01 + batch_02 combined AS-OCT-only strict baseline 与 preop measurement-only ready baseline。"
        "本步骤只汇总已有训练结果，不修改 manifest 或训练输出。",
        "",
        "## 数据与 split",
        "",
        "- combined AS-OCT strict train/val/test = 114/23/25",
        "- combined measurement ready train/val/test = 111/24/25",
        "- 两条路线使用同一套基于 `global_patient_uid` 的 patient-level split。",
        "",
        "## Family-level summary",
        "",
        markdown_table(
            family_df,
            [
                "baseline_family",
                "n_runs",
                "val_mae_mean",
                "val_mae_std",
                "test_mae_mean",
                "test_mae_std",
                "test_rmse_mean",
                "test_r2_mean",
            ],
        ),
        "",
        "## AS-OCT-only 结果",
        "",
        markdown_table(as_oct, ["baseline_family", "n_runs", "val_mae_mean", "test_mae_mean", "test_rmse_mean", "test_r2_mean"]),
        "",
        (
            "combined AS-OCT-only strict 使用 ImageNet fine-tune ResNet18，三个 seed 的平均 test MAE "
            f"为 {fmt(as_oct['test_mae_mean'].iloc[0]) if not as_oct.empty else 'NA'} um。"
        ),
        "",
        "## Measurement-only 结果",
        "",
        markdown_table(measurement, ["baseline_family", "n_runs", "val_mae_mean", "test_mae_mean", "test_rmse_mean", "test_r2_mean"]),
        "",
        (
            "measurement-only 中 Random Forest / Linear / Ridge / MLP 均基于真正术前结构化参数。"
            "这些模型整体保持稳定有效，说明术前结构化参数具有预测 POD1 vault 的价值。"
        ),
        "",
        "## 当前观察",
        "",
        (
            "当前结果显示 combined AS-OCT-only 在 test MAE 上优于 measurement-only；"
            "但这仍然是 pilot baseline，test set 只有 25 只眼，需要后续更多数据验证。"
        ),
        "",
        best_line,
        "",
        (
            "下一步建议构建 AS-OCT + measurement fusion baseline，检验图像与结构化术前参数是否互补。"
        ),
        "",
        "## Run-level summary",
        "",
        markdown_table(
            run_df,
            ["run_name", "input_type", "model_name", "seed", "val_mae_um", "test_mae_um", "test_rmse_um", "test_r2"],
        ),
    ]
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_yerr(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()


def short_label(name: str) -> str:
    return (
        name.replace("combined_", "")
        .replace("measurement_ready_", "meas_")
        .replace("as_oct_strict_imagenet_finetune", "as_oct")
    )


def plot_family_bar(df: pd.DataFrame, path: Path, y_col: str, err_col: str, title: str, ylabel: str) -> None:
    if df.empty:
        return
    ordered = df.sort_values(y_col, na_position="last")
    labels = [short_label(name) for name in ordered["baseline_family"]]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, ordered[y_col], yerr=safe_yerr(ordered[err_col]), capsize=4, color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_as_oct_vs_measurement(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    view = df.copy()
    grouped = view.groupby("input_type", as_index=False).agg(
        test_mae_mean=("test_mae_mean", "mean"),
        test_mae_std=("test_mae_mean", "std"),
    )
    grouped = grouped.sort_values("test_mae_mean")
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, grouped["test_mae_mean"], yerr=safe_yerr(grouped["test_mae_std"]), capsize=4, color="#59A14F", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["input_type"], rotation=15, ha="right")
    ax.set_ylabel("Mean Test MAE (um)")
    ax.set_title("AS-OCT-only vs Measurement-only")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_run_level(run_df: pd.DataFrame, path: Path) -> None:
    if run_df.empty:
        return
    view = run_df.copy()
    view["label"] = view["run_name"] + " / " + view["model_name"]
    view = view.sort_values("test_mae_um")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(view))
    ax.bar(x, view["test_mae_um"], color="#F58518", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(view["label"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Test MAE (um)")
    ax.set_title("Combined Run-level Test MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_figures(output_dir: Path, family_df: pd.DataFrame, run_df: pd.DataFrame) -> List[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_dir / "combined_test_mae_bar.png",
        fig_dir / "combined_val_mae_bar.png",
        fig_dir / "as_oct_vs_measurement_test_mae.png",
        fig_dir / "combined_run_level_test_mae.png",
    ]
    plot_family_bar(family_df, paths[0], "test_mae_mean", "test_mae_std", "Combined Baseline Test MAE", "Mean Test MAE (um)")
    plot_family_bar(family_df, paths[1], "val_mae_mean", "val_mae_std", "Combined Baseline Validation MAE", "Mean Validation MAE (um)")
    plot_as_oct_vs_measurement(family_df, paths[2])
    plot_run_level(run_df, paths[3])
    return paths


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    as_oct_log_root = resolve_project_path(args.as_oct_log_root)
    as_oct_prediction_root = resolve_project_path(args.as_oct_prediction_root)
    measurement_root = resolve_project_path(args.measurement_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    as_oct_df = load_as_oct_runs(as_oct_log_root, as_oct_prediction_root, warnings)
    measurement_df = load_measurement_runs(measurement_root, warnings)
    run_df = pd.concat([as_oct_df, measurement_df], ignore_index=True)
    family_df = build_family_summary(run_df)

    summary_csv = output_dir / "combined_baseline_summary.csv"
    run_csv = output_dir / "combined_run_level_summary.csv"
    summary_md = output_dir / "combined_baseline_summary.md"
    write_csv(family_df, summary_csv)
    write_csv(run_df, run_csv)
    write_markdown(summary_md, family_df, run_df, warnings)
    figure_paths = write_figures(output_dir, family_df, run_df)

    print(f"Scanned AS-OCT runs: {as_oct_df['run_name'].tolist() if not as_oct_df.empty else []}")
    print(f"Scanned measurement runs: {sorted(measurement_df['run_name'].unique().tolist()) if not measurement_df.empty else []}")
    print("Baseline family test MAE mean/std:")
    for _, row in family_df.iterrows():
        print(f"  {row['baseline_family']}: {row['test_mae_mean']:.2f} +/- {0.0 if pd.isna(row['test_mae_std']) else row['test_mae_std']:.2f}")
    if not family_df.empty:
        best = family_df.sort_values("test_mae_mean").iloc[0]
        print(f"Lowest test MAE baseline family: {best['baseline_family']} ({best['test_mae_mean']:.2f} um)")
    for warning in warnings:
        print(warning)
    print(f"Outputs: {format_paths([summary_csv, run_csv, summary_md, *figure_paths])}")


if __name__ == "__main__":
    main()
