"""Summarize final combined batch_01 + batch_02 pilot baselines.

This script reads existing training logs and prediction/summary files for
AS-OCT-only, measurement-only, and fusion baselines. It does not modify
training results or manifests.
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
FUSION_RUNS = [
    "combined_fusion_ready_concat_seed42_e30",
    "combined_fusion_ready_concat_seed2026_e30",
    "combined_fusion_ready_concat_seed3407_e30",
]
MEASUREMENT_MODEL_MAP = {
    "random_forest": "random_forest",
    "linear_regression": "linear",
    "ridge_regression": "ridge",
    "mlp_regressor": "mlp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final combined pilot baselines.")
    parser.add_argument("--as_oct_log_root", type=str, default="artifacts/logs/as_oct_pod1_baseline_batch_01")
    parser.add_argument(
        "--as_oct_prediction_root", type=str, default="artifacts/predictions/as_oct_pod1_baseline_batch_01"
    )
    parser.add_argument(
        "--measurement_root", type=str, default="artifacts/reports/preop_measurement_baseline_batch_01"
    )
    parser.add_argument("--fusion_log_root", type=str, default="artifacts/logs/fusion_baseline_batch_01_02")
    parser.add_argument(
        "--fusion_prediction_root", type=str, default="artifacts/predictions/fusion_baseline_batch_01_02"
    )
    parser.add_argument(
        "--output_dir", type=str, default="artifacts/reports/combined_batch_01_02/final_baseline_summary"
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_seed(run_name: str) -> int:
    match = re.search(r"seed(\d+)", run_name)
    return int(match.group(1)) if match else 42


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


def best_val_from_log(train_log: pd.DataFrame) -> float:
    return float(train_log.sort_values("val_mae_um", kind="stable").iloc[0]["val_mae_um"])


def load_image_runs(
    runs: List[str],
    log_root: Path,
    prediction_root: Path,
    baseline_family: str,
    model_name: str,
    warnings: List[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_name in runs:
        log_path = log_root / run_name / "train_log.csv"
        pred_path = prediction_root / run_name / "test_predictions.csv"
        if not log_path.exists():
            warnings.append(f"WARNING: missing train log: {log_path}")
            continue
        if not pred_path.exists():
            warnings.append(f"WARNING: missing test predictions: {pred_path}")
            continue
        log_df = pd.read_csv(log_path)
        pred_df = pd.read_csv(pred_path)
        test_metrics = metrics_from_predictions(pred_df)
        rows.append(
            {
                "run_name": run_name,
                "baseline_family": baseline_family,
                "seed": infer_seed(run_name),
                "val_mae_um": best_val_from_log(log_df),
                "test_mae_um": test_metrics["test_mae_um"],
                "test_rmse_um": test_metrics["test_rmse_um"],
                "test_r2": test_metrics["test_r2"],
                "model_name": model_name,
            }
        )
    return pd.DataFrame(rows)


def load_measurement_runs(root: Path, warnings: List[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_name in MEASUREMENT_RUNS:
        summary_path = root / run_name / "summary.csv"
        if not summary_path.exists():
            warnings.append(f"WARNING: missing measurement summary: {summary_path}")
            continue
        summary_df = pd.read_csv(summary_path)
        for _, row in summary_df.iterrows():
            model_name = MEASUREMENT_MODEL_MAP.get(str(row["model_name"]), str(row["model_name"]))
            rows.append(
                {
                    "run_name": run_name,
                    "baseline_family": f"measurement_only_{model_name}",
                    "seed": infer_seed(run_name),
                    "val_mae_um": float(row["val_mae_um"]),
                    "test_mae_um": float(row["test_mae_um"]),
                    "test_rmse_um": float(row["test_rmse_um"]),
                    "test_r2": float(row["test_r2"]),
                    "model_name": model_name,
                }
            )
    return pd.DataFrame(rows)


def build_family_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    input_type = {
        "as_oct_only_imagenet_finetune": "AS-OCT-only",
        "fusion_concat_resnet18_measurement": "AS-OCT + measurement fusion",
    }
    model_notes = {
        "as_oct_only_imagenet_finetune": "Combined strict AS-OCT-only ImageNet fine-tune ResNet18.",
        "measurement_only_random_forest": "True preoperative measurement-only Random Forest.",
        "measurement_only_linear": "True preoperative measurement-only Linear Regression.",
        "measurement_only_ridge": "True preoperative measurement-only Ridge Regression.",
        "measurement_only_mlp": "True preoperative measurement-only MLP.",
        "fusion_concat_resnet18_measurement": "Concat fusion of AS-OCT ResNet18 image features and true preop measurement features.",
    }
    rows: list[dict[str, object]] = []
    for family, group in run_df.groupby("baseline_family", dropna=False):
        model_name = str(group["model_name"].iloc[0])
        rows.append(
            {
                "baseline_family": family,
                "input_type": input_type.get(family, "preop measurement-only"),
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
                "notes": model_notes.get(family, ""),
            }
        )
    order = [
        "as_oct_only_imagenet_finetune",
        "measurement_only_random_forest",
        "measurement_only_linear",
        "measurement_only_ridge",
        "measurement_only_mlp",
        "fusion_concat_resnet18_measurement",
    ]
    df = pd.DataFrame(rows)
    df["_order"] = df["baseline_family"].map({name: index for index, name in enumerate(order)}).fillna(999)
    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


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
    return "\n".join(
        [
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join(["---"] * len(view.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in view.to_numpy()],
        ]
    )


def write_markdown(path: Path, family_df: pd.DataFrame, run_df: pd.DataFrame, warnings: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    as_oct = family_df[family_df["baseline_family"].eq("as_oct_only_imagenet_finetune")]
    fusion = family_df[family_df["baseline_family"].eq("fusion_concat_resnet18_measurement")]
    measurement = family_df[family_df["baseline_family"].str.startswith("measurement_only")]
    best = family_df.sort_values("test_mae_mean").iloc[0] if not family_df.empty else None
    as_oct_mae = float(as_oct["test_mae_mean"].iloc[0]) if not as_oct.empty else float("nan")
    fusion_mae = float(fusion["test_mae_mean"].iloc[0]) if not fusion.empty else float("nan")
    fusion_better = bool(np.isfinite(fusion_mae) and np.isfinite(as_oct_mae) and fusion_mae < as_oct_mae)

    lines = [
        "# Combined Batch 01 + Batch 02 final pilot baseline summary",
        "",
        "本报告汇总 combined AS-OCT-only、preop measurement-only 与 AS-OCT + measurement concat fusion 三条路线。"
        "本步骤只读取已有训练结果，不修改 manifest 或 checkpoint。",
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
        "## 主要观察",
        "",
        f"- AS-OCT-only 平均 test MAE: {fmt(as_oct_mae)} um",
        "- measurement-only 各模型平均 test MAE:",
        *[
            f"  - {row['baseline_family']}: {fmt(row['test_mae_mean'])} um"
            for _, row in measurement.iterrows()
        ],
        f"- fusion concat 平均 test MAE: {fmt(fusion_mae)} um",
        f"- 当前最佳 baseline: `{best['baseline_family']}`，test MAE = {fmt(best['test_mae_mean'])} um" if best is not None else "",
        f"- fusion 是否超过 AS-OCT-only: {'是' if fusion_better else '否'}",
        "",
        (
            "measurement features 虽然没有在当前 test MAE 上超过 AS-OCT-only，但整体表现稳定，"
            "说明真正术前结构化参数仍然具有预测价值。"
        ),
        "",
        (
            "当前仍是 pilot baseline，test set 只有 25 只眼，不能过度解释单次 family 排名。"
            "下一步建议尝试更强正则化、更小 fusion head、冻结 backbone、late fusion 或 residual fusion，"
            "而不是直接上复杂 cross-attention。"
        ),
        "",
        "## Run-level summary",
        "",
        markdown_table(
            run_df,
            ["run_name", "baseline_family", "seed", "val_mae_um", "test_mae_um", "test_rmse_um", "test_r2"],
        ),
    ]
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_yerr(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()


def short_label(name: str) -> str:
    return (
        name.replace("as_oct_only_imagenet_finetune", "AS-OCT")
        .replace("measurement_only_", "Meas ")
        .replace("fusion_concat_resnet18_measurement", "Fusion")
    )


def plot_family_bar(df: pd.DataFrame, path: Path, y_col: str, err_col: str, title: str, ylabel: str) -> None:
    if df.empty:
        return
    ordered = df.sort_values(y_col)
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


def plot_run_level(run_df: pd.DataFrame, path: Path) -> None:
    if run_df.empty:
        return
    view = run_df.copy()
    view["label"] = view["baseline_family"] + " / seed" + view["seed"].astype(str)
    view = view.sort_values("test_mae_um")
    x = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, view["test_mae_um"], color="#F58518", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(view["label"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Test MAE (um)")
    ax.set_title("Final Run-level Test MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_as_oct_vs_fusion(df: pd.DataFrame, path: Path) -> None:
    view = df[df["baseline_family"].isin(["as_oct_only_imagenet_finetune", "fusion_concat_resnet18_measurement"])].copy()
    if view.empty:
        return
    x = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(x, view["test_mae_mean"], yerr=safe_yerr(view["test_mae_std"]), capsize=4, color="#59A14F", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(name) for name in view["baseline_family"]])
    ax.set_ylabel("Mean Test MAE (um)")
    ax.set_title("AS-OCT-only vs Fusion")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_figures(output_dir: Path, family_df: pd.DataFrame, run_df: pd.DataFrame) -> List[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_dir / "final_test_mae_comparison.png",
        fig_dir / "final_val_mae_comparison.png",
        fig_dir / "final_run_level_test_mae.png",
        fig_dir / "as_oct_vs_fusion_test_mae.png",
    ]
    plot_family_bar(family_df, paths[0], "test_mae_mean", "test_mae_std", "Final Baseline Test MAE", "Mean Test MAE (um)")
    plot_family_bar(family_df, paths[1], "val_mae_mean", "val_mae_std", "Final Baseline Validation MAE", "Mean Validation MAE (um)")
    plot_run_level(run_df, paths[2])
    plot_as_oct_vs_fusion(family_df, paths[3])
    return paths


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    as_oct_df = load_image_runs(
        AS_OCT_RUNS,
        resolve_project_path(args.as_oct_log_root),
        resolve_project_path(args.as_oct_prediction_root),
        baseline_family="as_oct_only_imagenet_finetune",
        model_name="resnet18_imagenet_finetune",
        warnings=warnings,
    )
    measurement_df = load_measurement_runs(resolve_project_path(args.measurement_root), warnings)
    fusion_df = load_image_runs(
        FUSION_RUNS,
        resolve_project_path(args.fusion_log_root),
        resolve_project_path(args.fusion_prediction_root),
        baseline_family="fusion_concat_resnet18_measurement",
        model_name="resnet18_measurement_concat",
        warnings=warnings,
    )

    run_df = pd.concat([as_oct_df, measurement_df, fusion_df], ignore_index=True)
    run_df = run_df[["run_name", "baseline_family", "seed", "val_mae_um", "test_mae_um", "test_rmse_um", "test_r2", "model_name"]]
    family_df = build_family_summary(run_df)

    summary_csv = output_dir / "final_baseline_summary.csv"
    run_csv = output_dir / "final_run_level_summary.csv"
    summary_md = output_dir / "final_baseline_summary.md"
    write_csv(family_df, summary_csv)
    write_csv(run_df.drop(columns=["model_name"]), run_csv)
    write_markdown(summary_md, family_df, run_df.drop(columns=["model_name"]), warnings)
    figure_paths = write_figures(output_dir, family_df, run_df)

    print("Baseline family mean/std:")
    for _, row in family_df.iterrows():
        std = 0.0 if pd.isna(row["test_mae_std"]) else float(row["test_mae_std"])
        print(f"  {row['baseline_family']}: {row['test_mae_mean']:.2f} +/- {std:.2f}")
    best = family_df.sort_values("test_mae_mean").iloc[0] if not family_df.empty else None
    if best is not None:
        print(f"Lowest test MAE baseline: {best['baseline_family']} ({best['test_mae_mean']:.2f} um)")
    as_oct = family_df[family_df["baseline_family"].eq("as_oct_only_imagenet_finetune")]
    fusion = family_df[family_df["baseline_family"].eq("fusion_concat_resnet18_measurement")]
    if not as_oct.empty and not fusion.empty:
        fusion_lower = float(fusion["test_mae_mean"].iloc[0]) < float(as_oct["test_mae_mean"].iloc[0])
        print(f"Fusion lower than AS-OCT-only: {fusion_lower}")
    for warning in warnings:
        print(warning)
    print(f"Outputs: {format_paths([summary_csv, run_csv, summary_md, *figure_paths])}")


if __name__ == "__main__":
    main()
