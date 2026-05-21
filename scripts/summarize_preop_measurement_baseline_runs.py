"""Summarize preop measurement-only baseline runs.

This script reads existing preoperative measurement-only baseline summaries and
writes aggregate tables, figures, and a concise Chinese summary. It does not
modify run outputs, manifests, or training code.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize preop measurement-only baseline runs.")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="artifacts/reports/preop_measurement_baseline_batch_01",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/reports/preop_measurement_baseline_batch_01/summary",
    )
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


def warn(message: str) -> None:
    print(f"Warning: {message}")


def infer_dataset_type(run_name: str) -> str:
    if "strict" in run_name:
        return "strict"
    if "ready" in run_name:
        return "ready"
    return "unknown"


def infer_seed(run_name: str) -> int:
    match = SEED_PATTERN.search(run_name)
    return int(match.group("seed")) if match else 42


def scan_run_summaries(runs_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for summary_path in sorted(runs_dir.glob("*/summary.csv")):
        run_name = summary_path.parent.name
        if run_name == "summary":
            continue
        try:
            df = pd.read_csv(summary_path)
        except Exception as exc:
            warn(f"could not read {relative_path(summary_path)}: {exc}")
            continue
        df = df.copy()
        df.insert(0, "seed", infer_seed(run_name))
        df.insert(0, "dataset_type", infer_dataset_type(run_name))
        df.insert(0, "run_name", run_name)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_group_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    grouped = all_df.groupby(["dataset_type", "model_name"], dropna=False)
    rows: List[Dict[str, object]] = []
    for (dataset_type, model_name), group in grouped:
        rows.append(
            {
                "dataset_type": dataset_type,
                "model_name": model_name,
                "n_runs": int(len(group)),
                "val_mae_mean": float(group["val_mae_um"].mean()),
                "val_mae_std": float(group["val_mae_um"].std(ddof=1)),
                "test_mae_mean": float(group["test_mae_um"].mean()),
                "test_mae_std": float(group["test_mae_um"].std(ddof=1)),
                "test_rmse_mean": float(group["test_rmse_um"].mean()),
                "test_rmse_std": float(group["test_rmse_um"].std(ddof=1)),
                "test_r2_mean": float(group["test_r2"].mean()),
                "test_r2_std": float(group["test_r2"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_type", "test_mae_mean"], kind="stable")


def build_best_val_selected(all_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run_name, group in all_df.groupby("run_name", dropna=False):
        best = group.sort_values("val_mae_um", kind="stable").iloc[0]
        rows.append(
            {
                "run_name": run_name,
                "dataset_type": best["dataset_type"],
                "seed": int(best["seed"]),
                "selected_model_by_val": best["model_name"],
                "selected_val_mae_um": float(best["val_mae_um"]),
                "corresponding_test_mae_um": float(best["test_mae_um"]),
                "corresponding_test_rmse_um": float(best["test_rmse_um"]),
                "corresponding_test_r2": float(best["test_r2"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_type", "run_name"], kind="stable")


def fmt(value: object, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            digits = 3 if column.endswith("r2") or "_r2_" in column else 2
            display[column] = display[column].map(lambda value: fmt(value, digits=digits))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines) + "\n"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_test_mae(group_df: pd.DataFrame, dataset_type: str, path: Path) -> None:
    df = group_df[group_df["dataset_type"].eq(dataset_type)].sort_values("test_mae_mean", kind="stable")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(df["model_name"], df["test_mae_mean"], yerr=df["test_mae_std"], capsize=4, color="#4C78A8")
    ax.set_title(f"{dataset_type.title()} Manifest: Test MAE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Test MAE (um)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def plot_val_vs_test(group_df: pd.DataFrame, dataset_type: str, path: Path) -> None:
    df = group_df[group_df["dataset_type"].eq(dataset_type)].sort_values("val_mae_mean", kind="stable")
    if df.empty:
        return
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, df["val_mae_mean"], width=width, yerr=df["val_mae_std"], capsize=4, label="Val MAE", color="#4C78A8")
    ax.bar(x + width / 2, df["test_mae_mean"], width=width, yerr=df["test_mae_std"], capsize=4, label="Test MAE", color="#F58518")
    ax.set_title(f"{dataset_type.title()} Manifest: Validation vs Test MAE")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (um)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def plot_best_selected(best_df: pd.DataFrame, path: Path) -> None:
    df = best_df.sort_values(["dataset_type", "seed"], kind="stable")
    labels = [f"{row.dataset_type}\nseed {row.seed}\n{row.selected_model_by_val}" for row in df.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C78A8" if dataset == "ready" else "#F58518" for dataset in df["dataset_type"]]
    ax.bar(labels, df["corresponding_test_mae_um"], color=colors)
    ax.set_title("Test MAE of Validation-Selected Models")
    ax.set_xlabel("Run")
    ax.set_ylabel("Test MAE (um)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def write_summary_md(
    path: Path,
    all_df: pd.DataFrame,
    group_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    ready = group_df[group_df["dataset_type"].eq("ready")]
    strict = group_df[group_df["dataset_type"].eq("strict")]
    ready_runs = all_df[all_df["dataset_type"].eq("ready")]["run_name"].nunique()
    strict_runs = all_df[all_df["dataset_type"].eq("strict")]["run_name"].nunique()
    lines = [
        "# Preop measurement-only baseline summary",
        "",
        "本总结汇总 batch_01 preop measurement-only POD1 vault regression baseline。输入仅使用真正术前 CASIA2 2DAnalysis measurement features，不使用 AS-OCT image、UBM 或术后 2DAnalysis measurement。",
        "",
        "## 数据规模",
        "",
        "- ready manifest: 81 samples，train/val/test = 56/12/13；包含 measurement_ready 与 measurement_ready_with_confirmed_outlier。",
        "- strict manifest: 69 samples，train/val/test = 47/11/11；仅包含 measurement_ready。",
        f"- 当前汇总 ready runs: {ready_runs}；strict runs: {strict_runs}。",
        "",
        "## Ready manifest 各模型平均 test MAE",
        "",
    ]
    for row in ready.sort_values("test_mae_mean").to_dict(orient="records"):
        lines.append(
            f"- {row['model_name']}: test MAE = {fmt(row['test_mae_mean'])} ± {fmt(row['test_mae_std'])} um"
        )
    lines.extend(["", "## Strict manifest 各模型平均 test MAE", ""])
    for row in strict.sort_values("test_mae_mean").to_dict(orient="records"):
        lines.append(
            f"- {row['model_name']}: test MAE = {fmt(row['test_mae_mean'])} ± {fmt(row['test_mae_std'])} um"
        )
    lines.extend(
        [
            "",
            "## 观察与解释",
            "",
            "- MLP 在部分 run 中 test MAE 较低，但此前训练时出现 convergence warning，因此其结果需要谨慎解释，并建议后续增加收敛诊断或重复实验。",
            "- Linear/Ridge 模型结构简单，结果较稳定且可解释，适合作为 measurement-only baseline 的参考线。",
            "- Random Forest 在部分 run 中 validation MAE 较好，但 test MAE 明显偏高，提示存在 val-test gap，小样本下可能对验证集选择较敏感。",
            "- ready 与 strict 的 val/test 样本数量都较小，当前结果不能过度解释；尤其 test split 只有 13 或 11 只眼。",
            "- 下一步应与 AS-OCT-only baseline 进行统一汇总对比，评估图像特征与术前测量特征的互补性。",
            "",
            "## Best validation-selected models",
            "",
            markdown_table(best_df),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = resolve_project_path(args.runs_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"

    all_df = scan_run_summaries(runs_dir)
    if all_df.empty:
        raise SystemExit(f"No readable summary.csv files found under {runs_dir}")
    group_df = build_group_summary(all_df)
    best_df = build_best_val_selected(all_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "summary_all_runs.csv"
    group_path = output_dir / "group_summary_by_dataset_and_model.csv"
    best_path = output_dir / "best_val_selected_summary.csv"
    md_path = output_dir / "summary.md"
    all_df.to_csv(all_path, index=False, encoding="utf-8")
    group_df.to_csv(group_path, index=False, encoding="utf-8")
    best_df.to_csv(best_path, index=False, encoding="utf-8")
    write_summary_md(md_path, all_df, group_df, best_df)

    plot_model_test_mae(group_df, "ready", figures_dir / "ready_model_test_mae_bar.png")
    plot_model_test_mae(group_df, "strict", figures_dir / "strict_model_test_mae_bar.png")
    plot_val_vs_test(group_df, "ready", figures_dir / "ready_val_vs_test_mae_by_model.png")
    plot_val_vs_test(group_df, "strict", figures_dir / "strict_val_vs_test_mae_by_model.png")
    plot_best_selected(best_df, figures_dir / "best_val_selected_test_mae_by_run.png")

    run_count = all_df["run_name"].nunique()
    ready_count = all_df[all_df["dataset_type"].eq("ready")]["run_name"].nunique()
    strict_count = all_df[all_df["dataset_type"].eq("strict")]["run_name"].nunique()
    print(f"Scanned runs: {run_count}")
    print(f"Model result rows: {len(all_df)}")
    print(f"Ready runs: {ready_count}")
    print(f"Strict runs: {strict_count}")
    for dataset_type, group in group_df.groupby("dataset_type"):
        best = group.sort_values("test_mae_mean", kind="stable").iloc[0]
        print(
            f"{dataset_type} best mean test MAE model: "
            f"{best['model_name']} ({best['test_mae_mean']:.2f} um)"
        )
    print(f"All-run summary: {relative_path(all_path)}")
    print(f"Group summary: {relative_path(group_path)}")
    print(f"Best-val selected summary: {relative_path(best_path)}")
    print(f"Markdown summary: {relative_path(md_path)}")
    print(f"Figures directory: {relative_path(figures_dir)}")


if __name__ == "__main__":
    main()
