"""Analyze AS-OCT-only POD1 baseline prediction errors.

This script is for reporting and experiment analysis only. It reads existing
manifests, summaries, and prediction CSVs, then writes error-analysis tables,
figures, and a concise Chinese report. It does not modify predictions,
checkpoints, or training artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOCUS_RUNS = [
    "as_oct_pod1_clean_resnet18_imagenet_e30",
    "as_oct_pod1_clean_resnet18_random_e30",
    "as_oct_pod1_clean_resnet18_imagenet_seed2026_e30",
    "as_oct_pod1_clean_resnet18_imagenet_seed3407_e30",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AS-OCT-only POD1 prediction errors.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/summary.csv",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/error_analysis",
    )
    parser.add_argument("--runs", nargs="*", default=FOCUS_RUNS)
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


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {relative_path(path)}")


def read_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return pd.read_csv(summary_path)


def load_test_error_details(summary_df: pd.DataFrame, predictions_dir: Path, runs: Iterable[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    family_map = dict(zip(summary_df["run_name"], summary_df.get("experiment_family", "")))
    for run_name in runs:
        pred_path = predictions_dir / run_name / "test_predictions.csv"
        if not pred_path.exists():
            warn(f"missing test prediction file for {run_name}, skipped: {relative_path(pred_path)}")
            continue
        df = pd.read_csv(pred_path)
        required = {"sample_id", "patient_id", "eye_side", "vault_label_um", "pred_vault_um", "abs_error_um", "label_qc_flag", "oct_path"}
        missing = sorted(required.difference(df.columns))
        if missing:
            warn(f"{run_name} prediction file missing columns {missing}, skipped")
            continue
        df = df.copy()
        df["run_name"] = run_name
        df["experiment_family"] = family_map.get(run_name, "")
        df["vault_label_um"] = pd.to_numeric(df["vault_label_um"], errors="coerce")
        df["pred_vault_um"] = pd.to_numeric(df["pred_vault_um"], errors="coerce")
        df["abs_error_um"] = pd.to_numeric(df["abs_error_um"], errors="coerce")
        df["signed_error_um"] = df["pred_vault_um"] - df["vault_label_um"]
        rows.append(
            df[
                [
                    "run_name",
                    "experiment_family",
                    "sample_id",
                    "patient_id",
                    "eye_side",
                    "vault_label_um",
                    "pred_vault_um",
                    "abs_error_um",
                    "signed_error_um",
                    "label_qc_flag",
                    "oct_path",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_by_sample(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sample_id, group in detail_df.groupby("sample_id", dropna=False):
        group = group.sort_values("abs_error_um", ascending=False, kind="stable")
        first = group.iloc[0]
        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": first["patient_id"],
                "eye_side": first["eye_side"],
                "vault_label_um": first["vault_label_um"],
                "mean_abs_error_um": float(group["abs_error_um"].mean()),
                "std_abs_error_um": float(group["abs_error_um"].std(ddof=1)),
                "mean_signed_error_um": float(group["signed_error_um"].mean()),
                "n_runs": int(len(group)),
                "worst_run_name": first["run_name"],
                "max_abs_error_um": float(first["abs_error_um"]),
                "oct_path": first["oct_path"],
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_error_um", ascending=False, kind="stable")


def vault_range(value: float) -> str:
    if value < 500:
        return "low vault (<500 um)"
    if value <= 800:
        return "medium vault (500-800 um)"
    return "high vault (>800 um)"


def summarize_by_vault_range(sample_summary_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_summary_df.copy()
    df["vault_range"] = df["vault_label_um"].map(vault_range)
    rows: List[Dict[str, object]] = []
    for group_name, group in df.groupby("vault_range", dropna=False):
        rows.append(
            {
                "vault_range": group_name,
                "n": int(len(group)),
                "mean_abs_error_um": float(group["mean_abs_error_um"].mean()),
                "mean_signed_error_um": float(group["mean_signed_error_um"].mean()),
                "std_abs_error_um": float(group["mean_abs_error_um"].std(ddof=1)),
            }
        )
    order = {
        "low vault (<500 um)": 0,
        "medium vault (500-800 um)": 1,
        "high vault (>800 um)": 2,
    }
    result = pd.DataFrame(rows)
    result["order"] = result["vault_range"].map(order)
    return result.sort_values("order", kind="stable").drop(columns=["order"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved table: {relative_path(path)}")


def plot_test_abs_error_by_sample(sample_summary_df: pd.DataFrame, output_path: Path) -> None:
    df = sample_summary_df.sort_values("mean_abs_error_um", ascending=False, kind="stable")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["sample_id"], df["mean_abs_error_um"], color="#4C78A8")
    ax.set_title("Mean Test Absolute Error by Sample")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Mean absolute error (um)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def plot_test_signed_error_by_sample(sample_summary_df: pd.DataFrame, output_path: Path) -> None:
    df = sample_summary_df.sort_values("mean_signed_error_um", kind="stable")
    colors = ["#D62728" if value > 0 else "#4C78A8" for value in df["mean_signed_error_um"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["sample_id"], df["mean_signed_error_um"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Mean Test Signed Error by Sample")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Mean signed error (um)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def plot_pred_vs_gt_labeled(pred_df: pd.DataFrame, run_name: str, output_path: Path) -> None:
    labels = pd.to_numeric(pred_df["vault_label_um"], errors="coerce")
    preds = pd.to_numeric(pred_df["pred_vault_um"], errors="coerce")
    valid = labels.notna() & preds.notna()
    labels = labels[valid]
    preds = preds[valid]
    sample_ids = pred_df.loc[valid, "sample_id"].astype(str)
    lower = min(labels.min(), preds.min()) - 40
    upper = max(labels.max(), preds.max()) + 40

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(labels, preds, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.6)
    for x, y, sample_id in zip(labels, preds, sample_ids):
        ax.annotate(sample_id.replace("_POD1", ""), (x, y), fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.plot([lower, upper], [lower, upper], color="#D62728", linestyle="--", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title("Predicted vs Ground Truth Vault with Sample Labels")
    ax.set_xlabel("Ground truth POD1 vault (um)")
    ax.set_ylabel("Predicted POD1 vault (um)")
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)


def plot_error_vs_ground_truth(detail_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in detail_df.groupby("experiment_family"):
        ax.scatter(group["vault_label_um"], group["signed_error_um"], alpha=0.7, label=family)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Signed Error vs Ground Truth Vault")
    ax.set_xlabel("Ground truth POD1 vault (um)")
    ax.set_ylabel("Signed error (pred - label, um)")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)


def plot_mean_error_by_vault_range(range_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range_df["vault_range"], range_df["mean_abs_error_um"], color="#4C78A8")
    ax.set_title("Mean Absolute Error by Ground Truth Vault Range")
    ax.set_xlabel("Ground truth vault range")
    ax.set_ylabel("Mean absolute error (um)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def bias_text(mean_signed_error: float) -> str:
    if mean_signed_error > 20:
        return "整体呈现一定高估趋势"
    if mean_signed_error < -20:
        return "整体呈现一定低估趋势"
    return "整体 signed error 接近 0，未见强烈单向偏差"


def write_report(
    detail_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    range_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    top = sample_summary_df.head(5)
    mean_signed_error = float(detail_df["signed_error_um"].mean())
    random_errors = set(
        sample_summary_df[
            sample_summary_df["worst_run_name"].astype(str).str.contains("random", regex=False)
        ]["sample_id"]
    )
    imagenet_errors = set(
        sample_summary_df[
            sample_summary_df["worst_run_name"].astype(str).str.contains("imagenet", regex=False)
        ]["sample_id"]
    )
    overlap_text = "有部分重叠" if random_errors & imagenet_errors else "在 worst-run 维度上重叠不明显"
    best_val_run = summary_df.sort_values("best_val_mae_um", kind="stable").iloc[0]["run_name"]

    lines = [
        "# AS-OCT-only POD1 baseline prediction error analysis",
        "",
        "本分析基于 clean strict manifest 的 test split prediction 文件，重点比较 random initialization 与 ImageNet fine-tuning runs 的误差模式。该分析仅用于组会汇报和后续论文实验分析，不用于训练。",
        "",
        "## 当前测试集最大误差样本",
        "",
    ]
    for row in top.to_dict(orient="records"):
        lines.append(
            f"- `{row['sample_id']}`: mean abs error = {row['mean_abs_error_um']:.2f} um, "
            f"mean signed error = {row['mean_signed_error_um']:.2f} um, worst run = `{row['worst_run_name']}`."
        )
    lines.extend(
        [
            "",
            "## 系统性高估/低估",
            "",
            f"- 所有纳入 run 的平均 signed error 为 {mean_signed_error:.2f} um，{bias_text(mean_signed_error)}。",
            "- 仍需结合单个样本图像质量、真实 vault 区间和训练 seed 进一步判断偏差来源。",
            "",
            "## 不同 vault 区间的误差",
            "",
        ]
    )
    for row in range_df.to_dict(orient="records"):
        lines.append(
            f"- {row['vault_range']}: n={row['n']}, mean abs error={row['mean_abs_error_um']:.2f} um, "
            f"mean signed error={row['mean_signed_error_um']:.2f} um."
        )
    lines.extend(
        [
            "",
            "## random init 与 ImageNet fine-tune 的错误模式",
            "",
            f"- 当前 worst-run 样本集合中，random init 与 ImageNet fine-tune 的错误样本{overlap_text}。",
            "- ImageNet fine-tune 在 validation MAE 上更稳定，但 test set 较小，个别 test 样本仍可能改变不同 run 的排序。",
            f"- pred-vs-gt 标注图默认使用 best validation run `{best_val_run}`。",
            "",
            "## 对下一步实验的启示",
            "",
            "- 优先检查 mean abs error 最高的样本，确认图像质量、术前路径、POD1 label 和 scan consistency。",
            "- 对高 vault 与低 vault 样本分别评估误差，避免模型只学习到中间区间。",
            "- 后续可加入术前 2DAnalysis measurements 和 clinical features，观察是否能降低系统性误差。",
            "- 当前 test split 只有 13 只眼，所有结论应作为 pilot error analysis，而非最终临床结论。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {relative_path(output_path)}")


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    summary_path = resolve_project_path(args.summary)
    predictions_dir = resolve_project_path(args.predictions_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"

    summary_df = pd.read_csv(summary_path)
    if manifest_path.exists():
        pd.read_csv(manifest_path)

    detail_df = load_test_error_details(summary_df=summary_df, predictions_dir=predictions_dir, runs=args.runs)
    if detail_df.empty:
        raise SystemExit("No prediction files were available for error analysis.")

    sample_summary_df = summarize_by_sample(detail_df)
    top_error_df = sample_summary_df.head(10).copy()
    range_df = summarize_by_vault_range(sample_summary_df)

    detail_path = output_dir / "test_error_detail_all_runs.csv"
    sample_summary_path = output_dir / "test_error_summary_by_sample.csv"
    top_error_path = output_dir / "top_error_samples.csv"
    range_path = output_dir / "bias_summary_by_vault_range.csv"
    report_path = output_dir / "error_analysis.md"
    write_csv(detail_df, detail_path)
    write_csv(sample_summary_df, sample_summary_path)
    write_csv(top_error_df, top_error_path)
    write_csv(range_df, range_path)

    plot_test_abs_error_by_sample(sample_summary_df, figures_dir / "test_abs_error_by_sample.png")
    plot_test_signed_error_by_sample(sample_summary_df, figures_dir / "test_signed_error_by_sample.png")
    best_val_run = str(summary_df[summary_df["run_name"].isin(args.runs)].sort_values("best_val_mae_um", kind="stable").iloc[0]["run_name"])
    best_val_pred = pd.read_csv(predictions_dir / best_val_run / "test_predictions.csv")
    plot_pred_vs_gt_labeled(best_val_pred, best_val_run, figures_dir / "pred_vs_gt_with_sample_labels_best_val_run.png")
    plot_error_vs_ground_truth(detail_df, figures_dir / "error_vs_ground_truth_vault.png")
    plot_mean_error_by_vault_range(range_df, figures_dir / "mean_error_by_vault_range.png")
    write_report(detail_df, sample_summary_df, range_df, summary_df[summary_df["run_name"].isin(args.runs)], report_path)

    analyzed_runs = detail_df["run_name"].nunique()
    test_samples = sample_summary_df["sample_id"].nunique()
    mean_signed_error = float(detail_df["signed_error_um"].mean())
    print(f"Analyzed runs: {analyzed_runs}")
    print(f"Test samples: {test_samples}")
    print("Top 5 mean absolute error samples:")
    for row in sample_summary_df.head(5).to_dict(orient="records"):
        print(f"  {row['sample_id']}: {row['mean_abs_error_um']:.2f} um")
    print(f"Bias trend: {bias_text(mean_signed_error)} (mean signed error={mean_signed_error:.2f} um)")
    print(f"Output directory: {relative_path(output_dir)}")


if __name__ == "__main__":
    main()
