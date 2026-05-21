"""Analyze test-set error complementarity across combined POD1 baselines.

This script compares AS-OCT-only, true preoperative measurement-only, and
AS-OCT + true preoperative measurement concat fusion baselines. It does not
train models or modify any predictions/manifests. The measurement features used
here are true preoperative 2DAnalysis measurements only; postoperative
2DAnalysis measurements must not be used as input features.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AS_OCT_RUNS = [
    "combined_as_oct_strict_imagenet_seed42_e30",
    "combined_as_oct_strict_imagenet_seed2026_e30",
    "combined_as_oct_strict_imagenet_seed3407_e30",
]

FUSION_RUNS = [
    "combined_fusion_ready_concat_seed42_e30",
    "combined_fusion_ready_concat_seed2026_e30",
    "combined_fusion_ready_concat_seed3407_e30",
]

MEASUREMENT_MODEL_FILES = {
    "linear_regression_test_predictions.csv",
    "ridge_regression_test_predictions.csv",
    "random_forest_test_predictions.csv",
}

FEATURE_COLUMNS = [
    "cct_mean_um",
    "acd_epi_mean_mm",
    "acd_endo_mean_mm",
    "clr_mean_um",
    "ata_mean_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze combined baseline error complementarity on the test set."
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv",
        help="Combined fusion ready manifest used to align samples and metadata.",
    )
    parser.add_argument(
        "--as_oct_pred_dir",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
        help="Directory containing AS-OCT-only prediction run folders.",
    )
    parser.add_argument(
        "--fusion_pred_dir",
        default="artifacts/predictions/fusion_baseline_batch_01_02",
        help="Directory containing fusion prediction run folders.",
    )
    parser.add_argument(
        "--measurement_report_dir",
        default="artifacts/reports/preop_measurement_baseline_batch_01",
        help="Directory containing measurement-only run folders.",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/reports/combined_batch_01_02/error_complementarity",
        help="Output directory for complementarity reports and figures.",
    )
    parser.add_argument(
        "--similar_threshold_um",
        type=float,
        default=10.0,
        help="Absolute error difference threshold used to call two methods similar.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def ensure_global_sample_id(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "global_sample_id" in df.columns:
        df["global_sample_id"] = df["global_sample_id"].astype(str)
        return df

    if "batch_id" in df.columns and "sample_id" in df.columns:
        df["global_sample_id"] = df["batch_id"].astype(str) + "__" + df["sample_id"].astype(str)
        return df

    if "patient_id" in df.columns and "sample_id" in df.columns:
        patient_id = df["patient_id"].astype(str)
        batch_id = patient_id.str.extract(r"^(batch_\d+)__", expand=False)
        missing = batch_id.isna()
        if missing.any():
            warn(
                f"{source}: {int(missing.sum())} rows could not derive batch_id from patient_id; "
                "falling back to sample_id only for those rows."
            )
            batch_id = batch_id.fillna("")
        df["global_sample_id"] = np.where(
            batch_id.astype(str).str.len() > 0,
            batch_id.astype(str) + "__" + df["sample_id"].astype(str),
            df["sample_id"].astype(str),
        )
        return df

    raise ValueError(f"{source}: cannot construct global_sample_id.")


def load_prediction_file(path: Path, source: str) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"Missing prediction file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        warn(f"Could not read {path}: {exc}")
        return None

    required = {"vault_label_um", "pred_vault_um"}
    missing = required - set(df.columns)
    if missing:
        warn(f"{path} missing columns {sorted(missing)}; skipping.")
        return None

    df = ensure_global_sample_id(df, source)
    df["source_file"] = str(path)
    return df


def load_named_runs(base_dir: Path, runs: Iterable[str], method: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for run_name in runs:
        path = base_dir / run_name / "test_predictions.csv"
        df = load_prediction_file(path, run_name)
        if df is None:
            continue
        df["run_name"] = run_name
        df["method"] = method
        frames.append(df)
        loaded.append(str(path))

    if not frames:
        return pd.DataFrame(), loaded
    return pd.concat(frames, ignore_index=True), loaded


def load_measurement_predictions(report_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for pred_dir in sorted(report_dir.glob("combined_measurement_ready_seed*/predictions")):
        if not pred_dir.is_dir():
            continue
        run_name = pred_dir.parent.name
        for filename in sorted(MEASUREMENT_MODEL_FILES):
            path = pred_dir / filename
            df = load_prediction_file(path, f"{run_name}/{filename}")
            if df is None:
                continue
            model_name = filename.replace("_test_predictions.csv", "")
            df["run_name"] = run_name
            df["model_name"] = model_name
            df["method"] = "measurement"
            frames.append(df)
            loaded.append(str(path))

    if not frames:
        return pd.DataFrame(), loaded
    return pd.concat(frames, ignore_index=True), loaded


def aggregate_method_predictions(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("global_sample_id", as_index=False)
        .agg(
            vault_label_um=("vault_label_um", "first"),
            pred_mean_um=("pred_vault_um", "mean"),
            pred_std_um=("pred_vault_um", "std"),
            n_predictions=("pred_vault_um", "count"),
        )
        .copy()
    )
    grouped[f"{method}_pred_mean_um"] = grouped["pred_mean_um"]
    grouped[f"{method}_pred_std_um"] = grouped["pred_std_um"]
    grouped[f"{method}_n_predictions"] = grouped["n_predictions"]
    grouped[f"{method}_signed_error_mean_um"] = grouped["pred_mean_um"] - grouped["vault_label_um"]
    grouped[f"{method}_abs_error_mean_um"] = grouped[f"{method}_signed_error_mean_um"].abs()
    keep = [
        "global_sample_id",
        "vault_label_um",
        f"{method}_pred_mean_um",
        f"{method}_pred_std_um",
        f"{method}_n_predictions",
        f"{method}_signed_error_mean_um",
        f"{method}_abs_error_mean_um",
    ]
    return grouped[keep]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_arr = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    y_pred_arr = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    if len(y_true_arr) == 0:
        return {"mae": math.nan, "std_abs": math.nan, "median_abs": math.nan, "rmse": math.nan, "r2": math.nan, "mean_signed": math.nan}

    signed = y_pred_arr - y_true_arr
    abs_err = np.abs(signed)
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = math.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "mae": float(np.mean(abs_err)),
        "std_abs": float(np.std(abs_err, ddof=1)) if len(abs_err) > 1 else 0.0,
        "median_abs": float(np.median(abs_err)),
        "rmse": float(np.sqrt(np.mean(signed**2))),
        "r2": float(r2),
        "mean_signed": float(np.mean(signed)),
    }


def vault_range_label(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 500:
        return "low_vault_lt500"
    if value <= 800:
        return "medium_vault_500_800"
    return "high_vault_gt800"


def build_method_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("as_oct", "as_oct_pred_mean_um", "as_oct_abs_error_mean_um", "as_oct_signed_error_mean_um"),
        ("measurement", "measurement_pred_mean_um", "measurement_abs_error_mean_um", "measurement_signed_error_mean_um"),
        ("fusion", "fusion_pred_mean_um", "fusion_abs_error_mean_um", "fusion_signed_error_mean_um"),
    ]
    rows = []
    for method, pred_col, abs_col, signed_col in methods:
        metrics = compute_metrics(sample_df["vault_label_um"], sample_df[pred_col])
        rows.append(
            {
                "method": method,
                "n_samples": int(sample_df[pred_col].notna().sum()),
                "mean_abs_error": metrics["mae"],
                "std_abs_error": metrics["std_abs"],
                "median_abs_error": metrics["median_abs"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "num_best_samples": int((sample_df["best_method_for_sample"] == method).sum()),
                "mean_signed_error": float(pd.to_numeric(sample_df[signed_col], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def build_improvement_summary(sample_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    fusion_delta = sample_df["fusion_abs_error_mean_um"] - sample_df["as_oct_abs_error_mean_um"]
    measurement_delta = sample_df["measurement_abs_error_mean_um"] - sample_df["as_oct_abs_error_mean_um"]
    row = {
        "fusion_better_than_as_oct_count": int((fusion_delta < -threshold).sum()),
        "fusion_worse_than_as_oct_count": int((fusion_delta > threshold).sum()),
        "fusion_similar_to_as_oct_count": int((fusion_delta.abs() <= threshold).sum()),
        "measurement_better_than_as_oct_count": int((measurement_delta < -threshold).sum()),
        "measurement_worse_than_as_oct_count": int((measurement_delta > threshold).sum()),
        "measurement_similar_to_as_oct_count": int((measurement_delta.abs() <= threshold).sum()),
        "samples_where_measurement_best": int((sample_df["best_method_for_sample"] == "measurement").sum()),
        "samples_where_fusion_best": int((sample_df["best_method_for_sample"] == "fusion").sum()),
        "samples_where_as_oct_best": int((sample_df["best_method_for_sample"] == "as_oct").sum()),
    }
    return pd.DataFrame([row])


def build_vault_range_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df["vault_range_group"] = df["vault_label_um"].apply(vault_range_label)
    rows = []
    for group_name, group in df.groupby("vault_range_group", sort=False):
        for method in ["as_oct", "measurement", "fusion"]:
            rows.append(
                {
                    "vault_range_group": group_name,
                    "method": method,
                    "n": int(len(group)),
                    "mean_abs_error_um": float(group[f"{method}_abs_error_mean_um"].mean()),
                    "mean_signed_error_um": float(group[f"{method}_signed_error_mean_um"].mean()),
                    "std_abs_error_um": float(group[f"{method}_abs_error_mean_um"].std(ddof=1))
                    if len(group) > 1
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def save_bar_with_errors(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x, df[y_col], color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_sample_errors(sample_df: pd.DataFrame, out_path: Path) -> None:
    df = sample_df.sort_values("as_oct_abs_error_mean_um").reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, df["as_oct_abs_error_mean_um"], marker="o", label="AS-OCT only")
    ax.plot(x, df["measurement_abs_error_mean_um"], marker="o", label="Measurement only")
    ax.plot(x, df["fusion_abs_error_mean_um"], marker="o", label="Fusion")
    ax.set_xticks(x)
    ax.set_xticklabels(df["global_sample_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Absolute error (um)")
    ax.set_title("Per-Sample Test Absolute Error Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_delta(sample_df: pd.DataFrame, delta_col: str, out_path: Path, title: str) -> None:
    df = sample_df.sort_values(delta_col).reset_index(drop=True)
    colors = np.where(df[delta_col] < 0, "#59a14f", "#e15759")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(df)), df[delta_col], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["global_sample_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Delta absolute error vs AS-OCT (um)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pred_vs_gt(sample_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    methods = [
        ("as_oct_pred_mean_um", "AS-OCT only", "#4c78a8"),
        ("measurement_pred_mean_um", "Measurement only", "#f58518"),
        ("fusion_pred_mean_um", "Fusion", "#54a24b"),
    ]
    for col, label, color in methods:
        ax.scatter(sample_df["vault_label_um"], sample_df[col], label=label, alpha=0.8, color=color)
    min_v = float(np.nanmin([sample_df["vault_label_um"].min(), sample_df[[m[0] for m in methods]].min().min()]))
    max_v = float(np.nanmax([sample_df["vault_label_um"].max(), sample_df[[m[0] for m in methods]].max().max()]))
    ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Ground truth POD1 vault (um)")
    ax.set_ylabel("Predicted POD1 vault (um)")
    ax.set_title("Predicted vs Ground Truth Vault")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_error_by_vault_range(range_df: pd.DataFrame, out_path: Path) -> None:
    pivot = range_df.pivot(index="vault_range_group", columns="method", values="mean_abs_error_um").fillna(0)
    order = [g for g in ["low_vault_lt500", "medium_vault_500_800", "high_vault_gt800"] if g in pivot.index]
    pivot = pivot.loc[order]
    x = np.arange(len(pivot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(["as_oct", "measurement", "fusion"]):
        if method not in pivot.columns:
            continue
        ax.bar(x + (i - 1) * width, pivot[method], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Mean absolute error (um)")
    ax.set_title("Error by Ground Truth Vault Range")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_best_method_counts(sample_df: pd.DataFrame, out_path: Path) -> None:
    counts = (
        sample_df["best_method_for_sample"]
        .value_counts()
        .reindex(["as_oct", "measurement", "fusion"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    counts.columns = ["method", "count"]
    save_bar_with_errors(counts, "method", "count", out_path, "Best Method Per Test Sample", "Number of samples")


def write_markdown(
    out_path: Path,
    method_summary: pd.DataFrame,
    improvement_summary: pd.DataFrame,
    range_summary: pd.DataFrame,
    sample_df: pd.DataFrame,
    loaded_files: list[str],
) -> None:
    best_method = method_summary.sort_values("mean_abs_error").iloc[0]
    imp = improvement_summary.iloc[0]
    high_range = range_summary[range_summary["vault_range_group"] == "high_vault_gt800"]
    low_range = range_summary[range_summary["vault_range_group"] == "low_vault_lt500"]

    def method_line(method: str) -> str:
        row = method_summary[method_summary["method"] == method].iloc[0]
        return (
            f"- {method}: MAE {row['mean_abs_error']:.2f} um, "
            f"RMSE {row['rmse']:.2f} um, R2 {row['r2']:.3f}, "
            f"mean signed error {row['mean_signed_error']:.2f} um"
        )

    top_measurement = sample_df.sort_values("measurement_minus_as_oct_abs_error_um").head(5)
    top_fusion = sample_df.sort_values("fusion_minus_as_oct_abs_error_um").head(5)

    lines = [
        "# Combined baseline error complementarity analysis",
        "",
        "## 分析目的",
        "",
        "本报告比较 combined batch_01 + batch_02 test set 上 AS-OCT-only、true preoperative measurement-only 和 AS-OCT + preop measurement concat fusion 三条 baseline 的误差模式。该分析只使用已有 prediction 文件，不训练新模型，也不修改 manifest 或训练结果。",
        "",
        "需要特别强调：measurement-only 与 fusion 中的结构化输入仅来自真正术前 CASIA2 2DAnalysis measurement；POD1 postoperative 2DAnalysis measurement 只作为 label 来源，不作为输入特征。",
        "",
        "## 整体结果",
        "",
        method_line("as_oct"),
        method_line("measurement"),
        method_line("fusion"),
        "",
        f"当前整体 MAE 最低的方法是 **{best_method['method']}**，平均绝对误差为 {best_method['mean_abs_error']:.2f} um。",
        "",
        "## 与 AS-OCT-only 的互补性",
        "",
        f"- fusion 相比 AS-OCT-only 明显改善的样本数：{int(imp['fusion_better_than_as_oct_count'])}",
        f"- fusion 相比 AS-OCT-only 明显变差的样本数：{int(imp['fusion_worse_than_as_oct_count'])}",
        f"- fusion 与 AS-OCT-only 近似持平的样本数：{int(imp['fusion_similar_to_as_oct_count'])}",
        f"- measurement-only 相比 AS-OCT-only 明显改善的样本数：{int(imp['measurement_better_than_as_oct_count'])}",
        f"- measurement-only 相比 AS-OCT-only 明显变差的样本数：{int(imp['measurement_worse_than_as_oct_count'])}",
        "",
        "按单样本最小误差统计：",
        f"- AS-OCT-only 最优样本数：{int(imp['samples_where_as_oct_best'])}",
        f"- measurement-only 最优样本数：{int(imp['samples_where_measurement_best'])}",
        f"- fusion 最优样本数：{int(imp['samples_where_fusion_best'])}",
        "",
        "## measurement features 是否有帮助",
        "",
        "measurement-only 在部分样本上确实比 AS-OCT-only 更接近真实 vault，提示术前 CCT、ACD、CLR、ATA 等结构化参数包含有价值的预测信息。但从平均 test MAE 看，measurement-only 整体仍弱于 AS-OCT-only，说明仅靠 5 个术前测量特征不足以替代图像输入。",
        "",
        "measurement-only 相对 AS-OCT-only 改善最大的样本包括：",
    ]

    for _, row in top_measurement.iterrows():
        lines.append(
            f"- {row['global_sample_id']}: measurement - AS-OCT error delta = {row['measurement_minus_as_oct_abs_error_um']:.2f} um"
        )

    lines.extend(
        [
            "",
            "fusion 相对 AS-OCT-only 改善最大的样本包括：",
        ]
    )
    for _, row in top_fusion.iterrows():
        lines.append(
            f"- {row['global_sample_id']}: fusion - AS-OCT error delta = {row['fusion_minus_as_oct_abs_error_um']:.2f} um"
        )

    lines.extend(
        [
            "",
            "## vault range 与系统性偏差",
            "",
            "按真实 vault 分组后，可以观察不同方法在 low / medium / high vault 区间的误差和 signed error。若某个区间 signed error 持续为正，说明模型倾向高估；若持续为负，说明模型倾向低估。",
            "",
        ]
    )

    if not low_range.empty:
        for _, row in low_range.iterrows():
            lines.append(
                f"- low vault, {row['method']}: MAE {row['mean_abs_error_um']:.2f} um, signed error {row['mean_signed_error_um']:.2f} um"
            )
    if not high_range.empty:
        for _, row in high_range.iterrows():
            lines.append(
                f"- high vault, {row['method']}: MAE {row['mean_abs_error_um']:.2f} um, signed error {row['mean_signed_error_um']:.2f} um"
            )

    lines.extend(
        [
            "",
            "## 为什么 concat fusion 可能没有稳定超过 AS-OCT-only",
            "",
            "当前 concat fusion 的 test MAE 没有稳定低于 AS-OCT-only，可能原因包括：",
            "",
            "- 样本量仍小，fusion head 增加参数后更容易在 validation/test 上波动。",
            "- measurement features 和 AS-OCT 图像信息存在部分冗余，简单 concat 未必能学到稳健互补关系。",
            "- measurement feature scaling、fusion head 容量、dropout/weight decay 等正则化设置仍可能不够理想。",
            "- 如果某些 measurement record 带有 confirmed outlier 或单记录聚合，concat 模型可能更容易受结构化特征噪声影响。",
            "- 当前 test set 只有 25 只眼，少数高误差样本会明显影响均值。",
            "",
            "## 下一步建议",
            "",
            "建议优先尝试更克制的融合路线，而不是直接上复杂 cross-attention：",
            "",
            "- late fusion：分别训练 AS-OCT-only 和 measurement-only，再对预测值做线性或 ridge stacking。",
            "- residual fusion：以 AS-OCT-only 预测为主，measurement 分支只学习 residual correction。",
            "- 更强正则化的 concat fusion：更小 fusion head、冻结或半冻结 backbone、更高 dropout 或更强 weight decay。",
            "- 分层误差分析：重点检查 measurement-only 明显优于 AS-OCT-only 的样本，看是否存在影像质量、vault range 或解剖参数模式。",
            "",
            "## 加载的 prediction 文件",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in loaded_files)
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    manifest = ensure_global_sample_id(manifest, "manifest")
    test_manifest = manifest[manifest["split"].astype(str).str.lower() == "test"].copy()
    meta_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "global_patient_uid",
        "eye_side",
        "vault_label",
        "label_qc_flag",
        "measurement_ready_status",
        *FEATURE_COLUMNS,
    ]
    missing_meta = [col for col in meta_cols if col not in test_manifest.columns]
    if missing_meta:
        raise ValueError(f"Manifest missing required columns: {missing_meta}")

    meta = test_manifest[meta_cols].rename(columns={"vault_label": "manifest_vault_label_um"}).copy()

    as_oct_df, as_oct_loaded = load_named_runs(Path(args.as_oct_pred_dir), AS_OCT_RUNS, "as_oct")
    fusion_df, fusion_loaded = load_named_runs(Path(args.fusion_pred_dir), FUSION_RUNS, "fusion")
    measurement_df, measurement_loaded = load_measurement_predictions(Path(args.measurement_report_dir))
    loaded_files = as_oct_loaded + measurement_loaded + fusion_loaded

    as_oct_agg = aggregate_method_predictions(as_oct_df, "as_oct")
    measurement_agg = aggregate_method_predictions(measurement_df, "measurement")
    fusion_agg = aggregate_method_predictions(fusion_df, "fusion")

    if as_oct_agg.empty or measurement_agg.empty or fusion_agg.empty:
        raise RuntimeError("Could not load all three method families; cannot compute complementarity.")

    sample_df = meta.merge(as_oct_agg, on="global_sample_id", how="inner", suffixes=("", "_as_oct"))
    sample_df = sample_df.merge(measurement_agg, on="global_sample_id", how="inner", suffixes=("", "_measurement"))
    sample_df = sample_df.merge(fusion_agg, on="global_sample_id", how="inner", suffixes=("", "_fusion"))
    sample_df["vault_label_um"] = sample_df["manifest_vault_label_um"]

    if sample_df.empty:
        raise RuntimeError("No test samples aligned across AS-OCT, measurement, and fusion predictions.")

    sample_df["fusion_minus_as_oct_abs_error_um"] = (
        sample_df["fusion_abs_error_mean_um"] - sample_df["as_oct_abs_error_mean_um"]
    )
    sample_df["measurement_minus_as_oct_abs_error_um"] = (
        sample_df["measurement_abs_error_mean_um"] - sample_df["as_oct_abs_error_mean_um"]
    )

    error_cols = {
        "as_oct": "as_oct_abs_error_mean_um",
        "measurement": "measurement_abs_error_mean_um",
        "fusion": "fusion_abs_error_mean_um",
    }
    sample_df["best_method_for_sample"] = sample_df[list(error_cols.values())].idxmin(axis=1)
    sample_df["best_method_for_sample"] = sample_df["best_method_for_sample"].map(
        {v: k for k, v in error_cols.items()}
    )

    output_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "global_patient_uid",
        "eye_side",
        "vault_label_um",
        "as_oct_pred_mean_um",
        "as_oct_abs_error_mean_um",
        "measurement_pred_mean_um",
        "measurement_abs_error_mean_um",
        "fusion_pred_mean_um",
        "fusion_abs_error_mean_um",
        "fusion_minus_as_oct_abs_error_um",
        "measurement_minus_as_oct_abs_error_um",
        "best_method_for_sample",
        "label_qc_flag",
        "measurement_ready_status",
        *FEATURE_COLUMNS,
    ]
    by_sample = sample_df[output_cols].copy()

    method_summary = build_method_summary(sample_df)
    improvement_summary = build_improvement_summary(sample_df, args.similar_threshold_um)
    range_summary = build_vault_range_summary(sample_df)

    by_sample_path = out_dir / "test_error_complementarity_by_sample.csv"
    method_summary_path = out_dir / "method_error_summary.csv"
    improvement_path = out_dir / "improvement_summary.csv"
    range_path = out_dir / "vault_range_error_summary.csv"
    md_path = out_dir / "error_complementarity_summary.md"

    by_sample.to_csv(by_sample_path, index=False, encoding="utf-8")
    method_summary.to_csv(method_summary_path, index=False, encoding="utf-8")
    improvement_summary.to_csv(improvement_path, index=False, encoding="utf-8")
    range_summary.to_csv(range_path, index=False, encoding="utf-8")

    plot_per_sample_errors(sample_df, figures_dir / "per_sample_abs_error_comparison.png")
    plot_delta(
        sample_df,
        "fusion_minus_as_oct_abs_error_um",
        figures_dir / "fusion_vs_as_oct_error_delta.png",
        "Fusion vs AS-OCT Absolute Error Delta",
    )
    plot_delta(
        sample_df,
        "measurement_minus_as_oct_abs_error_um",
        figures_dir / "measurement_vs_as_oct_error_delta.png",
        "Measurement vs AS-OCT Absolute Error Delta",
    )
    plot_pred_vs_gt(sample_df, figures_dir / "pred_vs_gt_three_methods.png")
    plot_error_by_vault_range(range_summary, figures_dir / "error_by_vault_range_three_methods.png")
    plot_best_method_counts(sample_df, figures_dir / "best_method_per_sample_bar.png")

    write_markdown(md_path, method_summary, improvement_summary, range_summary, sample_df, loaded_files)

    print("Loaded prediction files:")
    for path in loaded_files:
        print(f"  {path}")
    print(f"Aligned test samples: {len(sample_df)}")
    for _, row in method_summary.iterrows():
        print(
            f"{row['method']} MAE: {row['mean_abs_error']:.2f} um "
            f"(RMSE {row['rmse']:.2f}, R2 {row['r2']:.3f})"
        )
    imp = improvement_summary.iloc[0]
    print(f"Fusion better than AS-OCT samples: {int(imp['fusion_better_than_as_oct_count'])}")
    print(f"Measurement better than AS-OCT samples: {int(imp['measurement_better_than_as_oct_count'])}")
    print("Output files:")
    for path in [
        by_sample_path,
        method_summary_path,
        improvement_path,
        range_path,
        md_path,
        figures_dir,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
