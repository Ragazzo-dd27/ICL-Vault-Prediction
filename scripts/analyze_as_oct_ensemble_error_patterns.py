"""Analyze AS-OCT seed ensemble error patterns for the combined test set.

This script reads existing AS-OCT-only test predictions and manifests only. It
does not retrain models and does not modify manifests, predictions, checkpoints,
or training code.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_RUNS = {
    "seed42": "combined_as_oct_strict_imagenet_seed42_e30/test_predictions.csv",
    "seed2026": "combined_as_oct_strict_imagenet_seed2026_e30/test_predictions.csv",
    "seed3407": "combined_as_oct_strict_imagenet_seed3407_e30/test_predictions.csv",
}

LABEL_CANDIDATES = ["vault_label_um", "vault_label", "label", "y_true", "target"]
PRED_CANDIDATES = ["pred_vault_um", "pred_um", "prediction", "pred", "y_pred"]
MEASUREMENT_METADATA = [
    "measurement_ready_status",
    "cct_mean_um",
    "acd_epi_mean_mm",
    "acd_endo_mean_mm",
    "clr_mean_um",
    "ata_mean_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AS-OCT-only seed ensemble test error patterns."
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv",
        help="Combined AS-OCT strict manifest.",
    )
    parser.add_argument(
        "--pred_root",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
        help="Prediction root containing combined AS-OCT run directories.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/reports/combined_batch_01_02/as_oct_error_analysis",
        help="Output directory for error analysis files.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of top error samples to export.")
    parser.add_argument("--low_threshold", type=float, default=500.0, help="Low vault threshold in um.")
    parser.add_argument("--high_threshold", type=float, default=800.0, help="High vault threshold in um.")
    parser.add_argument(
        "--measurement_metadata",
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv",
        help="Optional fusion/measurement manifest for preop measurement metadata.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_first_column(df: pd.DataFrame, candidates: list[str], file_path: Path) -> str:
    lower_to_col = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_col:
            return lower_to_col[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates} in {file_path}")


def derive_global_sample_id(df: pd.DataFrame) -> pd.Series:
    if "global_sample_id" in df.columns:
        return df["global_sample_id"].astype(str)
    if "batch_id" in df.columns and "sample_id" in df.columns:
        return df["batch_id"].astype(str) + "__" + df["sample_id"].astype(str)
    if "patient_id" in df.columns and "sample_id" in df.columns:
        batch_id = df["patient_id"].astype(str).str.extract(r"^(batch_\d+)__", expand=False)
        return np.where(
            batch_id.notna(),
            batch_id.astype(str) + "__" + df["sample_id"].astype(str),
            df["sample_id"].astype(str),
        )
    raise ValueError("Cannot derive global_sample_id from prediction file.")


def find_prediction_files(pred_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for seed_name, rel_path in SEED_RUNS.items():
        path = pred_root / rel_path
        if path.exists():
            found[seed_name] = path

    missing = set(SEED_RUNS) - set(found)
    if missing:
        all_files = list(pred_root.rglob("test_predictions.csv"))
        for seed_name in sorted(missing):
            seed_num = seed_name.replace("seed", "")
            pattern = re.compile(rf"combined_as_oct.*strict.*imagenet.*seed{seed_num}", re.IGNORECASE)
            matches = [path for path in all_files if pattern.search(str(path))]
            if matches:
                found[seed_name] = sorted(matches)[0]
    missing = set(SEED_RUNS) - set(found)
    if missing:
        raise FileNotFoundError(f"Missing AS-OCT prediction files for: {sorted(missing)}")
    return found


def load_prediction(path: Path, seed_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = find_first_column(df, LABEL_CANDIDATES, path)
    pred_col = find_first_column(df, PRED_CANDIDATES, path)
    out = pd.DataFrame(
        {
            "global_sample_id": derive_global_sample_id(df),
            "sample_id": df["sample_id"].astype(str) if "sample_id" in df.columns else "",
            "vault_label_um": pd.to_numeric(df[label_col], errors="coerce"),
            f"pred_{seed_name}_um": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    for col in ["patient_id", "eye_side", "split", "label_qc_flag", "oct_path"]:
        if col in df.columns:
            out[col] = df[col]
    return out


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = math.nan if ss_tot == 0 else float(1 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def vault_range(value: float, low_threshold: float, high_threshold: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < low_threshold:
        return "low"
    if value <= high_threshold:
        return "medium"
    return "high"


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        m = metrics(sub["vault_label_um"], sub["pred_ensemble_um"])
        rows.append(
            {
                group_col: group,
                "n_samples": int(len(sub)),
                "mae_um": m["mae"],
                "rmse_um": m["rmse"],
                "mean_signed_error_um": float(sub["signed_error_um"].mean()),
                "median_abs_error_um": float(sub["abs_error_um"].median()),
                "std_abs_error_um": float(sub["abs_error_um"].std(ddof=1)) if len(sub) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_vault_range(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["low", "medium", "high"]
    for group in order:
        sub = df[df["vault_range"] == group]
        if sub.empty:
            continue
        m = metrics(sub["vault_label_um"], sub["pred_ensemble_um"])
        rows.append(
            {
                "vault_range": group,
                "n_samples": int(len(sub)),
                "mae_um": m["mae"],
                "rmse_um": m["rmse"],
                "mean_signed_error_um": float(sub["signed_error_um"].mean()),
                "median_abs_error_um": float(sub["abs_error_um"].median()),
                "std_abs_error_um": float(sub["abs_error_um"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "min_label_um": float(sub["vault_label_um"].min()),
                "max_label_um": float(sub["vault_label_um"].max()),
                "overestimation_count": int((sub["signed_error_um"] > 0).sum()),
                "underestimation_count": int((sub["signed_error_um"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_review_fields(top_df: pd.DataFrame) -> pd.DataFrame:
    top_df = top_df.copy()
    max_rank = max(1, len(top_df))
    top_df["review_priority"] = np.where(top_df["error_rank_desc"] <= min(5, max_rank), "high", "medium")

    focuses = []
    for _, row in top_df.iterrows():
        focus = ["high_abs_error"]
        if row["vault_range"] == "low" and row["signed_error_um"] > 0:
            focus.append("low_vault_overestimation")
        if row["vault_range"] == "high" and row["signed_error_um"] < 0:
            focus.append("high_vault_underestimation")
        if str(row.get("label_qc_flag", "")).lower() not in {"", "nan", "ok"}:
            focus.append("label_qc_check_needed")
        focus.append("image_quality_check_needed")
        focuses.append(";".join(focus))
    top_df["possible_review_focus"] = focuses
    top_df["manual_review_comment"] = ""
    return top_df


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def create_review_package(top_df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    package_dir = out_dir / "top_error_review_package"
    image_dir = package_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    warnings = []
    for _, row in top_df.iterrows():
        src = Path(str(row.get("oct_path", "")))
        copied = ""
        if src.exists():
            suffix = src.suffix if src.suffix else ".jpg"
            dst_name = (
                f"rank{int(row['error_rank_desc']):02d}_"
                f"abs{row['abs_error_um']:.0f}um_"
                f"{safe_filename(row['global_sample_id'])}{suffix}"
            )
            dst = image_dir / dst_name
            try:
                shutil.copy2(src, dst)
                copied = str(dst)
            except Exception as exc:  # pragma: no cover - defensive file copy guard
                warnings.append(f"copy failed for {src}: {exc}")
        else:
            warnings.append(f"missing OCT image: {src}")
        rows.append(
            {
                "rank": int(row["error_rank_desc"]),
                "global_sample_id": row["global_sample_id"],
                "vault_label_um": row["vault_label_um"],
                "pred_ensemble_um": row["pred_ensemble_um"],
                "abs_error_um": row["abs_error_um"],
                "source_image_path": row.get("oct_path", ""),
                "copied_image_path": copied,
            }
        )
    index = pd.DataFrame(rows)
    index.to_csv(package_dir / "top_error_review_index.csv", index=False, encoding="utf-8")
    return index, warnings


def plot_pred_vs_gt(df: pd.DataFrame, top_df: pd.DataFrame, out_path: Path, m: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.scatter(df["vault_label_um"], df["pred_ensemble_um"], s=26, color="#9fbad6", alpha=0.75, edgecolor="white", linewidth=0.3)
    top5 = top_df.head(5)
    ax.scatter(top5["vault_label_um"], top5["pred_ensemble_um"], s=48, color="#b22222", alpha=0.9, edgecolor="black", linewidth=0.4, label="Top 5 errors")
    for idx, (_, row) in enumerate(top5.iterrows(), start=1):
        ax.annotate(str(idx), (row["vault_label_um"], row["pred_ensemble_um"]), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#7f0000")

    min_v = float(min(df["vault_label_um"].min(), df["pred_ensemble_um"].min()))
    max_v = float(max(df["vault_label_um"].max(), df["pred_ensemble_um"].max()))
    pad = (max_v - min_v) * 0.08
    lims = [min_v - pad, max_v + pad]
    ax.plot(lims, lims, linestyle="--", color="#404040", linewidth=0.9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground-truth vault (um)")
    ax.set_ylabel("Predicted vault (um)")
    ax.set_title("AS-OCT Ensemble Prediction vs. Ground Truth")
    ax.grid(color="#d9d9d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.04,
        0.96,
        f"MAE = {m['mae']:.2f} um\nRMSE = {m['rmse']:.2f} um\nR2 = {m['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bfbfbf", "linewidth": 0.6},
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_abs_error_by_range(df: pd.DataFrame, out_path: Path) -> None:
    groups = [g for g in ["low", "medium", "high"] if g in set(df["vault_range"])]
    data = [df.loc[df["vault_range"] == g, "abs_error_um"].to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.boxplot(data, labels=groups, widths=0.5, showfliers=False, patch_artist=True, boxprops={"facecolor": "#d9e8f5", "edgecolor": "#4c78a8"}, medianprops={"color": "#1f4e79"})
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, color="#4c78a8", alpha=0.75, edgecolor="white", linewidth=0.25)
        ax.text(i, max(vals) + 8 if len(vals) else 0, f"n={len(vals)}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Vault range")
    ax.set_ylabel("Absolute error (um)")
    ax.set_title("Absolute Error by Vault Range")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_signed_error(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(df["vault_label_um"], df["signed_error_um"], s=28, color="#4c78a8", alpha=0.78, edgecolor="white", linewidth=0.3)
    ax.axhline(0, linestyle="--", color="#404040", linewidth=0.9)
    ax.set_xlabel("Ground-truth vault (um)")
    ax.set_ylabel("Signed error (pred - label, um)")
    ax.set_title("Signed Error vs. Ground Truth Vault")
    ax.grid(color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_top_errors(top_df: pd.DataFrame, out_path: Path) -> None:
    df = top_df.sort_values("abs_error_um", ascending=True).copy()
    labels = [str(x).replace("batch_01__", "b01__").replace("batch_02__", "b02__") for x in df["global_sample_id"]]
    fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.35 * len(df) + 0.8)))
    y = np.arange(len(df))
    ax.barh(y, df["abs_error_um"], color="#b7b7b7", edgecolor="black", linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Absolute error (um)")
    ax.set_title("Top AS-OCT Ensemble Error Samples")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_val = float(df["abs_error_um"].max())
    ax.set_xlim(0, max_val * 1.18)
    for yi, value in zip(y, df["abs_error_um"]):
        ax.text(value + max_val * 0.015, yi, f"{value:.1f}", va="center", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_markdown(
    out_path: Path,
    df: pd.DataFrame,
    top_df: pd.DataFrame,
    range_summary: pd.DataFrame,
    batch_summary: pd.DataFrame,
    label_qc_summary: pd.DataFrame,
    m: dict[str, float],
) -> None:
    def md_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "_No records._"
        formatted = table.copy()
        for col in formatted.columns:
            if pd.api.types.is_float_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        cols = list(formatted.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in formatted.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(lines)

    max_error = top_df["abs_error_um"].max()
    low = range_summary[range_summary["vault_range"] == "low"]
    high = range_summary[range_summary["vault_range"] == "high"]
    trend_lines = []
    if not low.empty:
        low_signed = float(low.iloc[0]["mean_signed_error_um"])
        trend_lines.append(f"- Low vault mean signed error: {low_signed:.2f} um.")
    if not high.empty:
        high_signed = float(high.iloc[0]["mean_signed_error_um"])
        trend_lines.append(f"- High vault mean signed error: {high_signed:.2f} um.")

    lines = [
        "# AS-OCT seed ensemble error pattern analysis",
        "",
        "## 总体结果",
        "",
        f"AS-OCT seed ensemble 在 combined strict test set 上共分析 {len(df)} 个样本。整体 MAE 为 {m['mae']:.2f} um，RMSE 为 {m['rmse']:.2f} um，R2 为 {m['r2']:.3f}。",
        "",
        f"Top-error 表导出了 {len(top_df)} 个样本，最大绝对误差为 {max_error:.2f} um。这些样本建议作为下一轮人工复查的优先对象，重点检查术前 AS-OCT 图像质量、输入图像是否匹配、POD1 vault label 是否可靠，以及是否存在极端 vault 区间的系统性偏差。",
        "",
        "## Vault range error",
        "",
        md_table(range_summary),
        "",
        "Signed error 中正值表示 overestimation，负值表示 underestimation。",
        *trend_lines,
        "",
        "## Batch difference",
        "",
        md_table(batch_summary),
        "",
        "当前 batch-level 差异应谨慎解释，因为 test set 总量仍然较小，少数高误差样本会显著影响均值。",
        "",
        "## Label QC and measurement metadata",
        "",
    ]
    if label_qc_summary.empty:
        lines.append("当前输出未发现可用于分组的 label_qc_flag 字段。")
    else:
        lines.append(md_table(label_qc_summary))
    lines.extend(
        [
            "",
            "如果高误差集中在特定 label_qc_flag 或 measurement_ready_status，需要优先复查该组样本的 label 和输入图像。当前分析不使用 postoperative 2DAnalysis measurement 作为输入特征。",
            "",
            "## Recommended top-error review samples",
            "",
        ]
    )
    for _, row in top_df.head(10).iterrows():
        lines.append(
            f"- rank {int(row['error_rank_desc']):02d}: {row['global_sample_id']}, "
            f"label {row['vault_label_um']:.1f} um, pred {row['pred_ensemble_um']:.1f} um, "
            f"abs error {row['abs_error_um']:.1f} um, focus: {row['possible_review_focus']}"
        )
    lines.extend(
        [
            "",
            "## Suggested wording for the paper",
            "",
            "The AS-OCT seed ensemble achieved the strongest overall performance in the combined pilot cohort. Error analysis suggested that large errors were concentrated in a small number of samples and should be reviewed for image quality, input-label alignment, and extreme vault ranges. Due to the limited test set size, range- and batch-specific trends should be interpreted as exploratory rather than definitive.",
            "",
        ]
    )
    # Avoid optional tabulate dependency failures by falling back if needed.
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    out_dir = Path(args.output_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pred_root = Path(args.pred_root)
    pred_files = find_prediction_files(pred_root)
    print("Prediction files loaded:")
    loaded_frames = []
    for seed_name, path in sorted(pred_files.items()):
        print(f"  {seed_name}: {path}")
        loaded_frames.append(load_prediction(path, seed_name))

    ensemble = loaded_frames[0]
    for frame in loaded_frames[1:]:
        ensemble = ensemble.merge(frame[["global_sample_id", "vault_label_um", f"pred_{frame.columns[3].split('_')[1]}_um"]] if False else frame, on="global_sample_id", how="inner", suffixes=("", "_dup"))
        for col in list(ensemble.columns):
            if col.endswith("_dup"):
                base = col[:-4]
                if base not in ensemble.columns:
                    ensemble[base] = ensemble[col]
                ensemble = ensemble.drop(columns=[col])

    pred_cols = ["pred_seed42_um", "pred_seed2026_um", "pred_seed3407_um"]
    missing_pred = [col for col in pred_cols if col not in ensemble.columns]
    if missing_pred:
        raise ValueError(f"Missing seed prediction columns after merge: {missing_pred}")

    manifest = pd.read_csv(args.manifest)
    if "global_sample_id" not in manifest.columns:
        manifest["global_sample_id"] = derive_global_sample_id(manifest)
    manifest_meta_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "patient_uid",
        "global_patient_uid",
        "eye_side",
        "split",
        "oct_path",
        "label_qc_flag",
    ]
    manifest_meta_cols = [col for col in manifest_meta_cols if col in manifest.columns]
    meta = manifest[manifest_meta_cols].drop_duplicates("global_sample_id")

    df = meta.merge(ensemble[["global_sample_id", "vault_label_um", *pred_cols]], on="global_sample_id", how="inner")

    measurement_path = Path(args.measurement_metadata)
    if measurement_path.exists():
        measurement = pd.read_csv(measurement_path)
        if "global_sample_id" not in measurement.columns:
            measurement["global_sample_id"] = derive_global_sample_id(measurement)
        meta_cols = ["global_sample_id", *[col for col in MEASUREMENT_METADATA if col in measurement.columns]]
        df = df.merge(measurement[meta_cols].drop_duplicates("global_sample_id"), on="global_sample_id", how="left")
    else:
        print(f"WARNING: optional measurement metadata file not found: {measurement_path}")

    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if df.empty:
        raise RuntimeError("No aligned test samples found.")

    df["pred_ensemble_um"] = df[pred_cols].mean(axis=1)
    df["signed_error_um"] = df["pred_ensemble_um"] - df["vault_label_um"]
    df["abs_error_um"] = df["signed_error_um"].abs()
    df["squared_error_um"] = df["signed_error_um"] ** 2
    df["vault_range"] = df["vault_label_um"].apply(lambda x: vault_range(x, args.low_threshold, args.high_threshold))
    df = df.sort_values("abs_error_um", ascending=False).reset_index(drop=True)
    df["error_rank_desc"] = np.arange(1, len(df) + 1)

    m = metrics(df["vault_label_um"], df["pred_ensemble_um"])
    top_df = add_review_fields(df.head(args.top_k))
    range_summary = summarize_vault_range(df)
    batch_summary = summarize_group(df, "batch_id")
    label_qc_summary = summarize_group(df, "label_qc_flag") if "label_qc_flag" in df.columns else pd.DataFrame()

    by_sample_path = out_dir / "as_oct_ensemble_test_error_by_sample.csv"
    top_path = out_dir / "as_oct_ensemble_top_error_samples.csv"
    range_path = out_dir / "vault_range_error_summary.csv"
    batch_path = out_dir / "batch_error_summary.csv"
    label_qc_path = out_dir / "label_qc_error_summary.csv"
    md_path = out_dir / "as_oct_error_analysis_summary.md"

    output_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "patient_uid",
        "global_patient_uid",
        "eye_side",
        "split",
        "vault_label_um",
        *pred_cols,
        "pred_ensemble_um",
        "signed_error_um",
        "abs_error_um",
        "squared_error_um",
        "vault_range",
        "error_rank_desc",
        "oct_path",
        "label_qc_flag",
        *[col for col in MEASUREMENT_METADATA if col in df.columns],
    ]
    output_cols = [col for col in output_cols if col in df.columns]

    df[output_cols].to_csv(by_sample_path, index=False, encoding="utf-8")
    top_df[[col for col in [*output_cols, "review_priority", "possible_review_focus", "manual_review_comment"] if col in top_df.columns]].to_csv(top_path, index=False, encoding="utf-8")
    range_summary.to_csv(range_path, index=False, encoding="utf-8")
    batch_summary.to_csv(batch_path, index=False, encoding="utf-8")
    label_qc_summary.to_csv(label_qc_path, index=False, encoding="utf-8")

    review_index, copy_warnings = create_review_package(top_df, out_dir)
    for warning in copy_warnings:
        print(f"WARNING: {warning}")

    plot_pred_vs_gt(df, top_df, figures_dir / "pred_vs_gt_with_top_errors_highlighted.png", m)
    plot_abs_error_by_range(df, figures_dir / "abs_error_by_vault_range.png")
    plot_signed_error(df, figures_dir / "signed_error_vs_ground_truth.png")
    plot_top_errors(top_df, figures_dir / "top_error_samples_bar.png")
    write_markdown(md_path, df, top_df, range_summary, batch_summary, label_qc_summary, m)

    print(f"Aligned test samples: {len(df)}")
    print(f"AS-OCT ensemble MAE/RMSE/R2: {m['mae']:.2f} / {m['rmse']:.2f} / {m['r2']:.3f}")
    print("Top error samples:")
    for _, row in top_df.head(args.top_k).iterrows():
        print(
            f"  rank {int(row['error_rank_desc']):02d}: {row['global_sample_id']} "
            f"abs_error={row['abs_error_um']:.2f} um"
        )
    print("Vault range summary:")
    print(range_summary.to_string(index=False))
    print("Batch summary:")
    print(batch_summary.to_string(index=False))
    print("Output files:")
    for path in [
        by_sample_path,
        top_path,
        range_path,
        batch_path,
        label_qc_path,
        md_path,
        out_dir / "top_error_review_package" / "top_error_review_index.csv",
        figures_dir,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
