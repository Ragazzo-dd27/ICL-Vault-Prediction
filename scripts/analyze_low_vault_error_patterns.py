"""Analyze low-vault overestimation patterns for the AS-OCT seed ensemble.

This script reads existing manifests and prediction/error tables only. It does
not train models and does not modify manifests, predictions, checkpoints, or
training results.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "cct_mean_um",
    "acd_epi_mean_mm",
    "acd_endo_mean_mm",
    "clr_mean_um",
    "ata_mean_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze low-vault AS-OCT ensemble error patterns.")
    parser.add_argument(
        "--as_oct_error_csv",
        default="artifacts/reports/combined_batch_01_02/as_oct_error_analysis/as_oct_ensemble_test_error_by_sample.csv",
    )
    parser.add_argument(
        "--as_oct_manifest",
        default="data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv",
    )
    parser.add_argument(
        "--fusion_manifest",
        default="data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv",
    )
    parser.add_argument(
        "--error_complementarity_csv",
        default="artifacts/reports/combined_batch_01_02/error_complementarity/test_error_complementarity_by_sample.csv",
    )
    parser.add_argument(
        "--late_fusion_csv",
        default="artifacts/reports/combined_batch_01_02/late_fusion_analysis/late_fusion_test_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/reports/combined_batch_01_02/low_vault_error_analysis",
    )
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    return parser.parse_args()


def vault_range(value: float, low_threshold: float, high_threshold: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < low_threshold:
        return "low"
    if value <= high_threshold:
        return "medium"
    return "high"


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if len(true) == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan, "mean_signed": math.nan, "median_abs": math.nan}
    signed = pred - true
    abs_err = np.abs(signed)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "n": int(len(true)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(signed**2))),
        "r2": math.nan if ss_tot == 0 else float(1 - ss_res / ss_tot),
        "mean_signed": float(np.mean(signed)),
        "median_abs": float(np.median(abs_err)),
    }


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def read_optional(path: Path, description: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: optional {description} not found: {path}")
        return None
    return pd.read_csv(path)


def build_range_distribution(manifest: pd.DataFrame, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    label_col = "vault_label" if "vault_label" in manifest.columns else "vault_label_um"
    df = manifest.copy()
    df["vault_label_um"] = pd.to_numeric(df[label_col], errors="coerce")
    df["vault_range"] = df["vault_label_um"].apply(lambda x: vault_range(x, low_threshold, high_threshold))
    total_by_split = df.groupby("split")["global_sample_id"].count().rename("split_total")
    rows = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        total = int(total_by_split.get(split, 0))
        for group in ["low", "medium", "high"]:
            count = int((sub["vault_range"] == group).sum())
            rows.append(
                {
                    "split": split,
                    "vault_range": group,
                    "n_samples": count,
                    "split_total": total,
                    "proportion": count / total if total else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_as_oct_error_by_range(error_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["low", "medium", "high"]:
        sub = error_df[error_df["vault_range"] == group]
        if sub.empty:
            continue
        m = metrics(sub["vault_label_um"], sub["pred_ensemble_um"])
        over = int((sub["signed_error_um"] > 0).sum())
        under = int((sub["signed_error_um"] < 0).sum())
        rows.append(
            {
                "vault_range": group,
                "n_samples": int(len(sub)),
                "mae_um": m["mae"],
                "rmse_um": m["rmse"],
                "mean_signed_error_um": m["mean_signed"],
                "median_abs_error_um": m["median_abs"],
                "overestimation_count": over,
                "underestimation_count": under,
                "overestimation_ratio": over / len(sub) if len(sub) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def add_measurement_metadata(low_df: pd.DataFrame, fusion_manifest: pd.DataFrame | None) -> pd.DataFrame:
    if fusion_manifest is None:
        return low_df
    cols = ["global_sample_id", "measurement_ready_status", *FEATURE_COLUMNS]
    cols = [col for col in cols if col in fusion_manifest.columns]
    return low_df.merge(fusion_manifest[cols].drop_duplicates("global_sample_id"), on="global_sample_id", how="left", suffixes=("", "_fusion"))


def build_method_comparison(
    low_ids: set[str],
    error_df: pd.DataFrame,
    complementarity: pd.DataFrame | None,
    late_fusion: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []

    def add_method(method: str, df: pd.DataFrame, pred_col: str, label_col: str = "vault_label_um") -> None:
        sub = df[df["global_sample_id"].isin(low_ids)].copy()
        if sub.empty or pred_col not in sub.columns:
            return
        m = metrics(sub[label_col], sub[pred_col])
        rows.append(
            {
                "method": method,
                "n_samples": m["n"],
                "mae_um": m["mae"],
                "rmse_um": m["rmse"],
                "r2": m["r2"],
                "mean_signed_error_um": m["mean_signed"],
                "median_abs_error_um": m["median_abs"],
            }
        )

    add_method("AS-OCT seed ensemble", error_df, "pred_ensemble_um")
    if complementarity is not None:
        add_method("measurement ensemble", complementarity, "measurement_pred_mean_um")
        add_method("concat fusion ensemble", complementarity, "fusion_pred_mean_um")
    if late_fusion is not None:
        add_method("weighted late fusion", late_fusion, "late_fusion_pred_um")
        add_method("three-way weighted fusion", late_fusion, "three_way_fusion_pred_um")
        add_method("residual correction", late_fusion, "residual_fusion_pred_um")
    return pd.DataFrame(rows)


def build_patient052_summary(error_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.DataFrame:
    patient_mask = error_df[["global_sample_id", "sample_id", "global_patient_uid", "patient_uid"]].astype(str).apply(
        lambda row: row.str.contains("patient_052", case=False, na=False).any(), axis=1
    )
    p052 = error_df[patient_mask].copy()
    low_no_p052 = low_df[~low_df["global_sample_id"].isin(set(p052["global_sample_id"]))].copy()
    all_no_p052 = error_df[~error_df["global_sample_id"].isin(set(p052["global_sample_id"]))].copy()
    low_metric = metrics(low_df["vault_label_um"], low_df["pred_ensemble_um"])
    low_no_metric = metrics(low_no_p052["vault_label_um"], low_no_p052["pred_ensemble_um"])
    all_metric = metrics(error_df["vault_label_um"], error_df["pred_ensemble_um"])
    all_no_metric = metrics(all_no_p052["vault_label_um"], all_no_p052["pred_ensemble_um"])
    max_low_error = float(low_df["abs_error_um"].max()) if not low_df.empty else math.nan
    rows = []
    for _, row in p052.iterrows():
        rows.append(
            {
                "global_sample_id": row["global_sample_id"],
                "eye_side": row.get("eye_side", ""),
                "vault_label_um": row["vault_label_um"],
                "pred_ensemble_um": row["pred_ensemble_um"],
                "signed_error_um": row["signed_error_um"],
                "abs_error_um": row["abs_error_um"],
                "vault_range": row["vault_range"],
                "is_low_vault_max_error": bool(row["abs_error_um"] == max_low_error),
                "low_vault_mae_with_patient052": low_metric["mae"],
                "low_vault_mae_without_patient052": low_no_metric["mae"],
                "overall_mae_with_patient052": all_metric["mae"],
                "overall_mae_without_patient052": all_no_metric["mae"],
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data available._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    cols = list(formatted.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def plot_distribution(dist: pd.DataFrame, out_path: Path) -> None:
    pivot = dist.pivot(index="split", columns="vault_range", values="n_samples").reindex(["train", "val", "test"]).fillna(0)
    pivot = pivot[[col for col in ["low", "medium", "high"] if col in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(5.2, 3.4), color=["#9ecae1", "#bdbdbd", "#fdae6b"], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of samples")
    ax.set_title("Vault Range Distribution by Split")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Vault range", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_signed_error(error_df: pd.DataFrame, out_path: Path) -> None:
    groups = [g for g in ["low", "medium", "high"] if g in set(error_df["vault_range"])]
    data = [error_df.loc[error_df["vault_range"] == g, "signed_error_um"].to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.boxplot(data, labels=groups, showfliers=False, patch_artist=True, boxprops={"facecolor": "#e5e5e5", "edgecolor": "#555555"}, medianprops={"color": "#111111"})
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.035, len(vals)), vals, s=26, color="#4c78a8", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="#404040", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Vault range")
    ax.set_ylabel("Signed error (pred - label, um)")
    ax.set_title("Signed Error by Vault Range")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_low_vault_bars(low_df: pd.DataFrame, out_path: Path) -> None:
    df = low_df.sort_values("vault_label_um").copy()
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, df["vault_label_um"], width=width, label="Ground truth", color="#bdbdbd", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, df["pred_ensemble_um"], width=width, label="AS-OCT ensemble", color="#4c78a8", edgecolor="black", linewidth=0.4)
    labels = [sid.replace("batch_01__", "b01__").replace("batch_02__", "b02__") for sid in df["global_sample_id"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    for i, sid in enumerate(df["global_sample_id"]):
        if "patient_052" in sid:
            ax.annotate("patient_052", xy=(i, df.iloc[i]["pred_ensemble_um"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color="#b22222")
    ax.set_ylabel("Vault (um)")
    ax.set_title("Low-Vault Test Samples: Label vs Prediction")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_method_comparison(method_df: pd.DataFrame, out_path: Path) -> None:
    if method_df.empty:
        return
    df = method_df.sort_values("mae_um")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(df))
    ax.barh(y, df["mae_um"], color="#bdbdbd", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Low-vault test MAE (um)")
    ax.set_title("Method Comparison on Low-Vault Test Samples")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    max_val = float(df["mae_um"].max())
    for yi, value in zip(y, df["mae_um"]):
        ax.text(value + max_val * 0.015, yi, f"{value:.1f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_summary(
    out_path: Path,
    dist: pd.DataFrame,
    error_by_range: pd.DataFrame,
    low_df: pd.DataFrame,
    method_df: pd.DataFrame,
    patient052_df: pd.DataFrame,
) -> None:
    low_row = error_by_range[error_by_range["vault_range"] == "low"]
    low_mae = float(low_row.iloc[0]["mae_um"]) if not low_row.empty else math.nan
    low_signed = float(low_row.iloc[0]["mean_signed_error_um"]) if not low_row.empty else math.nan
    p052_low_with = float(patient052_df["low_vault_mae_with_patient052"].iloc[0]) if not patient052_df.empty else math.nan
    p052_low_without = float(patient052_df["low_vault_mae_without_patient052"].iloc[0]) if not patient052_df.empty else math.nan
    p052_all_with = float(patient052_df["overall_mae_with_patient052"].iloc[0]) if not patient052_df.empty else math.nan
    p052_all_without = float(patient052_df["overall_mae_without_patient052"].iloc[0]) if not patient052_df.empty else math.nan

    lines = [
        "# Low-vault AS-OCT error pattern analysis",
        "",
        "## 分析目的",
        "",
        "本分析围绕 AS-OCT seed ensemble 在 low-vault test cases 上的 overestimation 问题展开。所有结果均基于已有 manifest 和 prediction/error tables，不训练新模型，不修改任何数据或训练结果。",
        "",
        "## Vault range distribution",
        "",
        md_table(dist),
        "",
        "## AS-OCT ensemble error by vault range",
        "",
        md_table(error_by_range),
        "",
        f"Low-vault test samples 的 AS-OCT ensemble MAE 为 {low_mae:.2f} um，mean signed error 为 {low_signed:.2f} um。正的 signed error 表示模型倾向高估，因此当前结果支持 low-vault overestimation 这一观察。",
        "",
        "## patient_052 impact",
        "",
        md_table(patient052_df),
        "",
        f"包含 patient_052 时 low-vault MAE 为 {p052_low_with:.2f} um；排除 patient_052 后 low-vault MAE 为 {p052_low_without:.2f} um。整体 test MAE 从 {p052_all_with:.2f} um 变为 {p052_all_without:.2f} um。医生已确认 patient_052 标签、图像、眼别、日期和 visit 对齐无误，因此 patient_052 必须保留为真实有效的模型失败病例。",
        "",
        "## Measurement / fusion on low vault",
        "",
        md_table(method_df),
        "",
        "如果 measurement 或 fusion 在 low-vault 子集上没有稳定优于 AS-OCT，则说明结构化术前参数虽然可能包含局部互补信息，但当前融合方式还没有可靠解决 low-vault overestimation。",
        "",
        "## Suggested paper wording",
        "",
        "A range-stratified error analysis suggested that low-vault cases had larger positive signed errors, indicating a tendency toward overestimation. The largest two errors came from patient_052 in batch_02; clinical review confirmed correct labels, AS-OCT images, eye laterality, dates, and visit alignment. Therefore, these samples were retained in the primary analysis and interpreted as valid model failure cases rather than data exclusions.",
        "",
        "## Next steps",
        "",
        "- 保留 patient_052，不直接删除 top-error samples；",
        "- 扩大 low-vault 样本量；",
        "- 做 vault-range-aware evaluation；",
        "- 探索 low-vault-sensitive training strategy，例如分层采样、loss reweighting 或 calibration；",
        "- 暂不盲目上复杂 fusion，需要先验证简单且稳定的 range-aware 改进。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    as_oct_error_csv = Path(args.as_oct_error_csv)
    as_oct_manifest = Path(args.as_oct_manifest)
    require_file(as_oct_error_csv, "AS-OCT error CSV")
    require_file(as_oct_manifest, "AS-OCT combined strict manifest")

    error_df = pd.read_csv(as_oct_error_csv)
    manifest = pd.read_csv(as_oct_manifest)
    fusion_manifest = read_optional(Path(args.fusion_manifest), "fusion ready manifest")
    complementarity = read_optional(Path(args.error_complementarity_csv), "error complementarity table")
    late_fusion = read_optional(Path(args.late_fusion_csv), "late fusion prediction table")

    if "vault_range" not in error_df.columns:
        error_df["vault_range"] = error_df["vault_label_um"].apply(lambda x: vault_range(x, args.low_threshold, args.high_threshold))
    dist = build_range_distribution(manifest, args.low_threshold, args.high_threshold)
    error_by_range = build_as_oct_error_by_range(error_df)
    low_df = error_df[error_df["vault_label_um"] < args.low_threshold].copy()
    low_df = add_measurement_metadata(low_df, fusion_manifest)
    seed_cols = [col for col in ["pred_seed42_um", "pred_seed2026_um", "pred_seed3407_um"] if col in low_df.columns]
    low_df["seed_pred_std_um"] = low_df[seed_cols].astype(float).std(axis=1, ddof=1) if seed_cols else np.nan
    low_df["whether_patient052"] = low_df[["global_sample_id", "sample_id", "global_patient_uid", "patient_uid"]].astype(str).apply(
        lambda row: row.str.contains("patient_052", case=False, na=False).any(), axis=1
    )

    low_ids = set(low_df["global_sample_id"].astype(str))
    method_df = build_method_comparison(low_ids, error_df, complementarity, late_fusion)
    patient052_df = build_patient052_summary(error_df, low_df)

    dist_path = out_dir / "vault_range_distribution_by_split.csv"
    error_range_path = out_dir / "as_oct_error_by_vault_range.csv"
    low_samples_path = out_dir / "low_vault_test_samples.csv"
    method_path = out_dir / "low_vault_method_comparison.csv"
    p052_path = out_dir / "patient052_low_vault_case_summary.csv"
    md_path = out_dir / "low_vault_error_analysis_summary.md"

    dist.to_csv(dist_path, index=False, encoding="utf-8")
    error_by_range.to_csv(error_range_path, index=False, encoding="utf-8")
    low_cols = [
        "global_sample_id",
        "sample_id",
        "batch_id",
        "global_patient_uid",
        "patient_uid",
        "eye_side",
        "vault_label_um",
        "pred_ensemble_um",
        "signed_error_um",
        "abs_error_um",
        "pred_seed42_um",
        "pred_seed2026_um",
        "pred_seed3407_um",
        "seed_pred_std_um",
        "label_qc_flag",
        "oct_path",
        *FEATURE_COLUMNS,
        "measurement_ready_status",
        "whether_patient052",
    ]
    low_df[[col for col in low_cols if col in low_df.columns]].to_csv(low_samples_path, index=False, encoding="utf-8")
    method_df.to_csv(method_path, index=False, encoding="utf-8")
    patient052_df.to_csv(p052_path, index=False, encoding="utf-8")

    plot_distribution(dist, fig_dir / "vault_range_distribution_by_split.png")
    plot_signed_error(error_df, fig_dir / "signed_error_by_vault_range.png")
    plot_low_vault_bars(low_df, fig_dir / "low_vault_predictions_bar.png")
    plot_method_comparison(method_df, fig_dir / "low_vault_method_comparison.png")
    write_summary(md_path, dist, error_by_range, low_df, method_df, patient052_df)

    print("Vault range distribution:")
    print(dist.pivot(index="split", columns="vault_range", values="n_samples").reindex(["train", "val", "test"]).fillna(0).to_string())
    low_row = error_by_range[error_by_range["vault_range"] == "low"].iloc[0]
    print(f"Low-vault AS-OCT MAE: {low_row['mae_um']:.2f} um")
    print(f"Low-vault AS-OCT mean signed error: {low_row['mean_signed_error_um']:.2f} um")
    print("patient_052 low-vault errors:")
    if patient052_df.empty:
        print("  no patient_052 rows found")
    else:
        for _, row in patient052_df.iterrows():
            print(f"  {row['global_sample_id']}: abs_error={row['abs_error_um']:.2f} um, signed_error={row['signed_error_um']:.2f} um")
    if not method_df.empty:
        print("Low-vault method comparison:")
        print(method_df[["method", "mae_um", "mean_signed_error_um"]].to_string(index=False))
    print("Output files:")
    for path in [dist_path, error_range_path, low_samples_path, method_path, p052_path, md_path, fig_dir]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
