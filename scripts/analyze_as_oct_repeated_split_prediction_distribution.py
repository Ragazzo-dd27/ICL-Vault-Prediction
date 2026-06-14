"""Analyze AS-OCT repeated split prediction distributions.

This diagnostic script reads already completed AS-OCT-only standard repeated
split predictions and checks whether predictions are compressed toward the
center of the vault distribution. It does not retrain models and does not modify
existing prediction, manifest, split, checkpoint, or paper files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    "artifacts/reports/combined_batch_01_02/repeated_patient_split_stability/"
    "as_oct_only_seed42_standard/as_oct_repeated_split_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    "artifacts/reports/combined_batch_01_02/repeated_patient_split_stability/"
    "as_oct_only_seed42_standard/diagnostics"
)
RANGE_ORDER = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prediction distributions for AS-OCT standard repeated split predictions."
    )
    parser.add_argument("--predictions_csv", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    parser.add_argument(
        "--narrow_ratio_threshold",
        type=float,
        default=0.80,
        help="Flag prediction range as narrow if pred_range / label_range is below this threshold.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def label_pred_columns(df: pd.DataFrame) -> tuple[str, str]:
    label_candidates = ["vault_label_um", "vault_label", "label", "y_true", "target"]
    pred_candidates = ["pred_vault_um", "pred_um", "prediction", "pred", "y_pred"]
    label_col = next((col for col in label_candidates if col in df.columns), None)
    pred_col = next((col for col in pred_candidates if col in df.columns), None)
    if label_col is None or pred_col is None:
        raise KeyError(f"Could not infer label/prediction columns from {list(df.columns)}")
    return label_col, pred_col


def assign_vault_range(values: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select([numeric < low_threshold, numeric <= high_threshold], ["low", "medium"], default="high"),
        index=values.index,
    )


def prepare_predictions(df: pd.DataFrame, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    out = df.copy()
    label_col, pred_col = label_pred_columns(out)
    out["vault_label_um"] = pd.to_numeric(out[label_col], errors="coerce")
    out["pred_vault_um"] = pd.to_numeric(out[pred_col], errors="coerce")
    out = out.dropna(subset=["vault_label_um", "pred_vault_um"]).copy()
    out["signed_error_um"] = out["pred_vault_um"] - out["vault_label_um"]
    out["abs_error_um"] = out["signed_error_um"].abs()
    if "vault_range" not in out.columns:
        out["vault_range"] = assign_vault_range(out["vault_label_um"], low_threshold, high_threshold)
    if "split_seed" not in out.columns:
        out["split_seed"] = -1
    if "run_name" not in out.columns:
        out["run_name"] = out["split_seed"].map(lambda seed: f"split_{seed}")
    if "split" not in out.columns:
        out["split"] = "test"
    return out


def distribution_by_split(df: pd.DataFrame, narrow_ratio_threshold: float) -> pd.DataFrame:
    rows = []
    for (split_seed, run_name, split), group in df.groupby(["split_seed", "run_name", "split"], dropna=False):
        label_range = float(group["vault_label_um"].max() - group["vault_label_um"].min())
        pred_range = float(group["pred_vault_um"].max() - group["pred_vault_um"].min())
        range_ratio = pred_range / label_range if label_range > 0 else np.nan
        rows.append(
            {
                "split_seed": split_seed,
                "run_name": run_name,
                "split": split,
                "n_samples": len(group),
                "label_min_um": float(group["vault_label_um"].min()),
                "label_max_um": float(group["vault_label_um"].max()),
                "label_mean_um": float(group["vault_label_um"].mean()),
                "label_std_um": float(group["vault_label_um"].std(ddof=1)) if len(group) > 1 else np.nan,
                "prediction_min_um": float(group["pred_vault_um"].min()),
                "prediction_max_um": float(group["pred_vault_um"].max()),
                "prediction_mean_um": float(group["pred_vault_um"].mean()),
                "prediction_std_um": float(group["pred_vault_um"].std(ddof=1)) if len(group) > 1 else np.nan,
                "label_range_um": label_range,
                "prediction_range_um": pred_range,
                "prediction_to_label_range_ratio": range_ratio,
                "prediction_range_narrower_than_label": bool(range_ratio < narrow_ratio_threshold)
                if np.isfinite(range_ratio)
                else False,
                "signed_error_mean_um": float(group["signed_error_um"].mean()),
                "signed_error_std_um": float(group["signed_error_um"].std(ddof=1)) if len(group) > 1 else np.nan,
                "mae_um": float(group["abs_error_um"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split_seed", "split"])


def distribution_by_vault_range(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_seed, run_name, vault_range), group in df.groupby(["split_seed", "run_name", "vault_range"], dropna=False):
        rows.append(
            {
                "split_seed": split_seed,
                "run_name": run_name,
                "vault_range": vault_range,
                "n": len(group),
                "label_mean_um": float(group["vault_label_um"].mean()),
                "prediction_mean_um": float(group["pred_vault_um"].mean()),
                "mean_signed_error_um": float(group["signed_error_um"].mean()),
                "mae_um": float(group["abs_error_um"].mean()),
                "overestimation_count": int((group["signed_error_um"] > 0).sum()),
                "underestimation_count": int((group["signed_error_um"] < 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["vault_range"] = pd.Categorical(out["vault_range"], categories=RANGE_ORDER, ordered=True)
    return out.sort_values(["split_seed", "vault_range"]).reset_index(drop=True)


def case_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "split_seed",
        "run_name",
        "global_sample_id",
        "sample_id",
        "patient_id",
        "eye_side",
        "vault_label_um",
        "pred_vault_um",
        "signed_error_um",
        "abs_error_um",
        "vault_range",
        "label_qc_flag",
        "oct_path",
    ]
    return [col for col in cols if col in df.columns]


def md_table(df: pd.DataFrame, columns: List[str] | None = None) -> List[str]:
    if columns is not None:
        df = df[columns]
    if df.empty:
        return ["_None_", ""]
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            text_df[col] = text_df[col].astype(object).where(text_df[col].notna(), "").astype(str)
    lines = ["| " + " | ".join(text_df.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(text_df.columns)) + " |")
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text_df.columns) + " |")
    lines.append("")
    return lines


def write_summary(
    path: Path,
    input_path: Path,
    split_dist: pd.DataFrame,
    range_dist: pd.DataFrame,
    high_under: pd.DataFrame,
    low_over: pd.DataFrame,
) -> None:
    overall_range_ratio = split_dist["prediction_to_label_range_ratio"].mean()
    narrow_count = int(split_dist["prediction_range_narrower_than_label"].sum())
    low_summary = range_dist[range_dist["vault_range"].astype(str) == "low"]
    high_summary = range_dist[range_dist["vault_range"].astype(str) == "high"]
    seed_mae = split_dist[["split_seed", "mae_um"]].sort_values("mae_um", ascending=False)
    seed3407 = split_dist[split_dist["split_seed"].astype(str).eq("3407")]
    range_counts_source = range_dist.copy()
    range_counts_source["vault_range"] = range_counts_source["vault_range"].astype(str)
    range_counts = range_counts_source.pivot(index="split_seed", columns="vault_range", values="n").reset_index()

    lines = [
        "# AS-OCT repeated split prediction distribution diagnostics",
        "",
        "本诊断基于已经完成的 AS-OCT-only standard repeated split predictions，不重新训练模型，不修改已有文件。",
        "",
        f"- 输入 prediction 文件: `{input_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- repeated split 数: {split_dist['split_seed'].nunique()}",
        f"- prediction range / label range 平均比例: {overall_range_ratio:.2f}",
        f"- prediction range 明显窄于 label range 的 split 数: {narrow_count}",
        "",
        "## Per-split Distribution",
        "",
    ]
    lines.extend(
        md_table(
            split_dist,
            [
                "split_seed",
                "n_samples",
                "label_min_um",
                "label_max_um",
                "prediction_min_um",
                "prediction_max_um",
                "prediction_to_label_range_ratio",
                "signed_error_mean_um",
                "mae_um",
            ],
        )
    )
    lines.extend(["## Vault Range Counts by Split", ""])
    lines.extend(md_table(range_counts))

    lines.extend(["## Vault Range Error Pattern", ""])
    range_mean = (
        range_dist.groupby("vault_range", observed=False)
        .agg(
            n_mean=("n", "mean"),
            mae_mean=("mae_um", "mean"),
            signed_mean=("mean_signed_error_um", "mean"),
            overestimation_mean=("overestimation_count", "mean"),
            underestimation_mean=("underestimation_count", "mean"),
        )
        .reset_index()
    )
    lines.extend(md_table(range_mean))

    low_signed = low_summary["mean_signed_error_um"].mean() if not low_summary.empty else np.nan
    high_signed = high_summary["mean_signed_error_um"].mean() if not high_summary.empty else np.nan
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- Low-vault mean signed error across repeated splits: {low_signed:.2f} um.",
            f"- High-vault mean signed error across repeated splits: {high_signed:.2f} um.",
        ]
    )
    if low_signed > 0:
        lines.append("- 结果提示存在 low-vault overestimation 倾向。")
    else:
        lines.append("- 当前 repeated split 结果未显示稳定的 low-vault overestimation。")
    if high_signed < 0:
        lines.append("- 结果提示存在 high-vault underestimation 倾向。")
    else:
        lines.append("- 当前 repeated split 结果未显示稳定的 high-vault underestimation。")
    if narrow_count > 0:
        lines.append("- 至少部分 split 的 prediction range 明显窄于 label range，支持 regression-to-the-mean 的诊断。")
    else:
        lines.append("- prediction range 未普遍明显窄于 label range，但 range-level signed error 仍需结合解读。")

    if not seed3407.empty:
        seed3407_mae = float(seed3407.iloc[0]["mae_um"])
        seed3407_counts = range_counts[range_counts["split_seed"].astype(str).eq("3407")]
        high_n = int(seed3407_counts.iloc[0].get("high", 0)) if not seed3407_counts.empty else 0
        lines.append(
            f"- seed3407 test MAE 为 {seed3407_mae:.2f} um，其 test set high-vault 样本数为 {high_n}；"
            "如果该 split 的 high-vault underestimation 较明显，可能解释其较高 MAE。"
        )
    lines.extend(
        [
            "",
            "## Highest-MAE Splits",
            "",
        ]
    )
    lines.extend(md_table(seed_mae.head(5)))
    lines.extend(
        [
            "## Case Tables",
            "",
            f"- high-vault underestimation cases: {len(high_under)} rows",
            f"- low-vault overestimation cases: {len(low_over)} rows",
            "",
            "完整明细见同目录 CSV。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.predictions_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    raw = pd.read_csv(input_path)
    pred = prepare_predictions(raw, args.low_threshold, args.high_threshold)

    split_dist = distribution_by_split(pred, args.narrow_ratio_threshold)
    range_dist = distribution_by_vault_range(pred)
    high_under = pred[(pred["vault_range"].eq("high")) & (pred["signed_error_um"] < 0)].copy()
    low_over = pred[(pred["vault_range"].eq("low")) & (pred["signed_error_um"] > 0)].copy()
    high_under = high_under.sort_values("signed_error_um", ascending=True)
    low_over = low_over.sort_values("signed_error_um", ascending=False)

    split_path = output_dir / "prediction_distribution_by_split.csv"
    range_path = output_dir / "prediction_distribution_by_vault_range.csv"
    high_path = output_dir / "high_vault_underestimation_cases.csv"
    low_path = output_dir / "low_vault_overestimation_cases.csv"
    summary_path = output_dir / "prediction_distribution_diagnostic_summary.md"

    split_dist.to_csv(split_path, index=False, encoding="utf-8")
    range_dist.to_csv(range_path, index=False, encoding="utf-8")
    high_under[case_columns(high_under)].to_csv(high_path, index=False, encoding="utf-8")
    low_over[case_columns(low_over)].to_csv(low_path, index=False, encoding="utf-8")
    write_summary(summary_path, input_path, split_dist, range_dist, high_under, low_over)

    print(f"Input predictions: {input_path}")
    print(f"Rows analyzed: {len(pred)}")
    print("Prediction distribution by split:")
    print(
        split_dist[
            [
                "split_seed",
                "n_samples",
                "label_range_um",
                "prediction_range_um",
                "prediction_to_label_range_ratio",
                "signed_error_mean_um",
                "mae_um",
            ]
        ].to_string(index=False)
    )
    print("\nVault range summary mean across splits:")
    print(
        range_dist.groupby("vault_range", observed=False)[["n", "mae_um", "mean_signed_error_um"]]
        .mean()
        .reset_index()
        .to_string(index=False)
    )
    print(f"\nHigh-vault underestimation cases: {len(high_under)}")
    print(f"Low-vault overestimation cases: {len(low_over)}")
    print("Output files:")
    for path in [split_path, range_path, high_path, low_path, summary_path]:
        print(path)


if __name__ == "__main__":
    main()
