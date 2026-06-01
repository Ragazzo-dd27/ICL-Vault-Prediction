"""Sensitivity analysis for excluding patient_052 top-error samples.

This script recomputes AS-OCT seed ensemble metrics from the existing error
table only. It does not modify manifests, predictions, checkpoints, or training
results. Excluding patient_052 is a diagnostic sensitivity analysis, not the
primary result.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AS-OCT ensemble metric sensitivity to excluding patient_052."
    )
    parser.add_argument(
        "--error_csv",
        default="artifacts/reports/combined_batch_01_02/as_oct_error_analysis/as_oct_ensemble_test_error_by_sample.csv",
        help="AS-OCT ensemble per-sample test error table.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/reports/combined_batch_01_02/as_oct_error_analysis/patient052_sensitivity",
        help="Output directory for sensitivity analysis.",
    )
    parser.add_argument("--patient_pattern", default="patient_052", help="Patient identifier pattern to exclude.")
    return parser.parse_args()


def metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = pd.to_numeric(df["vault_label_um"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(df["pred_ensemble_um"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    signed = y_pred - y_true
    abs_err = np.abs(signed)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n_samples": int(len(y_true)),
        "mae_um": float(np.mean(abs_err)),
        "rmse_um": float(np.sqrt(np.mean(signed**2))),
        "r2": math.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
        "mean_signed_error_um": float(np.mean(signed)),
        "median_abs_error_um": float(np.median(abs_err)),
    }


def contains_patient(row: pd.Series, pattern: str) -> bool:
    cols = ["global_patient_uid", "patient_uid", "global_sample_id", "sample_id"]
    pattern = pattern.lower()
    return any(pattern in str(row.get(col, "")).lower() for col in cols)


def build_summary(df: pd.DataFrame, excluded: pd.DataFrame, keep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, sub, excluded_samples in [
        ("original_test_set", df, ""),
        ("exclude_patient_052", keep, ";".join(excluded["global_sample_id"].astype(str))),
    ]:
        row = {"setting": setting, **metrics(sub), "excluded_samples": excluded_samples}
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_batch_impact(df: pd.DataFrame, keep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, sub in [("original_test_set", df), ("exclude_patient_052", keep)]:
        for batch_id, group in sub.groupby("batch_id", dropna=False):
            rows.append({"setting": source, "batch_id": batch_id, **metrics(group)})
    return pd.DataFrame(rows)


def plot_mae_before_after(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    labels = ["Original", "Exclude patient_052"]
    values = summary["mae_um"].to_numpy(dtype=float)
    colors = ["#4c78a8", "#b7b7b7"]
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for i, value in enumerate(values):
        ax.text(i, value + max(values) * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Test MAE (um)")
    ax.set_title("MAE Before and After Excluding patient_052")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pred_vs_gt(df: pd.DataFrame, excluded: pd.DataFrame, keep: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    panels = [
        ("Original test set", df, excluded),
        ("Exclude patient_052", keep, pd.DataFrame()),
    ]
    all_min = float(min(df["vault_label_um"].min(), df["pred_ensemble_um"].min()))
    all_max = float(max(df["vault_label_um"].max(), df["pred_ensemble_um"].max()))
    pad = (all_max - all_min) * 0.08
    lims = [all_min - pad, all_max + pad]
    for ax, (title, sub, highlight) in zip(axes, panels):
        ax.scatter(sub["vault_label_um"], sub["pred_ensemble_um"], s=28, color="#4c78a8", alpha=0.72, edgecolor="white", linewidth=0.3)
        if not highlight.empty:
            ax.scatter(highlight["vault_label_um"], highlight["pred_ensemble_um"], s=60, color="#b22222", alpha=0.9, edgecolor="black", linewidth=0.4, label="patient_052")
            ax.legend(frameon=False, loc="lower right")
        ax.plot(lims, lims, linestyle="--", color="#404040", linewidth=0.9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(title)
        ax.set_xlabel("Ground-truth vault (um)")
        ax.grid(color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Predicted vault (um)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_markdown(
    out_path: Path,
    summary: pd.DataFrame,
    batch_summary: pd.DataFrame,
    excluded: pd.DataFrame,
) -> None:
    original = summary[summary["setting"] == "original_test_set"].iloc[0]
    after = summary[summary["setting"] == "exclude_patient_052"].iloc[0]
    delta_mae = float(original["mae_um"] - after["mae_um"])
    batch02 = batch_summary[batch_summary["batch_id"].astype(str) == "batch_02"]

    lines = [
        "# patient_052 exclusion sensitivity analysis",
        "",
        "## Purpose",
        "",
        "This analysis evaluates how AS-OCT seed ensemble metrics change when the two patient_052 test samples are excluded. This is a sensitivity analysis only. It is not the primary result, and samples should not be excluded unless clinical review confirms a label, image, eye-side, date, or visit alignment problem.",
        "",
        "## Overall metrics",
        "",
        f"- Original test set: n={int(original['n_samples'])}, MAE={original['mae_um']:.2f} um, RMSE={original['rmse_um']:.2f} um, R2={original['r2']:.3f}.",
        f"- Excluding patient_052: n={int(after['n_samples'])}, MAE={after['mae_um']:.2f} um, RMSE={after['rmse_um']:.2f} um, R2={after['r2']:.3f}.",
        f"- MAE decrease after exclusion: {delta_mae:.2f} um.",
        "",
        "## Excluded samples",
        "",
    ]
    for _, row in excluded.iterrows():
        lines.append(
            f"- {row['global_sample_id']}: label={row['vault_label_um']:.1f} um, "
            f"prediction={row['pred_ensemble_um']:.1f} um, abs error={row['abs_error_um']:.1f} um."
        )

    lines.extend(
        [
            "",
            "## Batch-level impact",
            "",
        ]
    )
    for _, row in batch_summary.iterrows():
        lines.append(
            f"- {row['setting']}, {row['batch_id']}: n={int(row['n_samples'])}, "
            f"MAE={row['mae_um']:.2f} um, RMSE={row['rmse_um']:.2f} um, "
            f"mean signed error={row['mean_signed_error_um']:.2f} um."
        )

    if not batch02.empty:
        orig_b02 = batch02[batch02["setting"] == "original_test_set"]
        after_b02 = batch02[batch02["setting"] == "exclude_patient_052"]
        if not orig_b02.empty and not after_b02.empty:
            lines.append("")
            lines.append(
                f"For batch_02, MAE changed from {orig_b02.iloc[0]['mae_um']:.2f} um "
                f"to {after_b02.iloc[0]['mae_um']:.2f} um after excluding patient_052."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "patient_052 contributes substantially to the aggregate test MAE because both eyes are among the largest errors. However, high model error alone is not sufficient justification for exclusion. The primary result should retain patient_052 unless manual review confirms a label, image quality, eye-side, date, or visit alignment issue.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    error_csv = Path(args.error_csv)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not error_csv.exists():
        raise FileNotFoundError(f"Missing input error CSV: {error_csv}")

    df = pd.read_csv(error_csv)
    required = {"global_sample_id", "vault_label_um", "pred_ensemble_um", "abs_error_um"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input error CSV missing columns: {sorted(missing)}")

    exclude_mask = df.apply(lambda row: contains_patient(row, args.patient_pattern), axis=1)
    excluded = df[exclude_mask].copy()
    keep = df[~exclude_mask].copy()
    summary = build_summary(df, excluded, keep)
    batch_summary = summarize_batch_impact(df, keep)

    summary_path = out_dir / "patient052_exclusion_sensitivity_summary.csv"
    excluded_path = out_dir / "excluded_patient052_samples.csv"
    batch_summary_path = out_dir / "patient052_exclusion_batch_summary.csv"
    md_path = out_dir / "patient052_exclusion_sensitivity_summary.md"
    mae_fig = fig_dir / "mae_before_after_patient052_exclusion.png"
    scatter_fig = fig_dir / "pred_vs_gt_before_after_patient052_exclusion.png"

    summary.to_csv(summary_path, index=False, encoding="utf-8")
    keep_cols = [
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
        "vault_range",
        "oct_path",
        "label_qc_flag",
        "measurement_ready_status",
    ]
    excluded[[col for col in keep_cols if col in excluded.columns]].to_csv(excluded_path, index=False, encoding="utf-8")
    batch_summary.to_csv(batch_summary_path, index=False, encoding="utf-8")
    write_markdown(md_path, summary, batch_summary, excluded)
    plot_mae_before_after(summary, mae_fig)
    plot_pred_vs_gt(df, excluded, keep, scatter_fig)

    original = summary[summary["setting"] == "original_test_set"].iloc[0]
    after = summary[summary["setting"] == "exclude_patient_052"].iloc[0]
    print(
        f"Original: n={int(original['n_samples'])}, MAE={original['mae_um']:.2f}, "
        f"RMSE={original['rmse_um']:.2f}, R2={original['r2']:.3f}"
    )
    print("Excluded samples:")
    for _, row in excluded.iterrows():
        print(f"  {row['global_sample_id']} abs_error={row['abs_error_um']:.2f} um")
    print(
        f"After exclusion: n={int(after['n_samples'])}, MAE={after['mae_um']:.2f}, "
        f"RMSE={after['rmse_um']:.2f}, R2={after['r2']:.3f}"
    )
    print("Output files:")
    for path in [summary_path, excluded_path, batch_summary_path, md_path, mae_fig, scatter_fig]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
