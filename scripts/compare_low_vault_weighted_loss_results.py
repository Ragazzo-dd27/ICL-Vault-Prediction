"""Compare low-vault weighted-loss AS-OCT pilot runs.

This script reads existing predictions and ensemble error tables only. It does
not modify manifests, predictions, checkpoints, or training results, and it does
not train new models.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_CANDIDATES = ["vault_label_um", "vault_label", "label", "y_true", "target"]
PRED_CANDIDATES = ["pred_vault_um", "pred_um", "prediction", "pred", "y_pred"]

METHODS = [
    {
        "method": "original_seed42",
        "pred_col": "original_seed42_pred_um",
        "loss_weight_mode": "none",
        "low_weight": 1.0,
        "medium_weight": 1.0,
        "high_weight": 1.0,
    },
    {
        "method": "low_weight_1p5_seed42",
        "pred_col": "loww1p5_seed42_pred_um",
        "loss_weight_mode": "vault_range_low",
        "low_weight": 1.5,
        "medium_weight": 1.0,
        "high_weight": 1.0,
    },
    {
        "method": "low_weight_2p0_seed42",
        "pred_col": "loww2_seed42_pred_um",
        "loss_weight_mode": "vault_range_low",
        "low_weight": 2.0,
        "medium_weight": 1.0,
        "high_weight": 1.0,
    },
    {
        "method": "extreme_weight_1p5_seed42",
        "pred_col": "extremew1p5_seed42_pred_um",
        "loss_weight_mode": "vault_range_extreme",
        "low_weight": 1.5,
        "medium_weight": 1.0,
        "high_weight": 1.5,
    },
    {
        "method": "as_oct_seed_ensemble",
        "pred_col": "ensemble_pred_um",
        "loss_weight_mode": "seed_ensemble",
        "low_weight": math.nan,
        "medium_weight": math.nan,
        "high_weight": math.nan,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare low-vault weighted-loss AS-OCT pilot results.")
    parser.add_argument(
        "--original_pred",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01/combined_as_oct_strict_imagenet_seed42_e30/test_predictions.csv",
    )
    parser.add_argument(
        "--loww1p5_pred",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01/combined_as_oct_loww1p5_seed42_e30/test_predictions.csv",
    )
    parser.add_argument(
        "--loww2_pred",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01/combined_as_oct_loww2_seed42_e30/test_predictions.csv",
    )
    parser.add_argument(
        "--extremew1p5_pred",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01/combined_as_oct_extremew1p5_seed42_e30/test_predictions.csv",
    )
    parser.add_argument(
        "--ensemble_error_csv",
        default="artifacts/reports/combined_batch_01_02/as_oct_error_analysis/as_oct_ensemble_test_error_by_sample.csv",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/vault_as_oct_only_pod1_manifest_combined_strict.csv",
    )
    parser.add_argument(
        "--pred_root",
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/reports/combined_batch_01_02/low_vault_weighted_loss_comparison",
    )
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    return parser.parse_args()


def find_original_prediction(default_path: Path, pred_root: Path) -> Path:
    if default_path.exists():
        return default_path
    candidates = []
    pattern = re.compile(r"combined_as_oct.*strict.*imagenet.*seed42.*e30", re.IGNORECASE)
    exclude = re.compile(r"loww|extremew|smoke|full_sensitivity", re.IGNORECASE)
    for path in pred_root.rglob("test_predictions.csv"):
        text = str(path)
        if pattern.search(text) and not exclude.search(text):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not locate original seed42 prediction. Tried {default_path}")
    return sorted(candidates)[0]


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def find_first_column(df: pd.DataFrame, candidates: list[str], path: Path) -> str:
    lower = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"Could not find any of {candidates} in {path}")


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
    raise ValueError("Cannot derive global_sample_id from prediction table.")


def read_prediction(path: Path, pred_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = find_first_column(df, LABEL_CANDIDATES, path)
    pred_col = find_first_column(df, PRED_CANDIDATES, path)
    return pd.DataFrame(
        {
            "global_sample_id": derive_global_sample_id(df),
            "vault_label_um": pd.to_numeric(df[label_col], errors="coerce"),
            pred_name: pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )


def read_ensemble(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "global_sample_id" not in df.columns:
        raise ValueError("Ensemble error table must contain global_sample_id.")
    if "ensemble_pred_um" in df.columns:
        return df[["global_sample_id", "ensemble_pred_um"]].copy()
    if "pred_ensemble_um" in df.columns:
        return df[["global_sample_id", "pred_ensemble_um"]].rename(columns={"pred_ensemble_um": "ensemble_pred_um"})
    raise ValueError("Ensemble error table must contain pred_ensemble_um or ensemble_pred_um.")


def vault_range(labels: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    out = pd.Series("medium", index=labels.index)
    out.loc[labels < low_threshold] = "low"
    out.loc[labels > high_threshold] = "high"
    return out


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if len(true) == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan, "mean_signed": math.nan}
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
    }


def build_sample_table(predictions: dict[str, pd.DataFrame], low_threshold: float, high_threshold: float) -> pd.DataFrame:
    df = predictions["original_seed42"]
    for method in ["low_weight_1p5_seed42", "low_weight_2p0_seed42", "extreme_weight_1p5_seed42"]:
        df = df.merge(predictions[method], on=["global_sample_id", "vault_label_um"], how="inner")
    df = df.merge(predictions["as_oct_seed_ensemble"], on="global_sample_id", how="inner")
    df["vault_range"] = vault_range(df["vault_label_um"], low_threshold, high_threshold)
    for spec in METHODS:
        pred_col = spec["pred_col"]
        prefix = pred_col.replace("_pred_um", "")
        df[f"{prefix}_abs_error_um"] = (df[pred_col] - df["vault_label_um"]).abs()
        df[f"{prefix}_signed_error_um"] = df[pred_col] - df["vault_label_um"]
    df["whether_patient052"] = df["global_sample_id"].astype(str).str.contains("patient_052", case=False, na=False)
    return df.sort_values("global_sample_id").reset_index(drop=True)


def build_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    original_row: dict[str, float] | None = None
    rows = []
    for spec in METHODS:
        method = spec["method"]
        pred_col = spec["pred_col"]
        overall = metrics(sample_df["vault_label_um"], sample_df[pred_col])
        row = {
            "method": method,
            "loss_weight_mode": spec["loss_weight_mode"],
            "low_weight": spec["low_weight"],
            "medium_weight": spec["medium_weight"],
            "high_weight": spec["high_weight"],
            "n_samples": overall["n"],
            "overall_mae_um": overall["mae"],
            "overall_rmse_um": overall["rmse"],
            "overall_r2": overall["r2"],
        }
        for group in ["low", "medium", "high"]:
            sub = sample_df[sample_df["vault_range"] == group]
            m = metrics(sub["vault_label_um"], sub[pred_col])
            signed = sub[pred_col] - sub["vault_label_um"]
            row[f"{group}_n"] = m["n"]
            row[f"{group}_mae_um"] = m["mae"]
            row[f"{group}_mean_signed_error_um"] = m["mean_signed"]
            if group == "low":
                row["low_overestimation_count"] = int((signed > 0).sum())
        if method == "original_seed42":
            original_row = row.copy()
        rows.append(row)

    if original_row is None:
        raise RuntimeError("Missing original_seed42 summary row.")
    for row in rows:
        row["overall_delta_vs_original_um"] = row["overall_mae_um"] - original_row["overall_mae_um"]
        row["low_delta_vs_original_um"] = row["low_mae_um"] - original_row["low_mae_um"]
        row["high_delta_vs_original_um"] = row["high_mae_um"] - original_row["high_mae_um"]
    return pd.DataFrame(rows)


def plot_overall_range(summary: pd.DataFrame, out_path: Path) -> None:
    groups = [("overall", "overall_mae_um"), ("low", "low_mae_um"), ("medium", "medium_mae_um"), ("high", "high_mae_um")]
    methods = list(summary["method"])
    x = np.arange(len(groups))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    colors = ["#4c78a8", "#f58518", "#eeca3b", "#b279a2", "#54a24b"]
    for i, method in enumerate(methods):
        row = summary[summary["method"] == method].iloc[0]
        values = [row[col] for _, col in groups]
        ax.bar(x + (i - 2) * width, values, width=width, label=method, color=colors[i], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in groups])
    ax.set_ylabel("MAE (um)")
    ax.set_title("Weighted-Loss Pilot MAE Comparison")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_low_signed(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    colors = ["#4c78a8", "#f58518", "#eeca3b", "#b279a2", "#54a24b"]
    ax.bar(summary["method"], summary["low_mean_signed_error_um"], color=colors, edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="#404040", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Low-vault mean signed error (um)")
    ax.set_title("Low-Vault Mean Signed Error")
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(summary["method"], rotation=25, ha="right", fontsize=6.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_delta(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary[summary["method"].isin(["low_weight_1p5_seed42", "low_weight_2p0_seed42", "extreme_weight_1p5_seed42"])].copy()
    groups = [
        ("overall", "overall_delta_vs_original_um"),
        ("low", "low_delta_vs_original_um"),
        ("medium", "medium_mae_um"),
        ("high", "high_delta_vs_original_um"),
    ]
    original = summary[summary["method"] == "original_seed42"].iloc[0]
    df["medium_delta_vs_original_um"] = df["medium_mae_um"] - original["medium_mae_um"]
    groups[2] = ("medium", "medium_delta_vs_original_um")
    x = np.arange(len(groups))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    colors = ["#f58518", "#eeca3b", "#b279a2"]
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[col] for _, col in groups]
        ax.bar(x + (i - 1) * width, values, width=width, label=row["method"], color=colors[i], edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="#404040", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in groups])
    ax.set_ylabel("MAE delta vs original seed42 (um)")
    ax.set_title("Weighted Settings: Delta vs Original Seed42")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_patient052(sample_df: pd.DataFrame, out_path: Path) -> None:
    p052 = sample_df[sample_df["whether_patient052"]].copy()
    if p052.empty:
        return
    methods = [spec["method"] for spec in METHODS]
    pred_cols = [spec["pred_col"] for spec in METHODS]
    rows = []
    for _, row in p052.iterrows():
        for method, pred_col in zip(methods, pred_cols):
            rows.append(
                {
                    "sample": row["global_sample_id"].replace("batch_02__", "b02__"),
                    "method": method,
                    "abs_error": abs(row[pred_col] - row["vault_label_um"]),
                    "signed_error": row[pred_col] - row["vault_label_um"],
                }
            )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    for ax, metric, title in [(axes[0], "abs_error", "Abs Error"), (axes[1], "signed_error", "Signed Error")]:
        pivot = df.pivot(index="sample", columns="method", values=metric)
        pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="#404040", linewidth=0.8)
        ax.set_title(f"patient_052 {title}")
        ax.set_ylabel(f"{title} (um)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    original = summary[summary["method"] == "original_seed42"].iloc[0]
    ensemble = summary[summary["method"] == "as_oct_seed_ensemble"].iloc[0]
    weighted = summary[summary["method"] != "as_oct_seed_ensemble"].copy()
    weighted = weighted[weighted["method"] != "original_seed42"]
    best_low = weighted.sort_values("low_mae_um").iloc[0]
    best_overall = weighted.sort_values("overall_mae_um").iloc[0]

    lines = [
        "# Low-vault weighted-loss pilot comparison",
        "",
        "## Summary",
        "",
        f"- Original seed42: overall MAE {original['overall_mae_um']:.2f} um, low-vault MAE {original['low_mae_um']:.2f} um.",
        f"- Best weighted low-vault setting: {best_low['method']} with low-vault MAE {best_low['low_mae_um']:.2f} um.",
        f"- Best weighted overall setting: {best_overall['method']} with overall MAE {best_overall['overall_mae_um']:.2f} um.",
        f"- AS-OCT seed ensemble remains strongest overall: overall MAE {ensemble['overall_mae_um']:.2f} um.",
        "",
        "## Interpretation",
        "",
        "low_weight=1.5 most clearly reduces low-vault MAE and low-vault positive signed error, but it worsens overall, medium-vault, and especially high-vault performance relative to original seed42. low_weight=2.0 also improves low-vault MAE relative to original seed42, but its overall performance is worse. extreme_weight=1.5 is more balanced overall than low-only weighting, but it does not solve low-vault overestimation.",
        "",
        "## Recommendation",
        "",
        "Do not promote weighted loss to the main result at this stage. The current evidence suggests weighted loss is useful as an exploratory strategy for low-vault overestimation, but it trades off other vault ranges and remains weaker than the AS-OCT seed ensemble. It is not recommended to immediately run seed2026/3407 for these settings unless a small, pre-specified range-aware experiment is planned.",
        "",
        "Suggested next steps: keep this as an exploratory training strategy, consider a milder grid or balanced sampler, and prioritize vault-range-aware evaluation/calibration before expanding to more seeds.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original_seed42": find_original_prediction(Path(args.original_pred), Path(args.pred_root)),
        "low_weight_1p5_seed42": Path(args.loww1p5_pred),
        "low_weight_2p0_seed42": Path(args.loww2_pred),
        "extreme_weight_1p5_seed42": Path(args.extremew1p5_pred),
    }
    for method, path in paths.items():
        require_file(path, f"{method} prediction")
    require_file(Path(args.ensemble_error_csv), "AS-OCT ensemble error table")

    predictions = {
        "original_seed42": read_prediction(paths["original_seed42"], "original_seed42_pred_um"),
        "low_weight_1p5_seed42": read_prediction(paths["low_weight_1p5_seed42"], "loww1p5_seed42_pred_um"),
        "low_weight_2p0_seed42": read_prediction(paths["low_weight_2p0_seed42"], "loww2_seed42_pred_um"),
        "extreme_weight_1p5_seed42": read_prediction(paths["extreme_weight_1p5_seed42"], "extremew1p5_seed42_pred_um"),
        "as_oct_seed_ensemble": read_ensemble(Path(args.ensemble_error_csv)),
    }
    sample_df = build_sample_table(predictions, args.low_threshold, args.high_threshold)
    summary = build_summary(sample_df)

    summary_path = out_dir / "weighted_loss_pilot_summary.csv"
    sample_path = out_dir / "weighted_loss_pilot_by_sample.csv"
    md_path = out_dir / "weighted_loss_pilot_summary.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    write_markdown(summary, md_path)

    plot_overall_range(summary, fig_dir / "weighted_loss_overall_range_mae_comparison.png")
    plot_low_signed(summary, fig_dir / "weighted_loss_low_signed_error_comparison.png")
    plot_delta(summary, fig_dir / "weighted_loss_delta_vs_original.png")
    plot_patient052(sample_df, fig_dir / "weighted_loss_patient052_errors.png")

    print("Prediction paths:")
    for method, path in paths.items():
        print(f"  {method}: {path}")
    print(f"  as_oct_seed_ensemble: {args.ensemble_error_csv}")
    print("Method MAE summary:")
    print(summary[["method", "overall_mae_um", "low_mae_um", "medium_mae_um", "high_mae_um", "low_mean_signed_error_um"]].to_string(index=False))
    print("Output files:")
    for path in [summary_path, sample_path, md_path, fig_dir]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
