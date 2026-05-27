"""Create paper-ready result figures for the combined POD1 vault baselines.

The script reads existing result summaries and prediction CSV files only. It
does not modify manifests, training logs, prediction files, checkpoints, or
paper source files.
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


AS_OCT_RUNS = [
    "combined_as_oct_strict_imagenet_seed42_e30",
    "combined_as_oct_strict_imagenet_seed2026_e30",
    "combined_as_oct_strict_imagenet_seed3407_e30",
]

BASELINE_DISPLAY = {
    "as_oct_only_imagenet_finetune": "AS-OCT-only\nImageNet fine-tune",
    "fusion_concat_resnet18_measurement": "Fusion concat\nResNet18 + measurement",
    "measurement_only_linear": "Measurement-only\nLinear",
    "measurement_only_random_forest": "Measurement-only\nRandom Forest",
    "measurement_only_ridge": "Measurement-only\nRidge",
    "measurement_only_mlp": "Measurement-only\nMLP",
}

BASELINE_PRIORITY = [
    "as_oct_only_imagenet_finetune",
    "fusion_concat_resnet18_measurement",
    "measurement_only_linear",
    "measurement_only_random_forest",
    "measurement_only_ridge",
    "measurement_only_mlp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-ready baseline comparison and prediction scatter figures."
    )
    parser.add_argument(
        "--summary_csv",
        default="artifacts/reports/combined_batch_01_02/final_baseline_summary/final_baseline_summary.csv",
        help="Final baseline family summary CSV.",
    )
    parser.add_argument(
        "--run_summary_csv",
        default="artifacts/reports/combined_batch_01_02/final_baseline_summary/final_run_level_summary.csv",
        help="Final run-level summary CSV.",
    )
    parser.add_argument(
        "--predictions_root",
        default="artifacts/predictions",
        help="Root directory containing prediction files.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/figures",
        help="Directory where paper figures will be saved.",
    )
    parser.add_argument(
        "--include_seed_ensemble",
        type=str_to_bool,
        default=True,
        help="Whether to add AS-OCT seed ensemble as a diagnostic result in Figure A.",
    )
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
    return df["sample_id"].astype(str)


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    if len(y_true_arr) == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}
    err = y_pred_arr - y_true_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2 = math.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_as_oct_ensemble(predictions_root: Path) -> tuple[pd.DataFrame | None, list[str]]:
    frames: list[pd.DataFrame] = []
    loaded: list[str] = []
    for run_name in AS_OCT_RUNS:
        path = predictions_root / "as_oct_pod1_baseline_batch_01" / run_name / "test_predictions.csv"
        if not path.exists():
            print(f"WARNING: AS-OCT ensemble prediction missing: {path}")
            continue
        df = pd.read_csv(path)
        if not {"vault_label_um", "pred_vault_um", "sample_id"}.issubset(df.columns):
            print(f"WARNING: AS-OCT prediction file has incompatible columns, skipping: {path}")
            continue
        df = df.copy()
        df["global_sample_id"] = derive_global_sample_id(df)
        df["run_name"] = run_name
        frames.append(df)
        loaded.append(str(path))

    if len(frames) < 2:
        return None, loaded

    stacked = pd.concat(frames, ignore_index=True)
    ensemble = (
        stacked.groupby("global_sample_id", as_index=False)
        .agg(
            sample_id=("sample_id", "first"),
            vault_label_um=("vault_label_um", "first"),
            pred_vault_um=("pred_vault_um", "mean"),
            n_predictions=("pred_vault_um", "count"),
        )
        .copy()
    )
    ensemble["source_name"] = "AS-OCT seed ensemble"
    return ensemble, loaded


def load_best_as_oct_single_run(run_summary: pd.DataFrame, predictions_root: Path) -> tuple[pd.DataFrame, str, str]:
    as_oct = run_summary[
        run_summary["baseline_family"].astype(str).str.contains("as_oct", case=False, na=False)
    ].copy()
    if as_oct.empty:
        raise ValueError("Could not find AS-OCT rows in run-level summary.")
    best = as_oct.sort_values("test_mae_um").iloc[0]
    run_name = str(best["run_name"])
    path = predictions_root / "as_oct_pod1_baseline_batch_01" / run_name / "test_predictions.csv"
    require_file(path, "best AS-OCT single-run test predictions")
    df = pd.read_csv(path)
    df["global_sample_id"] = derive_global_sample_id(df)
    df["source_name"] = run_name
    return df, run_name, str(path)


def build_figure_a_data(summary: pd.DataFrame, include_seed_ensemble: bool, predictions_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    used_baselines: list[str] = []
    for family in BASELINE_PRIORITY:
        match = summary[summary["baseline_family"].astype(str).str.lower() == family.lower()]
        if match.empty:
            contains = summary[
                summary["baseline_family"].astype(str).str.contains(family.replace("_", ".*"), case=False, na=False)
            ]
            match = contains.head(1)
        if match.empty:
            print(f"WARNING: baseline not found in summary: {family}")
            continue
        row = match.iloc[0]
        rows.append(
            {
                "baseline_family": family,
                "display_name": BASELINE_DISPLAY.get(family, str(row["baseline_family"])),
                "test_mae_mean": float(row["test_mae_mean"]),
                "test_mae_std": float(row["test_mae_std"]) if "test_mae_std" in row and pd.notna(row["test_mae_std"]) else math.nan,
                "n_runs": int(row["n_runs"]) if "n_runs" in row and pd.notna(row["n_runs"]) else math.nan,
                "category": "as_oct" if family.startswith("as_oct") else "fusion" if family.startswith("fusion") else "measurement",
            }
        )
        used_baselines.append(BASELINE_DISPLAY.get(family, family).replace("\n", " "))

    if include_seed_ensemble:
        ensemble_df, loaded = load_as_oct_ensemble(predictions_root)
        if ensemble_df is not None:
            m = metrics(ensemble_df["vault_label_um"], ensemble_df["pred_vault_um"])
            rows.append(
                {
                    "baseline_family": "as_oct_seed_ensemble_diagnostic",
                    "display_name": "AS-OCT seed ensemble\n(diagnostic)",
                    "test_mae_mean": m["mae"],
                    "test_mae_std": math.nan,
                    "n_runs": int(ensemble_df["n_predictions"].median()),
                    "category": "ensemble",
                }
            )
            used_baselines.append("AS-OCT seed ensemble (diagnostic)")
            print("Figure A added AS-OCT seed ensemble diagnostic result from:")
            for path in loaded:
                print(f"  {path}")
        else:
            print("WARNING: AS-OCT seed ensemble diagnostic result was requested but could not be built.")

    fig_df = pd.DataFrame(rows)
    if fig_df.empty:
        raise ValueError("No baselines available for Figure A.")
    fig_df = fig_df.sort_values("test_mae_mean", ascending=True).reset_index(drop=True)
    return fig_df, used_baselines


def plot_main_baseline_comparison(fig_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    color_map = {
        "as_oct": "#2f5597",
        "fusion": "#6b8e23",
        "measurement": "#b7b7b7",
        "ensemble": "#1f4e79",
    }
    colors = [color_map.get(cat, "#b7b7b7") for cat in fig_df["category"]]

    height = max(3.0, 0.42 * len(fig_df) + 1.2)
    fig, ax = plt.subplots(figsize=(5.8, height))
    y = np.arange(len(fig_df))
    ax.barh(y, fig_df["test_mae_mean"], color=colors, edgecolor="black", linewidth=0.4, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(fig_df["display_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (µm)")
    ax.set_title("Main Baseline Comparison on the Combined Pilot Cohort", pad=7)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = float(fig_df["test_mae_mean"].max())
    ax.set_xlim(0, max_val * 1.18)
    for yi, row in fig_df.iterrows():
        label = f"{row['test_mae_mean']:.2f}"
        if pd.notna(row["n_runs"]):
            label += f"  (n={int(row['n_runs'])})"
        ax.text(row["test_mae_mean"] + max_val * 0.015, yi, label, va="center", ha="left", fontsize=7.3)

    fig.tight_layout()
    png = output_dir / "paper_main_baseline_comparison.png"
    pdf = output_dir / "paper_main_baseline_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def select_prediction_for_figure_b(
    run_summary: pd.DataFrame,
    predictions_root: Path,
    include_seed_ensemble: bool,
) -> tuple[pd.DataFrame, str, list[str], bool]:
    if include_seed_ensemble:
        ensemble_df, loaded = load_as_oct_ensemble(predictions_root)
        if ensemble_df is not None:
            return ensemble_df, "AS-OCT seed ensemble", loaded, True
        print("WARNING: Ensemble prediction not available; falling back to best AS-OCT single run.")

    df, run_name, path = load_best_as_oct_single_run(run_summary, predictions_root)
    return df, run_name, [path], False


def plot_prediction_scatter(pred_df: pd.DataFrame, source_name: str, output_dir: Path) -> tuple[Path, Path, dict[str, float]]:
    if not {"vault_label_um", "pred_vault_um"}.issubset(pred_df.columns):
        raise ValueError("Prediction dataframe must contain vault_label_um and pred_vault_um.")
    m = metrics(pred_df["vault_label_um"], pred_df["pred_vault_um"])

    x = pd.to_numeric(pred_df["vault_label_um"], errors="coerce")
    y = pd.to_numeric(pred_df["pred_vault_um"], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    min_v = float(min(x.min(), y.min()))
    max_v = float(max(x.max(), y.max()))
    pad = (max_v - min_v) * 0.08 if max_v > min_v else 50.0
    lims = (min_v - pad, max_v + pad)

    fig, ax = plt.subplots(figsize=(3.55, 3.35))
    ax.scatter(x, y, s=28, color="#2f5597", alpha=0.72, edgecolor="white", linewidth=0.35)
    ax.plot(lims, lims, linestyle="--", color="#4d4d4d", linewidth=0.9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth vault (µm)")
    ax.set_ylabel("Predicted vault (µm)")
    ax.set_title("Prediction vs. Ground Truth for the Best AS-OCT Model", pad=7)
    ax.grid(color="#d9d9d9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    text = f"MAE = {m['mae']:.2f} µm\nRMSE = {m['rmse']:.2f} µm\nR² = {m['r2']:.3f}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.3,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#bfbfbf", "linewidth": 0.6},
    )
    ax.text(
        0.98,
        0.04,
        source_name,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=6.8,
        color="#404040",
    )

    fig.tight_layout()
    png = output_dir / "paper_best_model_pred_vs_gt.png"
    pdf = output_dir / "paper_best_model_pred_vs_gt.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf, m


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    summary_csv = Path(args.summary_csv)
    run_summary_csv = Path(args.run_summary_csv)
    predictions_root = Path(args.predictions_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_file(summary_csv, "baseline summary CSV")
    require_file(run_summary_csv, "run-level summary CSV")
    if not predictions_root.exists():
        raise FileNotFoundError(f"Missing predictions root: {predictions_root}")

    summary = pd.read_csv(summary_csv)
    run_summary = pd.read_csv(run_summary_csv)
    print(f"Read summary CSV: {summary_csv}")
    print(f"Read run-level summary CSV: {run_summary_csv}")

    fig_a_df, used_baselines = build_figure_a_data(summary, args.include_seed_ensemble, predictions_root)
    fig_a_png, fig_a_pdf = plot_main_baseline_comparison(fig_a_df, output_dir)

    pred_df, source_name, pred_files, used_ensemble = select_prediction_for_figure_b(
        run_summary, predictions_root, args.include_seed_ensemble
    )
    print("Figure B prediction source files:")
    for path in pred_files:
        print(f"  {path}")
    fig_b_png, fig_b_pdf, fig_b_metrics = plot_prediction_scatter(pred_df, source_name, output_dir)

    print("Figure A baselines:")
    for name in used_baselines:
        print(f"  {name}")
    print(f"Figure B model/run: {source_name}")
    if used_ensemble:
        print("Figure B used AS-OCT seed ensemble predictions.")
    else:
        print("Figure B fell back to the best AS-OCT-only single run.")
    print(
        "Figure B metrics: "
        f"MAE={fig_b_metrics['mae']:.2f} um, "
        f"RMSE={fig_b_metrics['rmse']:.2f} um, "
        f"R2={fig_b_metrics['r2']:.3f}"
    )
    print("Saved paper figures:")
    print(f"  {fig_a_png}")
    print(f"  {fig_a_pdf}")
    print(f"  {fig_b_png}")
    print(f"  {fig_b_pdf}")


if __name__ == "__main__":
    main()
