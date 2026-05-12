"""Plot AS-OCT-only POD1 baseline result figures for reports.

The script is read-only with respect to logs, predictions, manifests, and
checkpoints. It uses matplotlib only and writes 300 dpi PNG figures for meeting
slides and manuscript drafts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AS-OCT-only POD1 baseline results.")
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01",
        help="Directory containing summary.csv and group_summary.csv.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
        help="Directory containing per-run prediction CSV files.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="artifacts/logs/as_oct_pod1_baseline_batch_01",
        help="Directory containing per-run train_log.csv files.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
        help="Clean strict manifest used for split label distribution.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/figures",
        help="Output figure directory.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def warn(message: str) -> None:
    print(f"Warning: {message}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {relative_path(path)}")


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"missing file, skipped: {relative_path(path)}")
        return None
    return pd.read_csv(path)


def short_run_name(run_name: str) -> str:
    name = run_name.replace("as_oct_pod1_clean_resnet18_", "")
    name = name.replace("_e30", "")
    name = name.replace("imagenet_seed", "img_s")
    name = name.replace("imagenet_freeze", "img_freeze")
    name = name.replace("imagenet", "img")
    name = name.replace("random", "random")
    return name


def finite_error(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.notna())


def plot_group_bar(
    group_df: pd.DataFrame,
    value_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if group_df.empty:
        warn(f"group summary is empty, skipped {output_path.name}")
        return

    families = group_df["experiment_family"].astype(str).tolist()
    values = pd.to_numeric(group_df[value_column], errors="coerce")
    stds = pd.to_numeric(group_df[std_column], errors="coerce") if std_column in group_df.columns else pd.Series([0] * len(values))
    yerr = [0 if pd.isna(value) else value for value in stds]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(families, values, yerr=yerr, capsize=4, color=["#4C78A8", "#F58518", "#54A24B"][: len(families)])
    ax.set_title(title)
    ax.set_xlabel("Experiment family")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def plot_run_val_test(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        warn("summary is empty, skipped run val/test bar")
        return

    df = summary_df.sort_values("best_val_mae_um", kind="stable").copy()
    x = range(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], df["best_val_mae_um"], width=width, label="Best val MAE", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], df["test_mae_um"], width=width, label="Test MAE", color="#F58518")
    ax.set_title("Validation and Test MAE by Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("MAE (um)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([short_run_name(name) for name in df["run_name"]], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def plot_clean_vs_full(summary_df: pd.DataFrame, output_path: Path) -> None:
    required_runs = [
        "as_oct_pod1_clean_resnet18_imagenet_e30",
        "as_oct_pod1_full_resnet18_imagenet_seed42_e30",
    ]
    missing = [run_name for run_name in required_runs if run_name not in set(summary_df["run_name"].astype(str))]
    if missing:
        warn(f"missing clean/full sensitivity run(s), skipped {output_path.name}: {', '.join(missing)}")
        return

    df = summary_df[summary_df["run_name"].isin(required_runs)].copy()
    df["display_name"] = df["run_name"].map(
        {
            "as_oct_pod1_clean_resnet18_imagenet_e30": "Clean manifest",
            "as_oct_pod1_full_resnet18_imagenet_seed42_e30": "Full manifest",
        }
    )
    df["order"] = df["run_name"].map({required_runs[0]: 0, required_runs[1]: 1})
    df = df.sort_values("order", kind="stable")

    x = range(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar([i - width / 2 for i in x], df["best_val_mae_um"], width=width, label="Best val MAE", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], df["test_mae_um"], width=width, label="Test MAE", color="#F58518")
    ax.set_title("Clean vs Full Manifest Sensitivity")
    ax.set_xlabel("Manifest variant")
    ax.set_ylabel("MAE (um)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["display_name"])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def regression_metrics(pred_df: pd.DataFrame) -> tuple[float, float, float]:
    labels = pd.to_numeric(pred_df["vault_label_um"], errors="coerce")
    preds = pd.to_numeric(pred_df["pred_vault_um"], errors="coerce")
    valid = labels.notna() & preds.notna()
    labels = labels[valid]
    preds = preds[valid]
    errors = preds - labels
    mae = float(errors.abs().mean())
    rmse = float((errors.pow(2).mean()) ** 0.5)
    ss_tot = float((labels - labels.mean()).pow(2).sum())
    r2 = float("nan") if ss_tot <= 0 else 1.0 - float(errors.pow(2).sum()) / ss_tot
    return mae, rmse, r2


def plot_pred_vs_gt(pred_df: pd.DataFrame, run_name: str, output_path: Path, title_suffix: str) -> None:
    labels = pd.to_numeric(pred_df["vault_label_um"], errors="coerce")
    preds = pd.to_numeric(pred_df["pred_vault_um"], errors="coerce")
    valid = labels.notna() & preds.notna()
    labels = labels[valid]
    preds = preds[valid]
    if labels.empty:
        warn(f"no valid predictions for {run_name}, skipped {output_path.name}")
        return

    mae, rmse, r2 = regression_metrics(pred_df)
    lower = min(labels.min(), preds.min()) - 40
    upper = max(labels.max(), preds.max()) + 40

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(labels, preds, color="#4C78A8", alpha=0.8, edgecolor="white", linewidth=0.6)
    ax.plot([lower, upper], [lower, upper], color="#D62728", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(f"Predicted vs Ground Truth Vault ({title_suffix})")
    ax.set_xlabel("Ground truth POD1 vault (um)")
    ax.set_ylabel("Predicted POD1 vault (um)")
    ax.text(
        0.04,
        0.96,
        f"Run: {short_run_name(run_name)}\nMAE={mae:.1f} um\nRMSE={rmse:.1f} um\nR2={r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)


def plot_abs_error_distribution(summary_df: pd.DataFrame, predictions_dir: Path, output_path: Path) -> None:
    data: List[pd.Series] = []
    labels: List[str] = []
    for run_name in summary_df["run_name"].astype(str):
        pred_path = predictions_dir / run_name / "test_predictions.csv"
        pred_df = safe_read_csv(pred_path)
        if pred_df is None or "abs_error_um" not in pred_df.columns:
            warn(f"missing abs_error_um for {run_name}, skipped in distribution")
            continue
        errors = pd.to_numeric(pred_df["abs_error_um"], errors="coerce").dropna()
        if errors.empty:
            continue
        data.append(errors)
        labels.append(short_run_name(run_name))

    if not data:
        warn("no error distributions available, skipped test_abs_error_distribution.png")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=True)
    for index, errors in enumerate(data, start=1):
        ax.scatter([index] * len(errors), errors, alpha=0.45, s=18, color="#4C78A8")
    ax.set_title("Test Absolute Error Distribution")
    ax.set_xlabel("Run")
    ax.set_ylabel("Absolute error (um)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def plot_split_label_distribution(manifest_df: pd.DataFrame, output_path: Path) -> None:
    splits = ["train", "val", "test"]
    data = [
        pd.to_numeric(manifest_df.loc[manifest_df["split"] == split, "vault_label"], errors="coerce").dropna()
        for split in splits
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot(data, labels=splits, showfliers=True)
    for index, values in enumerate(data, start=1):
        ax.scatter([index] * len(values), values, alpha=0.5, s=18, color="#4C78A8")
    ax.set_title("POD1 Vault Label Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("POD1 vault label (um)")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)


def plot_training_curve(log_df: pd.DataFrame, output_path: Path) -> None:
    if log_df.empty:
        warn("training log is empty, skipped training curve")
        return
    best_index = pd.to_numeric(log_df["val_mae_um"], errors="coerce").idxmin()
    best_row = log_df.loc[best_index]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(log_df["epoch"], log_df["train_mae_um"], marker="o", markersize=3, label="Train MAE", color="#4C78A8")
    ax.plot(log_df["epoch"], log_df["val_mae_um"], marker="o", markersize=3, label="Val MAE", color="#F58518")
    ax.axvline(best_row["epoch"], color="#D62728", linestyle="--", linewidth=1.2, label=f"Best epoch {int(best_row['epoch'])}")
    ax.set_title("Training Curve: ImageNet Fine-tuning Seed 42")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (um)")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    reports_dir = resolve_project_path(args.reports_dir)
    predictions_dir = resolve_project_path(args.predictions_dir)
    logs_dir = resolve_project_path(args.logs_dir)
    manifest_path = resolve_project_path(args.manifest)
    figures_dir = resolve_project_path(args.figures_dir)

    summary_df = safe_read_csv(reports_dir / "summary.csv")
    group_df = safe_read_csv(reports_dir / "group_summary.csv")
    manifest_df = safe_read_csv(manifest_path)
    if summary_df is None or group_df is None:
        raise SystemExit(1)

    plot_group_bar(
        group_df=group_df,
        value_column="best_val_mae_mean",
        std_column="best_val_mae_std",
        ylabel="Best validation MAE (um)",
        title="Best Validation MAE by Experiment Family",
        output_path=figures_dir / "group_val_mae_bar.png",
    )
    plot_group_bar(
        group_df=group_df,
        value_column="test_mae_mean",
        std_column="test_mae_std",
        ylabel="Test MAE (um)",
        title="Test MAE by Experiment Family",
        output_path=figures_dir / "group_test_mae_bar.png",
    )
    plot_run_val_test(summary_df=summary_df, output_path=figures_dir / "run_val_test_mae_bar.png")
    plot_clean_vs_full(summary_df=summary_df, output_path=figures_dir / "clean_vs_full_val_test_mae.png")

    best_val_run = str(summary_df.sort_values("best_val_mae_um", kind="stable").iloc[0]["run_name"])
    best_val_pred = safe_read_csv(predictions_dir / best_val_run / "test_predictions.csv")
    if best_val_pred is not None:
        plot_pred_vs_gt(
            pred_df=best_val_pred,
            run_name=best_val_run,
            output_path=figures_dir / "pred_vs_gt_best_val_run.png",
            title_suffix="Best Val Run",
        )

    best_test_run = str(summary_df.sort_values("test_mae_um", kind="stable").iloc[0]["run_name"])
    best_test_pred = safe_read_csv(predictions_dir / best_test_run / "test_predictions.csv")
    if best_test_pred is not None:
        plot_pred_vs_gt(
            pred_df=best_test_pred,
            run_name=best_test_run,
            output_path=figures_dir / "pred_vs_gt_best_test_run.png",
            title_suffix="Best Test Run",
        )

    plot_abs_error_distribution(
        summary_df=summary_df,
        predictions_dir=predictions_dir,
        output_path=figures_dir / "test_abs_error_distribution.png",
    )

    if manifest_df is not None:
        plot_split_label_distribution(
            manifest_df=manifest_df,
            output_path=figures_dir / "split_label_distribution.png",
        )

    curve_log = safe_read_csv(logs_dir / "as_oct_pod1_clean_resnet18_imagenet_e30" / "train_log.csv")
    if curve_log is not None:
        plot_training_curve(
            log_df=curve_log,
            output_path=figures_dir / "training_curve_imagenet_finetune_seed42.png",
        )


if __name__ == "__main__":
    main()
