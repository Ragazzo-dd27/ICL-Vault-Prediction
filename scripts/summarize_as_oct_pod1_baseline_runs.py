"""Summarize AS-OCT-only POD1 baseline runs into CSV and Markdown tables.

This script is read-only with respect to training logs, predictions, and
checkpoints. It recomputes prediction metrics for reporting and writes compact
summary artifacts for meeting notes and manuscript drafts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AS-OCT-only POD1 baseline runs.")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="artifacts/logs/as_oct_pod1_baseline_batch_01",
        help="Directory containing per-run train_log.csv files.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="artifacts/predictions/as_oct_pod1_baseline_batch_01",
        help="Directory containing per-run val/test prediction CSV files.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01",
        help="Directory for summary.csv and summary.md.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_run_config(run_name: str) -> Dict[str, object]:
    pretrained = "imagenet" in run_name
    freeze_backbone = "freeze" in run_name
    seed_match = SEED_PATTERN.search(run_name)
    seed = int(seed_match.group("seed")) if seed_match else 42
    if "random" in run_name:
        experiment_family = "random_init"
        notes = "random initialization baseline"
    elif "imagenet_freeze" in run_name:
        experiment_family = "imagenet_freeze"
        notes = "ImageNet pretrained frozen backbone"
    elif "imagenet" in run_name:
        experiment_family = "imagenet_finetune"
        notes = "ImageNet pretrained full fine-tuning"
    else:
        experiment_family = "unknown"
        notes = "baseline run"
    return {
        "experiment_family": experiment_family,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "seed": seed,
        "notes": notes,
    }


def discover_runs(logs_dir: Path) -> List[str]:
    train_logs = sorted(logs_dir.glob("*/train_log.csv"))
    return [path.parent.name for path in train_logs]


def regression_metrics(predictions_df: pd.DataFrame) -> Dict[str, float]:
    labels = pd.to_numeric(predictions_df["vault_label_um"], errors="coerce")
    preds = pd.to_numeric(predictions_df["pred_vault_um"], errors="coerce")
    valid = labels.notna() & preds.notna()
    labels = labels[valid]
    preds = preds[valid]
    if labels.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    errors = preds - labels
    mae = float(errors.abs().mean())
    rmse = float((errors.pow(2).mean()) ** 0.5)
    ss_tot = float((labels - labels.mean()).pow(2).sum())
    if ss_tot <= 0:
        r2 = float("nan")
    else:
        ss_res = float(errors.pow(2).sum())
        r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def summarize_run(
    run_name: str,
    logs_dir: Path,
    predictions_dir: Path,
) -> Dict[str, object]:
    log_path = logs_dir / run_name / "train_log.csv"
    test_predictions_path = predictions_dir / run_name / "test_predictions.csv"
    for path in (log_path, test_predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file for {run_name}: {path}")

    log_df = pd.read_csv(log_path)
    test_predictions_df = pd.read_csv(test_predictions_path)

    best_index = pd.to_numeric(log_df["val_mae_um"], errors="coerce").idxmin()
    best_row = log_df.loc[best_index]
    test_metrics = regression_metrics(test_predictions_df)
    config = infer_run_config(run_name)

    return {
        "run_name": run_name,
        "experiment_family": config["experiment_family"],
        "pretrained": config["pretrained"],
        "freeze_backbone": config["freeze_backbone"],
        "seed": config["seed"],
        "best_epoch": int(best_row["epoch"]),
        "best_val_mae_um": float(best_row["val_mae_um"]),
        "best_val_rmse_um": float(best_row["val_rmse_um"]),
        "best_val_r2": float(best_row["val_r2"]),
        "test_mae_um": test_metrics["mae"],
        "test_rmse_um": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "notes": config["notes"],
    }


def format_float(value: object, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def format_std(value: object) -> str:
    formatted = format_float(value)
    return formatted if formatted else "NA"


def markdown_table(summary_df: pd.DataFrame) -> str:
    display = summary_df.copy()
    for column in [
        "best_val_mae_um",
        "best_val_rmse_um",
        "best_val_r2",
        "test_mae_um",
        "test_rmse_um",
        "test_r2",
        "best_val_mae_mean",
        "best_val_mae_std",
        "test_mae_mean",
        "test_mae_std",
        "test_rmse_mean",
        "test_rmse_std",
        "test_r2_mean",
        "test_r2_std",
    ]:
        if column in display.columns:
            digits = 3 if "_r2" in column else 2
            display[column] = display[column].map(lambda value: format_float(value, digits=digits))

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines) + "\n"


def build_group_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = summary_df.groupby("experiment_family", dropna=False)
    rows: List[Dict[str, object]] = []
    for family, group in grouped:
        rows.append(
            {
                "experiment_family": family,
                "n_runs": int(len(group)),
                "best_val_mae_mean": float(group["best_val_mae_um"].mean()),
                "best_val_mae_std": float(group["best_val_mae_um"].std(ddof=1)),
                "test_mae_mean": float(group["test_mae_um"].mean()),
                "test_mae_std": float(group["test_mae_um"].std(ddof=1)),
                "test_rmse_mean": float(group["test_rmse_um"].mean()),
                "test_rmse_std": float(group["test_rmse_um"].std(ddof=1)),
                "test_r2_mean": float(group["test_r2"].mean()),
                "test_r2_std": float(group["test_r2"].std(ddof=1)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "experiment_family",
            "n_runs",
            "best_val_mae_mean",
            "best_val_mae_std",
            "test_mae_mean",
            "test_mae_std",
            "test_rmse_mean",
            "test_rmse_std",
            "test_r2_mean",
            "test_r2_std",
        ],
    ).sort_values(by="test_mae_mean", kind="stable")


def write_outputs(summary_df: pd.DataFrame, group_summary_df: pd.DataFrame, reports_dir: Path) -> tuple[Path, Path, Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = reports_dir / "summary.csv"
    summary_md = reports_dir / "summary.md"
    group_summary_csv = reports_dir / "group_summary.csv"
    group_summary_md = reports_dir / "group_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_md.write_text(markdown_table(summary_df), encoding="utf-8")
    group_summary_df.to_csv(group_summary_csv, index=False, encoding="utf-8")
    group_summary_md.write_text(markdown_table(group_summary_df), encoding="utf-8")
    return summary_csv, summary_md, group_summary_csv, group_summary_md


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    logs_dir = resolve_project_path(args.logs_dir)
    predictions_dir = resolve_project_path(args.predictions_dir)
    reports_dir = resolve_project_path(args.reports_dir)
    run_names = discover_runs(logs_dir)
    if not run_names:
        raise FileNotFoundError(f"No run train logs found under: {logs_dir}")

    rows: List[Dict[str, object]] = []
    for run_name in run_names:
        row = summarize_run(
            run_name=run_name,
            logs_dir=logs_dir,
            predictions_dir=predictions_dir,
        )
        rows.append(row)
        print(
            f"{run_name}: "
            f"best_val_mae={row['best_val_mae_um']:.2f} um, "
            f"test_mae={row['test_mae_um']:.2f} um"
        )

    summary_df = pd.DataFrame(
        rows,
        columns=[
            "run_name",
            "experiment_family",
            "pretrained",
            "freeze_backbone",
            "seed",
            "best_epoch",
            "best_val_mae_um",
            "best_val_rmse_um",
            "best_val_r2",
            "test_mae_um",
            "test_rmse_um",
            "test_r2",
            "notes",
        ],
    )
    summary_df = summary_df.sort_values(by="best_val_mae_um", kind="stable")
    group_summary_df = build_group_summary(summary_df)
    summary_csv, summary_md, group_summary_csv, group_summary_md = write_outputs(
        summary_df,
        group_summary_df=group_summary_df,
        reports_dir=reports_dir,
    )

    best_val = summary_df.iloc[0]
    best_test = summary_df.sort_values(by="test_mae_um", kind="stable").iloc[0]
    print("Family summaries:")
    for row in group_summary_df.to_dict(orient="records"):
        print(
            f"{row['experiment_family']}: "
            f"best_val_mae={format_float(row['best_val_mae_mean'])} +/- {format_std(row['best_val_mae_std'])} um, "
            f"test_mae={format_float(row['test_mae_mean'])} +/- {format_std(row['test_mae_std'])} um"
        )
    print(f"Best val run: {best_val['run_name']} ({best_val['best_val_mae_um']:.2f} um)")
    print(f"Best test run: {best_test['run_name']} ({best_test['test_mae_um']:.2f} um)")
    print(f"Summary CSV: {relative_path(summary_csv)}")
    print(f"Summary Markdown: {relative_path(summary_md)}")
    print(f"Group summary CSV: {relative_path(group_summary_csv)}")
    print(f"Group summary Markdown: {relative_path(group_summary_md)}")


if __name__ == "__main__":
    main()
