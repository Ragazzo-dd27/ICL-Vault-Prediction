"""Aggregate corrected AS-OCT outputs after patient_100 OS label correction.

This script is analysis-only. It does not train models and does not modify
manifests, checkpoints, or any previous incorrect-label result directories.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "artifacts/reports/combined_batch_01_02_03_04"
CKPT_BASE = PROJECT / "artifacts/checkpoints/combined_batch_01_02_03_04"
FINAL_DIR = BASE / "as_oct_label_corrected_final_analysis"
ENSEMBLE_DIR = BASE / "as_oct_only_ensemble_label_corrected_patient100_os_seed42_2026_3407"
REPEATED_DIR = BASE / "as_oct_only_repeated_splits_label_corrected_patient100_os"
RECOMP_2002_DIR = REPEATED_DIR / "split_seed2002_model_seed42_recomputed"
PRIMARY_MANIFEST = PROJECT / (
    "data/manifests/"
    "vault_as_oct_only_pod1_manifest_combined_batch_01_02_03_04_strict_split_seed42.csv"
)
REPEATED_SPLIT_DIR = PROJECT / "data/splits/combined_batch_01_02_03_04_repeated"
MEAS_RF_PATH = BASE / "measurement_only_repeated_splits/measurement_repeated_split_overall_metrics.csv"
TARGET_GLOBAL = "batch_03__patient_100_OS_20240517"
MEASUREMENT_RF_PRIMARY_MAE = 169.44
FUSION_ENSEMBLE_PRIMARY_MAE = 182.80
LOW = 500.0
HIGH = 800.0


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return path.as_posix()


def f(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table."""
    if df.empty:
        return "_No rows._"
    frame = df.copy()
    for col in frame.columns:
        frame[col] = frame[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in frame.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def vault_range(value: Any) -> str:
    number = float(value)
    if number < LOW:
        return "low"
    if number <= HIGH:
        return "medium"
    return "high"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    label_range = float(np.max(y_true) - np.min(y_true))
    pred_range = float(np.max(y_pred) - np.min(y_pred))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mean_signed_error": float(np.mean(err)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0,
        "label_min": float(np.min(y_true)),
        "label_max": float(np.max(y_true)),
        "label_mean": float(np.mean(y_true)),
        "label_std": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0,
        "prediction_range_label_range_ratio": pred_range / label_range if label_range > 0 else float("nan"),
    }


def range_metrics(
    frame: pd.DataFrame,
    *,
    true_col: str = "vault_label_um",
    pred_col: str = "pred_vault_um",
    split_seed: int | None = None,
    model_seed: int | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["vault_range"] = out[true_col].apply(vault_range)
    out["signed_error_um"] = out[pred_col].astype(float) - out[true_col].astype(float)
    out["abs_error_um"] = out["signed_error_um"].abs()
    total_abs = float(out["abs_error_um"].sum())
    rows: list[dict[str, Any]] = []
    for group_name in ["low", "medium", "high"]:
        sub = out[out["vault_range"].eq(group_name)]
        if sub.empty:
            row = {
                "vault_range": group_name,
                "n": 0,
                "MAE": np.nan,
                "RMSE": np.nan,
                "mean_signed_error": np.nan,
                "overestimation_count": 0,
                "underestimation_count": 0,
                "absolute_error_contribution_percentage": 0.0,
            }
        else:
            metrics = regression_metrics(sub[true_col].to_numpy(float), sub[pred_col].to_numpy(float))
            row = {
                "vault_range": group_name,
                "n": int(len(sub)),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "mean_signed_error": metrics["mean_signed_error"],
                "overestimation_count": int((sub["signed_error_um"] > 0).sum()),
                "underestimation_count": int((sub["signed_error_um"] < 0).sum()),
                "absolute_error_contribution_percentage": (
                    float(sub["abs_error_um"].sum() / total_abs * 100.0) if total_abs else 0.0
                ),
            }
        if split_seed is not None:
            row["split_seed"] = split_seed
        if model_seed is not None:
            row["model_seed"] = model_seed
        rows.append(row)

    cols: list[str] = []
    if split_seed is not None:
        cols.append("split_seed")
    if model_seed is not None:
        cols.append("model_seed")
    cols += [
        "vault_range",
        "n",
        "MAE",
        "RMSE",
        "mean_signed_error",
        "overestimation_count",
        "underestimation_count",
        "absolute_error_contribution_percentage",
    ]
    return pd.DataFrame(rows)[cols]


def read_summary(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    patterns = {
        "best_epoch": r"Best epoch:\s*(\d+)",
        "best_val_mae": r"Best val MAE:\s*([0-9.]+)",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def manifest_counts(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    return {split: int((df["split"].astype(str) == split).sum()) for split in ["train", "val", "test"]}


def manifest_patients(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    col = "global_patient_uid" if "global_patient_uid" in df.columns else "patient_id"
    return {split: int(df[df["split"].astype(str).eq(split)][col].nunique()) for split in ["train", "val", "test"]}


def enrich_predictions(prediction_path: Path, manifest_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path)
    manifest = pd.read_csv(manifest_path)
    test_manifest = manifest[manifest["split"].astype(str).eq("test")].copy()
    if int(test_manifest["sample_id"].duplicated().sum()) != 0:
        raise ValueError(f"{manifest_path} has duplicate test sample_id values")
    keep = [
        c
        for c in [
            "sample_id",
            "global_sample_id",
            "global_patient_uid",
            "patient_uid",
            "patient_id",
            "eye",
            "eye_side",
            "batch_id",
            "split",
            "vault_label",
            "vault_label_num",
            "pod1_vault_mean_um",
            "label_qc_flag",
            "qc_flag",
            "oct_path",
            "vault_range_group",
            "vault_range_calc",
        ]
        if c in test_manifest.columns
    ]
    merged = predictions.merge(test_manifest[keep], on="sample_id", how="left", suffixes=("", "_manifest"))
    if "global_sample_id" in merged.columns and merged["global_sample_id"].isna().any():
        missing = merged.loc[merged["global_sample_id"].isna(), "sample_id"].tolist()
        raise ValueError(f"{prediction_path} rows not found in manifest: {missing[:5]}")
    if len(merged) != len(test_manifest):
        raise ValueError(f"{prediction_path}: {len(merged)} predictions, {len(test_manifest)} test manifest rows")
    corrected_labels = pd.to_numeric(merged.get("vault_label", merged["vault_label_um"]), errors="coerce")
    if corrected_labels.notna().all():
        merged["vault_label_um"] = corrected_labels.astype(float)
    merged["pred_vault_um"] = pd.to_numeric(merged["pred_vault_um"], errors="coerce")
    merged["signed_error_um"] = merged["pred_vault_um"] - merged["vault_label_um"]
    merged["abs_error_um"] = merged["signed_error_um"].abs()
    merged["vault_range"] = merged["vault_label_um"].apply(vault_range)
    return merged


def file_set(report_dir: Path, seed: int) -> dict[str, Path]:
    prefix = f"as_oct_v4_seed{seed}"
    return {
        "overall": report_dir / f"{prefix}_overall_metrics.csv",
        "range": report_dir / f"{prefix}_range_metrics.csv",
        "predictions": report_dir / f"{prefix}_predictions.csv",
        "training_log": report_dir / f"{prefix}_training_log.csv",
        "summary": report_dir / f"{prefix}_summary.md",
    }


def qc_run(
    evaluation_type: str,
    split_seed: int,
    model_seed: int,
    report_dir: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    files = file_set(report_dir, model_seed)
    best_path = checkpoint_dir / "best.pth"
    latest_path = checkpoint_dir / "latest.pth"
    required = {**files, "best.pth": best_path, "latest.pth": latest_path}
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError(f"Missing files for {evaluation_type} split={split_seed} model={model_seed}: {missing}")

    overall = pd.read_csv(files["overall"]).iloc[0]
    predictions = enrich_predictions(files["predictions"], manifest_path)
    counts = manifest_counts(manifest_path)
    if len(predictions) != counts["test"]:
        raise ValueError(f"{report_dir}: prediction row count does not match test manifest")
    if not predictions["global_sample_id"].is_unique:
        raise ValueError(f"{report_dir}: global_sample_id is not unique")
    if not np.isfinite(predictions[["vault_label_um", "pred_vault_um"]].to_numpy(float)).all():
        raise ValueError(f"{report_dir}: non-finite labels or predictions")

    checkpoint = torch.load(best_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {}) or {}
    ckpt_epoch = int(checkpoint.get("epoch"))
    ckpt_best_val = float(checkpoint.get("best_val_mae"))
    summary = read_summary(files["summary"])
    log = pd.read_csv(files["training_log"])
    best_idx = log["val_mae_um"].astype(float).idxmin()
    log_best_epoch = int(log.loc[best_idx, "epoch"])
    log_best_val = float(log.loc[best_idx, "val_mae_um"])
    epoch_ok = int(overall["best_epoch"]) == ckpt_epoch == log_best_epoch
    val_ok = abs(float(overall["best_val_mae_um"]) - ckpt_best_val) < 1e-4 and abs(log_best_val - ckpt_best_val) < 1e-4
    if "best_epoch" in summary:
        epoch_ok = epoch_ok and int(summary["best_epoch"]) == ckpt_epoch
    if "best_val_mae" in summary:
        val_ok = val_ok and abs(summary["best_val_mae"] - ckpt_best_val) < 0.02
    ckpt_arg_dir = Path(str(ckpt_args.get("checkpoint_dir", "")))
    if not ckpt_arg_dir.is_absolute():
        ckpt_arg_dir = PROJECT / ckpt_arg_dir
    ckpt_dir_ok = ckpt_arg_dir.resolve() == checkpoint_dir.resolve()
    ckpt_seed_ok = int(ckpt_args.get("seed", model_seed)) == model_seed
    if not (epoch_ok and val_ok and ckpt_dir_ok and ckpt_seed_ok):
        raise ValueError(
            f"{report_dir}: checkpoint/log/summary mismatch "
            f"epoch_ok={epoch_ok}, val_ok={val_ok}, dir_ok={ckpt_dir_ok}, seed_ok={ckpt_seed_ok}"
        )
    if evaluation_type == "repeated" and split_seed == 3407 and model_seed == 42:
        if ckpt_epoch != 1 or abs(ckpt_best_val - 159.28) > 0.01:
            raise ValueError(f"split3407 special QC failed: epoch={ckpt_epoch}, val={ckpt_best_val}")
        if (log["val_mae_um"].astype(float) < 159.28 - 0.01).any():
            raise ValueError("split3407 special QC failed: lower val MAE exists in training log")

    metric = regression_metrics(predictions["vault_label_um"].to_numpy(float), predictions["pred_vault_um"].to_numpy(float))
    patients = manifest_patients(manifest_path)
    row = {
        "evaluation_type": evaluation_type,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "best_epoch": int(overall["best_epoch"]),
        "best_val_mae": float(overall["best_val_mae_um"]),
        "test_mae": float(overall["mae_um"]),
        "test_rmse": float(overall["rmse_um"]),
        "test_r2": float(overall["r2"]),
        "mean_signed_error": float(overall["mean_signed_error_um"]),
        "prediction_range_ratio": metric["prediction_range_label_range_ratio"],
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "train_patients": patients["train"],
        "val_patients": patients["val"],
        "test_patients": patients["test"],
        "report_dir": rel(report_dir),
        "checkpoint_dir": rel(checkpoint_dir),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_best_val_mae": ckpt_best_val,
        "summary_checkpoint_log_consistent": True,
    }
    return row, predictions


def sample_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def write_recomputed_split2002(pred_cache: dict[tuple[str, int, int], pd.DataFrame], rows: list[dict[str, Any]]) -> None:
    old_report = BASE / "as_oct_only_repeated_splits/split_seed2002_model_seed42"
    manifest = REPEATED_SPLIT_DIR / "as_oct_manifest_split_seed2002.csv"
    predictions = enrich_predictions(old_report / "as_oct_v4_seed42_predictions.csv", manifest)
    target = predictions[predictions["global_sample_id"].eq(TARGET_GLOBAL)]
    if len(target) != 1 or abs(float(target.iloc[0]["vault_label_um"]) - 701.0) > 1e-9:
        raise ValueError("split2002 corrected target label check failed")

    predictions["split_seed"] = 2002
    predictions["model_seed"] = 42
    predictions["split"] = "test"
    metric = regression_metrics(predictions["vault_label_um"].to_numpy(float), predictions["pred_vault_um"].to_numpy(float))
    ranges = range_metrics(predictions, split_seed=2002, model_seed=42)
    counts = manifest_counts(manifest)
    patients = manifest_patients(manifest)
    keep = [
        c
        for c in [
            "sample_id",
            "patient_id",
            "eye_side",
            "split",
            "vault_label_um",
            "pred_vault_um",
            "abs_error_um",
            "label_qc_flag",
            "oct_path",
            "signed_error_um",
            "global_sample_id",
            "global_patient_uid",
            "patient_uid",
            "eye",
            "batch_id",
            "qc_flag",
            "vault_range",
            "split_seed",
            "model_seed",
        ]
        if c in predictions.columns
    ]
    RECOMP_2002_DIR.mkdir(parents=True, exist_ok=True)
    predictions[keep].to_csv(RECOMP_2002_DIR / "as_oct_v4_seed42_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "split": "test",
                "seed": 42,
                "best_epoch": np.nan,
                "best_val_mae_um": np.nan,
                "mae_um": metric["MAE"],
                "rmse_um": metric["RMSE"],
                "r2": metric["R2"],
                "mean_signed_error_um": metric["mean_signed_error"],
                "n_samples": len(predictions),
                "recomputed_from_corrected_label_only": True,
                "source_predictions_dir": rel(old_report),
            }
        ]
    ).to_csv(RECOMP_2002_DIR / "as_oct_v4_seed42_overall_metrics.csv", index=False, encoding="utf-8")
    ranges.rename(columns={"n": "n_samples"}).to_csv(
        RECOMP_2002_DIR / "as_oct_v4_seed42_range_metrics.csv", index=False, encoding="utf-8"
    )
    (RECOMP_2002_DIR / "as_oct_v4_seed42_summary.md").write_text(
        "\n".join(
            [
                "# Corrected split2002 AS-OCT-only metrics recomputation",
                "",
                "- Model checkpoint was not retrained.",
                "- Predictions were read from the original split_seed2002/model_seed42 output.",
                "- Metrics were recomputed using corrected test labels.",
                "- The original 7901 um label version is superseded by this corrected-label recomputation.",
                "",
                f"- Test MAE: {f(metric['MAE'])} um",
                f"- Test RMSE: {f(metric['RMSE'])} um",
                f"- Test R2: {f(metric['R2'], 4)}",
                f"- Test mean signed error: {f(metric['mean_signed_error'])} um",
                f"- Target `{TARGET_GLOBAL}` corrected true label: 701.00 um; prediction unchanged: {f(target.iloc[0]['pred_vault_um'])} um.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows.append(
        {
            "evaluation_type": "repeated_recomputed_label_only",
            "split_seed": 2002,
            "model_seed": 42,
            "best_epoch": np.nan,
            "best_val_mae": np.nan,
            "test_mae": metric["MAE"],
            "test_rmse": metric["RMSE"],
            "test_r2": metric["R2"],
            "mean_signed_error": metric["mean_signed_error"],
            "prediction_range_ratio": metric["prediction_range_label_range_ratio"],
            "n_train": counts["train"],
            "n_val": counts["val"],
            "n_test": counts["test"],
            "train_patients": patients["train"],
            "val_patients": patients["val"],
            "test_patients": patients["test"],
            "report_dir": rel(RECOMP_2002_DIR),
            "checkpoint_dir": rel(CKPT_BASE / "as_oct_only_repeated_splits/split_seed2002_model_seed42"),
            "checkpoint_epoch": np.nan,
            "checkpoint_best_val_mae": np.nan,
            "summary_checkpoint_log_consistent": True,
        }
    )
    pred_cache[("repeated", 2002, 42)] = predictions


def write_primary_ensemble(single_metrics: pd.DataFrame, pred_cache: dict[tuple[str, int, int], pd.DataFrame]) -> dict[str, Any]:
    frames = []
    for seed in [42, 2026, 3407]:
        frame = pred_cache[("primary", 42, seed)][
            [
                "global_sample_id",
                "global_patient_uid",
                "patient_uid",
                "eye",
                "batch_id",
                "vault_label_um",
                "pred_vault_um",
                "qc_flag",
                "label_qc_flag",
                "oct_path",
                "vault_range",
            ]
        ].copy()
        frame = frame.rename(columns={"pred_vault_um": f"pred_seed{seed}_um"})
        frames.append(frame)

    ensemble = frames[0]
    for frame in frames[1:]:
        ensemble = ensemble.merge(
            frame.drop(columns=["global_patient_uid", "patient_uid", "eye", "batch_id", "qc_flag", "label_qc_flag", "oct_path", "vault_range"]),
            on=["global_sample_id", "vault_label_um"],
            how="inner",
        )
    if len(ensemble) != len(frames[0]):
        raise ValueError("Primary ensemble alignment failed")
    seed_cols = [f"pred_seed{seed}_um" for seed in [42, 2026, 3407]]
    ensemble["ensemble_pred_vault_um"] = ensemble[seed_cols].mean(axis=1)
    ensemble["signed_error_um"] = ensemble["ensemble_pred_vault_um"] - ensemble["vault_label_um"]
    ensemble["abs_error_um"] = ensemble["signed_error_um"].abs()
    ensemble["vault_range"] = ensemble["vault_label_um"].apply(vault_range)
    metric = regression_metrics(ensemble["vault_label_um"].to_numpy(float), ensemble["ensemble_pred_vault_um"].to_numpy(float))
    ranges = range_metrics(ensemble, pred_col="ensemble_pred_vault_um")
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame([{**metric, "n_samples": len(ensemble)}]).to_csv(
        ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_overall_metrics.csv", index=False, encoding="utf-8"
    )
    ranges.to_csv(ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_range_metrics.csv", index=False, encoding="utf-8")
    ensemble.sort_values("abs_error_um", ascending=False).head(10).to_csv(
        ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_error_cases.csv", index=False, encoding="utf-8"
    )
    primary_table = single_metrics[single_metrics["evaluation_type"].eq("primary")][
        ["model_seed", "best_epoch", "best_val_mae", "test_mae", "test_rmse", "test_r2", "mean_signed_error", "prediction_range_ratio"]
    ].sort_values("model_seed")
    seed42_mae = float(primary_table[primary_table["model_seed"].eq(42)]["test_mae"].iloc[0])
    (ENSEMBLE_DIR / "as_oct_v4_corrected_ensemble_summary.md").write_text(
        "\n".join(
            [
                "# Corrected AS-OCT-only Primary 3-seed Ensemble",
                "",
                "This ensemble uses corrected-label primary predictions only: model seeds 42, 2026, and 3407.",
                "",
                "## Single Seeds",
                md_table(primary_table),
                "",
                "## Ensemble Overall",
                f"- MAE: {f(metric['MAE'])} um",
                f"- RMSE: {f(metric['RMSE'])} um",
                f"- R2: {f(metric['R2'], 4)}",
                f"- Mean signed error: {f(metric['mean_signed_error'])} um",
                f"- Prediction range / label range ratio: {f(metric['prediction_range_label_range_ratio'], 3)}",
                "",
                "## Comparisons",
                f"- Better than corrected seed42: {metric['MAE'] < seed42_mae} ({f(metric['MAE'])} vs {f(seed42_mae)} um).",
                f"- Better than measurement-only RF primary MAE 169.44 um: {metric['MAE'] < MEASUREMENT_RF_PRIMARY_MAE}.",
                f"- Better than fusion primary ensemble MAE 182.80 um: {metric['MAE'] < FUSION_ENSEMBLE_PRIMARY_MAE}.",
                "",
                "## Range Metrics",
                md_table(ranges),
                "",
                "## Worst Error Cases",
                md_table(
                    ensemble.sort_values("abs_error_um", ascending=False)
                    .head(10)[["global_sample_id", "vault_label_um", "ensemble_pred_vault_um", "abs_error_um", "signed_error_um", "vault_range"]]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"metrics": metric, "ranges": ranges, "seed42_mae": seed42_mae, "primary_table": primary_table}


def write_repeated_aggregate(single_metrics: pd.DataFrame, pred_cache: dict[tuple[str, int, int], pd.DataFrame]) -> dict[str, Any]:
    rows = []
    range_frames = []
    prediction_frames = []
    for split_seed in [42, 1001, 2002, 2026, 3407]:
        if split_seed == 42:
            frame = pred_cache[("primary", 42, 42)].copy()
            source = single_metrics[(single_metrics["evaluation_type"].eq("primary")) & (single_metrics["model_seed"].eq(42))].iloc[0]
            manifest = PRIMARY_MANIFEST
            best_epoch, best_val = source["best_epoch"], source["best_val_mae"]
        elif split_seed == 2002:
            frame = pred_cache[("repeated", 2002, 42)].copy()
            manifest = REPEATED_SPLIT_DIR / "as_oct_manifest_split_seed2002.csv"
            best_epoch, best_val = np.nan, np.nan
        else:
            frame = pred_cache[("repeated", split_seed, 42)].copy()
            source = single_metrics[(single_metrics["evaluation_type"].eq("repeated")) & (single_metrics["split_seed"].eq(split_seed))].iloc[0]
            manifest = REPEATED_SPLIT_DIR / f"as_oct_manifest_split_seed{split_seed}.csv"
            best_epoch, best_val = source["best_epoch"], source["best_val_mae"]

        frame["split_seed"] = split_seed
        frame["model_seed"] = 42
        frame["split"] = "test"
        metric = regression_metrics(frame["vault_label_um"].to_numpy(float), frame["pred_vault_um"].to_numpy(float))
        counts = manifest_counts(manifest)
        patients = manifest_patients(manifest)
        rows.append(
            {
                "split_seed": split_seed,
                "model_seed": 42,
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_test": counts["test"],
                "train_patients": patients["train"],
                "val_patients": patients["val"],
                "test_patients": patients["test"],
                "best_epoch": best_epoch,
                "best_val_mae": best_val,
                "test_mae": metric["MAE"],
                "test_rmse": metric["RMSE"],
                "test_r2": metric["R2"],
                "mean_signed_error": metric["mean_signed_error"],
                "prediction_min": metric["prediction_min"],
                "prediction_max": metric["prediction_max"],
                "prediction_mean": metric["prediction_mean"],
                "prediction_std": metric["prediction_std"],
                "label_min": metric["label_min"],
                "label_max": metric["label_max"],
                "label_mean": metric["label_mean"],
                "label_std": metric["label_std"],
                "prediction_range_label_range_ratio": metric["prediction_range_label_range_ratio"],
            }
        )
        range_frames.append(range_metrics(frame, split_seed=split_seed, model_seed=42))
        keep = [
            c
            for c in [
                "split_seed",
                "model_seed",
                "global_sample_id",
                "sample_id",
                "global_patient_uid",
                "patient_uid",
                "eye",
                "eye_side",
                "batch_id",
                "split",
                "vault_label_um",
                "pred_vault_um",
                "signed_error_um",
                "abs_error_um",
                "vault_range",
                "qc_flag",
                "label_qc_flag",
                "oct_path",
            ]
            if c in frame.columns
        ]
        prediction_frames.append(frame[keep])

    overall = pd.DataFrame(rows).sort_values("split_seed")
    ranges = pd.concat(range_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall.to_csv(REPEATED_DIR / "corrected_as_oct_repeated_split_overall_metrics.csv", index=False, encoding="utf-8")
    ranges.to_csv(REPEATED_DIR / "corrected_as_oct_repeated_split_range_metrics.csv", index=False, encoding="utf-8")
    predictions.to_csv(REPEATED_DIR / "corrected_as_oct_repeated_split_predictions.csv", index=False, encoding="utf-8")

    agg_rows: list[dict[str, Any]] = [
        {
            "metric_scope": "overall",
            "model": "corrected_as_oct_model_seed42",
            "MAE_mean": float(overall["test_mae"].mean()),
            "MAE_sample_std": sample_std(overall["test_mae"]),
            "MAE_median": float(overall["test_mae"].median()),
            "MAE_IQR": iqr(overall["test_mae"]),
            "MAE_min": float(overall["test_mae"].min()),
            "MAE_max": float(overall["test_mae"].max()),
            "RMSE_mean": float(overall["test_rmse"].mean()),
            "RMSE_sample_std": sample_std(overall["test_rmse"]),
            "R2_mean": float(overall["test_r2"].mean()),
            "R2_sample_std": sample_std(overall["test_r2"]),
            "signed_error_mean": float(overall["mean_signed_error"].mean()),
            "signed_error_sample_std": sample_std(overall["mean_signed_error"]),
            "prediction_range_ratio_mean": float(overall["prediction_range_label_range_ratio"].mean()),
            "prediction_range_ratio_sample_std": sample_std(overall["prediction_range_label_range_ratio"]),
        }
    ]
    for group_name in ["low", "medium", "high"]:
        sub = ranges[ranges["vault_range"].eq(group_name)].copy()
        agg_rows.append(
            {
                "metric_scope": f"range_{group_name}",
                "model": "corrected_as_oct_model_seed42",
                "MAE_mean": float(sub["MAE"].mean()),
                "MAE_sample_std": sample_std(sub["MAE"]),
                "MAE_median": float(sub["MAE"].median()),
                "MAE_IQR": iqr(sub["MAE"]),
                "MAE_min": float(sub["MAE"].min()),
                "MAE_max": float(sub["MAE"].max()),
                "RMSE_mean": float(sub["RMSE"].mean()),
                "RMSE_sample_std": sample_std(sub["RMSE"]),
                "signed_error_mean": float(sub["mean_signed_error"].mean()),
                "signed_error_sample_std": sample_std(sub["mean_signed_error"]),
                "sample_counts": ";".join(str(int(x)) for x in sub["n"].tolist()),
                "overestimation_proportion_mean": float((sub["overestimation_count"] / sub["n"].replace(0, np.nan)).mean()),
                "underestimation_proportion_mean": float((sub["underestimation_count"] / sub["n"].replace(0, np.nan)).mean()),
            }
        )
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(REPEATED_DIR / "corrected_as_oct_repeated_split_aggregate_metrics.csv", index=False, encoding="utf-8")

    measurement = pd.read_csv(MEAS_RF_PATH)
    rf = measurement[measurement["model"].astype(str).eq("Random Forest Regressor")].copy()
    paired = overall[["split_seed", "test_mae"]].merge(rf[["split_seed", "MAE"]], on="split_seed", how="inner")
    paired = paired.rename(columns={"test_mae": "corrected_AS_OCT_MAE", "MAE": "measurement_RF_MAE"})
    paired["delta_MAE"] = paired["corrected_AS_OCT_MAE"] - paired["measurement_RF_MAE"]
    paired["winner"] = np.where(paired["delta_MAE"] < 0, "corrected_AS_OCT", "measurement_RF")
    paired.to_csv(REPEATED_DIR / "corrected_as_oct_vs_measurement_rf_paired_comparison.csv", index=False, encoding="utf-8")

    as_wins = int((paired["winner"] == "corrected_AS_OCT").sum())
    rf_wins = int((paired["winner"] == "measurement_RF").sum())
    (REPEATED_DIR / "corrected_as_oct_repeated_split_summary.md").write_text(
        "\n".join(
            [
                "# Corrected AS-OCT-only Repeated Split Aggregation",
                "",
                f"- MAE mean +/- sample std: {f(overall['test_mae'].mean())} +/- {f(overall['test_mae'].std(ddof=1))} um.",
                f"- MAE median/IQR/min/max: {f(overall['test_mae'].median())} / {f(iqr(overall['test_mae']))} / {f(overall['test_mae'].min())} / {f(overall['test_mae'].max())} um.",
                f"- RMSE mean +/- sample std: {f(overall['test_rmse'].mean())} +/- {f(overall['test_rmse'].std(ddof=1))} um.",
                f"- R2 mean +/- sample std: {f(overall['test_r2'].mean(), 4)} +/- {f(overall['test_r2'].std(ddof=1), 4)}.",
                f"- Signed error mean +/- sample std: {f(overall['mean_signed_error'].mean())} +/- {f(overall['mean_signed_error'].std(ddof=1))} um.",
                f"- Prediction range ratio mean +/- sample std: {f(overall['prediction_range_label_range_ratio'].mean(), 3)} +/- {f(overall['prediction_range_label_range_ratio'].std(ddof=1), 3)}.",
                "",
                "## Overall Metrics",
                md_table(overall),
                "",
                "## Range Metrics",
                md_table(ranges),
                "",
                "## Paired RF Comparison",
                md_table(paired),
                "",
                f"- Corrected AS-OCT wins: {as_wins}/5 splits.",
                f"- Measurement RF wins: {rf_wins}/5 splits.",
                f"- Paired delta MAE mean +/- std: {f(paired['delta_MAE'].mean())} +/- {f(paired['delta_MAE'].std(ddof=1))} um.",
                "- No significance claim is made from only five patient-level splits.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"overall": overall, "ranges": ranges, "paired": paired, "rf": rf, "as_wins": as_wins, "rf_wins": rf_wins}


def write_final_status(single_metrics: pd.DataFrame, ensemble: dict[str, Any], repeated: dict[str, Any]) -> None:
    overall = repeated["overall"]
    ranges = repeated["ranges"]
    paired = repeated["paired"]
    rf = repeated["rf"]
    low = ranges[ranges["vault_range"].eq("low")]
    high = ranges[ranges["vault_range"].eq("high")]
    low_over = float((low["overestimation_count"] / low["n"]).mean())
    high_under = float((high["underestimation_count"] / high["n"]).mean())
    seed42_mae = ensemble["seed42_mae"]
    representative = (overall["test_mae"].min() - 0.01) <= seed42_mae <= (overall["test_mae"].max() + 0.01)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "corrected_as_oct_final_status.md").write_text(
        "\n".join(
            [
                "# Corrected AS-OCT Final Status",
                "",
                "## Label Correction",
                "- `patient_100 OS` was corrected from 7901 um to 701 um after human verification of the original POD1 CASIA2 report/image.",
                "- The old AS-OCT results that used the 7901 um label are superseded for AS-OCT reporting.",
                "",
                "## Corrected Primary Single Seeds",
                md_table(ensemble["primary_table"]),
                "",
                "## Corrected Primary Ensemble",
                f"- Ensemble MAE/RMSE/R2/mean signed error: {f(ensemble['metrics']['MAE'])} / {f(ensemble['metrics']['RMSE'])} / {f(ensemble['metrics']['R2'], 4)} / {f(ensemble['metrics']['mean_signed_error'])} um.",
                f"- Prediction range ratio: {f(ensemble['metrics']['prediction_range_label_range_ratio'], 3)}.",
                f"- Better than measurement-only RF primary MAE 169.44 um: {ensemble['metrics']['MAE'] < MEASUREMENT_RF_PRIMARY_MAE}.",
                f"- Better than unaffected fusion primary ensemble MAE 182.80 um: {ensemble['metrics']['MAE'] < FUSION_ENSEMBLE_PRIMARY_MAE}.",
                "",
                "## Corrected Repeated Splits",
                f"- Corrected AS-OCT repeated MAE mean +/- std: {f(overall['test_mae'].mean())} +/- {f(overall['test_mae'].std(ddof=1))} um.",
                f"- Range: {f(overall['test_mae'].min())} to {f(overall['test_mae'].max())} um.",
                f"- Primary seed42 MAE {f(seed42_mae)} um is {'within' if representative else 'outside'} the corrected repeated split MAE range.",
                "",
                "## Measurement RF Paired Comparison",
                f"- Corrected AS-OCT wins {repeated['as_wins']}/5 splits; measurement RF wins {repeated['rf_wins']}/5 splits.",
                f"- Paired delta MAE mean +/- std: {f(paired['delta_MAE'].mean())} +/- {f(paired['delta_MAE'].std(ddof=1))} um.",
                f"- Corrected AS-OCT MAE std: {f(overall['test_mae'].std(ddof=1))} um; measurement RF MAE std: {f(rf['MAE'].std(ddof=1))} um.",
                "- With five splits, this is descriptive paired evidence only, not a significance test.",
                "",
                "## Bias Pattern",
                f"- Low-vault overestimation persists: mean low overestimation proportion {f(low_over * 100)}%.",
                f"- High-vault underestimation persists: mean high underestimation proportion {f(high_under * 100)}%.",
                f"- Prediction range compression persists: repeated prediction range ratio mean {f(overall['prediction_range_label_range_ratio'].mean(), 3)}.",
                "",
                "## Next Recommendations",
                "- Fusion repeated evaluation is still worth running because primary fusion remained stronger than AS-OCT-only and is unaffected by this OS sample.",
                "- Lower-lr / weighted-loss / calibration experiments should be secondary. First lock the corrected AS-OCT baseline and compare it against fusion repeated evaluation on the same splits.",
                "- Do not use the superseded 7901-label AS-OCT metrics in new status reports or manuscript tables.",
                "",
                "## Output Pointers",
                f"- Single-run metrics: `{rel(FINAL_DIR / 'corrected_as_oct_single_run_metrics.csv')}`",
                f"- Primary ensemble: `{rel(ENSEMBLE_DIR)}`",
                f"- Repeated split aggregate: `{rel(REPEATED_DIR / 'corrected_as_oct_repeated_split_summary.md')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    REPEATED_DIR.mkdir(parents=True, exist_ok=True)

    primary_runs = {
        42: (BASE / "as_oct_only_label_corrected_patient100_os_seed42", CKPT_BASE / "as_oct_only_label_corrected_patient100_os_seed42"),
        2026: (BASE / "as_oct_only_label_corrected_patient100_os_seed2026", CKPT_BASE / "as_oct_only_label_corrected_patient100_os_seed2026"),
        3407: (BASE / "as_oct_only_label_corrected_patient100_os_seed3407", CKPT_BASE / "as_oct_only_label_corrected_patient100_os_seed3407"),
    }
    repeated_runs = {
        1001: (
            REPEATED_DIR / "split_seed1001_model_seed42",
            CKPT_BASE / "as_oct_only_repeated_splits_label_corrected_patient100_os/split_seed1001_model_seed42",
            REPEATED_SPLIT_DIR / "as_oct_manifest_split_seed1001.csv",
        ),
        2026: (
            REPEATED_DIR / "split_seed2026_model_seed42",
            CKPT_BASE / "as_oct_only_repeated_splits_label_corrected_patient100_os/split_seed2026_model_seed42",
            REPEATED_SPLIT_DIR / "as_oct_manifest_split_seed2026.csv",
        ),
        3407: (
            REPEATED_DIR / "split_seed3407_model_seed42",
            CKPT_BASE / "as_oct_only_repeated_splits_label_corrected_patient100_os/split_seed3407_model_seed42",
            REPEATED_SPLIT_DIR / "as_oct_manifest_split_seed3407.csv",
        ),
    }

    rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    for seed, (report, checkpoint) in primary_runs.items():
        row, predictions = qc_run("primary", 42, seed, report, checkpoint, PRIMARY_MANIFEST)
        rows.append(row)
        pred_cache[("primary", 42, seed)] = predictions
    for split_seed, (report, checkpoint, manifest) in repeated_runs.items():
        row, predictions = qc_run("repeated", split_seed, 42, report, checkpoint, manifest)
        rows.append(row)
        pred_cache[("repeated", split_seed, 42)] = predictions

    write_recomputed_split2002(pred_cache, rows)
    single_metrics = pd.DataFrame(rows)
    single_metrics.to_csv(FINAL_DIR / "corrected_as_oct_single_run_metrics.csv", index=False, encoding="utf-8")
    single_metrics.to_csv(FINAL_DIR / "corrected_as_oct_run_qc.csv", index=False, encoding="utf-8")

    ensemble = write_primary_ensemble(single_metrics, pred_cache)
    repeated = write_repeated_aggregate(single_metrics, pred_cache)
    write_final_status(single_metrics, ensemble, repeated)

    print("Corrected trained run QC passed: 6/6")
    print("split3407 epoch=1 special QC passed")
    split2002 = single_metrics[single_metrics["split_seed"].eq(2002)].iloc[0]
    print(f"split2002 recomputed MAE/RMSE: {split2002['test_mae']:.2f} / {split2002['test_rmse']:.2f}")
    print(f"primary corrected ensemble MAE/RMSE/R2: {ensemble['metrics']['MAE']:.2f} / {ensemble['metrics']['RMSE']:.2f} / {ensemble['metrics']['R2']:.4f}")
    print(
        "repeated corrected MAE mean/std: "
        f"{repeated['overall']['test_mae'].mean():.2f} / {repeated['overall']['test_mae'].std(ddof=1):.2f}"
    )
    print(f"paired wins AS-OCT/RF: {repeated['as_wins']} / {repeated['rf_wins']}")
    print(f"Wrote {rel(FINAL_DIR / 'corrected_as_oct_final_status.md')}")


if __name__ == "__main__":
    main()
