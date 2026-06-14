"""Aggregate combined v4 fusion repeated split results.

This script is analysis-only. It does not train models.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "artifacts/reports/combined_batch_01_02_03_04"
OUT_DIR = BASE / "fusion_repeated_splits_fixed_model_seed42"
SPLIT_DIR = PROJECT / "data/splits/combined_batch_01_02_03_04_repeated"
PRIMARY_REPORT = BASE / "fusion_baseline_seed42"
MEASUREMENT_OVERALL = BASE / "measurement_only_repeated_splits/measurement_repeated_split_overall_metrics.csv"
CORRECTED_AS_OCT_OVERALL = (
    BASE
    / "as_oct_only_repeated_splits_label_corrected_patient100_os/corrected_as_oct_repeated_split_overall_metrics.csv"
)
FEATURE_COLUMNS = ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]
LOW = 500.0
HIGH = 800.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fusion repeated split outputs.")
    parser.add_argument("--split_seeds", default="42,1001,2002,2026,3407")
    parser.add_argument("--model_seed", type=int, default=42)
    return parser.parse_args()


def split_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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
    if df.empty:
        return "_No rows._"
    frame = df.copy()
    for col in frame.columns:
        frame[col] = frame[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in frame.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def vault_range(value: float) -> str:
    if value < LOW:
        return "low"
    if value <= HIGH:
        return "medium"
    return "high"


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = pd.to_numeric(y_true, errors="coerce").astype(float)
    y_pred = pd.to_numeric(y_pred, errors="coerce").astype(float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    label_range = float(y_true.max() - y_true.min())
    pred_range = float(y_pred.max() - y_pred.min())
    return {
        "test_mae": float(err.abs().mean()),
        "test_rmse": float(np.sqrt(np.mean(err**2))),
        "test_r2": float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot),
        "mean_signed_error": float(err.mean()),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
        "prediction_mean": float(y_pred.mean()),
        "prediction_std": float(y_pred.std(ddof=1)),
        "label_min": float(y_true.min()),
        "label_max": float(y_true.max()),
        "label_mean": float(y_true.mean()),
        "label_std": float(y_true.std(ddof=1)),
        "prediction_range_label_range_ratio": pred_range / label_range if label_range > 0 else float("nan"),
    }


def manifest_for(seed: int) -> Path:
    return SPLIT_DIR / f"fusion_manifest_split_seed{seed}.csv"


def report_dir_for(seed: int, model_seed: int) -> Path:
    if seed == 42:
        return PRIMARY_REPORT
    return OUT_DIR / f"split_seed{seed}_model_seed{model_seed}"


def prefix_for(model_seed: int) -> str:
    return f"fusion_v4_seed{model_seed}"


def read_manifest_qc(seed: int) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    path = manifest_for(seed)
    df = pd.read_csv(path)
    leakage = int((df.groupby("global_patient_uid")["split"].nunique() > 1).sum())
    duplicate = int(df["global_sample_id"].duplicated().sum())
    missing_features = int(df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").isna().sum().sum())
    if leakage or duplicate or missing_features:
        raise ValueError(
            f"Manifest QC failed for split_seed={seed}: leakage={leakage}, duplicate={duplicate}, missing_features={missing_features}"
        )
    eyes = {split: int((df["split"].astype(str) == split).sum()) for split in ["train", "val", "test"]}
    patients = {
        split: int(df.loc[df["split"].astype(str).eq(split), "global_patient_uid"].nunique())
        for split in ["train", "val", "test"]
    }
    return df, eyes, patients


def read_run(seed: int, model_seed: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest, eyes, patients = read_manifest_qc(seed)
    report = report_dir_for(seed, model_seed)
    prefix = prefix_for(model_seed)
    predictions_path = report / f"{prefix}_predictions.csv"
    overall_path = report / f"{prefix}_overall_metrics.csv"
    range_path = report / f"{prefix}_range_metrics.csv"
    log_path = report / f"{prefix}_training_log.csv"
    summary_path = report / f"{prefix}_summary.md"
    for path in [predictions_path, overall_path, range_path, log_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    predictions = pd.read_csv(predictions_path)
    if len(predictions) != eyes["test"]:
        raise ValueError(f"split_seed={seed}: expected {eyes['test']} predictions, got {len(predictions)}")
    if not predictions["global_sample_id"].is_unique:
        raise ValueError(f"split_seed={seed}: duplicate prediction global_sample_id")
    test_manifest = manifest[manifest["split"].astype(str).eq("test")].copy()
    check = predictions[["global_sample_id", "vault_label_um"]].merge(
        test_manifest[["global_sample_id", "vault_label"]], on="global_sample_id", how="outer", indicator=True
    )
    if not check["_merge"].eq("both").all():
        raise ValueError(f"split_seed={seed}: predictions do not match test manifest sample set")
    if not np.allclose(check["vault_label_um"].astype(float), check["vault_label"].astype(float), atol=1e-4):
        raise ValueError(f"split_seed={seed}: prediction labels differ from manifest labels")
    overall_raw = pd.read_csv(overall_path).iloc[0].to_dict()
    train_log = pd.read_csv(log_path)
    best_row = train_log.loc[train_log["val_mae_um"].idxmin()]
    computed = regression_metrics(predictions["vault_label_um"], predictions["pred_vault_um"])
    row = {
        "split_seed": seed,
        "model_seed": model_seed,
        "n_train": eyes["train"],
        "n_val": eyes["val"],
        "n_test": eyes["test"],
        "train_patients": patients["train"],
        "val_patients": patients["val"],
        "test_patients": patients["test"],
        "best_epoch": int(overall_raw.get("best_epoch", best_row["epoch"])),
        "best_val_mae": float(overall_raw.get("best_val_mae_um", best_row["val_mae_um"])),
        **computed,
        "report_dir": rel(report),
    }
    ranges = range_metrics(predictions)
    ranges.insert(0, "model_seed", model_seed)
    ranges.insert(0, "split_seed", seed)
    predictions = predictions.copy()
    predictions.insert(0, "model_seed", model_seed)
    predictions.insert(0, "split_seed", seed)
    return row, ranges, predictions


def range_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["vault_label_um"] = pd.to_numeric(df["vault_label_um"], errors="coerce")
    df["pred_vault_um"] = pd.to_numeric(df["pred_vault_um"], errors="coerce")
    df["signed_error_um"] = df["pred_vault_um"] - df["vault_label_um"]
    df["abs_error_um"] = df["signed_error_um"].abs()
    df["vault_range"] = df["vault_label_um"].map(vault_range)
    total_abs = float(df["abs_error_um"].sum())
    rows = []
    for name in ["low", "medium", "high"]:
        sub = df[df["vault_range"].eq(name)]
        if sub.empty:
            rows.append(
                {
                    "vault_range": name,
                    "n": 0,
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "mean_signed_error": float("nan"),
                    "overestimation_count": 0,
                    "underestimation_count": 0,
                    "absolute_error_contribution_pct": 0.0,
                }
            )
            continue
        rows.append(
            {
                "vault_range": name,
                "n": int(len(sub)),
                "MAE": float(sub["abs_error_um"].mean()),
                "RMSE": float(np.sqrt(np.mean(sub["signed_error_um"] ** 2))),
                "mean_signed_error": float(sub["signed_error_um"].mean()),
                "overestimation_count": int((sub["signed_error_um"] > 0).sum()),
                "underestimation_count": int((sub["signed_error_um"] < 0).sum()),
                "absolute_error_contribution_pct": float(sub["abs_error_um"].sum() / total_abs * 100.0) if total_abs > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def aggregate(overall: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric_scope": "overall",
            "model": "fusion_model_seed42",
            "MAE_mean": float(overall["test_mae"].mean()),
            "MAE_sample_std": float(overall["test_mae"].std(ddof=1)),
            "MAE_median": float(overall["test_mae"].median()),
            "MAE_IQR": iqr(overall["test_mae"]),
            "MAE_min": float(overall["test_mae"].min()),
            "MAE_max": float(overall["test_mae"].max()),
            "RMSE_mean": float(overall["test_rmse"].mean()),
            "RMSE_sample_std": float(overall["test_rmse"].std(ddof=1)),
            "R2_mean": float(overall["test_r2"].mean()),
            "R2_sample_std": float(overall["test_r2"].std(ddof=1)),
            "signed_error_mean": float(overall["mean_signed_error"].mean()),
            "signed_error_sample_std": float(overall["mean_signed_error"].std(ddof=1)),
            "prediction_range_ratio_mean": float(overall["prediction_range_label_range_ratio"].mean()),
            "prediction_range_ratio_sample_std": float(overall["prediction_range_label_range_ratio"].std(ddof=1)),
            "sample_counts": "",
            "overestimation_proportion_mean": float("nan"),
            "underestimation_proportion_mean": float("nan"),
        }
    ]
    for name in ["low", "medium", "high"]:
        sub = ranges[ranges["vault_range"].eq(name)].copy()
        rows.append(
            {
                "metric_scope": f"range_{name}",
                "model": "fusion_model_seed42",
                "MAE_mean": float(sub["MAE"].mean()),
                "MAE_sample_std": float(sub["MAE"].std(ddof=1)),
                "MAE_median": float(sub["MAE"].median()),
                "MAE_IQR": iqr(sub["MAE"]),
                "MAE_min": float(sub["MAE"].min()),
                "MAE_max": float(sub["MAE"].max()),
                "RMSE_mean": float(sub["RMSE"].mean()),
                "RMSE_sample_std": float(sub["RMSE"].std(ddof=1)),
                "R2_mean": float("nan"),
                "R2_sample_std": float("nan"),
                "signed_error_mean": float(sub["mean_signed_error"].mean()),
                "signed_error_sample_std": float(sub["mean_signed_error"].std(ddof=1)),
                "prediction_range_ratio_mean": float("nan"),
                "prediction_range_ratio_sample_std": float("nan"),
                "sample_counts": ";".join(str(int(value)) for value in sub["n"].tolist()),
                "overestimation_proportion_mean": float((sub["overestimation_count"] / sub["n"].replace(0, np.nan)).mean()),
                "underestimation_proportion_mean": float((sub["underestimation_count"] / sub["n"].replace(0, np.nan)).mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(overall: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    measurement = pd.read_csv(MEASUREMENT_OVERALL)
    rf = measurement[measurement["model"].astype(str).eq("Random Forest Regressor")].copy()
    vs_rf = overall[["split_seed", "test_mae"]].merge(rf[["split_seed", "MAE"]], on="split_seed", how="inner")
    vs_rf = vs_rf.rename(columns={"test_mae": "fusion_MAE", "MAE": "measurement_RF_MAE"})
    vs_rf["delta_MAE"] = vs_rf["fusion_MAE"] - vs_rf["measurement_RF_MAE"]
    vs_rf["winner"] = np.where(vs_rf["delta_MAE"] < 0, "fusion", "measurement_RF")

    as_oct = pd.read_csv(CORRECTED_AS_OCT_OVERALL)
    vs_as = overall[["split_seed", "test_mae"]].merge(as_oct[["split_seed", "test_mae"]], on="split_seed", how="inner")
    vs_as = vs_as.rename(columns={"test_mae_x": "fusion_MAE", "test_mae_y": "corrected_AS_OCT_MAE"})
    vs_as["delta_MAE"] = vs_as["fusion_MAE"] - vs_as["corrected_AS_OCT_MAE"]
    vs_as["winner"] = np.where(vs_as["delta_MAE"] < 0, "fusion", "corrected_AS_OCT")
    return vs_rf, vs_as


def write_summary(overall: pd.DataFrame, ranges: pd.DataFrame, agg: pd.DataFrame, vs_rf: pd.DataFrame, vs_as: pd.DataFrame) -> None:
    rf_wins = int((vs_rf["winner"] == "measurement_RF").sum())
    fusion_rf_wins = int((vs_rf["winner"] == "fusion").sum())
    as_wins = int((vs_as["winner"] == "corrected_AS_OCT").sum())
    fusion_as_wins = int((vs_as["winner"] == "fusion").sum())
    low = ranges[ranges["vault_range"].eq("low")]
    high = ranges[ranges["vault_range"].eq("high")]
    lines = [
        "# Fusion Repeated Split Aggregation",
        "",
        f"- Fusion repeated MAE mean +/- sample std: {f(overall['test_mae'].mean())} +/- {f(overall['test_mae'].std(ddof=1))} um.",
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
        "## Aggregate Metrics",
        md_table(agg),
        "",
        "## Paired Comparison: Measurement RF",
        md_table(vs_rf),
        f"- Fusion wins {fusion_rf_wins}/5; measurement RF wins {rf_wins}/5.",
        f"- Fusion - RF delta MAE mean +/- std: {f(vs_rf['delta_MAE'].mean())} +/- {f(vs_rf['delta_MAE'].std(ddof=1))} um.",
        "",
        "## Paired Comparison: Corrected AS-OCT",
        md_table(vs_as),
        f"- Fusion wins {fusion_as_wins}/5; corrected AS-OCT wins {as_wins}/5.",
        f"- Fusion - corrected AS-OCT delta MAE mean +/- std: {f(vs_as['delta_MAE'].mean())} +/- {f(vs_as['delta_MAE'].std(ddof=1))} um.",
        "",
        "## Bias Pattern",
        f"- Low-vault overestimation proportion mean: {f((low['overestimation_count'] / low['n']).mean() * 100)}%.",
        f"- High-vault underestimation proportion mean: {f((high['underestimation_count'] / high['n']).mean() * 100)}%.",
        "- No exaggerated significance claim is made from five repeated patient-level splits.",
    ]
    (OUT_DIR / "fusion_repeated_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    range_frames = []
    prediction_frames = []
    for seed in split_seeds(args.split_seeds):
        row, ranges, predictions = read_run(seed, args.model_seed)
        rows.append(row)
        range_frames.append(ranges)
        prediction_frames.append(predictions)

    overall = pd.DataFrame(rows).sort_values("split_seed")
    ranges = pd.concat(range_frames, ignore_index=True).sort_values(["split_seed", "vault_range"])
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["split_seed", "global_sample_id"])
    agg = aggregate(overall, ranges)
    vs_rf, vs_as = paired_comparisons(overall)

    overall.to_csv(OUT_DIR / "fusion_repeated_split_overall_metrics.csv", index=False, encoding="utf-8")
    ranges.to_csv(OUT_DIR / "fusion_repeated_split_range_metrics.csv", index=False, encoding="utf-8")
    predictions.to_csv(OUT_DIR / "fusion_repeated_split_predictions.csv", index=False, encoding="utf-8")
    agg.to_csv(OUT_DIR / "fusion_repeated_split_aggregate_metrics.csv", index=False, encoding="utf-8")
    vs_rf.to_csv(OUT_DIR / "fusion_vs_measurement_rf_paired_comparison.csv", index=False, encoding="utf-8")
    vs_as.to_csv(OUT_DIR / "fusion_vs_corrected_as_oct_paired_comparison.csv", index=False, encoding="utf-8")
    write_summary(overall, ranges, agg, vs_rf, vs_as)
    print(f"Wrote {rel(OUT_DIR / 'fusion_repeated_split_summary.md')}")


if __name__ == "__main__":
    main()
