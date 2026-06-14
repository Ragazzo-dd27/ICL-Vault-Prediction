"""Aggregate combined v4 AS-OCT-only repeated split results.

This script does not train models. It reads the existing primary seed42 result
and completed repeated split outputs for split seeds 1001/2002/2026/3407, then
summarizes AS-OCT-only performance and compares it with measurement-only RF on
the same split seeds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data/splits/combined_batch_01_02_03_04_repeated"
BASE_REPORT_DIR = PROJECT_ROOT / "artifacts/reports/combined_batch_01_02_03_04"
OUTPUT_DIR = BASE_REPORT_DIR / "as_oct_only_repeated_splits"
PRIMARY_REPORT_DIR = BASE_REPORT_DIR / "as_oct_only_baseline_seed42"
MEASUREMENT_REPEATED = BASE_REPORT_DIR / (
    "measurement_only_repeated_splits/measurement_repeated_split_overall_metrics.csv"
)
SEEDS = [42, 1001, 2002, 2026, 3407]
MODEL_SEED = 42
RANGES = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate AS-OCT-only v4 repeated split outputs.")
    parser.add_argument("--split_seeds", default="42,1001,2002,2026,3407")
    parser.add_argument("--model_seed", type=int, default=42)
    return parser.parse_args()


def parse_seed_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def f(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def manifest_path(seed: int) -> Path:
    return SPLIT_DIR / f"as_oct_manifest_split_seed{seed}.csv"


def report_dir(seed: int, model_seed: int) -> Path:
    if seed == 42:
        return PRIMARY_REPORT_DIR
    return OUTPUT_DIR / f"split_seed{seed}_model_seed{model_seed}"


def prefix(model_seed: int) -> str:
    return f"as_oct_v4_seed{model_seed}"


def load_manifest(seed: int) -> pd.DataFrame:
    path = manifest_path(seed)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"global_sample_id", "global_patient_uid", "split", "vault_label", "oct_path", "sample_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def validate_manifest(seed: int, df: pd.DataFrame) -> None:
    leakage = int((df.groupby("global_patient_uid")["split"].nunique() > 1).sum())
    if leakage != 0:
        raise ValueError(f"seed{seed} patient leakage: {leakage}")
    duplicates = int(df["global_sample_id"].duplicated().sum())
    if duplicates != 0:
        raise ValueError(f"seed{seed} duplicate global_sample_id: {duplicates}")
    missing_images = []
    for value in df["oct_path"].fillna(""):
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            missing_images.append(str(value))
            if len(missing_images) >= 5:
                break
    if missing_images:
        raise FileNotFoundError(f"seed{seed} missing image examples: {missing_images}")


def seed42_primary_consistency() -> bool:
    repeated = load_manifest(42)
    primary = pd.read_csv(
        PROJECT_ROOT
        / "data/manifests/vault_as_oct_only_pod1_manifest_combined_batch_01_02_03_04_strict_split_seed42.csv"
    )
    cols = ["global_sample_id", "global_patient_uid", "split", "vault_label", "oct_path"]
    r = repeated[cols].sort_values("global_sample_id").reset_index(drop=True)
    p = primary[cols].sort_values("global_sample_id").reset_index(drop=True)
    if len(r) != len(p) or r["global_sample_id"].tolist() != p["global_sample_id"].tolist():
        return False
    for col in ["global_patient_uid", "split", "oct_path"]:
        if not r[col].astype(str).equals(p[col].astype(str)):
            return False
    diff = (pd.to_numeric(r["vault_label"], errors="coerce") - pd.to_numeric(p["vault_label"], errors="coerce")).abs().max()
    return bool(pd.notna(diff) and float(diff) <= 1e-9)


def vault_range(labels: pd.Series) -> pd.Series:
    values = pd.to_numeric(labels, errors="coerce")
    return pd.Series(
        np.select([values < 500, values <= 800], ["low", "medium"], default="high"),
        index=labels.index,
    )


def prediction_path(seed: int, model_seed: int) -> Path:
    return report_dir(seed, model_seed) / f"{prefix(model_seed)}_predictions.csv"


def overall_path(seed: int, model_seed: int) -> Path:
    return report_dir(seed, model_seed) / f"{prefix(model_seed)}_overall_metrics.csv"


def range_path(seed: int, model_seed: int) -> Path:
    return report_dir(seed, model_seed) / f"{prefix(model_seed)}_range_metrics.csv"


def load_predictions(seed: int, model_seed: int, manifest: pd.DataFrame) -> pd.DataFrame:
    path = prediction_path(seed, model_seed)
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path)
    test_manifest = manifest[manifest["split"] == "test"].copy()
    expected = len(test_manifest)
    if len(pred) != expected:
        raise ValueError(f"seed{seed} predictions rows={len(pred)}, expected test rows={expected}")
    meta_cols = [
        "sample_id",
        "global_sample_id",
        "global_patient_uid",
        "patient_uid",
        "eye",
        "batch_id",
        "qc_flag",
        "label_qc_flag",
        "oct_path",
    ]
    meta_cols = [col for col in meta_cols if col in manifest.columns]
    pred = pred.merge(manifest[meta_cols].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "_manifest"))
    if pred["global_sample_id"].isna().any():
        raise ValueError(f"seed{seed} predictions could not map all global_sample_id values")
    pred["split_seed"] = seed
    pred["model_seed"] = model_seed
    pred["vault_label_um"] = pd.to_numeric(pred["vault_label_um"], errors="coerce")
    pred["pred_vault_um"] = pd.to_numeric(pred["pred_vault_um"], errors="coerce")
    pred["signed_error_um"] = pred["pred_vault_um"] - pred["vault_label_um"]
    pred["abs_error_um"] = pred["signed_error_um"].abs()
    pred["vault_range"] = vault_range(pred["vault_label_um"])
    return pred


def metrics_from_predictions(pred: pd.DataFrame) -> Dict[str, float]:
    labels = pred["vault_label_um"].astype(float)
    preds = pred["pred_vault_um"].astype(float)
    errors = preds - labels
    ss_res = float((errors**2).sum())
    ss_tot = float(((labels - labels.mean()) ** 2).sum())
    pred_range = float(preds.max() - preds.min())
    label_range = float(labels.max() - labels.min())
    return {
        "test_mae": float(errors.abs().mean()),
        "test_rmse": float(np.sqrt((errors**2).mean())),
        "test_r2": float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot),
        "mean_signed_error": float(errors.mean()),
        "prediction_min": float(preds.min()),
        "prediction_max": float(preds.max()),
        "prediction_mean": float(preds.mean()),
        "prediction_std": float(preds.std(ddof=1)),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std(ddof=1)),
        "prediction_range_label_range_ratio": pred_range / label_range if label_range > 0 else float("nan"),
    }


def range_rows(seed: int, model_seed: int, pred: pd.DataFrame) -> List[Dict[str, Any]]:
    total_abs = float(pred["abs_error_um"].sum())
    rows = []
    for vr in RANGES:
        sub = pred[pred["vault_range"] == vr]
        if sub.empty:
            rows.append(
                {
                    "split_seed": seed,
                    "model_seed": model_seed,
                    "vault_range": vr,
                    "n": 0,
                    "MAE": float("nan"),
                    "RMSE": float("nan"),
                    "mean_signed_error": float("nan"),
                    "overestimation_count": 0,
                    "underestimation_count": 0,
                    "absolute_error_contribution_percentage": 0.0,
                }
            )
            continue
        err = sub["signed_error_um"]
        rows.append(
            {
                "split_seed": seed,
                "model_seed": model_seed,
                "vault_range": vr,
                "n": int(len(sub)),
                "MAE": float(sub["abs_error_um"].mean()),
                "RMSE": float(np.sqrt((err**2).mean())),
                "mean_signed_error": float(err.mean()),
                "overestimation_count": int((err > 0).sum()),
                "underestimation_count": int((err < 0).sum()),
                "absolute_error_contribution_percentage": float(sub["abs_error_um"].sum() / total_abs * 100) if total_abs else float("nan"),
            }
        )
    return rows


def load_overall(seed: int, model_seed: int, manifest: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, Any]:
    path = overall_path(seed, model_seed)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path).iloc[0]
    train = manifest[manifest["split"] == "train"]
    val = manifest[manifest["split"] == "val"]
    test = manifest[manifest["split"] == "test"]
    recomputed = metrics_from_predictions(pred)
    return {
        "split_seed": seed,
        "model_seed": model_seed,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_patients": int(train["global_patient_uid"].nunique()),
        "val_patients": int(val["global_patient_uid"].nunique()),
        "test_patients": int(test["global_patient_uid"].nunique()),
        "best_epoch": int(raw.get("best_epoch", 0)),
        "best_val_mae": float(raw.get("best_val_mae_um", float("nan"))),
        **recomputed,
    }


def iqr(series: pd.Series) -> float:
    q75, q25 = np.nanpercentile(series, [75, 25])
    return float(q75 - q25)


def aggregate(overall: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "section": "overall",
            "vault_range": "all",
            "MAE_mean": overall["test_mae"].mean(),
            "MAE_sample_std": overall["test_mae"].std(ddof=1),
            "MAE_median": overall["test_mae"].median(),
            "MAE_IQR": iqr(overall["test_mae"]),
            "MAE_min": overall["test_mae"].min(),
            "MAE_max": overall["test_mae"].max(),
            "RMSE_mean": overall["test_rmse"].mean(),
            "RMSE_sample_std": overall["test_rmse"].std(ddof=1),
            "R2_mean": overall["test_r2"].mean(),
            "R2_sample_std": overall["test_r2"].std(ddof=1),
            "signed_error_mean": overall["mean_signed_error"].mean(),
            "signed_error_sample_std": overall["mean_signed_error"].std(ddof=1),
            "best_val_mae_mean": overall["best_val_mae"].mean(),
            "best_val_mae_sample_std": overall["best_val_mae"].std(ddof=1),
            "prediction_range_ratio_mean": overall["prediction_range_label_range_ratio"].mean(),
            "prediction_range_ratio_sample_std": overall["prediction_range_label_range_ratio"].std(ddof=1),
            "overestimation_proportion": "",
            "underestimation_proportion": "",
        }
    )
    for vr in RANGES:
        sub = ranges[ranges["vault_range"] == vr]
        over_prop = ""
        under_prop = ""
        if vr == "low" and sub["n"].sum() > 0:
            over_prop = float(sub["overestimation_count"].sum() / sub["n"].sum())
        if vr == "high" and sub["n"].sum() > 0:
            under_prop = float(sub["underestimation_count"].sum() / sub["n"].sum())
        rows.append(
            {
                "section": "range",
                "vault_range": vr,
                "MAE_mean": sub["MAE"].mean(),
                "MAE_sample_std": sub["MAE"].std(ddof=1),
                "MAE_median": sub["MAE"].median(),
                "MAE_IQR": iqr(sub["MAE"]),
                "MAE_min": sub["MAE"].min(),
                "MAE_max": sub["MAE"].max(),
                "RMSE_mean": sub["RMSE"].mean(),
                "RMSE_sample_std": sub["RMSE"].std(ddof=1),
                "R2_mean": "",
                "R2_sample_std": "",
                "signed_error_mean": sub["mean_signed_error"].mean(),
                "signed_error_sample_std": sub["mean_signed_error"].std(ddof=1),
                "best_val_mae_mean": "",
                "best_val_mae_sample_std": "",
                "prediction_range_ratio_mean": "",
                "prediction_range_ratio_sample_std": "",
                "overestimation_proportion": over_prop,
                "underestimation_proportion": under_prop,
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(overall: pd.DataFrame) -> pd.DataFrame:
    measurement = pd.read_csv(MEASUREMENT_REPEATED)
    rf = measurement[measurement["model"] == "Random Forest Regressor"].copy()
    if rf.empty:
        raise ValueError("Could not find Random Forest Regressor rows in measurement repeated metrics")
    paired = overall[["split_seed", "test_mae"]].merge(
        rf[["split_seed", "MAE"]], on="split_seed", how="inner", validate="one_to_one"
    )
    paired = paired.rename(columns={"test_mae": "AS-OCT MAE", "MAE": "RF MAE"})
    paired["delta_MAE"] = paired["AS-OCT MAE"] - paired["RF MAE"]
    paired["winner"] = np.where(paired["delta_MAE"] < 0, "AS-OCT", "RF")
    return paired.sort_values("split_seed").reset_index(drop=True)


def write_summary(
    overall: pd.DataFrame,
    ranges: pd.DataFrame,
    agg: pd.DataFrame,
    paired: pd.DataFrame,
    seed42_match: bool,
) -> None:
    best_as = overall.loc[overall["test_mae"].idxmin()]
    worst_as = overall.loc[overall["test_mae"].idxmax()]
    measurement = pd.read_csv(MEASUREMENT_REPEATED)
    rf = measurement[measurement["model"] == "Random Forest Regressor"].copy()
    best_rf = rf.loc[rf["MAE"].idxmin()]
    worst_rf = rf.loc[rf["MAE"].idxmax()]
    overall_agg = agg[agg["section"] == "overall"].iloc[0]
    as_wins = int((paired["winner"] == "AS-OCT").sum())
    rf_wins = int((paired["winner"] == "RF").sum())
    low = ranges[ranges["vault_range"] == "low"]
    high = ranges[ranges["vault_range"] == "high"]
    low_over = bool(((low["overestimation_count"] / low["n"]) > 0.5).all())
    high_under = bool(((high["underestimation_count"] / high["n"]) > 0.5).all())
    range_compression = bool((overall["prediction_range_label_range_ratio"] < 0.75).all())
    lines = [
        "# Combined v4 AS-OCT-only repeated split aggregation",
        "",
        "## QC",
        markdown_table(
            ["check", "result"],
            [
                ["model_seed for all splits", "42"],
                ["split_seed separated from model_seed", "yes"],
                ["seed42 primary result reused", "yes"],
                ["seed42 assignment matches primary", "yes" if seed42_match else "no"],
                ["patient leakage", "0 for all manifests"],
                ["duplicate global_sample_id", "0 for all manifests"],
                ["image path exists", "yes for all manifests"],
                ["test prediction rows correct", "yes"],
                ["no prediction ensemble across splits", "yes"],
                ["test labels used for model selection/calibration", "no"],
            ],
        ),
        "",
        "## Overall Metrics By Split",
        markdown_table(
            [
                "split_seed",
                "model_seed",
                "n_train",
                "n_val",
                "n_test",
                "best epoch",
                "best val MAE",
                "test MAE",
                "RMSE",
                "R2",
                "mean signed",
                "range ratio",
            ],
            [
                [
                    int(r.split_seed),
                    int(r.model_seed),
                    int(r.n_train),
                    int(r.n_val),
                    int(r.n_test),
                    int(r.best_epoch),
                    f(r.best_val_mae),
                    f(r.test_mae),
                    f(r.test_rmse),
                    f(r.test_r2, 3),
                    f(r.mean_signed_error),
                    f(r.prediction_range_label_range_ratio, 3),
                ]
                for _, r in overall.sort_values("split_seed").iterrows()
            ],
        ),
        "",
        "## Aggregate Metrics",
        markdown_table(
            ["metric", "value"],
            [
                ["MAE mean +/- sample std", f"{f(overall_agg.MAE_mean)} +/- {f(overall_agg.MAE_sample_std)} um"],
                ["MAE median", f(overall_agg.MAE_median)],
                ["MAE IQR", f(overall_agg.MAE_IQR)],
                ["MAE min / max", f"{f(overall_agg.MAE_min)} / {f(overall_agg.MAE_max)}"],
                ["RMSE mean +/- sample std", f"{f(overall_agg.RMSE_mean)} +/- {f(overall_agg.RMSE_sample_std)} um"],
                ["R2 mean +/- sample std", f"{f(overall_agg.R2_mean, 3)} +/- {f(overall_agg.R2_sample_std, 3)}"],
                ["signed error mean +/- sample std", f"{f(overall_agg.signed_error_mean)} +/- {f(overall_agg.signed_error_sample_std)} um"],
                ["best val MAE mean +/- sample std", f"{f(overall_agg.best_val_mae_mean)} +/- {f(overall_agg.best_val_mae_sample_std)} um"],
                ["prediction range ratio mean +/- sample std", f"{f(overall_agg.prediction_range_ratio_mean, 3)} +/- {f(overall_agg.prediction_range_ratio_sample_std, 3)}"],
            ],
        ),
        "",
        "## Range Aggregate Metrics",
        markdown_table(
            ["range", "MAE mean +/- std", "signed error mean +/- std", "low over prop", "high under prop"],
            [
                [
                    r.vault_range,
                    f"{f(r.MAE_mean)} +/- {f(r.MAE_sample_std)}",
                    f"{f(r.signed_error_mean)} +/- {f(r.signed_error_sample_std)}",
                    "" if r.overestimation_proportion == "" else f(r.overestimation_proportion, 3),
                    "" if r.underestimation_proportion == "" else f(r.underestimation_proportion, 3),
                ]
                for _, r in agg[agg["section"] == "range"].iterrows()
            ],
        ),
        "",
        "## Paired Comparison With Measurement-only RF",
        markdown_table(
            ["split_seed", "AS-OCT MAE", "RF MAE", "delta", "winner"],
            [[int(r.split_seed), f(r["AS-OCT MAE"]), f(r["RF MAE"]), f(r.delta_MAE), r.winner] for _, r in paired.iterrows()],
        ),
        "",
        f"- AS-OCT wins: {as_wins} / {len(paired)} splits",
        f"- RF wins: {rf_wins} / {len(paired)} splits",
        f"- paired delta MAE mean +/- std: {f(paired['delta_MAE'].mean())} +/- {f(paired['delta_MAE'].std(ddof=1))} um",
        f"- AS-OCT easiest/hardest split: seed{int(best_as.split_seed)} / seed{int(worst_as.split_seed)}",
        f"- RF easiest/hardest split: seed{int(best_rf.split_seed)} / seed{int(worst_rf.split_seed)}",
        "- With only five paired splits, this comparison should be treated descriptively rather than as a formal significance test.",
        "",
        "## Final Answers",
        f"- AS-OCT repeated MAE mean +/- std: {f(overall_agg.MAE_mean)} +/- {f(overall_agg.MAE_sample_std)} um.",
        f"- Primary split seed42 MAE 201.91 um is {'within' if overall_agg.MAE_min <= 201.91 <= overall_agg.MAE_max else 'outside'} the repeated split MAE range {f(overall_agg.MAE_min)} / {f(overall_agg.MAE_max)} um.",
        f"- AS-OCT patient-level split sensitivity: MAE range {f(overall_agg.MAE_min)} to {f(overall_agg.MAE_max)} um.",
        f"- Low-vault overestimation stable across splits: {'yes' if low_over else 'no'}.",
        f"- High-vault underestimation stable across splits: {'yes' if high_under else 'no'}.",
        f"- Prediction range compression across splits: {'yes' if range_compression else 'no'}.",
        f"- Same-split comparison: RF is better in {rf_wins}/{len(paired)} splits; AS-OCT is better in {as_wins}/{len(paired)} splits.",
        "- Fusion repeated evaluation is worth running next to see whether fusion consistently narrows the AS-OCT vs RF gap on the same splits.",
        "- Lower-lr or range-aware loss remains worth exploring later because range compression and high-vault underestimation persist.",
    ]
    (OUTPUT_DIR / "as_oct_repeated_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.split_seeds)
    if args.model_seed != 42:
        raise ValueError("This aggregation expects model_seed=42 for all AS-OCT repeated split runs.")
    seed42_match = seed42_primary_consistency()
    if not seed42_match:
        raise ValueError("seed42 repeated manifest does not match the primary manifest; refusing to aggregate.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_rows: List[Dict[str, Any]] = []
    range_rows_all: List[Dict[str, Any]] = []
    pred_frames = []
    for seed in seeds:
        manifest = load_manifest(seed)
        validate_manifest(seed, manifest)
        pred = load_predictions(seed, args.model_seed, manifest)
        pred_frames.append(pred)
        overall_rows.append(load_overall(seed, args.model_seed, manifest, pred))
        range_rows_all.extend(range_rows(seed, args.model_seed, pred))

    overall = pd.DataFrame(overall_rows).sort_values("split_seed").reset_index(drop=True)
    ranges = pd.DataFrame(range_rows_all).sort_values(["split_seed", "vault_range"]).reset_index(drop=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    agg = aggregate(overall, ranges)
    paired = paired_comparison(overall)

    overall.to_csv(OUTPUT_DIR / "as_oct_repeated_split_overall_metrics.csv", index=False, encoding="utf-8")
    ranges.to_csv(OUTPUT_DIR / "as_oct_repeated_split_range_metrics.csv", index=False, encoding="utf-8")
    predictions.to_csv(OUTPUT_DIR / "as_oct_repeated_split_predictions.csv", index=False, encoding="utf-8")
    agg.to_csv(OUTPUT_DIR / "as_oct_repeated_split_aggregate_metrics.csv", index=False, encoding="utf-8")
    paired.to_csv(OUTPUT_DIR / "as_oct_vs_measurement_rf_paired_comparison.csv", index=False, encoding="utf-8")
    write_summary(overall, ranges, agg, paired, seed42_match)

    print(f"Wrote AS-OCT repeated split aggregation to {rel(OUTPUT_DIR)}")
    print(overall[["split_seed", "model_seed", "test_mae", "test_rmse", "test_r2", "prediction_range_label_range_ratio"]].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
