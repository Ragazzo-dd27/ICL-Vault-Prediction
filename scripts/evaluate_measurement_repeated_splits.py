"""Evaluate measurement-only baselines on repeated patient-level splits.

This script trains only traditional preoperative measurement-only models on
existing repeated patient-level splits. It does not train AS-OCT models and does
not modify manifests, split files, predictions, checkpoints, or paper text.

Input features are true preoperative 2DAnalysis measurements only. Postoperative
2DAnalysis measurements must not be used as input features.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ALIASES = {
    "cct_mean_um": ["cct_mean_um", "cct_um", "cct"],
    "acd_epi_mean_mm": ["acd_epi_mean_mm", "acd_epi_mm", "acd_epi"],
    "acd_endo_mean_mm": ["acd_endo_mean_mm", "acd_endo_mm", "acd_endo"],
    "clr_mean_um": ["clr_mean_um", "clr_um", "clr"],
    "ata_mean_mm": ["ata_mean_mm", "ata_mm", "ata"],
}
LABEL_ALIASES = ["vault_label", "vault_label_um", "pod1_vault_mean_um", "label", "vault_um"]
SPLIT_ORDER = ["train", "val", "test"]
RANGE_ORDER = ["low", "medium", "high"]
STANDARD_SPLIT_FILES = [
    "data/splits/repeated_patient_split_seed42.csv",
    "data/splits/repeated_patient_split_seed1001.csv",
    "data/splits/repeated_patient_split_seed2002.csv",
    "data/splits/repeated_patient_split_seed2026.csv",
    "data/splits/repeated_patient_split_seed3407.csv",
]
FORCED_SPLIT_FILES = [
    "data/splits/repeated_patient_split_patient052test_seed52052.csv",
    "data/splits/repeated_patient_split_patient052test_seed52053.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate measurement-only baselines on repeated patient-level splits."
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Combined measurement-only ready manifest. If omitted, auto-detect under data/manifests.",
    )
    parser.add_argument(
        "--split_files",
        default="",
        help="Optional comma-separated repeated split CSV paths. Default uses standard + patient_052 forced-test splits.",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/reports/combined_batch_01_02/repeated_patient_split_stability/measurement_only",
    )
    parser.add_argument("--low_threshold", type=float, default=500.0)
    parser.add_argument("--high_threshold", type=float, default=800.0)
    parser.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--rf_n_estimators", type=int, default=500)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=2)
    parser.add_argument("--rf_random_state", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def auto_find_manifest(user_manifest: str) -> Path:
    if user_manifest:
        path = resolve_path(user_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    preferred = PROJECT_ROOT / "data/manifests/vault_preop_measurement_only_pod1_manifest_combined_ready.csv"
    if preferred.exists():
        return preferred

    candidates = []
    for path in (PROJECT_ROOT / "data/manifests").glob("*measurement*combined*ready*.csv"):
        try:
            n_rows = len(pd.read_csv(path, usecols=["sample_id"]))
        except Exception:
            n_rows = -1
        candidates.append((abs(n_rows - 160), -n_rows, path))
    if not candidates:
        raise FileNotFoundError("Could not find combined measurement-only ready manifest under data/manifests.")
    return sorted(candidates)[0][2]


def parse_split_files(text: str) -> List[Path]:
    if text.strip():
        return [resolve_path(item.strip()) for item in text.split(",") if item.strip()]
    return [resolve_path(path) for path in STANDARD_SPLIT_FILES + FORCED_SPLIT_FILES]


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"Could not find any candidate column: {list(candidates)}")


def feature_mapping(df: pd.DataFrame) -> Dict[str, str]:
    return {canonical: find_column(df, aliases) for canonical, aliases in FEATURE_ALIASES.items()}


def prepare_manifest(df: pd.DataFrame, low_threshold: float, high_threshold: float) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    mapping = feature_mapping(out)
    for canonical, source_col in mapping.items():
        out[canonical] = pd.to_numeric(out[source_col], errors="coerce")
    label_col = find_column(out, LABEL_ALIASES)
    out["vault_um"] = pd.to_numeric(out[label_col], errors="coerce")
    if "global_patient_uid" not in out.columns:
        if {"batch_id", "patient_uid"}.issubset(out.columns):
            out["global_patient_uid"] = out["batch_id"].astype(str) + "__" + out["patient_uid"].astype(str)
        else:
            patient_col = find_column(out, ["patient_id", "patient_uid"])
            out["global_patient_uid"] = out[patient_col].astype(str)
    if "global_sample_id" not in out.columns:
        if {"batch_id", "sample_id"}.issubset(out.columns):
            out["global_sample_id"] = out["batch_id"].astype(str) + "__" + out["sample_id"].astype(str)
        else:
            out["global_sample_id"] = out["sample_id"].astype(str)
    if "patient_id" not in out.columns:
        out["patient_id"] = out["global_patient_uid"]
    if "patient_uid" not in out.columns:
        out["patient_uid"] = out["patient_id"]
    if "eye" not in out.columns:
        out["eye"] = out["eye_side"] if "eye_side" in out.columns else ""
    if "eye_side" not in out.columns:
        out["eye_side"] = out["eye"]

    missing_features = out[list(FEATURE_ALIASES)].isna().any(axis=1)
    missing_label = out["vault_um"].isna() | (out["vault_um"] <= 0)
    if missing_features.any() or missing_label.any():
        raise ValueError(
            f"Manifest has missing inputs/labels: missing_features={int(missing_features.sum())}, "
            f"missing_or_invalid_label={int(missing_label.sum())}"
        )
    out["vault_range"] = np.select(
        [out["vault_um"] < low_threshold, out["vault_um"] <= high_threshold],
        ["low", "medium"],
        default="high",
    )
    return out, mapping


def split_metadata(path: Path) -> Tuple[int, str]:
    name = path.name
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse seed from split filename: {name}")
    split_seed = int(match.group(1))
    split_type = "patient052_forced_test" if "patient052test" in name else "standard_repeated"
    return split_seed, split_type


def apply_split(manifest: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_path)
    if "global_patient_uid" not in split_df.columns:
        patient_col = find_column(split_df, ["patient_id", "patient_uid"])
        split_df["global_patient_uid"] = split_df[patient_col].astype(str)
    if "split" not in split_df.columns:
        raise KeyError(f"Split file has no split column: {split_path}")
    patient_split = split_df[["global_patient_uid", "split"]].drop_duplicates()
    leaked = patient_split.groupby("global_patient_uid")["split"].nunique()
    leaked = leaked[leaked > 1]
    if len(leaked):
        raise ValueError(f"Patient-level split file leaks patients across splits: {split_path}")
    merged = manifest.drop(columns=["split"], errors="ignore").merge(patient_split, on="global_patient_uid", how="left")
    if merged["split"].isna().any():
        missing = sorted(merged.loc[merged["split"].isna(), "global_patient_uid"].unique())
        raise ValueError(f"Measurement manifest patients missing from split {split_path}: {missing[:10]}")
    return merged


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae_um": float(mean_absolute_error(y_true, y_pred)),
        "rmse_um": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "mean_signed_error_um": float(np.mean(y_pred - y_true)),
    }


def range_metrics(y_true: np.ndarray, y_pred: np.ndarray, ranges: Iterable[str]) -> List[Dict[str, object]]:
    frame = pd.DataFrame({"y": y_true, "pred": y_pred, "vault_range": list(ranges)})
    frame["signed_error_um"] = frame["pred"] - frame["y"]
    frame["abs_error_um"] = frame["signed_error_um"].abs()
    rows = []
    for vault_range in RANGE_ORDER:
        group = frame[frame["vault_range"] == vault_range]
        if group.empty:
            rows.append(
                {
                    "vault_range": vault_range,
                    "n": 0,
                    "mae_um": np.nan,
                    "mean_signed_error_um": np.nan,
                    "overestimation_count": 0,
                }
            )
            continue
        rows.append(
            {
                "vault_range": vault_range,
                "n": len(group),
                "mae_um": float(group["abs_error_um"].mean()),
                "mean_signed_error_um": float(group["signed_error_um"].mean()),
                "overestimation_count": int((group["signed_error_um"] > 0).sum()),
            }
        )
    return rows


def split_xy(df: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    subset = df[df["split"] == split].copy()
    x = subset[list(FEATURE_ALIASES)].astype(float)
    y = subset["vault_um"].to_numpy(dtype=float)
    return x, y, subset


def train_models(
    split_df: pd.DataFrame,
    ridge_alphas: List[float],
    rf_n_estimators: int,
    rf_min_samples_leaf: int,
    rf_random_state: int,
) -> Dict[str, Dict[str, object]]:
    x_train, y_train, train_meta = split_xy(split_df, "train")
    x_val, y_val, val_meta = split_xy(split_df, "val")
    x_test, y_test, test_meta = split_xy(split_df, "test")
    if min(len(train_meta), len(val_meta), len(test_meta)) == 0:
        raise ValueError("Train/val/test split must all be non-empty.")

    models: Dict[str, Dict[str, object]] = {}

    linear = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    linear.fit(x_train, y_train)
    models["linear_regression"] = {
        "estimator": linear,
        "params": {},
        "val_pred": linear.predict(x_val),
        "test_pred": linear.predict(x_test),
    }

    best_ridge = None
    for alpha in ridge_alphas:
        ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
        ridge.fit(x_train, y_train)
        val_pred = ridge.predict(x_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        if best_ridge is None or val_mae < best_ridge["val_mae"]:
            best_ridge = {"estimator": ridge, "params": {"alpha": alpha}, "val_mae": val_mae, "val_pred": val_pred}
    assert best_ridge is not None
    best_ridge["test_pred"] = best_ridge["estimator"].predict(x_test)
    models["ridge_regression"] = best_ridge

    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=rf_random_state,
        n_jobs=1,
    )
    rf.fit(x_train, y_train)
    models["random_forest"] = {
        "estimator": rf,
        "params": {
            "n_estimators": rf_n_estimators,
            "min_samples_leaf": rf_min_samples_leaf,
            "random_state": rf_random_state,
        },
        "val_pred": rf.predict(x_val),
        "test_pred": rf.predict(x_test),
    }

    for result in models.values():
        result["train_meta"] = train_meta
        result["val_meta"] = val_meta
        result["test_meta"] = test_meta
        result["y_val"] = y_val
        result["y_test"] = y_test
    return models


def sample_counts(split_seed: int, split_type: str, split_path: Path, split_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        group = split_df[split_df["split"] == split]
        rows.append(
            {
                "split_seed": split_seed,
                "split_type": split_type,
                "split_file": relative_path(split_path),
                "split": split,
                "n_samples": len(group),
                "n_patients": group["global_patient_uid"].nunique(),
                "n_low": int((group["vault_range"] == "low").sum()),
                "n_medium": int((group["vault_range"] == "medium").sum()),
                "n_high": int((group["vault_range"] == "high").sum()),
            }
        )
    return pd.DataFrame(rows)


def patient052_location(split_seed: int, split_type: str, split_df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=split_df.index)
    for col in ["global_patient_uid", "patient_uid", "patient_id", "global_sample_id", "sample_id"]:
        if col in split_df.columns:
            mask = mask | split_df[col].astype(str).str.contains("patient_052", case=False, na=False)
    out = split_df.loc[
        mask,
        ["global_sample_id", "sample_id", "patient_id", "patient_uid", "global_patient_uid", "eye", "eye_side", "vault_um", "vault_range", "split"],
    ].copy()
    out.insert(0, "split_seed", split_seed)
    out.insert(1, "split_type", split_type)
    return out


def build_outputs_for_split(
    split_seed: int,
    split_type: str,
    split_path: Path,
    split_df: pd.DataFrame,
    models: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], pd.DataFrame]:
    overall_rows = []
    range_rows = []
    patient052_rows = []
    location = patient052_location(split_seed, split_type, split_df)

    for model_name, result in models.items():
        y_val = result["y_val"]
        y_test = result["y_test"]
        val_pred = result["val_pred"]
        test_pred = result["test_pred"]
        val_metrics = metrics(y_val, val_pred)
        test_metrics = metrics(y_test, test_pred)
        overall_rows.append(
            {
                "split_seed": split_seed,
                "split_type": split_type,
                "split_file": relative_path(split_path),
                "model_name": model_name,
                "best_params": result["params"],
                "val_mae_um": val_metrics["mae_um"],
                "test_mae_um": test_metrics["mae_um"],
                "test_rmse_um": test_metrics["rmse_um"],
                "test_r2": test_metrics["r2"],
                "test_mean_signed_error_um": test_metrics["mean_signed_error_um"],
            }
        )
        test_meta = result["test_meta"].copy()
        for row in range_metrics(y_test, test_pred, test_meta["vault_range"]):
            row.update({"split_seed": split_seed, "split_type": split_type, "model_name": model_name})
            range_rows.append(row)

        p52_test = test_meta[test_meta["global_sample_id"].isin(location.loc[location["split"] == "test", "global_sample_id"])]
        if not p52_test.empty:
            pred_by_id = dict(zip(test_meta["global_sample_id"], test_pred))
            for _, sample in p52_test.iterrows():
                pred = float(pred_by_id[sample["global_sample_id"]])
                label = float(sample["vault_um"])
                patient052_rows.append(
                    {
                        "split_seed": split_seed,
                        "split_type": split_type,
                        "model_name": model_name,
                        "global_sample_id": sample["global_sample_id"],
                        "sample_id": sample["sample_id"],
                        "patient_id": sample["patient_id"],
                        "global_patient_uid": sample["global_patient_uid"],
                        "eye": sample["eye"],
                        "eye_side": sample["eye_side"],
                        "vault_um": label,
                        "pred_vault_um": pred,
                        "abs_error_um": abs(pred - label),
                        "signed_error_um": pred - label,
                        "vault_range": sample["vault_range"],
                        "split": "test",
                    }
                )

    location["model_name"] = "location_only"
    location["pred_vault_um"] = np.nan
    location["abs_error_um"] = np.nan
    location["signed_error_um"] = np.nan
    location_cols = [
        "split_seed",
        "split_type",
        "model_name",
        "global_sample_id",
        "sample_id",
        "patient_id",
        "global_patient_uid",
        "eye",
        "eye_side",
        "vault_um",
        "pred_vault_um",
        "abs_error_um",
        "signed_error_um",
        "vault_range",
        "split",
    ]
    patient052_df = pd.concat([location[location_cols], pd.DataFrame(patient052_rows)], ignore_index=True)
    return overall_rows, range_rows, patient052_df


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
            text_df[col] = text_df[col].fillna("").astype(str)
    lines = ["| " + " | ".join(text_df.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(text_df.columns)) + " |")
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in text_df.columns) + " |")
    lines.append("")
    return lines


def mean_std_table(overall: pd.DataFrame, split_type: str) -> pd.DataFrame:
    subset = overall[overall["split_type"] == split_type]
    if subset.empty:
        return pd.DataFrame(columns=["model_name", "n_splits", "test_mae_mean", "test_mae_std"])
    return (
        subset.groupby("model_name")
        .agg(
            n_splits=("split_seed", "nunique"),
            test_mae_mean=("test_mae_um", "mean"),
            test_mae_std=("test_mae_um", "std"),
            test_rmse_mean=("test_rmse_um", "mean"),
            test_r2_mean=("test_r2", "mean"),
        )
        .reset_index()
        .sort_values("test_mae_mean")
    )


def write_summary(
    path: Path,
    manifest_path: Path,
    split_paths: List[Path],
    feature_map: Dict[str, str],
    counts: pd.DataFrame,
    overall: pd.DataFrame,
    range_df: pd.DataFrame,
    patient052: pd.DataFrame,
) -> None:
    standard_summary = mean_std_table(overall, "standard_repeated")
    forced_summary = overall[overall["split_type"] == "patient052_forced_test"].sort_values(["split_seed", "model_name"])
    low_range = range_df[range_df["vault_range"] == "low"].groupby(["split_type", "model_name"]).agg(
        low_mae_mean=("mae_um", "mean"),
        low_signed_mean=("mean_signed_error_um", "mean"),
    ).reset_index()

    lines = [
        "# Measurement-only repeated split stability evaluation",
        "",
        "本步骤只评估传统 measurement-only baseline 在 repeated patient-level splits 下的稳定性。",
        "没有训练 AS-OCT 模型，没有修改 manifest、split、prediction 或 checkpoint，也不替代主结果。",
        "",
        f"- 输入 measurement manifest: `{relative_path(manifest_path)}`",
        "- 输入特征为真正术前 2DAnalysis measurements：CCT, ACD Epi, ACD Endo, CLR, ATA。",
        "- 术后 2DAnalysis measurements 不作为输入特征。",
        "",
        "## Feature Mapping",
        "",
    ]
    for canonical, source in feature_map.items():
        lines.append(f"- {canonical}: `{source}`")
    lines.extend(["", "## Split Files", ""])
    for split_path in split_paths:
        lines.append(f"- `{relative_path(split_path)}`")

    lines.extend(["", "## Actual Train / Val / Test Sample Counts", ""])
    lines.extend(md_table(counts, ["split_type", "split_seed", "split", "n_samples", "n_patients", "n_low", "n_medium", "n_high"]))

    lines.extend(["## Standard Repeated Splits: Test MAE Mean +/- Std", ""])
    lines.extend(md_table(standard_summary, ["model_name", "n_splits", "test_mae_mean", "test_mae_std", "test_rmse_mean", "test_r2_mean"]))

    lines.extend(["## Patient 052 Forced-Test Splits: Test MAE", ""])
    if forced_summary.empty:
        lines.append("_No forced-test split results._")
        lines.append("")
    else:
        lines.extend(md_table(forced_summary, ["split_seed", "model_name", "test_mae_um", "test_rmse_um", "test_r2", "test_mean_signed_error_um"]))

    lines.extend(["## Vault Range Error Pattern", ""])
    lines.extend(md_table(low_range, ["split_type", "model_name", "low_mae_mean", "low_signed_mean"]))
    lines.append(
        "low / medium / high 的完整 range-level 指标见 `measurement_repeated_split_range_metrics.csv`。"
    )
    lines.append("")

    lines.extend(["## Patient 052", ""])
    p52_test = patient052[(patient052["split_type"] == "patient052_forced_test") & (patient052["split"] == "test")]
    if p52_test.empty:
        lines.append("patient_052 未出现在 test，或没有可报告的 forced-test prediction。")
    else:
        prediction_rows = p52_test[p52_test["model_name"] != "location_only"].copy()
        lines.extend(md_table(prediction_rows, ["split_seed", "model_name", "sample_id", "eye", "vault_um", "pred_vault_um", "abs_error_um", "signed_error_um"]))
    lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- 本分析用于 measurement-only repeated split stability evaluation，不替代 combined cohort 主结果。",
            "- patient_052 没有被删除；forced-test split 仅用于观察该模型失败病例进入 test 时的稳定性压力。",
            "- measurement-only 模型可提供结构化术前参数的稳定性参照，但不涉及 AS-OCT checkpoint 或图像模型训练。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = auto_find_manifest(args.manifest)
    split_paths = parse_split_files(args.split_files)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ridge_alphas = parse_float_list(args.ridge_alphas)

    manifest_raw = pd.read_csv(manifest_path)
    manifest, feature_map = prepare_manifest(manifest_raw, args.low_threshold, args.high_threshold)
    print(f"Measurement manifest: {manifest_path}")
    print(f"Manifest rows: {len(manifest)}")
    print("Feature mapping:")
    for canonical, source in feature_map.items():
        print(f"  {canonical} <- {source}")

    overall_rows: List[Dict[str, object]] = []
    range_rows: List[Dict[str, object]] = []
    count_frames = []
    patient052_frames = []

    for split_path in split_paths:
        if not split_path.exists():
            print(f"WARNING: split file missing, skipped: {split_path}")
            continue
        split_seed, split_type = split_metadata(split_path)
        split_df = apply_split(manifest, split_path)
        leaked = split_df.groupby("global_patient_uid")["split"].nunique()
        if (leaked > 1).any():
            raise RuntimeError(f"Patient leakage after mapping split: {split_path}")
        counts = sample_counts(split_seed, split_type, split_path, split_df)
        count_frames.append(counts)
        models = train_models(
            split_df,
            ridge_alphas=ridge_alphas,
            rf_n_estimators=args.rf_n_estimators,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
            rf_random_state=args.rf_random_state,
        )
        split_overall, split_range, split_patient052 = build_outputs_for_split(
            split_seed, split_type, split_path, split_df, models
        )
        overall_rows.extend(split_overall)
        range_rows.extend(split_range)
        patient052_frames.append(split_patient052)
        print(f"\n{split_type} seed {split_seed}:")
        print(counts[["split", "n_samples", "n_patients", "n_low", "n_medium", "n_high"]].to_string(index=False))
        for row in split_overall:
            print(f"  {row['model_name']}: test MAE={row['test_mae_um']:.2f} um")

    if not overall_rows:
        raise RuntimeError("No split results were generated.")

    overall_df = pd.DataFrame(overall_rows)
    range_df = pd.DataFrame(range_rows)
    counts_df = pd.concat(count_frames, ignore_index=True)
    patient052_df = pd.concat(patient052_frames, ignore_index=True) if patient052_frames else pd.DataFrame()

    overall_path = output_dir / "measurement_repeated_split_overall_metrics.csv"
    range_path = output_dir / "measurement_repeated_split_range_metrics.csv"
    patient052_path = output_dir / "measurement_repeated_split_patient052_cases.csv"
    counts_path = output_dir / "measurement_repeated_split_sample_counts.csv"
    summary_path = output_dir / "measurement_repeated_split_summary.md"

    overall_df.to_csv(overall_path, index=False, encoding="utf-8")
    range_df.to_csv(range_path, index=False, encoding="utf-8")
    patient052_df.to_csv(patient052_path, index=False, encoding="utf-8")
    counts_df.to_csv(counts_path, index=False, encoding="utf-8")
    write_summary(summary_path, manifest_path, split_paths, feature_map, counts_df, overall_df, range_df, patient052_df)

    print("\nStandard repeated split mean +/- std test MAE:")
    print(mean_std_table(overall_df, "standard_repeated")[["model_name", "test_mae_mean", "test_mae_std"]].to_string(index=False))
    print("\nPatient_052 forced-test test MAE:")
    forced = overall_df[overall_df["split_type"] == "patient052_forced_test"]
    if forced.empty:
        print("No forced-test results.")
    else:
        print(forced[["split_seed", "model_name", "test_mae_um"]].to_string(index=False))

    print("\nOutput files:")
    for path in [overall_path, range_path, patient052_path, counts_path, summary_path]:
        print(path)


if __name__ == "__main__":
    main()
