"""Build preop measurement-only POD1 vault regression manifests.

This script uses only true preoperative CASIA2 2DAnalysis measurements.
Postoperative 2DAnalysis measurements must not be used as preoperative input
features, because that would leak postoperative vault information into the
baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ["cct_um", "acd_epi_mm", "acd_endo_mm", "clr_um", "ata_mm"]
EXCLUDE_NOTE = "exclude_from_preop_measurement_baseline"
BASE_NOTE = "preop measurement-only manifest; postoperative measurements excluded."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preop measurement-only POD1 manifests.")
    parser.add_argument(
        "--measurements",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review_priority_checked_patient042_fixed.csv",
    )
    parser.add_argument(
        "--clean_manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    )
    parser.add_argument(
        "--pod1_labels",
        type=str,
        default="data/manifests/vault_label_candidates_batch_01_pod1_verified.csv",
    )
    parser.add_argument(
        "--validation_report",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/preop_measurements/preop_measurement_validation_report.csv",
    )
    parser.add_argument(
        "--ready_out",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_batch_01_ready.csv",
    )
    parser.add_argument(
        "--strict_out",
        type=str,
        default="data/manifests/vault_preop_measurement_only_pod1_manifest_batch_01_strict.csv",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalize_measurements(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "verify_status" not in normalized.columns and "verified" in normalized.columns:
        normalized["verify_status"] = normalized["verified"]
    if "is_preop" not in normalized.columns:
        normalized["is_preop"] = True
    normalized["is_preop"] = normalize_bool_series(normalized["is_preop"])
    normalized["has_vault"] = normalize_bool_series(normalized["has_vault"]) if "has_vault" in normalized.columns else False
    normalized["verify_status"] = normalized["verify_status"].fillna("").astype(str).str.strip().str.lower()
    normalized["extraction_method"] = normalized["extraction_method"].fillna("").astype(str).str.strip().str.lower()
    normalized["notes"] = normalized["notes"].fillna("").astype(str)
    for column in FEATURE_COLUMNS + ["vault_um", "analysis_index"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def is_excluded(df: pd.DataFrame) -> pd.Series:
    return df["verify_status"].eq("excluded") | df["notes"].str.contains(EXCLUDE_NOTE, case=False, regex=False)


def leakage_risk(df: pd.DataFrame) -> pd.Series:
    vault_present = df["vault_um"].notna()
    raw_text = df.get("vault_raw_text", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    raw_text_present = raw_text.ne("") & raw_text.ne("---") & raw_text.str.lower().ne("nan")
    return vault_present | df["has_vault"] | raw_text_present


def baseline_candidate_records(df: pd.DataFrame) -> pd.DataFrame:
    candidate = df[
        df["is_preop"]
        & df["verify_status"].eq("verified")
        & df["extraction_method"].eq("manual_verified")
        & ~is_excluded(df)
        & ~leakage_risk(df)
    ].copy()
    for column in FEATURE_COLUMNS:
        candidate = candidate[candidate[column].notna()]
    return candidate


def scan_value(group: pd.DataFrame, column: str, position: int) -> float | str:
    if len(group) <= position:
        return ""
    value = group.iloc[position][column]
    return "" if pd.isna(value) else float(value)


def join_unique(values: Iterable[object]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"]
    return ";".join(dict.fromkeys(clean))


def aggregate_measurements(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sample_id, group in candidate_df.groupby("sample_id", dropna=False):
        group = group.sort_values(by=["analysis_index", "exam_time", "image_path"], kind="stable", na_position="last")
        row: Dict[str, object] = {
            "sample_id": sample_id,
            "num_preop_measurement_records": int(len(group)),
            "measurement_source_images": join_unique(group["image_path"]),
            "measurement_crop_paths": join_unique(group["measurement_crop_path"]),
        }
        for source, prefix, unit in [
            ("cct_um", "cct", "um"),
            ("acd_epi_mm", "acd_epi", "mm"),
            ("acd_endo_mm", "acd_endo", "mm"),
            ("clr_um", "clr", "um"),
            ("ata_mm", "ata", "mm"),
        ]:
            row[f"{prefix}_mean_{unit}"] = float(group[source].mean())
            row[f"{prefix}_scan1_{unit}"] = scan_value(group, source, 0)
            row[f"{prefix}_scan2_{unit}"] = scan_value(group, source, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest(
    base_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    measurement_features_df: pd.DataFrame,
) -> pd.DataFrame:
    label_cols = [
        "sample_id",
        "pod1_vault_mean_um",
        "pod1_vault_range_um",
        "qc_flag",
        "label_status",
        "verify_status",
    ]
    labels = labels_df[label_cols].rename(columns={"qc_flag": "label_qc_flag", "verify_status": "label_verify_status"})
    validation = validation_df[["sample_id", "measurement_ready_status"]]

    merged = (
        base_df.merge(validation, on="sample_id", how="left")
        .merge(measurement_features_df, on="sample_id", how="left")
        .merge(labels, on="sample_id", how="left", suffixes=("", "_label"))
    )
    merged["vault_label"] = merged["pod1_vault_mean_um"]
    merged["verify_status"] = merged["label_verify_status"]
    merged["measurement_input_status"] = merged["measurement_ready_status"].map(
        {
            "measurement_ready": "ready",
            "measurement_ready_with_confirmed_outlier": "ready_with_confirmed_outlier",
        }
    )
    merged["notes"] = BASE_NOTE

    columns = [
        "sample_id",
        "patient_id",
        "patient_uid",
        "eye_side",
        "eye",
        "split",
        "cct_mean_um",
        "cct_scan1_um",
        "cct_scan2_um",
        "acd_epi_mean_mm",
        "acd_epi_scan1_mm",
        "acd_epi_scan2_mm",
        "acd_endo_mean_mm",
        "acd_endo_scan1_mm",
        "acd_endo_scan2_mm",
        "clr_mean_um",
        "clr_scan1_um",
        "clr_scan2_um",
        "ata_mean_mm",
        "ata_scan1_mm",
        "ata_scan2_mm",
        "num_preop_measurement_records",
        "measurement_ready_status",
        "measurement_input_status",
        "vault_label",
        "pod1_vault_mean_um",
        "pod1_vault_range_um",
        "label_qc_flag",
        "label_status",
        "verify_status",
        "measurement_source_images",
        "measurement_crop_paths",
        "notes",
    ]
    for column in columns:
        if column not in merged.columns:
            merged[column] = ""
    return merged[columns]


def validate_output(df: pd.DataFrame, candidate_df: pd.DataFrame, name: str) -> None:
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{name}: sample_id is not unique.")
    if df["split"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: split contains empty values.")
    vault_label = pd.to_numeric(df["vault_label"], errors="coerce")
    if vault_label.isna().any() or (vault_label <= 0).any():
        raise ValueError(f"{name}: vault_label must be non-empty and > 0.")
    for column in ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]:
        if pd.to_numeric(df[column], errors="coerce").isna().any():
            raise ValueError(f"{name}: {column} contains missing values.")
    used_samples = set(df["sample_id"].astype(str))
    used_records = candidate_df[candidate_df["sample_id"].astype(str).isin(used_samples)]
    if is_excluded(used_records).any():
        raise ValueError(f"{name}: excluded records entered the manifest.")
    if leakage_risk(used_records).any():
        raise ValueError(f"{name}: leakage risk records entered the manifest.")
    patient042_bad = used_records[
        used_records["patient_uid"].astype(str).eq("patient_042")
        & used_records["notes"].str.contains("possible_postop_record_originally_misclassified", regex=False)
    ]
    if not patient042_bad.empty:
        raise ValueError(f"{name}: patient_042 old postop records entered the manifest.")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def print_split_counts(df: pd.DataFrame, label: str) -> None:
    counts = df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    print(f"{label} train/val/test rows: {counts['train']} / {counts['val']} / {counts['test']}")


def main() -> None:
    args = parse_args()
    measurements_path = resolve_project_path(args.measurements)
    clean_manifest_path = resolve_project_path(args.clean_manifest)
    pod1_labels_path = resolve_project_path(args.pod1_labels)
    validation_report_path = resolve_project_path(args.validation_report)
    ready_out = resolve_project_path(args.ready_out)
    strict_out = resolve_project_path(args.strict_out)

    measurements_df = normalize_measurements(pd.read_csv(measurements_path))
    base_df = pd.read_csv(clean_manifest_path)
    labels_df = pd.read_csv(pod1_labels_path)
    validation_df = pd.read_csv(validation_report_path)

    candidate_df = baseline_candidate_records(measurements_df)
    features_df = aggregate_measurements(candidate_df)
    full_manifest = build_manifest(base_df, labels_df, validation_df, features_df)

    ready_df = full_manifest[
        full_manifest["measurement_ready_status"].isin(
            ["measurement_ready", "measurement_ready_with_confirmed_outlier"]
        )
    ].copy()
    strict_df = full_manifest[full_manifest["measurement_ready_status"].eq("measurement_ready")].copy()

    validate_output(ready_df, candidate_df, "ready manifest")
    validate_output(strict_df, candidate_df, "strict manifest")
    write_csv(ready_df, ready_out)
    write_csv(strict_df, strict_out)

    p42_ready = ready_df[ready_df["patient_uid"].astype(str).eq("patient_042")]
    p42_records = candidate_df[candidate_df["patient_uid"].astype(str).eq("patient_042")]
    p42_uses_new = bool(
        not p42_ready.empty
        and not p42_records.empty
        and p42_records["notes"].str.contains("newly_added_patient042_true_preop_2danalysis", regex=False).all()
    )
    leakage = bool(leakage_risk(candidate_df[candidate_df["sample_id"].isin(ready_df["sample_id"])]).any())

    print(f"Ready manifest rows: {len(ready_df)}")
    print(f"Strict manifest rows: {len(strict_df)}")
    print_split_counts(ready_df, "Ready manifest")
    print_split_counts(strict_df, "Strict manifest")
    print("measurement_ready_status distribution:")
    print(ready_df["measurement_ready_status"].value_counts().to_string())
    print(f"patient_042 samples included: {len(p42_ready)}; uses newly added true preop records: {p42_uses_new}")
    print(f"Leakage risk present: {leakage}")
    print(f"Ready output: {relative_path(ready_out)}")
    print(f"Strict output: {relative_path(strict_out)}")


if __name__ == "__main__":
    main()
