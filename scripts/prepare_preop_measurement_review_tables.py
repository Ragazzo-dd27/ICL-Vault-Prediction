"""Prepare true preoperative CASIA2 measurement review tables for batch_01.

This script only uses initial CASIA2 2DAnalysis records with is_preop == TRUE.
POD1 postoperative measurements must not be used as preoperative input
features, because that would leak postoperative label information into a
preoperative baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_MEASUREMENT_COLUMNS = ["cct_um", "acd_epi_mm", "acd_endo_mm", "clr_um", "ata_mm"]
OUTPUT_COLUMNS = [
    "sample_id",
    "patient_uid",
    "eye",
    "split",
    "exam_date",
    "exam_time",
    "analysis_index",
    "image_path",
    "measurement_crop_path",
    "cct_um",
    "acd_epi_mm",
    "acd_endo_mm",
    "vault_um",
    "clr_um",
    "ata_mm",
    "has_vault",
    "vault_raw_text",
    "extraction_method",
    "verify_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare batch_01 preop measurement review tables.")
    parser.add_argument(
        "--measurements_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_initial.csv",
    )
    parser.add_argument(
        "--clean_manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    )
    parser.add_argument(
        "--preop_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review.csv",
    )
    parser.add_argument(
        "--priority_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review_priority.csv",
    )
    parser.add_argument(
        "--status_out",
        type=str,
        default="artifacts/reports/as_oct_pod1_baseline_batch_01/preop_measurement_review_status.csv",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def numeric_present(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().any()


def verified_present(series: pd.Series) -> bool:
    return series.fillna("").astype(str).str.strip().str.lower().eq("verified").any()


def append_note(base: object, addition: str) -> str:
    text = "" if pd.isna(base) else str(base).strip()
    if not text:
        return addition
    if addition in text:
        return text
    return f"{text} | {addition}"


def build_sample_lookup(clean_df: pd.DataFrame) -> Dict[tuple[str, str], List[Dict[str, object]]]:
    lookup: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in clean_df.to_dict(orient="records"):
        key = (str(row["patient_uid"]), str(row["eye"]))
        lookup.setdefault(key, []).append(row)
    return lookup


def attach_clean_sample_columns(preop_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    clean_keys = clean_df[["sample_id", "patient_uid", "eye", "split"]].copy()
    merged = preop_df.merge(clean_keys, on=["patient_uid", "eye"], how="left")
    merged["sample_id"] = merged["sample_id"].fillna("")
    merged["split"] = merged["split"].fillna("")
    for column in OUTPUT_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    return merged[OUTPUT_COLUMNS].sort_values(
        by=["patient_uid", "eye", "exam_date", "exam_time", "analysis_index"],
        kind="stable",
        na_position="last",
    )


def build_status(clean_df: pd.DataFrame, preop_priority_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for row in clean_df.sort_values(by=["patient_uid", "eye", "sample_id"], kind="stable").to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        patient_uid = str(row["patient_uid"])
        eye = str(row["eye"])
        group = preop_priority_df[preop_priority_df["sample_id"] == sample_id]
        notes: List[str] = [
            "POD1 postoperative measurements must not be used as preoperative input features."
        ]

        if group.empty:
            status = "missing_preop_2danalysis"
            has_values = {column: False for column in KEY_MEASUREMENT_COLUMNS}
        else:
            has_values = {column: numeric_present(group[column]) for column in KEY_MEASUREMENT_COLUMNS}
            all_key_values = all(has_values.values())
            any_verified = verified_present(group["verify_status"])
            if all_key_values and any_verified:
                status = "measurement_ready"
            else:
                status = "manual_review_needed"

            if pd.to_numeric(group["vault_um"], errors="coerce").notna().any():
                notes.append("warning_preop_vault_um_present_review_for_leakage")

        rows.append(
            {
                "sample_id": sample_id,
                "patient_uid": patient_uid,
                "eye": eye,
                "split": row["split"],
                "num_preop_records": int(len(group)),
                "has_cct": has_values["cct_um"],
                "has_acd_epi": has_values["acd_epi_mm"],
                "has_acd_endo": has_values["acd_endo_mm"],
                "has_clr": has_values["clr_um"],
                "has_ata": has_values["ata_mm"],
                "measurement_ready_status": status,
                "notes": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    measurements_in = resolve_project_path(args.measurements_in)
    clean_manifest = resolve_project_path(args.clean_manifest)
    preop_out = resolve_project_path(args.preop_out)
    priority_out = resolve_project_path(args.priority_out)
    status_out = resolve_project_path(args.status_out)

    measurements_df = pd.read_csv(measurements_in)
    clean_df = pd.read_csv(clean_manifest)

    measurements_df["is_preop"] = normalize_bool_series(measurements_df["is_preop"])
    preop_df = measurements_df[measurements_df["is_preop"]].copy()

    if pd.to_numeric(preop_df["vault_um"], errors="coerce").notna().any():
        preop_df["notes"] = preop_df.apply(
            lambda row: append_note(row.get("notes", ""), "warning_preop_vault_um_present_review_for_leakage")
            if pd.notna(pd.to_numeric(pd.Series([row.get("vault_um")]), errors="coerce").iloc[0])
            else row.get("notes", ""),
            axis=1,
        )

    preop_review_df = attach_clean_sample_columns(preop_df, clean_df)
    priority_df = preop_review_df[preop_review_df["sample_id"].astype(str).str.strip().ne("")].copy()
    status_df = build_status(clean_df=clean_df, preop_priority_df=priority_df)

    write_csv(preop_review_df, preop_out)
    write_csv(priority_df, priority_out)
    write_csv(status_df, status_out)

    status_counts = status_df["measurement_ready_status"].value_counts().to_dict()
    print("Reminder: POD1 postoperative measurements must not be used as preoperative input features.")
    print(f"All preop records: {len(preop_review_df)}")
    print(f"Clean manifest samples: {len(clean_df)}")
    print(f"Priority preop review records: {len(priority_df)}")
    print(f"measurement_ready samples: {status_counts.get('measurement_ready', 0)}")
    print(f"manual_review_needed samples: {status_counts.get('manual_review_needed', 0)}")
    print(f"missing_preop_2danalysis samples: {status_counts.get('missing_preop_2danalysis', 0)}")
    print(f"Preop review output: {relative_path(preop_out)}")
    print(f"Priority review output: {relative_path(priority_out)}")
    print(f"Status output: {relative_path(status_out)}")


if __name__ == "__main__":
    main()
