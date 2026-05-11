"""Build eye-level POD1 verified vault labels from manually checked CASIA2 rows.

This is POD1 verified label construction for manual-review outputs. It is not
a formal training manifest builder, does not run OCR, and does not modify the
source CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build eye-level POD1 verified vault labels from checked manual-review rows."
    )
    parser.add_argument(
        "--pod1_checked_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_pod1_manual_review_checked.csv",
        help="Input POD1 manual-review CSV after human verification.",
    )
    parser.add_argument(
        "--labels_out",
        type=str,
        default="data/manifests/vault_label_candidates_batch_01_pod1_verified.csv",
        help="Output eye-level POD1 verified vault label CSV.",
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


def normalize_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def normalize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()


def sample_id_for(patient_uid: str, eye: str) -> str:
    return f"{patient_uid}_{eye}_POD1"


def join_unique(values: Iterable[object]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"]
    return ";".join(dict.fromkeys(clean))


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def mean_or_empty(series: pd.Series) -> float | str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return float(numeric.mean())


def sorted_scan_records(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(
        by=["analysis_index", "exam_time", "image_path"],
        kind="stable",
        na_position="last",
    )


def scan_value(group: pd.DataFrame, scan_position: int, column: str) -> float | str:
    if len(group) <= scan_position:
        return ""
    value = group.iloc[scan_position][column]
    return "" if pd.isna(value) else float(value)


def build_qc_flag(verified_group: pd.DataFrame) -> str:
    if verified_group.empty:
        return "missing_label"

    flags: List[str] = []
    vault_values = verified_group["vault_um"].dropna()
    if (vault_values <= 0).any():
        flags.append("invalid_vault_value")

    acd_pairs = verified_group[["acd_epi_mm", "acd_endo_mm"]].dropna(how="any")
    if not acd_pairs.empty and (acd_pairs["acd_epi_mm"] <= acd_pairs["acd_endo_mm"]).any():
        flags.append("acd_order_check")

    if len(verified_group) == 1:
        flags.append("single_record_only")
    elif not vault_values.empty:
        vault_range = float(vault_values.max() - vault_values.min())
        if vault_range > 100:
            flags.append("large_between_scan_difference")
        else:
            flags.append("ok")
    else:
        flags.append("missing_label")

    return "|".join(dict.fromkeys(flags))


def build_eye_level_rows(pod1_df: pd.DataFrame, verified_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    verified_groups = {
        key: group.copy()
        for key, group in verified_df.groupby(["patient_uid", "eye"], dropna=False)
    }

    for (patient_uid, eye), full_group in pod1_df.groupby(["patient_uid", "eye"], dropna=False):
        patient_uid = str(patient_uid)
        eye = str(eye)
        full_group = full_group.sort_values(
            by=["exam_date", "exam_time", "analysis_index", "image_path"],
            kind="stable",
            na_position="last",
        )
        verified_group = verified_groups.get((patient_uid, eye), pd.DataFrame(columns=pod1_df.columns))
        verified_group = sorted_scan_records(verified_group)

        vault_values = verified_group["vault_um"].dropna() if not verified_group.empty else pd.Series(dtype=float)
        has_verified_label = not vault_values.empty
        qc_flag = build_qc_flag(verified_group)

        notes = [
            "POD1 verified label construction only",
            "not a formal training manifest",
            "analysis_index rows are aggregated to one eye-level label",
        ]
        if "missing_label" in qc_flag:
            notes.append("no verified POD1 vault value passed filters")
        if "large_between_scan_difference" in qc_flag:
            notes.append("manual review recommended for between-scan vault difference")
        if len(verified_group) > 2:
            notes.append("more_than_two_records")

        rows.append(
            {
                "sample_id": sample_id_for(patient_uid, eye),
                "patient_uid": patient_uid,
                "eye": eye,
                "label_exam_date": first_nonempty(verified_group["exam_date"] if has_verified_label else full_group["exam_date"]),
                "label_exam_time": first_nonempty(verified_group["exam_time"] if has_verified_label else full_group["exam_time"]),
                "postop_day": 1,
                "num_pod1_records": int(len(verified_group)),
                "scan1_analysis_index": scan_value(verified_group, 0, "analysis_index") if has_verified_label else "",
                "scan2_analysis_index": scan_value(verified_group, 1, "analysis_index") if has_verified_label else "",
                "pod1_vault_scan1_um": scan_value(verified_group, 0, "vault_um") if has_verified_label else "",
                "pod1_vault_scan2_um": scan_value(verified_group, 1, "vault_um") if has_verified_label else "",
                "pod1_vault_mean_um": float(vault_values.mean()) if has_verified_label else "",
                "pod1_vault_median_um": float(vault_values.median()) if has_verified_label else "",
                "pod1_vault_min_um": float(vault_values.min()) if has_verified_label else "",
                "pod1_vault_max_um": float(vault_values.max()) if has_verified_label else "",
                "pod1_vault_range_um": float(vault_values.max() - vault_values.min()) if has_verified_label else "",
                "cct_mean_um": mean_or_empty(verified_group["cct_um"]) if has_verified_label else "",
                "acd_epi_mean_mm": mean_or_empty(verified_group["acd_epi_mm"]) if has_verified_label else "",
                "acd_endo_mean_mm": mean_or_empty(verified_group["acd_endo_mm"]) if has_verified_label else "",
                "clr_mean_um": mean_or_empty(verified_group["clr_um"]) if has_verified_label else "",
                "ata_mean_mm": mean_or_empty(verified_group["ata_mm"]) if has_verified_label else "",
                "vault_source_images": join_unique(verified_group["image_path"]) if has_verified_label else "",
                "measurement_crop_paths": join_unique(verified_group["measurement_crop_path"]) if has_verified_label else "",
                "label_status": "valid" if has_verified_label else "missing",
                "verify_status": "verified" if has_verified_label else "missing",
                "qc_flag": qc_flag,
                "notes": " | ".join(notes),
            }
        )

    output_df = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "patient_uid",
            "eye",
            "label_exam_date",
            "label_exam_time",
            "postop_day",
            "num_pod1_records",
            "scan1_analysis_index",
            "scan2_analysis_index",
            "pod1_vault_scan1_um",
            "pod1_vault_scan2_um",
            "pod1_vault_mean_um",
            "pod1_vault_median_um",
            "pod1_vault_min_um",
            "pod1_vault_max_um",
            "pod1_vault_range_um",
            "cct_mean_um",
            "acd_epi_mean_mm",
            "acd_endo_mean_mm",
            "clr_mean_um",
            "ata_mean_mm",
            "vault_source_images",
            "measurement_crop_paths",
            "label_status",
            "verify_status",
            "qc_flag",
            "notes",
        ],
    )
    return output_df.sort_values(by=["patient_uid", "eye", "sample_id"], kind="stable")


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def print_summary(input_records: int, verified_records: int, output_df: pd.DataFrame, output_path: Path) -> None:
    mean_values = pd.to_numeric(output_df["pod1_vault_mean_um"], errors="coerce").dropna()
    large_samples = output_df[
        output_df["qc_flag"].fillna("").astype(str).str.contains("large_between_scan_difference", regex=False)
    ]["sample_id"].tolist()
    empty_scan1_with_records = output_df[
        (pd.to_numeric(output_df["num_pod1_records"], errors="coerce").fillna(0) >= 1)
        & (output_df["pod1_vault_scan1_um"].fillna("").astype(str).str.strip() == "")
    ]

    print(f"Input POD1 records: {input_records}")
    print(f"Verified POD1 records after filters: {verified_records}")
    print(f"Output eye-level samples: {len(output_df)}")
    print("label_status distribution:")
    if output_df.empty:
        print("  none")
    else:
        for status, count in output_df["label_status"].value_counts().sort_index().items():
            print(f"  {status}: {count}")
    print("verify_status distribution:")
    if output_df.empty:
        print("  none")
    else:
        for status, count in output_df["verify_status"].value_counts().sort_index().items():
            print(f"  {status}: {count}")
    print("QC flag distribution:")
    if output_df.empty:
        print("  none")
    else:
        for flag, count in output_df["qc_flag"].value_counts().sort_index().items():
            print(f"  {flag}: {count}")

    if mean_values.empty:
        print("pod1_vault_mean_um stats: no verified values")
    else:
        print(
            "pod1_vault_mean_um stats: "
            f"mean={mean_values.mean():.2f}, "
            f"std={mean_values.std():.2f}, "
            f"min={mean_values.min():.2f}, "
            f"max={mean_values.max():.2f}"
        )
    print(
        "Empty pod1_vault_scan1_um with num_pod1_records >= 1: "
        f"{len(empty_scan1_with_records)}"
    )
    print(
        "large_between_scan_difference samples: "
        f"{', '.join(large_samples) if large_samples else 'none'}"
    )
    print(f"Output: {relative_path(output_path)}")


def main() -> None:
    args = parse_args()
    pod1_checked_in = resolve_project_path(args.pod1_checked_in)
    labels_out = resolve_project_path(args.labels_out)

    pod1_df = pd.read_csv(pod1_checked_in)
    normalize_text_columns(
        pod1_df,
        ["patient_uid", "eye", "exam_date", "exam_time", "verify_status", "extraction_method", "image_path", "measurement_crop_path"],
    )
    normalize_numeric_columns(
        pod1_df,
        [
            "analysis_index",
            "postop_day",
            "vault_um",
            "cct_um",
            "acd_epi_mm",
            "acd_endo_mm",
            "clr_um",
            "ata_mm",
        ],
    )
    pod1_df["is_postop"] = normalize_bool_series(pod1_df["is_postop"])
    pod1_df["has_vault"] = normalize_bool_series(pod1_df["has_vault"])

    pod1_records = pod1_df[(pod1_df["is_postop"]) & (pod1_df["postop_day"] == 1)].copy()
    verified_records = pod1_records[
        (pod1_records["has_vault"])
        & (pod1_records["vault_um"].notna())
        & (pod1_records["verify_status"].str.lower() == "verified")
        & (pod1_records["extraction_method"].str.lower() == "manual_verified")
    ].copy()

    output_df = build_eye_level_rows(pod1_df=pod1_records, verified_df=verified_records)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(labels_out, index=False, encoding="utf-8")

    print_summary(
        input_records=len(pod1_records),
        verified_records=len(verified_records),
        output_df=output_df,
        output_path=labels_out,
    )


if __name__ == "__main__":
    main()
