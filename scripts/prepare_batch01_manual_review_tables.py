"""Prepare batch_01 manual-review tables from initial CASIA2 2DAnalysis CSVs.

This helper only reshapes already extracted review metadata. It does not run
OCR, does not train a model, and does not modify the initial measurement or
label-candidate CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORT_COLUMNS = ["patient_uid", "eye", "exam_date", "exam_time", "analysis_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare priority manual-review tables for batch_01 2DAnalysis records."
    )
    parser.add_argument(
        "--measurements_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_initial.csv",
        help="Input batch_01 initial CASIA2 2DAnalysis measurement CSV.",
    )
    parser.add_argument(
        "--labels_in",
        type=str,
        default="data/manifests/vault_label_candidates_batch_01.csv",
        help="Input batch_01 vault label candidate CSV.",
    )
    parser.add_argument(
        "--summary_in",
        type=str,
        default="data/manifests/real_export_batch_01_summary.csv",
        help="Optional batch_01 patient summary CSV used only to flag missing UBM.",
    )
    parser.add_argument(
        "--pod1_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_pod1_manual_review.csv",
        help="Output CSV for postoperative day 1 manual vault review.",
    )
    parser.add_argument(
        "--preop_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review.csv",
        help="Output CSV for preoperative manual measurement review.",
    )
    parser.add_argument(
        "--overview_out",
        type=str,
        default="data/interim/batch_01_eye_level_sample_overview.csv",
        help="Output CSV for patient-eye sample overview.",
    )
    parser.add_argument(
        "--unclassified_out",
        type=str,
        default="data/interim/batch_01_unclassified_2danalysis_records.csv",
        help="Output CSV for unclassified 2DAnalysis records.",
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


def normalize_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sort_measurement_rows(df: pd.DataFrame) -> pd.DataFrame:
    sortable = df.copy()
    sortable["_analysis_index_sort"] = normalize_numeric_series(sortable.get("analysis_index", pd.Series(dtype=object)))
    sort_columns = [column for column in SORT_COLUMNS if column in sortable.columns]
    by = [column if column != "analysis_index" else "_analysis_index_sort" for column in sort_columns]
    sorted_df = sortable.sort_values(by=by, kind="stable", na_position="last")
    return sorted_df.drop(columns=["_analysis_index_sort"], errors="ignore")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def sample_id_for(patient_uid: str, eye: str, preop_exam_date: str) -> str:
    if preop_exam_date:
        return f"{patient_uid}_{eye}_{preop_exam_date.replace('-', '')}"
    return f"{patient_uid}_{eye}"


def load_ubm_flags(summary_in: Path) -> Dict[str, bool]:
    if not summary_in.exists():
        return {}

    summary_df = pd.read_csv(summary_in)
    if "patient_uid" not in summary_df.columns:
        return {}
    if "has_any_ubm" in summary_df.columns:
        flags = normalize_bool_series(summary_df["has_any_ubm"])
    else:
        ubm_columns = [
            column
            for column in ("num_ubm_horizontal_images", "num_ubm_vertical_images", "num_ubm_unknown_images")
            if column in summary_df.columns
        ]
        if not ubm_columns:
            return {}
        flags = summary_df[ubm_columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0

    return dict(zip(summary_df["patient_uid"].astype(str), flags.astype(bool)))


def build_overview(measurements_df: pd.DataFrame, labels_df: pd.DataFrame, ubm_flags: Dict[str, bool]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    measurements = measurements_df.copy()
    labels = labels_df.copy()
    measurements["is_preop"] = normalize_bool_series(measurements["is_preop"])
    measurements["is_postop"] = normalize_bool_series(measurements["is_postop"])
    measurements["postop_day"] = normalize_numeric_series(measurements["postop_day"])
    labels["postop_day"] = normalize_numeric_series(labels["postop_day"])

    grouped = measurements.groupby(["patient_uid", "eye"], dropna=False)
    label_grouped = {
        key: group.copy()
        for key, group in labels.groupby(["patient_uid", "eye"], dropna=False)
    }

    for (patient_uid, eye), group in grouped:
        patient_uid = str(patient_uid)
        eye = str(eye)
        preop_dates = sorted(
            str(value)
            for value in group.loc[group["is_preop"], "exam_date"].dropna().unique()
            if str(value)
        )
        preop_exam_date = preop_dates[0] if preop_dates else ""
        sample_id = sample_id_for(patient_uid, eye, preop_exam_date)
        label_group = label_grouped.get((patient_uid, eye), pd.DataFrame())
        postop_days = label_group["postop_day"].dropna() if not label_group.empty else pd.Series(dtype=float)
        has_ubm = bool(ubm_flags.get(patient_uid, False))

        notes: list[str] = [
            "manual review overview only",
            "POD1 candidates should be verified before label use",
        ]
        if not has_ubm:
            notes.append("patient has no UBM in batch_01 summary")
        if not preop_exam_date:
            notes.append("missing preop exam date")

        rows.append(
            {
                "patient_uid": patient_uid,
                "eye": eye,
                "sample_id": sample_id,
                "preop_exam_date": preop_exam_date,
                "num_preop_2d_analysis": int(group["is_preop"].sum()),
                "num_pod1_2d_analysis": int(((group["is_postop"]) & (group["postop_day"] == 1)).sum()),
                "has_pod1_candidate": bool((postop_days == 1).any()) if not postop_days.empty else False,
                "num_all_postop_candidates": int(len(label_group)),
                "min_postop_day": int(postop_days.min()) if not postop_days.empty else "",
                "max_postop_day": int(postop_days.max()) if not postop_days.empty else "",
                "has_ubm": has_ubm,
                "notes": " | ".join(notes),
            }
        )

    overview_df = pd.DataFrame(
        rows,
        columns=[
            "patient_uid",
            "eye",
            "sample_id",
            "preop_exam_date",
            "num_preop_2d_analysis",
            "num_pod1_2d_analysis",
            "has_pod1_candidate",
            "num_all_postop_candidates",
            "min_postop_day",
            "max_postop_day",
            "has_ubm",
            "notes",
        ],
    )
    return overview_df.sort_values(
        by=["patient_uid", "eye", "preop_exam_date", "sample_id"],
        kind="stable",
        na_position="last",
    )


def format_paths(paths: Iterable[Path]) -> str:
    return ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def main() -> None:
    args = parse_args()
    measurements_in = resolve_project_path(args.measurements_in)
    labels_in = resolve_project_path(args.labels_in)
    summary_in = resolve_project_path(args.summary_in)
    pod1_out = resolve_project_path(args.pod1_out)
    preop_out = resolve_project_path(args.preop_out)
    overview_out = resolve_project_path(args.overview_out)
    unclassified_out = resolve_project_path(args.unclassified_out)

    measurements_df = pd.read_csv(measurements_in)
    labels_df = pd.read_csv(labels_in)

    measurements_df["is_preop"] = normalize_bool_series(measurements_df["is_preop"])
    measurements_df["is_postop"] = normalize_bool_series(measurements_df["is_postop"])
    measurements_df["postop_day"] = normalize_numeric_series(measurements_df["postop_day"])

    pod1_df = sort_measurement_rows(
        measurements_df[(measurements_df["is_postop"]) & (measurements_df["postop_day"] == 1)].copy()
    )
    preop_df = sort_measurement_rows(measurements_df[measurements_df["is_preop"]].copy())
    unclassified_df = sort_measurement_rows(
        measurements_df[
            (measurements_df["eye"].fillna("").astype(str).str.lower() == "unknown")
            | (measurements_df["exam_date"].fillna("").astype(str).str.strip() == "")
            | ((~measurements_df["is_preop"]) & (~measurements_df["is_postop"]))
        ].copy()
    )

    ubm_flags = load_ubm_flags(summary_in)
    overview_df = build_overview(measurements_df=measurements_df, labels_df=labels_df, ubm_flags=ubm_flags)

    write_csv(pod1_df, pod1_out)
    write_csv(preop_df, preop_out)
    write_csv(overview_df, overview_out)
    write_csv(unclassified_df, unclassified_out)

    samples_with_pod1 = int(overview_df["has_pod1_candidate"].sum()) if not overview_df.empty else 0
    samples_missing_ubm = overview_df[~normalize_bool_series(overview_df["has_ubm"])] if not overview_df.empty else overview_df
    patients_missing_ubm = sorted(samples_missing_ubm["patient_uid"].unique().tolist()) if not samples_missing_ubm.empty else []

    print(f"Total 2DAnalysis records: {len(measurements_df)}")
    print(f"POD1 manual review records: {len(pod1_df)}")
    print(f"Preop manual review records: {len(preop_df)}")
    print(f"Eye-level samples: {len(overview_df)}")
    print(f"Unclassified records: {len(unclassified_df)}")
    print(f"Samples with POD1 candidate: {samples_with_pod1}")
    print(f"Patients missing UBM: {len(patients_missing_ubm)}")
    print(f"Samples missing UBM: {len(samples_missing_ubm)}")
    print(f"Patients missing UBM list: {', '.join(patients_missing_ubm) if patients_missing_ubm else 'none'}")
    print(f"Outputs: {format_paths([pod1_out, preop_out, overview_out, unclassified_out])}")


if __name__ == "__main__":
    main()
