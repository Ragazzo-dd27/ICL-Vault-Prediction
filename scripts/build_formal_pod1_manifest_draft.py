"""Build a batch_01 eye-level POD1 formal vault manifest draft.

This script aligns preoperative AS-OCT input candidates with POD1 verified
vault labels. It is a POD1 manifest draft, not the final training manifest:
UBM paths are only patient-level associated and still need eye/date/visit
alignment before any final multimodal training manifest is created.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT_PREFIX = "data/raw/real_export_batch_01/patients"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build batch_01 POD1 formal vault manifest draft from verified labels."
    )
    parser.add_argument(
        "--labels_in",
        type=str,
        default="data/manifests/vault_label_candidates_batch_01_pod1_verified.csv",
        help="Input eye-level POD1 verified vault label table.",
    )
    parser.add_argument(
        "--measurements_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_initial.csv",
        help="Input initial CASIA2 2DAnalysis measurement table.",
    )
    parser.add_argument(
        "--manifest_in",
        type=str,
        default="data/manifests/real_export_batch_01_manifest_initial.csv",
        help="Input initial real-export manifest used for preop image paths.",
    )
    parser.add_argument(
        "--summary_in",
        type=str,
        default="data/manifests/real_export_batch_01_summary.csv",
        help="Optional patient-level summary used to supplement UBM availability.",
    )
    parser.add_argument(
        "--manifest_out",
        type=str,
        default="data/manifests/formal_vault_manifest_batch_01_pod1_draft.csv",
        help="Output POD1 formal vault manifest draft CSV.",
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


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if "." in text:
        text = text.split(".", maxsplit=1)[0]
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 3:
            year, month, day = parts
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    if "-" in text:
        parts = text.split("-")
        if len(parts) == 3:
            year, month, day = parts
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def normalize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()


def split_paths(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def join_unique_paths(values: Iterable[object], prefix_raw_root: bool = False) -> str:
    paths: List[str] = []
    for value in values:
        for item in split_paths(value):
            path = item.replace("\\", "/")
            if prefix_raw_root and not path.startswith("data/") and not Path(path).is_absolute():
                path = f"{RAW_ROOT_PREFIX}/{path}"
            paths.append(path)
    return ";".join(dict.fromkeys(paths))


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def build_preop_lookup(measurements_df: pd.DataFrame) -> Dict[tuple[str, str], Dict[str, object]]:
    measurements = measurements_df.copy()
    measurements["is_preop"] = normalize_bool_series(measurements["is_preop"])
    measurements["exam_date_norm"] = measurements["exam_date"].map(normalize_date)
    preop = measurements[measurements["is_preop"]].copy()
    lookup: Dict[tuple[str, str], Dict[str, object]] = {}

    for (patient_uid, eye), group in preop.groupby(["patient_uid", "eye"], dropna=False):
        group = group.sort_values(
            by=["exam_date_norm", "exam_time", "analysis_index", "image_path"],
            kind="stable",
            na_position="last",
        )
        preop_dates = [value for value in group["exam_date_norm"].dropna().unique().tolist() if value]
        preop_exam_date = sorted(preop_dates)[0] if preop_dates else ""
        date_group = group[group["exam_date_norm"] == preop_exam_date] if preop_exam_date else group
        lookup[(str(patient_uid), str(eye))] = {
            "preop_exam_date": preop_exam_date,
            "num_preop_2d_analysis": int(len(date_group)),
            "preop_2d_analysis_paths": join_unique_paths(date_group["image_path"]),
        }

    return lookup


def build_manifest_preop_lookup(manifest_df: pd.DataFrame) -> Dict[tuple[str, str, str], Dict[str, object]]:
    manifest = manifest_df.copy()
    manifest["date_norm"] = manifest["date"].map(normalize_date)
    manifest["has_oct_raw"] = normalize_bool_series(manifest["has_oct_raw"])
    manifest["has_oct_2d_analysis"] = normalize_bool_series(manifest["has_oct_2d_analysis"])
    lookup: Dict[tuple[str, str, str], Dict[str, object]] = {}

    for (patient_uid, eye, date_norm), group in manifest.groupby(["patient_uid", "eye", "date_norm"], dropna=False):
        lookup[(str(patient_uid), str(eye), str(date_norm))] = {
            "preop_as_oct_raw_paths": join_unique_paths(group["oct_raw_paths"], prefix_raw_root=True),
            "preop_as_oct_2d_analysis_paths": join_unique_paths(
                group["oct_2d_analysis_paths"],
                prefix_raw_root=True,
            ),
            "has_preop_as_oct_raw": bool(group["has_oct_raw"].any()),
            "has_preop_2d_analysis": bool(group["has_oct_2d_analysis"].any()),
        }

    return lookup


def build_ubm_lookup(manifest_df: pd.DataFrame, summary_df: pd.DataFrame | None) -> Dict[str, Dict[str, object]]:
    manifest = manifest_df.copy()
    manifest["has_ubm_horizontal"] = normalize_bool_series(manifest["has_ubm_horizontal"])
    manifest["has_ubm_vertical"] = normalize_bool_series(manifest["has_ubm_vertical"])
    lookup: Dict[str, Dict[str, object]] = {}

    for patient_uid, group in manifest.groupby("patient_uid", dropna=False):
        horizontal_paths = join_unique_paths(group["ubm_horizontal_paths"], prefix_raw_root=True)
        vertical_paths = join_unique_paths(group["ubm_vertical_paths"], prefix_raw_root=True)
        has_horizontal = bool(group["has_ubm_horizontal"].any() or horizontal_paths)
        has_vertical = bool(group["has_ubm_vertical"].any() or vertical_paths)
        lookup[str(patient_uid)] = {
            "ubm_horizontal_paths": horizontal_paths,
            "ubm_vertical_paths": vertical_paths,
            "has_ubm_horizontal": has_horizontal,
            "has_ubm_vertical": has_vertical,
            "has_ubm": bool(has_horizontal or has_vertical),
        }

    if summary_df is not None and not summary_df.empty and "patient_uid" in summary_df.columns:
        summary = summary_df.copy()
        if "has_any_ubm" in summary.columns:
            summary["has_any_ubm"] = normalize_bool_series(summary["has_any_ubm"])
            for row in summary.to_dict(orient="records"):
                patient_uid = str(row["patient_uid"])
                entry = lookup.setdefault(
                    patient_uid,
                    {
                        "ubm_horizontal_paths": "",
                        "ubm_vertical_paths": "",
                        "has_ubm_horizontal": False,
                        "has_ubm_vertical": False,
                        "has_ubm": False,
                    },
                )
                entry["has_ubm"] = bool(entry["has_ubm"] or row["has_any_ubm"])

    for entry in lookup.values():
        entry["ubm_alignment_status"] = "patient_level_available" if entry["has_ubm"] else "missing"
    return lookup


def build_notes(label_qc_flag: str, has_ubm: bool) -> str:
    notes = [
        "This is a POD1 formal manifest draft, not the final training manifest.",
        "UBM is only patient-level associated and needs further eye/date/visit alignment.",
        "Large between-scan vault difference samples should be reviewed before final experiments.",
    ]
    if label_qc_flag == "large_between_scan_difference":
        notes.append("clinical_review_recommended_for_large_between_scan_difference")
    if not has_ubm:
        notes.append("missing_ubm")
    return " | ".join(notes)


def training_ready_status(has_preop_as_oct_raw: bool, pod1_vault_mean_um: object, verify_status: str) -> str:
    if not has_preop_as_oct_raw:
        return "missing_preop_as_oct"
    if pd.notna(pd.to_numeric(pd.Series([pod1_vault_mean_um]), errors="coerce").iloc[0]) and verify_status == "verified":
        return "image_label_ready"
    return "missing_verified_label"


def build_manifest(
    labels_df: pd.DataFrame,
    preop_lookup: Dict[tuple[str, str], Dict[str, object]],
    manifest_preop_lookup: Dict[tuple[str, str, str], Dict[str, object]],
    ubm_lookup: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for label in labels_df.sort_values(by=["patient_uid", "eye", "sample_id"], kind="stable").to_dict(orient="records"):
        patient_uid = str(label["patient_uid"])
        eye = str(label["eye"])
        preop_info = preop_lookup.get(
            (patient_uid, eye),
            {"preop_exam_date": "", "num_preop_2d_analysis": 0, "preop_2d_analysis_paths": ""},
        )
        preop_exam_date = str(preop_info["preop_exam_date"])
        manifest_info = manifest_preop_lookup.get(
            (patient_uid, eye, preop_exam_date),
            {
                "preop_as_oct_raw_paths": "",
                "preop_as_oct_2d_analysis_paths": preop_info["preop_2d_analysis_paths"],
                "has_preop_as_oct_raw": False,
                "has_preop_2d_analysis": bool(preop_info["preop_2d_analysis_paths"]),
            },
        )
        ubm_info = ubm_lookup.get(
            patient_uid,
            {
                "ubm_horizontal_paths": "",
                "ubm_vertical_paths": "",
                "has_ubm_horizontal": False,
                "has_ubm_vertical": False,
                "has_ubm": False,
                "ubm_alignment_status": "missing",
            },
        )
        label_qc_flag = str(label.get("qc_flag", ""))
        ready_status = training_ready_status(
            bool(manifest_info["has_preop_as_oct_raw"]),
            label.get("pod1_vault_mean_um", ""),
            str(label.get("verify_status", "")),
        )

        rows.append(
            {
                "sample_id": label["sample_id"],
                "patient_uid": patient_uid,
                "eye": eye,
                "preop_exam_date": preop_exam_date,
                "num_preop_2d_analysis": preop_info["num_preop_2d_analysis"],
                "preop_as_oct_raw_paths": manifest_info["preop_as_oct_raw_paths"],
                "preop_as_oct_2d_analysis_paths": manifest_info["preop_as_oct_2d_analysis_paths"],
                "has_preop_as_oct_raw": manifest_info["has_preop_as_oct_raw"],
                "has_preop_2d_analysis": manifest_info["has_preop_2d_analysis"],
                "has_ubm": ubm_info["has_ubm"],
                "ubm_horizontal_paths": ubm_info["ubm_horizontal_paths"],
                "ubm_vertical_paths": ubm_info["ubm_vertical_paths"],
                "ubm_alignment_status": ubm_info["ubm_alignment_status"],
                "label_exam_date": label.get("label_exam_date", ""),
                "label_exam_time": label.get("label_exam_time", ""),
                "postop_day": label.get("postop_day", ""),
                "pod1_vault_scan1_um": label.get("pod1_vault_scan1_um", ""),
                "pod1_vault_scan2_um": label.get("pod1_vault_scan2_um", ""),
                "pod1_vault_mean_um": label.get("pod1_vault_mean_um", ""),
                "pod1_vault_median_um": label.get("pod1_vault_median_um", ""),
                "pod1_vault_min_um": label.get("pod1_vault_min_um", ""),
                "pod1_vault_max_um": label.get("pod1_vault_max_um", ""),
                "pod1_vault_range_um": label.get("pod1_vault_range_um", ""),
                "label_qc_flag": label_qc_flag,
                "label_status": label.get("label_status", ""),
                "verify_status": label.get("verify_status", ""),
                "training_ready_status": ready_status,
                "split": "",
                "notes": build_notes(label_qc_flag=label_qc_flag, has_ubm=bool(ubm_info["has_ubm"])),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "patient_uid",
            "eye",
            "preop_exam_date",
            "num_preop_2d_analysis",
            "preop_as_oct_raw_paths",
            "preop_as_oct_2d_analysis_paths",
            "has_preop_as_oct_raw",
            "has_preop_2d_analysis",
            "has_ubm",
            "ubm_horizontal_paths",
            "ubm_vertical_paths",
            "ubm_alignment_status",
            "label_exam_date",
            "label_exam_time",
            "postop_day",
            "pod1_vault_scan1_um",
            "pod1_vault_scan2_um",
            "pod1_vault_mean_um",
            "pod1_vault_median_um",
            "pod1_vault_min_um",
            "pod1_vault_max_um",
            "pod1_vault_range_um",
            "label_qc_flag",
            "label_status",
            "verify_status",
            "training_ready_status",
            "split",
            "notes",
        ],
    )


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def print_summary(output_df: pd.DataFrame, output_path: Path) -> None:
    missing_ubm = output_df[~normalize_bool_series(output_df["has_ubm"])]
    large_samples = output_df[
        output_df["label_qc_flag"].fillna("").astype(str).str.contains("large_between_scan_difference", regex=False)
    ]["sample_id"].tolist()

    print(f"Output manifest rows: {len(output_df)}")
    print(f"Patients: {output_df['patient_uid'].nunique() if not output_df.empty else 0}")
    print(f"Eye-level samples: {output_df['sample_id'].nunique() if not output_df.empty else 0}")
    print(f"Samples with preop AS-OCT raw: {int(normalize_bool_series(output_df['has_preop_as_oct_raw']).sum())}")
    print(f"Samples with UBM: {int(normalize_bool_series(output_df['has_ubm']).sum())}")
    print(f"Patients missing UBM: {missing_ubm['patient_uid'].nunique() if not missing_ubm.empty else 0}")
    print(f"Samples missing UBM: {len(missing_ubm)}")
    print("training_ready_status distribution:")
    for status, count in output_df["training_ready_status"].value_counts().sort_index().items():
        print(f"  {status}: {count}")
    print("label_qc_flag distribution:")
    for flag, count in output_df["label_qc_flag"].value_counts().sort_index().items():
        print(f"  {flag}: {count}")
    print(
        "large_between_scan_difference samples: "
        f"{', '.join(large_samples) if large_samples else 'none'}"
    )
    print(f"Output: {relative_path(output_path)}")


def main() -> None:
    args = parse_args()
    labels_in = resolve_project_path(args.labels_in)
    measurements_in = resolve_project_path(args.measurements_in)
    manifest_in = resolve_project_path(args.manifest_in)
    summary_in = resolve_project_path(args.summary_in)
    manifest_out = resolve_project_path(args.manifest_out)

    labels_df = pd.read_csv(labels_in)
    measurements_df = pd.read_csv(measurements_in)
    export_manifest_df = pd.read_csv(manifest_in)
    summary_df = pd.read_csv(summary_in) if summary_in.exists() else None

    normalize_text_columns(labels_df, ["sample_id", "patient_uid", "eye", "label_status", "verify_status", "qc_flag"])
    normalize_text_columns(measurements_df, ["patient_uid", "eye", "exam_date", "exam_time", "image_path"])
    normalize_text_columns(export_manifest_df, ["patient_uid", "eye"])

    preop_lookup = build_preop_lookup(measurements_df)
    manifest_preop_lookup = build_manifest_preop_lookup(export_manifest_df)
    ubm_lookup = build_ubm_lookup(export_manifest_df, summary_df)
    output_df = build_manifest(
        labels_df=labels_df,
        preop_lookup=preop_lookup,
        manifest_preop_lookup=manifest_preop_lookup,
        ubm_lookup=ubm_lookup,
    )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(manifest_out, index=False, encoding="utf-8")
    print_summary(output_df=output_df, output_path=manifest_out)


if __name__ == "__main__":
    main()
