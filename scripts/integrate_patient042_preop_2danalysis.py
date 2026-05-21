"""Integrate newly supplied true preop CASIA2 2DAnalysis images for patient_042.

This is an incremental data-fix helper. Postoperative 2DAnalysis measurements
must not be used as preoperative input features, because doing so would leak
postoperative vault information into a preoperative measurement baseline.

The script does not modify raw images, POD1 labels, training scripts, or the
original checked table in place. It writes a backup copy, a new manual-review
CSV for the newly supplied true preop images, and a fixed checked table.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OCT_FILENAME_PATTERN = re.compile(
    r"^(?P<exam_id>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<eye>[A-Za-z]+)",
    re.IGNORECASE,
)
ANALYSIS_INDEX_PATTERN = re.compile(r"_(?P<analysis_index>\d{3})$")
EXCLUDE_NOTE = "exclude_from_preop_measurement_baseline; possible_postop_record_originally_misclassified"
NEW_PREOP_NOTE = "newly_added_patient042_true_preop_2danalysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate patient_042 true preop 2DAnalysis images.")
    parser.add_argument(
        "--checked_in",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review_priority_checked.csv",
    )
    parser.add_argument(
        "--clean_manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean_strict.csv",
    )
    parser.add_argument(
        "--raw_patient_dir",
        type=str,
        default="data/raw/real_export_batch_01/patient/patient_042",
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        default="artifacts/figures/casia2_measurement_crops_batch_01",
    )
    parser.add_argument(
        "--backup_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review_priority_checked_before_patient042_fix.csv",
    )
    parser.add_argument(
        "--new_records_out",
        type=str,
        default="data/interim/patient042_new_preop_2danalysis_records_for_manual_review.csv",
    )
    parser.add_argument(
        "--fixed_out",
        type=str,
        default="data/interim/casia2_2d_measurements_batch_01_preop_manual_review_priority_checked_patient042_fixed.csv",
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


def normalize_eye(value: str) -> str:
    token = (value or "").strip().lower()
    if token in {"r", "right", "od"}:
        return "OD"
    if token in {"l", "left", "os"}:
        return "OS"
    return "unknown"


def normalize_date(value: str) -> str:
    return f"{value[:4]}/{int(value[4:6])}/{int(value[6:])}"


def normalize_time(value: str) -> str:
    return f"{int(value[:2])}:{value[2:4]}:{value[4:]}"


def is_2d_analysis_path(path: Path) -> bool:
    joined = " ".join(part.lower() for part in path.parts)
    return path.suffix.lower() in IMAGE_EXTENSIONS and "2danalysis" in joined


def parse_filename_metadata(path: Path) -> Dict[str, object]:
    match = OCT_FILENAME_PATTERN.match(path.stem)
    analysis_match = ANALYSIS_INDEX_PATTERN.search(path.stem)
    if not match:
        return {
            "exam_id": "",
            "exam_date": "",
            "exam_time": "",
            "eye": "unknown",
            "analysis_index": "",
        }
    groups = match.groupdict()
    return {
        "exam_id": groups["exam_id"],
        "exam_date": normalize_date(groups["date"]),
        "exam_date_raw": groups["date"],
        "exam_time": normalize_time(groups["time"]),
        "eye": normalize_eye(groups["eye"]),
        "analysis_index": int(analysis_match.group("analysis_index")) if analysis_match else "",
    }


def discover_true_preop_paths(raw_patient_dir: Path) -> List[Path]:
    paths = [path for path in sorted(raw_patient_dir.rglob("*")) if path.is_file() and is_2d_analysis_path(path)]
    parsed = [(path, parse_filename_metadata(path)) for path in paths]
    parsed = [(path, meta) for path, meta in parsed if meta.get("exam_date_raw")]
    if not parsed:
        return []
    earliest_date = min(str(meta["exam_date_raw"]) for _, meta in parsed)
    true_preop = [path for path, meta in parsed if meta.get("exam_date_raw") == earliest_date]
    return sorted(true_preop)


def sanitize_stem_for_filename(stem: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)
    return sanitized.strip("_") or "unknown_image"


def build_crop_name(image_path: Path) -> str:
    try:
        relative = image_path.relative_to(PROJECT_ROOT / "data/raw/real_export_batch_01/patient")
        stem = "__".join(relative.with_suffix("").parts)
    except ValueError:
        stem = image_path.stem
    return f"{sanitize_stem_for_filename(stem)}_measurements.png"


def save_measurement_crop(image_path: Path, crop_dir: Path) -> str:
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / build_crop_name(image_path)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        top = max(int(height * 0.72), 0)
        crop = image.crop((0, top, width, height))
        crop.save(crop_path)
    return relative_path(crop_path)


def append_note(value: object, note: str) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text or text.lower() == "nan":
        return note
    if note in text:
        return text
    return f"{text} | {note}"


def ensure_checked_columns(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df.copy()
    if "verify_status" not in fixed.columns and "verified" in fixed.columns:
        fixed["verify_status"] = fixed["verified"]
    if "verified" not in fixed.columns and "verify_status" in fixed.columns:
        fixed["verified"] = fixed["verify_status"]
    if "verify_status" not in fixed.columns:
        fixed["verify_status"] = ""
    if "verified" not in fixed.columns:
        fixed["verified"] = fixed["verify_status"]
    return fixed


def sample_lookup(clean_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    patient_rows = clean_df[clean_df["patient_uid"].astype(str).eq("patient_042")]
    return {str(row["eye"]): row for row in patient_rows.to_dict(orient="records")}


def empty_like_record(columns: Iterable[str]) -> Dict[str, object]:
    return {column: "" for column in columns}


def build_new_records(paths: List[Path], checked_columns: List[str], clean_df: pd.DataFrame, crop_dir: Path) -> pd.DataFrame:
    lookup = sample_lookup(clean_df)
    records: List[Dict[str, object]] = []
    for path in paths:
        meta = parse_filename_metadata(path)
        eye = str(meta["eye"])
        sample = lookup.get(eye, {})
        record = empty_like_record(checked_columns)
        record.update(
            {
                "sample_id": sample.get("sample_id", f"patient_042_{eye}_POD1"),
                "patient_uid": "patient_042",
                "eye": eye,
                "split": sample.get("split", ""),
                "exam_date": meta["exam_date"],
                "exam_time": meta["exam_time"],
                "analysis_index": meta["analysis_index"],
                "image_path": relative_path(path),
                "measurement_crop_path": save_measurement_crop(path, crop_dir),
                "cct_um": "",
                "acd_epi_mm": "",
                "acd_endo_mm": "",
                "vault_um": "",
                "clr_um": "",
                "ata_mm": "",
                "has_vault": False,
                "vault_raw_text": "---",
                "extraction_method": "manual_pending",
                "verify_status": "pending",
                "verified": "pending",
                "notes": NEW_PREOP_NOTE,
                "is_preop": True,
                "is_postop": False,
                "postop_day": "",
                "exam_id": meta["exam_id"],
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=checked_columns)


def mark_old_patient042_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    fixed = df.copy()
    patient_mask = fixed["patient_uid"].astype(str).eq("patient_042")
    old_count = int(patient_mask.sum())
    vault_values = pd.to_numeric(fixed.loc[patient_mask, "vault_um"], errors="coerce") if "vault_um" in fixed.columns else pd.Series(dtype=float)
    notes = fixed.loc[patient_mask, "notes"].fillna("").astype(str) if "notes" in fixed.columns else pd.Series(dtype=str)
    exclude_mask = patient_mask.copy()
    if old_count:
        possible_postop = vault_values.notna().reindex(fixed.index, fill_value=False)
        note_postop = notes.str.contains("postoperative|postop|术后", case=False, regex=True).reindex(fixed.index, fill_value=False)
        exclude_mask = patient_mask & (possible_postop | note_postop | patient_mask)
        fixed.loc[exclude_mask, "verify_status"] = "excluded"
        fixed.loc[exclude_mask, "verified"] = "excluded"
        fixed.loc[exclude_mask, "notes"] = fixed.loc[exclude_mask, "notes"].map(lambda value: append_note(value, EXCLUDE_NOTE))
    return fixed, old_count, int(exclude_mask.sum())


def sort_records(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["patient_uid", "eye", "exam_date", "exam_time", "analysis_index"] if column in df.columns]
    return df.sort_values(sort_columns, kind="stable", na_position="last")


def main() -> None:
    args = parse_args()
    checked_in = resolve_project_path(args.checked_in)
    clean_manifest = resolve_project_path(args.clean_manifest)
    raw_patient_dir = resolve_project_path(args.raw_patient_dir)
    crop_dir = resolve_project_path(args.crop_dir)
    backup_out = resolve_project_path(args.backup_out)
    new_records_out = resolve_project_path(args.new_records_out)
    fixed_out = resolve_project_path(args.fixed_out)

    checked_df = ensure_checked_columns(pd.read_csv(checked_in))
    clean_df = pd.read_csv(clean_manifest)
    backup_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checked_in, backup_out)

    true_preop_paths = discover_true_preop_paths(raw_patient_dir)
    fixed_old_df, old_count, excluded_count = mark_old_patient042_records(checked_df)
    new_records_df = build_new_records(
        paths=true_preop_paths,
        checked_columns=list(fixed_old_df.columns),
        clean_df=clean_df,
        crop_dir=crop_dir,
    )
    fixed_df = sort_records(pd.concat([fixed_old_df, new_records_df], ignore_index=True))

    new_records_out.parent.mkdir(parents=True, exist_ok=True)
    fixed_out.parent.mkdir(parents=True, exist_ok=True)
    new_records_df.to_csv(new_records_out, index=False, encoding="utf-8")
    fixed_df.to_csv(fixed_out, index=False, encoding="utf-8")

    print("Reminder: postoperative 2DAnalysis measurements must not be used as preoperative input features.")
    print(f"Original checked rows: {len(checked_df)}")
    print(f"patient_042 old records: {old_count}")
    print(f"patient_042 old records excluded: {excluded_count}")
    print(f"New patient_042 true preop 2DAnalysis records: {len(new_records_df)}")
    print(f"New crops generated: {len(new_records_df)}")
    print(f"Fixed checked rows: {len(fixed_df)}")
    print(f"Backup output: {relative_path(backup_out)}")
    print(f"New records output: {relative_path(new_records_out)}")
    print(f"Fixed checked output: {relative_path(fixed_out)}")
    print(f"Crop directory: {relative_path(crop_dir)}")


if __name__ == "__main__":
    main()
