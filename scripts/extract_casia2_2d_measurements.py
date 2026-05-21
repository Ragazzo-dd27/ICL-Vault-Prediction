"""Extract structured CASIA2 2DAnalysis metadata into review tables.

This script is intentionally conservative. It does not train a model and does
not build a formal training manifest. The goal is to organize AS-OCT
2DAnalysis-derived metadata and candidate postoperative vault labels into
CSV tables that can be manually reviewed before any downstream dataset build.

Important constraints:
- Postoperative 2DAnalysis images can only be used as a label source.
- Postoperative 2DAnalysis images should not be used as model inputs for
  preoperative vault prediction.
- Automatically extracted measurements are not treated as verified labels.
- The current outputs are review tables, not a formal training manifest.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
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
PATIENT_UID_PREFIX_PATTERN = re.compile(r"^(patient_\d+)(?:\b|_)", re.IGNORECASE)
MEASUREMENT_PATTERNS = {
    "cct_um": re.compile(r"CCT\s*(?:\[[^\]]+\])?\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "acd_epi_mm": re.compile(r"ACD\s*\[?\s*Epi\.?\s*\]?\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "acd_endo_mm": re.compile(r"ACD\s*\[?\s*Endo\.?\s*\]?\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "vault_um": re.compile(r"Vault\s*(?:\[[^\]]+\])?\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "clr_um": re.compile(r"CLR\s*(?:\[[^\]]+\])?\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "ata_mm": re.compile(r"ATA\s*(?:\[[^\]]+\])?\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
}
VAULT_TEXT_PATTERN = re.compile(r"(Vault[^0-9\-]*[-]{2,}|Vault[^0-9]*\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class MeasurementRecord:
    patient_uid: str
    source_patient_folder: str
    image_path: str
    exam_id: str
    exam_date: str
    exam_time: str
    eye: str
    analysis_index: str
    visit_index: int
    is_preop: bool
    is_postop: bool
    postop_day: int | None
    cct_um: float | None
    acd_epi_mm: float | None
    acd_endo_mm: float | None
    vault_um: float | None
    clr_um: float | None
    ata_mm: float | None
    has_vault: bool
    vault_raw_text: str
    extraction_method: str
    verify_status: str
    measurement_crop_path: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CASIA2 2DAnalysis metadata and vault label candidates."
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw/real_export_demo",
        help="Root directory containing exported patient folders.",
    )
    parser.add_argument(
        "--manifest_in",
        type=str,
        default="data/manifests/real_export_manifest_initial.csv",
        help="Optional initial real-export manifest used to discover 2DAnalysis paths first.",
    )
    parser.add_argument(
        "--measurements_out",
        type=str,
        default="data/interim/casia2_2d_measurements_initial.csv",
        help="Output CSV path for the per-image 2DAnalysis measurement table.",
    )
    parser.add_argument(
        "--labels_out",
        type=str,
        default="data/manifests/vault_label_candidates.csv",
        help="Output CSV path for postoperative vault label candidates.",
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        default="artifacts/figures/casia2_measurement_crops",
        help="Directory used to save measurement-table crops for manual review.",
    )
    parser.add_argument(
        "--enable_ocr",
        action="store_true",
        help="Enable optional OCR extraction if pytesseract is available locally.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_eye(value: str) -> str:
    token = (value or "").strip().lower()
    if token in {"r", "right", "od"}:
        return "OD"
    if token in {"l", "left", "os"}:
        return "OS"
    return "unknown"


def normalize_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return ""


def normalize_time(value: str) -> str:
    if len(value) == 6 and value.isdigit():
        return f"{value[:2]}:{value[2:4]}:{value[4:]}"
    return ""


def parse_filename_metadata(path: Path) -> Dict[str, str]:
    match = OCT_FILENAME_PATTERN.match(path.stem)
    analysis_match = re.search(r"_(\d{3})$", path.stem)
    analysis_index = analysis_match.group(1) if analysis_match else ""
    if not match:
        return {
            "exam_id": "",
            "exam_date": "",
            "exam_time": "",
            "eye": "unknown",
            "analysis_index": analysis_index,
        }

    groups = match.groupdict()
    return {
        "exam_id": groups["exam_id"],
        "exam_date": normalize_date(groups["date"]),
        "exam_time": normalize_time(groups["time"]),
        "eye": normalize_eye(groups["eye"]),
        "analysis_index": analysis_index,
    }


def is_2d_analysis_path(path: Path) -> bool:
    joined = " ".join(part.lower() for part in path.parts)
    return any(keyword in joined for keyword in ("2danalysis", "2d_analysis", "2d", "analysis"))


def classify_image(path: Path) -> str:
    joined = " ".join(part.lower() for part in path.parts)
    if "ubm" in joined:
        if "horizontal" in joined or "\u6c34\u5e73" in joined:
            return "ubm_horizontal"
        if "vertical" in joined or "\u5782\u76f4" in joined:
            return "ubm_vertical"
        return "ubm_unknown"
    if is_2d_analysis_path(path):
        return "oct_2d_analysis"
    if "oct" in joined or "casia" in path.stem.lower():
        return "oct_raw"
    return "other"


def resolve_patient_uid(folder_name: str, index: int) -> str:
    if re.match(r"^patient_\d+$", folder_name, flags=re.IGNORECASE):
        return folder_name.lower()
    # Batch-specific exports should be processed independently before any merge.
    # Preserve anonymized patient_NNN prefixes when a folder has a suffix.
    prefix_match = PATIENT_UID_PREFIX_PATTERN.match(folder_name)
    if prefix_match:
        return prefix_match.group(1).lower()
    return f"patient_{index:03d}"


def parse_manifest_paths(manifest_in: Path, raw_root: Path) -> List[Path]:
    df = pd.read_csv(manifest_in)
    paths: list[Path] = []
    if "oct_2d_analysis_paths" not in df.columns:
        return paths

    for raw_value in df["oct_2d_analysis_paths"].fillna(""):
        for item in str(raw_value).split(";"):
            item = item.strip()
            if item:
                item_path = Path(item)
                if item_path.is_absolute():
                    paths.append(item_path)
                else:
                    paths.append(raw_root / item_path)
                    paths.append(PROJECT_ROOT / item_path)
    return paths


def scan_2d_analysis_paths(raw_root: Path) -> List[Path]:
    paths: list[Path] = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and is_2d_analysis_path(path):
            paths.append(path)
    return paths


def discover_2d_analysis_paths(raw_root: Path, manifest_in: Path) -> List[Path]:
    discovered: List[Path] = []

    if manifest_in.exists():
        discovered.extend(path for path in parse_manifest_paths(manifest_in, raw_root) if path.exists())

    if not discovered:
        discovered.extend(scan_2d_analysis_paths(raw_root))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in sorted(discovered):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def assign_patient_uids(raw_root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    patient_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    for index, patient_dir in enumerate(patient_dirs, start=1):
        mapping[patient_dir.name] = resolve_patient_uid(patient_dir.name, index)
    return mapping


def count_images_by_modality(raw_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            counts["total"] += 1
            counts[classify_image(path)] += 1
    return counts


def warn_if_common_patient_dir_typo(raw_root: Path) -> None:
    if raw_root.name.lower() != "patients":
        return

    typo_dir = raw_root.with_name("ptients")
    if typo_dir.exists():
        print(f"Warning: found likely typo directory '{typo_dir}'. Please rename/use 'patients'.")


def safe_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def extract_source_patient_folder(path: Path, raw_root: Path) -> str:
    try:
        relative = path.relative_to(raw_root)
        return relative.parts[0]
    except ValueError:
        return path.parent.name


def sanitize_stem_for_filename(stem: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)
    return sanitized.strip("_") or "unknown_image"


def build_crop_name(image_path: Path, raw_root: Path) -> str:
    try:
        relative = image_path.relative_to(raw_root)
        stem = "__".join(relative.with_suffix("").parts)
    except ValueError:
        stem = image_path.stem
    return f"{sanitize_stem_for_filename(stem)}_measurements.png"


def save_measurement_crop(image_path: Path, crop_dir: Path, raw_root: Path) -> str:
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / build_crop_name(image_path, raw_root)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        top = max(int(height * 0.72), 0)
        crop = image.crop((0, top, width, height))
        crop.save(crop_path)

    return safe_relative(crop_path)


def maybe_run_ocr(crop_path: Path) -> str:
    try:
        import pytesseract
    except Exception:
        return ""

    try:
        with Image.open(crop_path) as image:
            gray = ImageOps.grayscale(image)
            enhanced = ImageOps.autocontrast(gray)
            return pytesseract.image_to_string(enhanced)
    except Exception:
        return ""


def parse_numeric_value(text: str, field_name: str) -> float | None:
    if not text:
        return None

    pattern = MEASUREMENT_PATTERNS[field_name]
    match = pattern.search(text)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_vault_raw_text(text: str) -> str:
    if not text:
        return ""
    match = VAULT_TEXT_PATTERN.search(" ".join(text.split()))
    if match:
        return match.group(1).strip()
    return ""


def build_base_rows(paths: Iterable[Path], raw_root: Path, patient_uid_map: Dict[str, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(paths):
        source_patient_folder = extract_source_patient_folder(path, raw_root)
        patient_uid = patient_uid_map.get(source_patient_folder, resolve_patient_uid(source_patient_folder, 1))
        metadata = parse_filename_metadata(path)
        rows.append(
            {
                "patient_uid": patient_uid,
                "source_patient_folder": source_patient_folder,
                "path_obj": path,
                "image_path": safe_relative(path),
                "exam_id": metadata["exam_id"],
                "exam_date": metadata["exam_date"],
                "exam_time": metadata["exam_time"],
                "eye": metadata["eye"],
                "analysis_index": metadata["analysis_index"],
            }
        )
    return rows


def assign_visit_roles(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["patient_uid"]), str(row["eye"]))].append(row)

    for key, group_rows in grouped.items():
        group_rows.sort(
            key=lambda row: (
                str(row["exam_date"]),
                str(row["exam_time"]),
                str(row["image_path"]),
            )
        )

        unique_dates = sorted({str(row["exam_date"]) for row in group_rows if str(row["exam_date"])})
        preop_date = unique_dates[0] if unique_dates else ""
        first_postop_date = unique_dates[1] if len(unique_dates) > 1 else ""

        for visit_index, row in enumerate(group_rows):
            exam_date = str(row["exam_date"])
            row["visit_index"] = visit_index
            row["is_preop"] = bool(preop_date and exam_date == preop_date)
            row["is_postop"] = bool(first_postop_date and exam_date >= first_postop_date)

            if row["is_postop"] and exam_date and first_postop_date:
                exam_dt = datetime.strptime(exam_date, "%Y-%m-%d")
                first_postop_dt = datetime.strptime(first_postop_date, "%Y-%m-%d")
                row["postop_day"] = (exam_dt - first_postop_dt).days + 1
            else:
                row["postop_day"] = None

            notes: list[str] = [
                "visit_index starts at 0 within each patient_uid + eye group",
                "postoperative 2DAnalysis can only be used as label source, not model input",
                "current output is an initial review table, not a formal training manifest",
            ]

            if len(unique_dates) <= 1:
                notes.append("only one exam_date found for this patient_uid + eye; record is conservatively treated as preop")
            elif not row["is_preop"] and not row["is_postop"]:
                notes.append("visit role could not be confidently assigned")

            row["notes"] = " | ".join(notes)


def enrich_measurements(
    rows: List[Dict[str, object]],
    crop_dir: Path,
    raw_root: Path,
    enable_ocr: bool,
) -> List[MeasurementRecord]:
    records: List[MeasurementRecord] = []

    for row in rows:
        path_obj = row["path_obj"]
        crop_relative = save_measurement_crop(
            image_path=path_obj,
            crop_dir=crop_dir,
            raw_root=raw_root,
        )
        crop_path = PROJECT_ROOT / crop_relative

        ocr_text = maybe_run_ocr(crop_path) if enable_ocr else ""
        extraction_method = "manual_pending"
        verify_status = "pending"

        cct_um = acd_epi_mm = acd_endo_mm = vault_um = clr_um = ata_mm = None
        vault_raw_text = ""
        if ocr_text:
            extraction_method = "ocr_optional"
            verify_status = "uncertain"
            cct_um = parse_numeric_value(ocr_text, "cct_um")
            acd_epi_mm = parse_numeric_value(ocr_text, "acd_epi_mm")
            acd_endo_mm = parse_numeric_value(ocr_text, "acd_endo_mm")
            vault_um = parse_numeric_value(ocr_text, "vault_um")
            clr_um = parse_numeric_value(ocr_text, "clr_um")
            ata_mm = parse_numeric_value(ocr_text, "ata_mm")
            vault_raw_text = parse_vault_raw_text(ocr_text)

        has_vault = vault_um is not None

        if not has_vault and vault_raw_text and "---" in vault_raw_text:
            verify_status = "pending"

        records.append(
            MeasurementRecord(
                patient_uid=str(row["patient_uid"]),
                source_patient_folder=str(row["source_patient_folder"]),
                image_path=str(row["image_path"]),
                exam_id=str(row["exam_id"]),
                exam_date=str(row["exam_date"]),
                exam_time=str(row["exam_time"]),
                eye=str(row["eye"]),
                analysis_index=str(row["analysis_index"]),
                visit_index=int(row["visit_index"]),
                is_preop=bool(row["is_preop"]),
                is_postop=bool(row["is_postop"]),
                postop_day=row["postop_day"],
                cct_um=cct_um,
                acd_epi_mm=acd_epi_mm,
                acd_endo_mm=acd_endo_mm,
                vault_um=vault_um,
                clr_um=clr_um,
                ata_mm=ata_mm,
                has_vault=has_vault,
                vault_raw_text=vault_raw_text,
                extraction_method=extraction_method,
                verify_status=verify_status,
                measurement_crop_path=crop_relative,
                notes=str(row["notes"]),
            )
        )

    return records


def build_label_candidates(records: List[MeasurementRecord]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[MeasurementRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.patient_uid, record.eye)].append(record)

    candidates: List[Dict[str, object]] = []
    for (patient_uid, eye), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda record: (record.exam_date, record.exam_time, record.image_path))
        preop_dates = sorted({record.exam_date for record in ordered if record.is_preop and record.exam_date})
        preop_exam_date = preop_dates[0] if preop_dates else ""
        sample_id = f"{patient_uid}_{eye}_{preop_exam_date.replace('-', '')}" if preop_exam_date else f"{patient_uid}_{eye}"

        for record in ordered:
            if not record.is_postop:
                continue

            if record.has_vault:
                label_status = "valid"
            elif record.extraction_method == "ocr_optional" and record.vault_raw_text:
                label_status = "uncertain"
            else:
                label_status = "missing"

            notes = [
                "candidate vault labels come from postoperative 2DAnalysis images only",
                "current labels require manual review before any formal manifest build",
            ]
            if not record.has_vault:
                notes.append("no reliable vault value extracted automatically")

            candidates.append(
                {
                    "sample_id": sample_id,
                    "patient_uid": patient_uid,
                    "eye": eye,
                    "preop_exam_date": preop_exam_date,
                    "label_exam_date": record.exam_date,
                    "label_exam_time": record.exam_time,
                    "postop_day": record.postop_day,
                    "postop_vault_um": record.vault_um,
                    "vault_source_image": record.image_path,
                    "label_status": label_status,
                    "verify_status": record.verify_status,
                    "notes": " | ".join(notes),
                }
            )

    return candidates


def main() -> None:
    args = parse_args()
    raw_root = resolve_project_path(args.raw_root)
    manifest_in = resolve_project_path(args.manifest_in)
    measurements_out = resolve_project_path(args.measurements_out)
    labels_out = resolve_project_path(args.labels_out)
    crop_dir = resolve_project_path(args.crop_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw export root does not exist: {raw_root}")
    warn_if_common_patient_dir_typo(raw_root)

    patient_uid_map = assign_patient_uids(raw_root)
    image_counts = count_images_by_modality(raw_root)
    paths = discover_2d_analysis_paths(raw_root=raw_root, manifest_in=manifest_in)
    base_rows = build_base_rows(paths=paths, raw_root=raw_root, patient_uid_map=patient_uid_map)
    assign_visit_roles(base_rows)
    measurement_records = enrich_measurements(
        base_rows,
        crop_dir=crop_dir,
        raw_root=raw_root,
        enable_ocr=args.enable_ocr,
    )
    label_candidates = build_label_candidates(measurement_records)

    measurements_df = pd.DataFrame(
        [record.__dict__ for record in measurement_records],
        columns=[
            "patient_uid",
            "source_patient_folder",
            "image_path",
            "exam_id",
            "exam_date",
            "exam_time",
            "eye",
            "analysis_index",
            "visit_index",
            "is_preop",
            "is_postop",
            "postop_day",
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
            "measurement_crop_path",
            "notes",
        ],
    )
    labels_df = pd.DataFrame(
        label_candidates,
        columns=[
            "sample_id",
            "patient_uid",
            "eye",
            "preop_exam_date",
            "label_exam_date",
            "label_exam_time",
            "postop_day",
            "postop_vault_um",
            "vault_source_image",
            "label_status",
            "verify_status",
            "notes",
        ],
    )

    measurements_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    measurements_df.to_csv(measurements_out, index=False)
    labels_df.to_csv(labels_out, index=False)

    per_patient_counts = measurements_df.groupby("patient_uid").size().to_dict() if not measurements_df.empty else {}
    crop_count = int(measurements_df["measurement_crop_path"].astype(str).ne("").sum()) if not measurements_df.empty else 0
    preop_count = int(measurements_df["is_preop"].sum()) if not measurements_df.empty else 0
    postop_count = int(measurements_df["is_postop"].sum()) if not measurements_df.empty else 0
    valid_label_count = int((labels_df["label_status"] == "valid").sum()) if not labels_df.empty else 0

    print(f"Scanned patients: {len(patient_uid_map)}")
    print(f"Total images: {image_counts['total']}")
    print(f"OCT raw images: {image_counts['oct_raw']}")
    print(f"OCT 2DAnalysis images: {image_counts['oct_2d_analysis']}")
    print(f"UBM horizontal images: {image_counts['ubm_horizontal']}")
    print(f"UBM vertical images: {image_counts['ubm_vertical']}")
    print(f"UBM unknown images: {image_counts['ubm_unknown']}")
    print(f"Other images: {image_counts['other']}")
    print(f"2DAnalysis records: {len(measurement_records)}")
    print(f"Measurements output: {measurements_out.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Vault label candidates output: {labels_out.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Measurement crop directory: {crop_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Measurement crops generated: {crop_count}")
    for patient_uid, count in sorted(per_patient_counts.items()):
        print(f"{patient_uid} 2DAnalysis images: {count}")
    print(f"Preop records: {preop_count}")
    print(f"Postop records: {postop_count}")
    print(f"Candidate vault labels: {len(label_candidates)}")
    print(f"Valid vault values extracted automatically: {valid_label_count}")
    print("Review note: current outputs require manual verification and cannot be used as verified labels directly.")


if __name__ == "__main__":
    main()
