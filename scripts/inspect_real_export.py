"""Inspect a real hospital export and build an initial feasibility manifest.

This helper is intentionally conservative. It is not a formal training-manifest
builder and should only be used to validate whether the exported AS-OCT / UBM
files can be scanned, parsed, summarized, and previewed by the project code.

Current limitations:
- `vault_label` is unavailable and therefore left empty.
- Clinical features are unavailable and marked as missing.
- UBM files are only grouped at the patient level here; exact linkage between
  UBM images and OCT visit-eye records still requires manual confirmation or
  richer matching rules in a later pipeline.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PATIENT_UID_PATTERN = re.compile(r"^patient_\d+$", re.IGNORECASE)
OCT_FILENAME_PATTERN = re.compile(
    r"^(?P<exam_id>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<eye>[A-Za-z]+)",
    re.IGNORECASE,
)
NOTE_TEXT = "initial real-export manifest, not for formal training"
SUMMARY_NOTE_TEXT = "real-export feasibility check only; UBM-OCT linkage still needs confirmation"


@dataclass
class ImageRecord:
    path: Path
    relative_path: str
    patient_uid: str
    source_patient_folder: str
    modality: str
    exam_id: str = ""
    date: str = ""
    time: str = ""
    eye: str = ""
    is_2d_analysis: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a real ICL export and build an initial feasibility manifest."
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw/real_export_demo",
        help="Root directory that contains exported patient folders.",
    )
    parser.add_argument(
        "--manifest_out",
        type=str,
        default="data/manifests/real_export_manifest_initial.csv",
        help="Output path for the initial real-export manifest CSV.",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default="data/manifests/real_export_summary.csv",
        help="Output path for the patient-level summary CSV.",
    )
    parser.add_argument(
        "--figure_out",
        type=str,
        default="artifacts/figures/real_multimodal_example_patient001_deidentified.png",
        help="Output path for the de-identified multimodal example figure.",
    )
    parser.add_argument(
        "--paper_figure_out",
        type=str,
        default="artifacts/figures/real_multimodal_example_paper.png",
        help="Output path for the paper-ready de-identified multimodal example figure.",
    )
    parser.add_argument(
        "--example_patient",
        type=str,
        default="patient_001",
        help="Anonymous patient UID used for the example multimodal figure.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser()


def normalize_eye(value: str) -> str:
    token = (value or "").strip().lower()
    if token in {"od", "r", "right"}:
        return "OD"
    if token in {"os", "l", "left"}:
        return "OS"
    return ""


def detect_is_2d_analysis(path: Path) -> bool:
    joined = " ".join(part.lower() for part in path.parts)
    return any(keyword in joined for keyword in ("2danalysis", "2d_analysis", "2d", "analysis"))


def classify_ubm_orientation(path: Path) -> str:
    joined = " ".join(part.lower() for part in path.parts)

    if "horizontal" in joined or "\u6c34\u5e73" in joined:
        return "ubm_horizontal"
    if "vertical" in joined or "\u5782\u76f4" in joined:
        return "ubm_vertical"

    tokens = [token for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", joined) if token]
    if any(token == "h" for token in tokens):
        return "ubm_horizontal"
    if any(token == "v" for token in tokens):
        return "ubm_vertical"

    return "ubm_unknown"


def classify_image(path: Path) -> str:
    joined = " ".join(part.lower() for part in path.parts)
    if "ubm" in joined:
        return classify_ubm_orientation(path)
    if detect_is_2d_analysis(path):
        return "oct_2d_analysis"
    if "oct" in joined or "casia" in path.stem.lower():
        return "oct_raw"
    return "other"


def parse_oct_metadata(path: Path) -> Dict[str, str]:
    match = OCT_FILENAME_PATTERN.match(path.stem)
    if not match:
        return {"exam_id": "", "date": "", "time": "", "eye": ""}

    metadata = match.groupdict()
    return {
        "exam_id": metadata.get("exam_id", ""),
        "date": metadata.get("date", ""),
        "time": metadata.get("time", ""),
        "eye": normalize_eye(metadata.get("eye", "")),
    }


def iter_patient_dirs(raw_root: Path) -> List[Path]:
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def warn_if_common_patient_dir_typo(raw_root: Path) -> None:
    if raw_root.name.lower() != "patients":
        return

    typo_dir = raw_root.with_name("ptients")
    if typo_dir.exists():
        print(f"Warning: found likely typo directory '{typo_dir}'. Please rename/use 'patients'.")


def assign_patient_uid(folder_name: str, index: int) -> str:
    if PATIENT_UID_PATTERN.match(folder_name):
        return folder_name.lower()
    return f"patient_{index:03d}"


def gather_patient_records(patient_dir: Path, patient_uid: str, raw_root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for path in sorted(patient_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        modality = classify_image(path)
        metadata = parse_oct_metadata(path) if modality in {"oct_raw", "oct_2d_analysis"} else {}
        records.append(
            ImageRecord(
                path=path,
                relative_path=path.relative_to(raw_root).as_posix(),
                patient_uid=patient_uid,
                source_patient_folder=patient_dir.name,
                modality=modality,
                exam_id=metadata.get("exam_id", ""),
                date=metadata.get("date", ""),
                time=metadata.get("time", ""),
                eye=metadata.get("eye", ""),
                is_2d_analysis=(modality == "oct_2d_analysis"),
            )
        )
    return records


def join_paths(paths: Iterable[str]) -> str:
    ordered = sorted(path for path in paths if path)
    return ";".join(ordered)


def build_manifest_rows(patient_records: List[ImageRecord]) -> List[Dict[str, str]]:
    manifest_rows: List[Dict[str, str]] = []
    records_by_patient: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in patient_records:
        records_by_patient[record.patient_uid].append(record)

    for patient_uid in sorted(records_by_patient.keys()):
        records = records_by_patient[patient_uid]
        source_folder = records[0].source_patient_folder if records else ""

        patient_ubm_horizontal = [
            record.relative_path for record in records if record.modality == "ubm_horizontal"
        ]
        patient_ubm_vertical = [
            record.relative_path for record in records if record.modality == "ubm_vertical"
        ]
        patient_ubm_unknown = [
            record.relative_path for record in records if record.modality == "ubm_unknown"
        ]

        oct_records = [record for record in records if record.modality in {"oct_raw", "oct_2d_analysis"}]
        grouped_oct: Dict[tuple[str, str, str, str], List[ImageRecord]] = defaultdict(list)
        for record in oct_records:
            grouped_oct[(record.date, record.time, record.eye, record.exam_id)].append(record)

        if not grouped_oct:
            manifest_rows.append(
                {
                    "patient_uid": patient_uid,
                    "source_patient_folder": source_folder,
                    "date": "",
                    "time": "",
                    "eye": "",
                    "exam_id": "",
                    "oct_raw_paths": "",
                    "oct_2d_analysis_paths": "",
                    "ubm_horizontal_paths": join_paths(patient_ubm_horizontal),
                    "ubm_vertical_paths": join_paths(patient_ubm_vertical),
                    "ubm_unknown_paths": join_paths(patient_ubm_unknown),
                    "has_oct_raw": False,
                    "has_oct_2d_analysis": False,
                    "has_ubm_horizontal": bool(patient_ubm_horizontal),
                    "has_ubm_vertical": bool(patient_ubm_vertical),
                    "vault_label": "",
                    "clinical_features_status": "missing",
                    "notes": NOTE_TEXT,
                }
            )
            continue

        for date, time, eye, exam_id in sorted(grouped_oct.keys()):
            group = grouped_oct[(date, time, eye, exam_id)]
            oct_raw_paths = [record.relative_path for record in group if record.modality == "oct_raw"]
            oct_2d_paths = [record.relative_path for record in group if record.modality == "oct_2d_analysis"]

            manifest_rows.append(
                {
                    "patient_uid": patient_uid,
                    "source_patient_folder": source_folder,
                    "date": date,
                    "time": time,
                    "eye": eye or "",
                    "exam_id": exam_id,
                    "oct_raw_paths": join_paths(oct_raw_paths),
                    "oct_2d_analysis_paths": join_paths(oct_2d_paths),
                    "ubm_horizontal_paths": join_paths(patient_ubm_horizontal),
                    "ubm_vertical_paths": join_paths(patient_ubm_vertical),
                    "ubm_unknown_paths": join_paths(patient_ubm_unknown),
                    "has_oct_raw": bool(oct_raw_paths),
                    "has_oct_2d_analysis": bool(oct_2d_paths),
                    "has_ubm_horizontal": bool(patient_ubm_horizontal),
                    "has_ubm_vertical": bool(patient_ubm_vertical),
                    "vault_label": "",
                    "clinical_features_status": "missing",
                    "notes": NOTE_TEXT,
                }
            )

    return manifest_rows


def build_summary_rows(patient_records: List[ImageRecord], manifest_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    summary_rows: List[Dict[str, str]] = []
    records_by_patient: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in patient_records:
        records_by_patient[record.patient_uid].append(record)

    manifest_df = pd.DataFrame(manifest_rows)

    for patient_uid in sorted(records_by_patient.keys()):
        records = records_by_patient[patient_uid]
        source_folder = records[0].source_patient_folder if records else ""
        patient_manifest = manifest_df[manifest_df["patient_uid"] == patient_uid] if not manifest_df.empty else pd.DataFrame()

        num_oct_raw_images = sum(record.modality == "oct_raw" for record in records)
        num_oct_2d_images = sum(record.modality == "oct_2d_analysis" for record in records)
        num_ubm_horizontal_images = sum(record.modality == "ubm_horizontal" for record in records)
        num_ubm_vertical_images = sum(record.modality == "ubm_vertical" for record in records)
        num_ubm_unknown_images = sum(record.modality == "ubm_unknown" for record in records)

        summary_rows.append(
            {
                "patient_uid": patient_uid,
                "source_patient_folder": source_folder,
                "num_all_images": len(records),
                "num_oct_raw_images": num_oct_raw_images,
                "num_oct_2d_analysis_images": num_oct_2d_images,
                "num_oct_visit_eye_records": len(patient_manifest),
                "num_ubm_horizontal_images": num_ubm_horizontal_images,
                "num_ubm_vertical_images": num_ubm_vertical_images,
                "num_ubm_unknown_images": num_ubm_unknown_images,
                "has_any_oct": bool(num_oct_raw_images or num_oct_2d_images),
                "has_any_ubm": bool(
                    num_ubm_horizontal_images or num_ubm_vertical_images or num_ubm_unknown_images
                ),
                "notes": SUMMARY_NOTE_TEXT,
            }
        )

    return summary_rows


def load_image_for_panel(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image.load()
    return ImageOps.exif_transpose(image)


def deidentify_oct_2d_analysis_image(image: Image.Image) -> Image.Image:
    """Mask likely PHI regions for a presentation-safe 2DAnalysis sample.

    This figure is only intended for de-identified paper / report display. When
    the exact PHI location is uncertain, we use a conservative masking strategy
    and prioritize privacy by covering broad top and bottom report bands.
    """

    sanitized = image.convert("RGB").copy()
    width, height = sanitized.size

    top_band_height = max(int(height * 0.16), 120)
    bottom_band_height = max(int(height * 0.12), 90)

    draw = ImageDraw.Draw(sanitized)
    draw.rectangle((0, 0, width, min(top_band_height, height)), fill="white")
    draw.rectangle(
        (0, max(height - bottom_band_height, 0), width, height),
        fill="white",
    )
    return sanitized


def select_example_images(patient_records: List[ImageRecord], patient_uid: str) -> Dict[str, Path]:
    records = [record for record in patient_records if record.patient_uid == patient_uid]
    selected: Dict[str, Path] = {}
    preferred_modalities = ("oct_raw", "oct_2d_analysis", "ubm_horizontal", "ubm_vertical")
    for modality in preferred_modalities:
        candidates = sorted(record.path for record in records if record.modality == modality)
        if candidates:
            selected[modality] = candidates[0]
    return selected


def build_example_panels(selected: Dict[str, Path]) -> List[tuple[str, Path | None]]:
    return [
        ("AS-OCT Raw", selected.get("oct_raw")),
        ("AS-OCT 2DAnalysis", selected.get("oct_2d_analysis")),
        ("UBM Horizontal", selected.get("ubm_horizontal")),
        ("UBM Vertical", selected.get("ubm_vertical")),
    ]


def draw_example_panels(
    axes: List[plt.Axes],
    panels: List[tuple[str, Path | None]],
    title_fontsize: int,
) -> None:
    for axis, (title, image_path) in zip(axes, panels):
        axis.set_title(title, fontsize=title_fontsize, pad=6)
        axis.axis("off")

        if image_path is None:
            axis.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=16)
            continue

        try:
            image = load_image_for_panel(image_path)
            if title == "AS-OCT 2DAnalysis":
                image = deidentify_oct_2d_analysis_image(image)
            axis.imshow(image, cmap="gray" if image.mode in {"L", "I"} else None)
        except Exception:
            axis.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=16)


def render_example_figure(patient_records: List[ImageRecord], patient_uid: str, figure_out: Path) -> None:
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    selected = select_example_images(patient_records, patient_uid)
    panels = build_example_panels(selected)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Real Multimodal Export Example: {patient_uid}", fontsize=14)
    draw_example_panels(list(axes.flatten()), panels, title_fontsize=12)
    plt.tight_layout()
    fig.savefig(figure_out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_paper_example_figure(patient_records: List[ImageRecord], patient_uid: str, figure_out: Path) -> None:
    """Render a tighter paper-ready figure for LaTeX insertion.

    This version keeps the four per-panel titles, removes the large super-title,
    minimizes surrounding whitespace, and preserves conservative de-identifying
    masks on the AS-OCT 2DAnalysis panel before publication-facing export.
    """

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    selected = select_example_images(patient_records, patient_uid)
    panels = build_example_panels(selected)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    draw_example_panels(list(axes.flatten()), panels, title_fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.96, wspace=0.03, hspace=0.08)
    fig.savefig(figure_out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_example_figures(
    patient_records: List[ImageRecord],
    patient_uid: str,
    figure_out: Path,
    paper_figure_out: Path,
) -> None:
    render_example_figure(patient_records=patient_records, patient_uid=patient_uid, figure_out=figure_out)
    render_paper_example_figure(
        patient_records=patient_records,
        patient_uid=patient_uid,
        figure_out=paper_figure_out,
    )


def print_patient_stats(summary_df: pd.DataFrame) -> None:
    for row in summary_df.to_dict(orient="records"):
        print(
            " | ".join(
                [
                    f"patient_uid={row['patient_uid']}",
                    f"all_images={row['num_all_images']}",
                    f"oct_raw={row['num_oct_raw_images']}",
                    f"oct_2d={row['num_oct_2d_analysis_images']}",
                    f"visit_eye_records={row['num_oct_visit_eye_records']}",
                    f"ubm_h={row['num_ubm_horizontal_images']}",
                    f"ubm_v={row['num_ubm_vertical_images']}",
                    f"ubm_unknown={row['num_ubm_unknown_images']}",
                ]
            )
        )


def print_batch_summary(patient_dirs: List[Path], patient_records: List[ImageRecord], manifest_df: pd.DataFrame) -> None:
    modality_counts = defaultdict(int)
    for record in patient_records:
        modality_counts[record.modality] += 1

    print(f"Scanned patients: {len(patient_dirs)}")
    print(f"Total images: {len(patient_records)}")
    print(f"OCT raw images: {modality_counts['oct_raw']}")
    print(f"OCT 2DAnalysis images: {modality_counts['oct_2d_analysis']}")
    print(f"UBM horizontal images: {modality_counts['ubm_horizontal']}")
    print(f"UBM vertical images: {modality_counts['ubm_vertical']}")
    print(f"UBM unknown images: {modality_counts['ubm_unknown']}")
    print(f"Other images: {modality_counts['other']}")
    print(f"2DAnalysis records: {int(manifest_df['has_oct_2d_analysis'].sum()) if not manifest_df.empty else 0}")


def main() -> None:
    args = parse_args()
    raw_root = resolve_path(args.raw_root)
    manifest_out = resolve_path(args.manifest_out)
    summary_out = resolve_path(args.summary_out)
    figure_out = resolve_path(args.figure_out)
    paper_figure_out = resolve_path(args.paper_figure_out)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw export root does not exist: {raw_root}")
    warn_if_common_patient_dir_typo(raw_root)

    patient_dirs = iter_patient_dirs(raw_root)
    patient_records: List[ImageRecord] = []
    for index, patient_dir in enumerate(patient_dirs, start=1):
        patient_uid = assign_patient_uid(patient_dir.name, index)
        patient_records.extend(gather_patient_records(patient_dir=patient_dir, patient_uid=patient_uid, raw_root=raw_root))

    manifest_rows = build_manifest_rows(patient_records)
    summary_rows = build_summary_rows(patient_records, manifest_rows)

    manifest_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "patient_uid",
            "source_patient_folder",
            "date",
            "time",
            "eye",
            "exam_id",
            "oct_raw_paths",
            "oct_2d_analysis_paths",
            "ubm_horizontal_paths",
            "ubm_vertical_paths",
            "ubm_unknown_paths",
            "has_oct_raw",
            "has_oct_2d_analysis",
            "has_ubm_horizontal",
            "has_ubm_vertical",
            "vault_label",
            "clinical_features_status",
            "notes",
        ],
    )
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "patient_uid",
            "source_patient_folder",
            "num_all_images",
            "num_oct_raw_images",
            "num_oct_2d_analysis_images",
            "num_oct_visit_eye_records",
            "num_ubm_horizontal_images",
            "num_ubm_vertical_images",
            "num_ubm_unknown_images",
            "has_any_oct",
            "has_any_ubm",
            "notes",
        ],
    )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    render_example_figures(
        patient_records=patient_records,
        patient_uid=args.example_patient,
        figure_out=figure_out,
        paper_figure_out=paper_figure_out,
    )

    print_batch_summary(patient_dirs=patient_dirs, patient_records=patient_records, manifest_df=manifest_df)
    print(f"Manifest output: {manifest_out}")
    print(f"Summary output: {summary_out}")
    print(f"Figure output: {figure_out}")
    print(f"Paper figure output: {paper_figure_out}")
    print_patient_stats(summary_df)


if __name__ == "__main__":
    main()
