"""Build an eye-level OCT+ASP multimodal MCOA manifest from an image-level manifest."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an eye-level multimodal MCOA manifest CSV.")
    parser.add_argument("--input_manifest", type=str, default="data/manifests/mcoa_manifest_medium.csv")
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="data/manifests/mcoa_multimodal_manifest_medium.csv",
    )
    parser.add_argument(
        "--conflict_policy",
        type=str,
        default="drop",
        choices=("drop", "majority"),
    )
    return parser.parse_args()


def normalize_text(value: Optional[str]) -> str:
    return (value or "").strip()


def infer_oct_eye_key(image_path: str) -> str:
    stem = Path(image_path).stem.strip()
    parenthesized_match = re.match(r"^(\d+)\s*\((\d+)\)$", stem)
    if parenthesized_match:
        return parenthesized_match.group(1)
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def infer_oct_slice_index(image_path: str) -> int:
    stem = Path(image_path).stem.strip()
    parenthesized_match = re.match(r"^(\d+)\s*\((\d+)\)$", stem)
    if parenthesized_match:
        return int(parenthesized_match.group(2))
    underscored_match = re.match(r"^(\d+)_(\d+)$", stem)
    if underscored_match:
        return int(underscored_match.group(2))
    return 1


def resolve_group_split(splits: List[str], conflict_policy: str) -> Optional[str]:
    unique_splits = sorted(set(splits))
    if len(unique_splits) == 1:
        return unique_splits[0]
    if conflict_policy == "drop":
        return None
    split_counter = Counter(splits)
    return sorted(split_counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_asp_path_map() -> Dict[str, Dict[str, str]]:
    base = Path("data/public_datasets/mcoa_oct/MCOA_ Dataset/Images")
    asp_dirs = {
        "normal": base / "Normal Cornea" / "ASP",
        "opaque": base / "Opaque Cornea" / "ASP" / "ASP Original Images",
    }
    asp_path_map: Dict[str, Dict[str, str]] = {"normal": {}, "opaque": {}}
    for label, asp_dir in asp_dirs.items():
        if not asp_dir.exists():
            continue
        for path in asp_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                asp_path_map[label][path.stem.strip()] = str(path)
    return asp_path_map


def load_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return rows


def build_multimodal_rows(
    rows: List[Dict[str, str]],
    conflict_policy: str,
) -> Tuple[List[Dict[str, str]], int]:
    grouped_rows: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    dropped_conflicts = 0
    asp_path_map = build_asp_path_map()

    for row in rows:
        label = normalize_text(row.get("label"))
        image_path = normalize_text(row.get("image_path"))
        split = normalize_text(row.get("split"))
        if not label or not image_path:
            raise ValueError("Encountered row with empty label or image_path.")
        if split not in VALID_SPLITS:
            raise ValueError(f"Encountered invalid split {split!r}. Expected one of {VALID_SPLITS}.")
        eye_key = infer_oct_eye_key(image_path)
        grouped_rows[(label, eye_key)].append(row)

    multimodal_rows: List[Dict[str, str]] = []
    for label, eye_key in sorted(grouped_rows.keys()):
        group = grouped_rows[(label, eye_key)]
        ordered_group = sorted(
            group,
            key=lambda row: (
                infer_oct_slice_index(normalize_text(row["image_path"])),
                normalize_text(row["image_path"]),
            ),
        )
        resolved_split = resolve_group_split(
            splits=[normalize_text(row["split"]) for row in ordered_group],
            conflict_policy=conflict_policy,
        )
        if resolved_split is None:
            dropped_conflicts += 1
            continue

        oct_slice_paths = [normalize_text(row["image_path"]) for row in ordered_group]
        asp_path = asp_path_map.get(label, {}).get(eye_key)
        multimodal_rows.append(
            {
                "eye_id": f"{label}_{eye_key}",
                "label": label,
                "split": resolved_split,
                "num_oct_slices": str(len(oct_slice_paths)),
                "oct_slice_paths": "|".join(oct_slice_paths),
                "oct_slice_indices": "|".join(str(infer_oct_slice_index(path)) for path in oct_slice_paths),
                "asp_path": asp_path or "",
                "has_oct": "true",
                "has_asp": "true" if asp_path else "false",
                "source_sample_ids": "|".join(normalize_text(row.get("sample_id")) for row in ordered_group),
            }
        )
    return multimodal_rows, dropped_conflicts


def save_multimodal_manifest(output_path: Path, rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "eye_id",
        "label",
        "split",
        "num_oct_slices",
        "oct_slice_paths",
        "oct_slice_indices",
        "asp_path",
        "has_oct",
        "has_asp",
        "source_sample_ids",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)
    rows = load_rows(input_manifest)
    multimodal_rows, dropped_conflicts = build_multimodal_rows(rows=rows, conflict_policy=args.conflict_policy)
    save_multimodal_manifest(output_path=output_manifest, rows=multimodal_rows)

    split_counter = Counter(row["split"] for row in multimodal_rows)
    modality_counter = Counter((row["has_oct"], row["has_asp"]) for row in multimodal_rows)
    print("Built multimodal eye-level MCOA manifest.")
    print(f"Input manifest: {input_manifest}")
    print(f"Output manifest: {output_manifest}")
    print(f"Eye samples: {len(multimodal_rows)}")
    print(f"Split counts: {dict(split_counter)}")
    print(f"Modality counts: {dict(modality_counter)}")
    print(f"Dropped split-conflict eyes: {dropped_conflicts}")
    print("Conflict policy note: 'drop' avoids eye-level train/val leakage.")


if __name__ == "__main__":
    main()
