"""Build an eye-level MCOA manifest from an image-level manifest.

The current image-level MCOA manifests do not contain an explicit `eye_id`
field, and some image-level splits mix slices from the same eye across train
and val. This helper converts the image manifest into a minimal eye-level CSV
that can drive a true eye-level training path.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an eye-level MCOA manifest CSV.")
    parser.add_argument(
        "--input_manifest",
        type=str,
        default="data/manifests/mcoa_manifest_medium.csv",
        help="Path to the source image-level MCOA manifest.",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="data/manifests/mcoa_eye_manifest_medium.csv",
        help="Path to the output eye-level MCOA manifest.",
    )
    parser.add_argument(
        "--conflict_policy",
        type=str,
        default="drop",
        choices=("drop", "majority"),
        help=(
            "How to handle eyes whose slices span multiple splits. "
            "'drop' excludes them to avoid leakage; 'majority' keeps them "
            "using the most common split."
        ),
    )
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    return (value or "").strip()


def infer_eye_key(image_path: str) -> str:
    stem = Path(image_path).stem.strip()

    parenthesized_match = re.match(r"^(\d+)\s*\((\d+)\)$", stem)
    if parenthesized_match:
        return parenthesized_match.group(1)

    if "_" in stem:
        return stem.split("_", 1)[0]

    return stem


def infer_slice_index(image_path: str) -> int:
    stem = Path(image_path).stem.strip()

    parenthesized_match = re.match(r"^(\d+)\s*\((\d+)\)$", stem)
    if parenthesized_match:
        return int(parenthesized_match.group(2))

    underscored_match = re.match(r"^(\d+)_(\d+)$", stem)
    if underscored_match:
        return int(underscored_match.group(2))

    base_match = re.match(r"^(\d+)$", stem)
    if base_match:
        return 1

    numeric_tokens = re.findall(r"\d+", stem)
    if numeric_tokens:
        return int(numeric_tokens[-1])

    return 1


def resolve_group_split(splits: List[str], conflict_policy: str) -> str | None:
    unique_splits = sorted(set(splits))
    if len(unique_splits) == 1:
        return unique_splits[0]

    if conflict_policy == "drop":
        return None

    split_counter = Counter(splits)
    return sorted(split_counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def load_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return rows


def build_eye_rows(rows: List[Dict[str, str]], conflict_policy: str) -> Tuple[List[Dict[str, str]], int]:
    grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    dropped_conflicts = 0

    for row in rows:
        label = normalize_text(row.get("label"))
        image_path = normalize_text(row.get("image_path"))
        split = normalize_text(row.get("split"))

        if not label:
            raise ValueError("Encountered row with empty label.")
        if not image_path:
            raise ValueError("Encountered row with empty image_path.")
        if split not in VALID_SPLITS:
            raise ValueError(f"Encountered invalid split {split!r}. Expected one of {VALID_SPLITS}.")

        parent_dir = str(Path(image_path).parent)
        eye_key = infer_eye_key(image_path)
        grouped_rows[(label, parent_dir, eye_key)].append(row)

    eye_rows: List[Dict[str, str]] = []
    for label, parent_dir, eye_key in sorted(grouped_rows.keys()):
        group = grouped_rows[(label, parent_dir, eye_key)]
        ordered_group = sorted(
            group,
            key=lambda row: (
                infer_slice_index(normalize_text(row["image_path"])),
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

        first_sample_id = normalize_text(ordered_group[0].get("sample_id"))
        eye_rows.append(
            {
                "eye_id": f"{label}_{eye_key}",
                "label": label,
                "split": resolved_split,
                "num_slices": str(len(ordered_group)),
                "source_parent_dir": parent_dir,
                "source_sample_ids": "|".join(normalize_text(row.get("sample_id")) for row in ordered_group),
                "slice_paths": "|".join(normalize_text(row["image_path"]) for row in ordered_group),
                "slice_indices": "|".join(
                    str(infer_slice_index(normalize_text(row["image_path"]))) for row in ordered_group
                ),
                "source_manifest_first_sample_id": first_sample_id,
            }
        )

    return eye_rows, dropped_conflicts


def save_eye_manifest(output_path: Path, eye_rows: List[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "eye_id",
        "label",
        "split",
        "num_slices",
        "source_parent_dir",
        "source_sample_ids",
        "slice_paths",
        "slice_indices",
        "source_manifest_first_sample_id",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eye_rows)


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)

    rows = load_rows(input_manifest)
    eye_rows, dropped_conflicts = build_eye_rows(rows=rows, conflict_policy=args.conflict_policy)
    save_eye_manifest(output_path=output_manifest, eye_rows=eye_rows)

    split_counter = Counter(row["split"] for row in eye_rows)
    label_counter = Counter(row["label"] for row in eye_rows)

    print("Built eye-level MCOA manifest.")
    print(f"Input manifest: {input_manifest}")
    print(f"Output manifest: {output_manifest}")
    print(f"Eye samples: {len(eye_rows)}")
    print(f"Split counts: {dict(split_counter)}")
    print(f"Label counts: {dict(label_counter)}")
    print(f"Dropped split-conflict eyes: {dropped_conflicts}")
    print("Conflict policy note: 'drop' avoids eye-level train/val leakage.")


if __name__ == "__main__":
    main()
