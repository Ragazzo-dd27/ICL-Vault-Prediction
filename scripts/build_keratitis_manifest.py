"""Build a minimal V2 manifest for the keratitis OCT structure pretraining line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal keratitis structure manifest.")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/public_datasets/keratitis_oct/images",
        help="Directory containing keratitis OCT BMP images.",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="data/public_datasets/keratitis_oct/masks",
        help="Directory containing keratitis OCT LabelMe JSON annotations.",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="data/manifests/keratitis_structure_manifest.csv",
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cornea_segmentation",
        choices=("cornea_segmentation",),
        help="Structure task to expose in the manifest.",
    )
    return parser.parse_args()


def resolve_split(index: int) -> str:
    """Stable 80/20 split based on sorted sample order."""
    return "val" if index % 5 == 0 else "train"


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_manifest = Path(args.output_manifest)

    image_paths = {
        path.stem: path
        for path in sorted(images_dir.glob("*.bmp"))
        if path.is_file() and not path.name.startswith(".")
    }
    mask_paths = {
        path.stem: path
        for path in sorted(masks_dir.glob("*.json"))
        if path.is_file() and not path.name.startswith(".")
    }

    paired_ids = sorted(set(image_paths) & set(mask_paths), key=lambda value: int(value))
    records: list[dict[str, str]] = []
    for index, sample_id in enumerate(paired_ids):
        records.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_paths[sample_id].as_posix()),
                "mask_path": str(mask_paths[sample_id].as_posix()),
                "task": args.task,
                "split": resolve_split(index),
            }
        )

    df = pd.DataFrame(records)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest, index=False)

    print(f"paired_samples={len(df)}")
    print(f"train_samples={(df['split'] == 'train').sum() if not df.empty else 0}")
    print(f"val_samples={(df['split'] == 'val').sum() if not df.empty else 0}")
    print(f"output_manifest={output_manifest.as_posix()}")


if __name__ == "__main__":
    main()
