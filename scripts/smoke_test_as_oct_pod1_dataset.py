"""Smoke test the AS-OCT-only POD1 vault regression manifest.

This is not training. It checks the first AS-OCT-only POD1 smoke-test manifest,
image paths, labels, split filtering, optional VaultDataset loading, and batch
collation without modifying the manifest or training code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REQUIRED_COLUMNS = (
    "sample_id",
    "patient_id",
    "eye_side",
    "split",
    "oct_path",
    "has_oct",
    "vault_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test an AS-OCT-only POD1 vault regression manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/vault_as_oct_only_pod1_manifest_batch_01_clean.csv",
        help="AS-OCT-only POD1 manifest CSV to test.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to inspect.")
    parser.add_argument("--batch_size", type=int, default=4, help="DataLoader batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--image_size", type=int, default=224, help="Resize AS-OCT images to this square size.")
    parser.add_argument(
        "--strict_oct_only",
        action="store_true",
        help="Require that the dataset/batch does not load UBM tensors.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_data_path(value: object) -> Path | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_distribution(title: str, series: pd.Series) -> None:
    print(f"{title}:")
    if series.empty:
        print("  none")
        return
    for value, count in series.value_counts().sort_index().items():
        print(f"  {value}: {count}")


def check_required_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def describe_manifest(df: pd.DataFrame) -> None:
    labels = pd.to_numeric(df["vault_label"], errors="coerce").dropna()
    print(f"Manifest rows: {len(df)}")
    print_distribution("Split distribution", df["split"])
    if labels.empty:
        print("vault_label stats: no numeric labels")
    else:
        print(
            "vault_label stats: "
            f"mean={labels.mean():.2f}, "
            f"std={labels.std():.2f}, "
            f"min={labels.min():.2f}, "
            f"max={labels.max():.2f}"
        )


def check_oct_paths(df: pd.DataFrame) -> tuple[int, int]:
    missing = int(df["oct_path"].fillna("").astype(str).str.strip().eq("").sum())
    nonexistent = 0
    for value in df["oct_path"]:
        resolved = resolve_data_path(value)
        if resolved is None or not resolved.exists():
            nonexistent += 1
    return missing, nonexistent


def preview_pil_images(split_df: pd.DataFrame, max_count: int = 5) -> tuple[int, list[str]]:
    opened = 0
    failures: list[str] = []
    print(f"PIL image preview: first {min(max_count, len(split_df))} rows")
    for row in split_df.head(max_count).to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        oct_path = str(row["oct_path"])
        resolved = resolve_data_path(oct_path)
        if resolved is None:
            failures.append(f"{sample_id}: empty oct_path")
            print(f"  {sample_id} | missing oct_path | vault_label={row['vault_label']}")
            continue
        try:
            with Image.open(resolved) as image:
                image = ImageOps.exif_transpose(image)
                print(
                    "  "
                    f"{sample_id} | {oct_path} | size={image.size} | "
                    f"mode={image.mode} | vault_label={row['vault_label']}"
                )
            opened += 1
        except Exception as exc:
            failures.append(f"{sample_id}: {exc}")
            print(f"  {sample_id} | failed to open {oct_path}: {exc}")
    return opened, failures


def describe_value(value: Any) -> str:
    if value is None:
        return "None"
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={tuple(value.shape)})"
    if isinstance(value, dict):
        return f"dict(len={len(value)}, keys={list(value.keys())[:8]})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    if value is None or not hasattr(value, "shape"):
        return None
    return tuple(value.shape)


def bool_tensor_true_count(value: Any) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    try:
        total = int(value.numel())
        true_count = int(value.sum().item())
        return true_count, total
    except Exception:
        return (0, 0)


def build_oct_transform(torch_module: Any, image_size: int) -> Any:
    try:
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
    except Exception as exc:
        print(f"torchvision unavailable, using local PIL resize + tensor transform: {exc}")

        def _transform(image: Image.Image) -> Any:
            import numpy as np

            resized = image.resize((image_size, image_size))
            image_array = np.asarray(resized, dtype="float32") / 255.0
            return torch_module.from_numpy(image_array).permute(2, 0, 1)

        return _transform


def try_dataset_and_dataloader(
    manifest_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    strict_oct_only: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "torch_available": False,
        "dataset_instantiated": False,
        "dataloader_batch": False,
        "dataset_output_mode": "not_checked",
        "oct_image_shape": None,
        "batch_oct_images_shape": None,
        "batch_vault_labels_shape": None,
        "ubm_images_is_none": None,
        "ubm_available_true_count": None,
        "ubm_available_total": None,
        "strict_oct_only_ok": None,
        "message": "",
    }

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        result["message"] = f"torch unavailable, skipped VaultDataset/DataLoader checks: {exc}"
        print(result["message"])
        return result

    result["torch_available"] = True
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    try:
        from icl_vault.data.collate import collate_vault_batch
        from icl_vault.data.datasets import VaultDataset

        oct_transform = build_oct_transform(torch_module=torch, image_size=image_size)
        dataset = VaultDataset(manifest_path=str(manifest_path), split=split, oct_transform=oct_transform)
        result["dataset_instantiated"] = True
        print(f"VaultDataset instantiated: len={len(dataset)}")
        if len(dataset) == 0:
            result["message"] = "VaultDataset split is empty; skipped dataset[0] and DataLoader batch."
            print(result["message"])
            return result

        item = dataset[0]
        image_value = item.get("oct_image") if isinstance(item, dict) else None
        result["oct_image_shape"] = tensor_shape(image_value)
        result["dataset_output_mode"] = "image_tensor" if image_value is not None else "metadata_only"
        print("VaultDataset[0] keys/types:")
        for key in sorted(item.keys()):
            print(f"  {key}: {describe_value(item[key])}")
        print(f"VaultDataset[0] oct_image shape: {result['oct_image_shape']}")
        print(f"Dataset output mode: {result['dataset_output_mode']}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_vault_batch,
        )
        batch = next(iter(loader), None)
        if batch is None:
            result["message"] = "DataLoader returned no batch."
            print(result["message"])
            return result

        result["dataloader_batch"] = True
        result["batch_oct_images_shape"] = tensor_shape(batch.get("oct_images"))
        result["batch_vault_labels_shape"] = tensor_shape(batch.get("vault_labels"))
        result["ubm_images_is_none"] = batch.get("ubm_images") is None
        true_count, total = bool_tensor_true_count(batch.get("ubm_available"))
        result["ubm_available_true_count"] = true_count
        result["ubm_available_total"] = total
        result["strict_oct_only_ok"] = bool(result["ubm_images_is_none"] or true_count == 0)

        print("First DataLoader batch keys/types:")
        for key in sorted(batch.keys()):
            print(f"  {key}: {describe_value(batch[key])}")
        print(f"First DataLoader batch oct_images shape: {result['batch_oct_images_shape']}")
        print(f"First DataLoader batch vault_labels shape: {result['batch_vault_labels_shape']}")
        print(f"First DataLoader batch ubm_images is None: {result['ubm_images_is_none']}")
        print(f"First DataLoader batch ubm_available true/total: {true_count}/{total}")
        if isinstance(batch.get("meta"), dict):
            print("Batch meta field lengths:")
            for key, value in sorted(batch["meta"].items()):
                print(f"  meta.{key}: {describe_value(value)}")
        if strict_oct_only and not result["strict_oct_only_ok"]:
            result["message"] = "strict_oct_only check failed: UBM tensors or availability flags are present."
            print(result["message"])
        return result
    except Exception as exc:
        result["message"] = f"VaultDataset/DataLoader check failed: {exc}"
        print(result["message"])
        return result


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    df = pd.read_csv(manifest_path)
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
    describe_manifest(df)

    missing_columns = check_required_columns(df)
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print("SMOKE TEST FAILED")
        raise SystemExit(1)
    print("Required columns: ok")

    split_df = df[df["split"].fillna("").astype(str) == args.split].copy()
    print(f"Selected split: {args.split}")
    print(f"Selected split rows: {len(split_df)}")
    if split_df.empty:
        print("SMOKE TEST FAILED: selected split is empty")
        raise SystemExit(1)

    missing_oct_path, nonexistent_oct_path = check_oct_paths(df)
    print(f"Missing oct_path rows: {missing_oct_path}")
    print(f"Nonexistent oct_path rows: {nonexistent_oct_path}")

    pil_opened, pil_failures = preview_pil_images(split_df, max_count=5)
    print(f"PIL opened preview images: {pil_opened}")
    if pil_failures:
        print("PIL preview failures:")
        for failure in pil_failures:
            print(f"  {failure}")

    dataset_result = try_dataset_and_dataloader(
        manifest_path=manifest_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        strict_oct_only=args.strict_oct_only,
    )

    passed = (
        not missing_columns
        and missing_oct_path == 0
        and nonexistent_oct_path == 0
        and pil_opened == min(5, len(split_df))
    )
    if dataset_result["torch_available"]:
        passed = passed and dataset_result["dataset_instantiated"] and dataset_result["dataloader_batch"]
        expected_item_shape = (3, args.image_size, args.image_size)
        expected_batch_shape = (args.batch_size, 3, args.image_size, args.image_size)
        passed = passed and dataset_result["oct_image_shape"] == expected_item_shape
        passed = passed and dataset_result["batch_oct_images_shape"] == expected_batch_shape
        if args.strict_oct_only:
            passed = passed and bool(dataset_result["strict_oct_only_ok"])
    else:
        passed = False

    print(f"VaultDataset instantiated: {dataset_result['dataset_instantiated']}")
    print(f"DataLoader produced batch: {dataset_result['dataloader_batch']}")
    print(f"VaultDataset[0] oct_image shape: {dataset_result['oct_image_shape']}")
    print(f"First DataLoader batch oct_images shape: {dataset_result['batch_oct_images_shape']}")
    print(f"First DataLoader batch vault_labels shape: {dataset_result['batch_vault_labels_shape']}")
    print(f"First DataLoader batch ubm_images is None: {dataset_result['ubm_images_is_none']}")
    print(
        "First DataLoader batch ubm_available true/total: "
        f"{dataset_result['ubm_available_true_count']}/{dataset_result['ubm_available_total']}"
    )
    print(f"Dataset output mode: {dataset_result['dataset_output_mode']}")
    print(f"SMOKE TEST {'PASSED' if passed else 'FAILED'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
