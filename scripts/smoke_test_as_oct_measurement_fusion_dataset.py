"""Smoke test AS-OCT + true preop measurement fusion manifests.

This fusion dataset uses preoperative AS-OCT image and true preoperative
2DAnalysis measurements only. Postoperative 2DAnalysis measurements must not be
used as input features.

The script does not modify manifests, does not modify training code, does not
train models, and does not use UBM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "data/manifests/vault_as_oct_plus_preop_measurement_pod1_manifest_combined_ready.csv"
FEATURE_COLUMNS = ["cct_mean_um", "acd_epi_mean_mm", "acd_endo_mean_mm", "clr_mean_um", "ata_mean_mm"]
REQUIRED_COLUMNS = [
    "global_sample_id",
    "sample_id",
    "batch_id",
    "global_patient_uid",
    "split",
    "oct_path",
    "vault_label",
    *FEATURE_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test AS-OCT + measurement fusion dataset.")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def path_exists(path_text: object) -> bool:
    if pd.isna(path_text):
        return False
    text = str(path_text).strip()
    if not text or text.lower() == "nan":
        return False
    return resolve_project_path(text).exists()


def validate_manifest(df: pd.DataFrame) -> Dict[str, object]:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    labels = pd.to_numeric(df["vault_label"], errors="coerce") if "vault_label" in df.columns else pd.Series(dtype=float)
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce") if all(
        col in df.columns for col in FEATURE_COLUMNS
    ) else pd.DataFrame()
    patient_cross_split = bool((df.groupby("global_patient_uid")["split"].nunique() > 1).any()) if {
        "global_patient_uid",
        "split",
    }.issubset(df.columns) else True
    return {
        "missing_required_columns": missing_cols,
        "oct_path_missing_or_nonexistent": int((~df["oct_path"].map(path_exists)).sum()) if "oct_path" in df.columns else -1,
        "invalid_vault_label": int((labels.isna() | (labels <= 0)).sum()) if not labels.empty else -1,
        "missing_measurement_features": int(features.isna().any(axis=1).sum()) if not features.empty else -1,
        "global_sample_id_duplicates": int(df["global_sample_id"].duplicated().sum()) if "global_sample_id" in df.columns else -1,
        "global_patient_uid_cross_split": patient_cross_split,
    }


def print_manifest_summary(df: pd.DataFrame) -> None:
    labels = pd.to_numeric(df["vault_label"], errors="coerce")
    print(f"Manifest rows: {len(df)}")
    print(f"Split distribution: {df['split'].value_counts(dropna=False).to_dict()}")
    print(f"Batch distribution: {df['batch_id'].value_counts(dropna=False).to_dict()}")
    print(f"measurement_ready_status distribution: {df['measurement_ready_status'].value_counts(dropna=False).to_dict()}")
    print(f"label_qc_flag distribution: {df['label_qc_flag'].value_counts(dropna=False).to_dict()}")
    print(
        "vault_label mean/std/min/max: "
        f"{labels.mean():.2f} / {labels.std():.2f} / {labels.min():.2f} / {labels.max():.2f}"
    )


def preview_pil_images(df: pd.DataFrame, n: int = 5) -> bool:
    ok = True
    print("PIL preview of first images:")
    for _, row in df.head(n).iterrows():
        oct_path = resolve_project_path(row["oct_path"])
        try:
            with Image.open(oct_path) as image:
                image = ImageOps.exif_transpose(image)
                size = image.size
                mode = image.mode
            feature_values = {col: row[col] for col in FEATURE_COLUMNS}
            print(
                f"  {row['global_sample_id']} | {row['oct_path']} | size={size} | mode={mode} | "
                f"vault_label={row['vault_label']} | features={feature_values}"
            )
        except Exception as exc:
            ok = False
            print(f"  FAILED {row.get('global_sample_id', '')}: {oct_path} ({exc})")
    return ok


def train_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df["split"].astype(str).eq("train")].copy()
    features = train_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame({"mean": features.mean(), "std": features.std()})
    return stats


def try_import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset

        return torch, Dataset, DataLoader
    except Exception as exc:
        print(f"Torch unavailable; CSV/PIL checks only. Reason: {exc}")
        return None, None, None


def build_image_transform(torch, image_size: int):
    try:
        from torchvision import transforms

        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    except Exception as exc:
        print(f"torchvision transform unavailable; using PIL+torch fallback. Reason: {exc}")

        def fallback(image: Image.Image):
            image = image.resize((image_size, image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = np.stack([array, array, array], axis=-1)
            if array.shape[-1] == 4:
                array = array[..., :3]
            array = np.transpose(array, (2, 0, 1))
            return torch.from_numpy(array)

        return fallback


def make_fusion_dataset_class(torch, Dataset):
    class FusionDataset(Dataset):
        def __init__(self, df: pd.DataFrame, transform):
            self.df = df.reset_index(drop=True).copy()
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, index: int) -> Dict[str, object]:
            row = self.df.iloc[index]
            oct_path = resolve_project_path(row["oct_path"])
            with Image.open(oct_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                oct_image = self.transform(image) if self.transform else image
            measurement_values = pd.to_numeric(row[FEATURE_COLUMNS], errors="coerce").astype(float).to_numpy(dtype=np.float32)
            return {
                "global_sample_id": row["global_sample_id"],
                "sample_id": row["sample_id"],
                "batch_id": row["batch_id"],
                "global_patient_uid": row["global_patient_uid"],
                "split": row["split"],
                "oct_image": oct_image,
                "measurement_features": torch.tensor(measurement_values, dtype=torch.float32),
                "measurement_feature_names": FEATURE_COLUMNS,
                "vault_label": torch.tensor(float(row["vault_label"]), dtype=torch.float32),
                "oct_path": row["oct_path"],
                "metadata": {
                    "eye": row.get("eye", ""),
                    "eye_side": row.get("eye_side", ""),
                    "measurement_ready_status": row.get("measurement_ready_status", ""),
                    "label_qc_flag": row.get("label_qc_flag", ""),
                },
            }

    return FusionDataset


def fusion_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    import torch

    return {
        "global_sample_id": [item["global_sample_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "batch_id": [item["batch_id"] for item in batch],
        "global_patient_uid": [item["global_patient_uid"] for item in batch],
        "split": [item["split"] for item in batch],
        "oct_images": torch.stack([item["oct_image"] for item in batch], dim=0),
        "measurement_features": torch.stack([item["measurement_features"] for item in batch], dim=0),
        "measurement_feature_names": FEATURE_COLUMNS,
        "vault_labels": torch.stack([item["vault_label"] for item in batch], dim=0),
        "oct_path": [item["oct_path"] for item in batch],
        "meta": [item["metadata"] for item in batch],
    }


def run_dataloader_check(df: pd.DataFrame, split: str, image_size: int, batch_size: int, num_workers: int) -> bool:
    torch, Dataset, DataLoader = try_import_torch()
    if torch is None:
        return True
    split_df = df[df["split"].astype(str).eq(split)].copy()
    transform = build_image_transform(torch, image_size)
    FusionDataset = make_fusion_dataset_class(torch, Dataset)
    dataset = FusionDataset(split_df, transform=transform)
    if len(dataset) == 0:
        print("Dataset split is empty; DataLoader check failed.")
        return False
    first = dataset[0]
    print(f"Dataset[0] oct_image shape: {tuple(first['oct_image'].shape)}")
    print(f"Dataset[0] measurement_features shape: {tuple(first['measurement_features'].shape)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=fusion_collate)
    batch = next(iter(loader))
    print(f"First batch oct_images shape: {tuple(batch['oct_images'].shape)}")
    print(f"First batch measurement_features shape: {tuple(batch['measurement_features'].shape)}")
    print(f"First batch vault_labels shape: {tuple(batch['vault_labels'].shape)}")
    print(f"First batch global_sample_id: {batch['global_sample_id'][:min(5, len(batch['global_sample_id']))]}")
    print(f"Measurement feature names: {batch['measurement_feature_names']}")
    has_nan = bool(torch.isnan(batch["measurement_features"]).any() or torch.isnan(batch["vault_labels"]).any())
    print(f"Batch contains NaN: {has_nan}")
    expected_oct_shape = (min(batch_size, len(dataset)), 3, image_size, image_size)
    ok = (
        tuple(batch["oct_images"].shape) == expected_oct_shape
        and tuple(batch["measurement_features"].shape)[1] == len(FEATURE_COLUMNS)
        and tuple(batch["vault_labels"].shape) == (min(batch_size, len(dataset)),)
        and not has_nan
    )
    return ok


def main() -> None:
    args = parse_args()
    manifest_path = resolve_project_path(args.manifest)
    df = pd.read_csv(manifest_path)
    print(f"Manifest path: {manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
    print_manifest_summary(df)

    validation = validate_manifest(df)
    print(f"Validation checks: {validation}")
    missing_required = bool(validation["missing_required_columns"])
    selected = df[df["split"].astype(str).eq(args.split)].copy()
    print(f"Selected split '{args.split}' rows: {len(selected)}")
    stats = train_feature_stats(df)
    print("Train split feature mean/std (use train split only for scaler during training):")
    print(stats.to_string())

    pil_ok = preview_pil_images(selected, n=5)
    csv_ok = (
        not missing_required
        and validation["oct_path_missing_or_nonexistent"] == 0
        and validation["invalid_vault_label"] == 0
        and validation["missing_measurement_features"] == 0
        and validation["global_sample_id_duplicates"] == 0
        and not validation["global_patient_uid_cross_split"]
        and len(selected) > 0
    )
    dataloader_ok = run_dataloader_check(
        df=df,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    passed = bool(csv_ok and pil_ok and dataloader_ok)
    print("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")


if __name__ == "__main__":
    main()
