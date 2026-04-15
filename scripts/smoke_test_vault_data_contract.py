"""Minimal smoke test for the V2 vault data contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icl_vault.data.collate import collate_vault_batch
from icl_vault.data.datasets import VaultDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the V2 vault data contract.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/manifests/vault_manifest_example.csv",
        help="Path to the vault manifest CSV.",
    )
    parser.add_argument("--split", type=str, default=None, help="Optional split filter.")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Optional explicit log path. Defaults to artifacts/logs/<timestamped>.json",
    )
    return parser.parse_args()


def _tensor_summary(tensor: Any) -> Dict[str, Any]:
    if tensor is None:
        return {"present": False, "shape": None, "dtype": None}
    return {"present": True, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def build_log_path(manifest_path: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_stem = Path(manifest_path).stem
    return PROJECT_ROOT / "artifacts" / "logs" / f"vault_data_contract_smoketest_{manifest_stem}_{timestamp}.json"


def main() -> None:
    args = parse_args()

    dataset = VaultDataset(manifest_path=args.manifest_path, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_vault_batch,
    )

    batch = next(iter(loader), None)
    if batch is None:
        raise ValueError("The selected manifest/split produced no samples, so no batch could be built.")

    result = {
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "split": args.split,
        "dataset_summary": dataset.describe(),
        "dataset_size": len(dataset),
        "batch_size": len(batch["meta"]["sample_id"]),
        "batch_keys": list(batch.keys()),
        "tensor_shapes": batch["tensor_shapes"],
        "oct_images": _tensor_summary(batch["oct_images"]),
        "ubm_images": _tensor_summary(batch["ubm_images"]),
        "topography_images": _tensor_summary(batch["topography_images"]),
        "vault_labels_shape": list(batch["vault_labels"].shape),
        "label_available": batch["label_available"].tolist(),
        "has_oct": batch["has_oct"].tolist(),
        "has_ubm": batch["has_ubm"].tolist(),
        "has_topography": batch["has_topography"].tolist(),
        "oct_available": batch["oct_available"].tolist(),
        "ubm_available": batch["ubm_available"].tolist(),
        "topography_available": batch["topography_available"].tolist(),
        "sample_ids": batch["meta"]["sample_id"],
    }

    log_path = Path(args.log_path) if args.log_path else build_log_path(args.manifest_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Smoke test log written to: {log_path}")


if __name__ == "__main__":
    main()
