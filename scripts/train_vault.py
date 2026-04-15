"""V2 scaffold entrypoint for vault prediction training.

This script standardizes argument parsing and data loading for the V2 project.
It intentionally does not perform real model training yet.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icl_vault.data.datasets import VaultDataset
from icl_vault.data.collate import collate_vault_batch
from icl_vault.engine.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the V2 scaffold training entrypoint."""
    parser = argparse.ArgumentParser(description="V2 scaffold entrypoint for vault training.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/manifests/vault_manifest_example.csv",
        help="Path to the manifest CSV file.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Split name used for the training dataset.",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        help="Split name used for the validation dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Mini-batch size for train and validation dataloaders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for dataloaders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of the scaffold run.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set a basic random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> Tuple[VaultDataset, VaultDataset, DataLoader, DataLoader]:
    """Build train/val datasets and dataloaders from the manifest."""
    train_dataset = VaultDataset(manifest_path=args.manifest_path, split=args.train_split)
    val_dataset = VaultDataset(manifest_path=args.manifest_path, split=args.val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_vault_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_vault_batch,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def main() -> None:
    """Run the V2 vault training scaffold."""
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)

    print("V2 scaffold: train_vault.py is a placeholder training entrypoint.")
    print("Current step: scaffold trainer integration for the V2 main task.")
    print("No real multimodal model, forward pass, loss, optimizer, checkpoint, or metrics are wired yet.")
    print(f"Manifest: {args.manifest_path}")
    print(f"Train split: {args.train_split} | Val split: {args.val_split}")
    print(f"Batch size: {args.batch_size} | Num workers: {args.num_workers} | Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train summary: {train_dataset.describe()}")
    print(f"Val summary: {val_dataset.describe()}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    # Placeholder flow to prove the entrypoint can build and inspect one batch.
    train_batch = next(iter(train_loader), None)
    val_batch = next(iter(val_loader), None)
    print(f"Preview train batch size: {len(train_batch['meta']['sample_id']) if train_batch is not None else 0}")
    print(f"Preview val batch size: {len(val_batch['meta']['sample_id']) if val_batch is not None else 0}")
    print(f"Preview train batch keys: {list(train_batch.keys()) if train_batch is not None else []}")
    print(f"Preview val batch keys: {list(val_batch.keys()) if val_batch is not None else []}")
    print(f"Preview train tensor shapes: {train_batch['tensor_shapes'] if train_batch is not None else {}}")
    print(f"Preview val tensor shapes: {val_batch['tensor_shapes'] if val_batch is not None else {}}")

    trainer = Trainer(
        model=None,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
        config={"task": "vault_scaffold"},
    )
    trainer.fit()

    # TODO: Connect the V2 multimodal model once the dedicated model module exists.
    # TODO: Add clinical feature loading after the real manifest/schema is finalized.
    # TODO: Add optimizer, scheduler, metrics, checkpointing, and logging integration.
    # TODO: Replace model=None with the future V2 multimodal model entry.


if __name__ == "__main__":
    main()
