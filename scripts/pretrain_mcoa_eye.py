"""V2 training entrypoint for eye-level MCOA pretraining."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icl_vault.data.collate_mcoa import collate_mcoa_eye_batch
from icl_vault.data.datasets import MCOAEyeDataset
from icl_vault.engine.checkpoint import resolve_checkpoint_dir, save_checkpoint
from icl_vault.engine.logger import setup_experiment_logger
from icl_vault.models.public_pretrain import MCOAEyeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 entrypoint for eye-level MCOA pretraining.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/manifests/mcoa_eye_manifest_medium.csv",
        help="Path to the eye-level MCOA manifest CSV file.",
    )
    parser.add_argument("--train_split", type=str, default="train", help="Training split name.")
    parser.add_argument("--val_split", type=str, default="val", help="Validation split name.")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument(
        "--max_slices",
        type=int,
        default=16,
        help="Maximum number of slices kept per eye. Longer eyes are truncated; shorter eyes are padded.",
    )
    parser.add_argument(
        "--use_imagenet_pretrain",
        action="store_true",
        help="Use ImageNet-pretrained torchvision ResNet18 weights.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="artifacts/logs",
        help="Directory used to store the experiment log file.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    args: argparse.Namespace,
) -> Tuple[MCOAEyeDataset, MCOAEyeDataset, DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms()
    train_dataset = MCOAEyeDataset(
        manifest_path=args.manifest_path,
        split=args.train_split,
        transform=train_transform,
    )
    val_dataset = MCOAEyeDataset(
        manifest_path=args.manifest_path,
        split=args.val_split,
        transform=val_transform,
    )

    collate_fn = partial(collate_mcoa_eye_batch, max_slices=args.max_slices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        slice_mask = batch["slice_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(images=images, slice_mask=slice_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            slice_mask = batch["slice_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images=images, slice_mask=slice_mask)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    denom = max(total_samples, 1)
    return {
        "val_loss": total_loss / denom,
        "val_accuracy": total_correct / denom,
    }


def build_log_path(args: argparse.Namespace) -> Path:
    manifest_stem = Path(args.manifest_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mcoa_eye_smoketest_{manifest_stem}_{timestamp}.log"
    return Path(args.log_dir) / filename


def log_command(logger: logging.Logger) -> None:
    logger.info("Command: %s", " ".join(sys.argv))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = build_log_path(args)
    logger = setup_experiment_logger(log_path=log_path, name="icl_vault.mcoa_eye")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)
    num_classes = len(train_dataset.label_to_index)
    model = MCOAEyeClassifier(
        num_classes=num_classes,
        use_imagenet_pretrain=args.use_imagenet_pretrain,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_dir = resolve_checkpoint_dir()
    latest_checkpoint_path = checkpoint_dir / "mcoa_eye_latest.pth"
    best_checkpoint_path = checkpoint_dir / "mcoa_eye_best.pth"
    best_val_accuracy = float("-inf")
    init_mode = "ImageNet pretrained" if args.use_imagenet_pretrain else "from scratch"

    logger.info("V2 training path: pretrain_mcoa_eye.py is running eye-level MCOA pretraining.")
    logger.info("Manifest note: eye-level manifest groups multiple AS-OCT slices per eye.")
    logger.info("Log file: %s", log_path)
    logger.info("Python executable: %s", sys.executable)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("Torchvision version: %s", getattr(sys.modules.get("torchvision"), "__version__", "unknown"))
    log_command(logger)
    logger.info("Manifest: %s", args.manifest_path)
    logger.info("Train split: %s | Val split: %s", args.train_split, args.val_split)
    logger.info(
        f"Batch size: {args.batch_size} | Num workers: {args.num_workers} | "
        f"Seed: {args.seed} | Epochs: {args.epochs} | LR: {args.lr}"
    )
    logger.info("Max slices per eye: %s", args.max_slices)
    logger.info("Device: %s", device)
    logger.info("Model init: %s", init_mode)
    logger.info("Num classes: %s", num_classes)
    logger.info("Label mapping: %s", train_dataset.label_to_index)
    logger.info("Train eyes: %s", len(train_dataset))
    logger.info("Val eyes: %s", len(val_dataset))
    logger.info("Train summary: %s", train_dataset.describe())
    logger.info("Val summary: %s", val_dataset.describe())
    logger.info("Train batches per epoch: %s", len(train_loader))
    logger.info("Val batches per epoch: %s", len(val_loader))

    train_batch = next(iter(train_loader), None)
    val_batch = next(iter(val_loader), None)
    logger.info("Preview train batch keys: %s", list(train_batch.keys()) if train_batch is not None else [])
    logger.info("Preview val batch keys: %s", list(val_batch.keys()) if val_batch is not None else [])
    if train_batch is not None:
        logger.info("Preview train images shape: %s", tuple(train_batch["images"].shape))
        logger.info("Preview train slice mask shape: %s", tuple(train_batch["slice_mask"].shape))
    if val_batch is not None:
        logger.info("Preview val images shape: %s", tuple(val_batch["images"].shape))
        logger.info("Preview val slice mask shape: %s", tuple(val_batch["slice_mask"].shape))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_accuracy={val_metrics['val_accuracy']:.4f}"
        )
        latest_metrics = {"train_loss": train_loss, **val_metrics}
        saved_latest_path = save_checkpoint(
            path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=latest_metrics,
        )
        logger.info("Latest checkpoint saved to: %s", saved_latest_path)

        if val_metrics["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["val_accuracy"]
            saved_best_path = save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=latest_metrics,
            )
            logger.info("Best checkpoint updated: True -> %s", saved_best_path)
        else:
            logger.info("Best checkpoint updated: False")


if __name__ == "__main__":
    main()
