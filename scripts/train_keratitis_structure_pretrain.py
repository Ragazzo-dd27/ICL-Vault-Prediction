"""Minimal V2 structure pretraining entrypoint for keratitis OCT."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icl_vault.data.datasets import KeratitisStructureDataset
from icl_vault.engine.checkpoint import resolve_checkpoint_dir, save_checkpoint
from icl_vault.engine.logger import setup_experiment_logger
from icl_vault.models.public_pretrain import KeratitisStructureModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal keratitis OCT structure pretraining line.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/manifests/keratitis_structure_manifest.csv",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="artifacts/logs")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_log_path(args: argparse.Namespace) -> Path:
    manifest_stem = Path(args.manifest_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.log_dir) / f"keratitis_structure_smoketest_{manifest_stem}_{timestamp}.log"


def log_command(logger: logging.Logger) -> None:
    logger.info("Command: %s", " ".join(sys.argv))


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denominator + eps)).mean()


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
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
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
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score_from_logits(logits, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_dice += dice.item() * batch_size
            total_samples += batch_size

    denom = max(total_samples, 1)
    return {
        "val_loss": total_loss / denom,
        "val_dice": total_dice / denom,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_experiment_logger(build_log_path(args), name="icl_vault.keratitis_structure")

    train_dataset = KeratitisStructureDataset(
        manifest_path=args.manifest_path,
        split=args.train_split,
        image_size=args.image_size,
    )
    val_dataset = KeratitisStructureDataset(
        manifest_path=args.manifest_path,
        split=args.val_split,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = KeratitisStructureModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_dir = resolve_checkpoint_dir()
    latest_checkpoint_path = checkpoint_dir / "keratitis_structure_latest.pth"
    best_checkpoint_path = checkpoint_dir / "keratitis_structure_best.pth"
    best_val_dice = float("-inf")

    logger.info("V2 training path: train_keratitis_structure_pretrain.py is running keratitis OCT structure pretraining.")
    logger.info("This line is the AIDK-blocked substitute transition line, not AIDK itself.")
    logger.info("Task: cornea_segmentation")
    logger.info("Python executable: %s", sys.executable)
    logger.info("PyTorch version: %s", torch.__version__)
    log_command(logger)
    logger.info("Manifest: %s", args.manifest_path)
    logger.info("Train split: %s | Val split: %s", args.train_split, args.val_split)
    logger.info(
        "Batch size: %s | Num workers: %s | Seed: %s | Epochs: %s | LR: %s | Image size: %s",
        args.batch_size,
        args.num_workers,
        args.seed,
        args.epochs,
        args.lr,
        args.image_size,
    )
    logger.info("Device: %s", device)
    logger.info("Train samples: %s", len(train_dataset))
    logger.info("Val samples: %s", len(val_dataset))
    logger.info("Train summary: %s", train_dataset.describe())
    logger.info("Val summary: %s", val_dataset.describe())
    logger.info("Train batches per epoch: %s", len(train_loader))
    logger.info("Val batches per epoch: %s", len(val_loader))

    train_batch = next(iter(train_loader), None)
    if train_batch is not None:
        logger.info("Preview train image shape: %s", tuple(train_batch["image"].shape))
        logger.info("Preview train mask shape: %s", tuple(train_batch["mask"].shape))
        logger.info("Preview train sample ids: %s", train_batch["sample_id"])

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["val_dice"],
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

        if val_metrics["val_dice"] > best_val_dice:
            best_val_dice = val_metrics["val_dice"]
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
