"""V2 training entrypoint for eye-level MCOA OCT + ASP multimodal pretraining."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icl_vault.data.collate_mcoa_multimodal import collate_mcoa_multimodal_batch
from icl_vault.data.datasets import MCOAMultimodalDataset
from icl_vault.engine.checkpoint import resolve_checkpoint_dir, save_checkpoint
from icl_vault.engine.logger import setup_experiment_logger
from icl_vault.models.public_pretrain import MCOAMultimodalClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 entrypoint for multimodal MCOA pretraining.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/manifests/mcoa_multimodal_manifest_medium.csv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="oct_asp",
        choices=("oct_only", "asp_only", "oct_asp"),
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_slices", type=int, default=8)
    parser.add_argument("--use_imagenet_pretrain", action="store_true")
    parser.add_argument("--log_dir", type=str, default="artifacts/logs")
    parser.add_argument("--force_missing_asp_ratio", type=float, default=0.0)
    parser.add_argument("--force_missing_oct_ratio", type=float, default=0.0)
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
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return train_transform, val_transform


def build_dataloaders(
    args: argparse.Namespace,
) -> Tuple[MCOAMultimodalDataset, MCOAMultimodalDataset, DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms()
    train_dataset = MCOAMultimodalDataset(
        manifest_path=args.manifest_path,
        split=args.train_split,
        oct_transform=train_transform,
        asp_transform=train_transform,
    )
    val_dataset = MCOAMultimodalDataset(
        manifest_path=args.manifest_path,
        split=args.val_split,
        oct_transform=val_transform,
        asp_transform=val_transform,
    )

    collate_fn = partial(collate_mcoa_multimodal_batch, max_slices=args.max_slices)
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
    force_missing_oct_ratio: float = 0.0,
    force_missing_asp_ratio: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = apply_forced_missing_modalities(
            batch=batch,
            force_missing_oct_ratio=force_missing_oct_ratio,
            force_missing_asp_ratio=force_missing_asp_ratio,
        )
        oct_images = batch["oct_images"].to(device)
        asp_images = batch["asp_images"].to(device)
        oct_slice_mask = batch["oct_slice_mask"].to(device)
        has_oct = batch["has_oct"].to(device)
        has_asp = batch["has_asp"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(
            oct_images=oct_images,
            asp_images=asp_images,
            oct_slice_mask=oct_slice_mask,
            has_oct=has_oct,
            has_asp=has_asp,
        )
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
    force_missing_oct_ratio: float = 0.0,
    force_missing_asp_ratio: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = apply_forced_missing_modalities(
                batch=batch,
                force_missing_oct_ratio=force_missing_oct_ratio,
                force_missing_asp_ratio=force_missing_asp_ratio,
            )
            oct_images = batch["oct_images"].to(device)
            asp_images = batch["asp_images"].to(device)
            oct_slice_mask = batch["oct_slice_mask"].to(device)
            has_oct = batch["has_oct"].to(device)
            has_asp = batch["has_asp"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                oct_images=oct_images,
                asp_images=asp_images,
                oct_slice_mask=oct_slice_mask,
                has_oct=has_oct,
                has_asp=has_asp,
            )
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    denom = max(total_samples, 1)
    return {"val_loss": total_loss / denom, "val_accuracy": total_correct / denom}


def build_log_path(args: argparse.Namespace) -> Path:
    manifest_stem = Path(args.manifest_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_parts = [f"mode-{args.mode}"]
    if args.force_missing_oct_ratio > 0.0:
        suffix_parts.append(f"miss-oct-{str(args.force_missing_oct_ratio).replace('.', 'p')}")
    if args.force_missing_asp_ratio > 0.0:
        suffix_parts.append(f"miss-asp-{str(args.force_missing_asp_ratio).replace('.', 'p')}")
    suffix = "_".join(suffix_parts)
    return Path(args.log_dir) / f"mcoa_smoketest_{manifest_stem}_{suffix}_{timestamp}.log"


def log_command(logger: logging.Logger) -> None:
    logger.info("Command: %s", " ".join(sys.argv))


def apply_forced_missing_modalities(
    batch: dict[str, object],
    force_missing_oct_ratio: float,
    force_missing_asp_ratio: float,
) -> dict[str, object]:
    if force_missing_oct_ratio <= 0.0 and force_missing_asp_ratio <= 0.0:
        return batch

    updated_batch = dict(batch)
    labels = batch.get("labels")
    if not isinstance(labels, torch.Tensor):
        return batch

    batch_size = labels.size(0)
    if force_missing_oct_ratio > 0.0:
        oct_drop_mask = torch.rand(batch_size) < force_missing_oct_ratio
        updated_batch["has_oct"] = batch["has_oct"].clone()
        updated_batch["oct_images"] = batch["oct_images"].clone()
        updated_batch["oct_slice_mask"] = batch["oct_slice_mask"].clone()
        updated_batch["has_oct"][oct_drop_mask] = False
        updated_batch["oct_images"][oct_drop_mask] = 0
        updated_batch["oct_slice_mask"][oct_drop_mask] = False

    if force_missing_asp_ratio > 0.0:
        asp_drop_mask = torch.rand(batch_size) < force_missing_asp_ratio
        updated_batch["has_asp"] = batch["has_asp"].clone()
        updated_batch["asp_images"] = batch["asp_images"].clone()
        updated_batch["has_asp"][asp_drop_mask] = False
        updated_batch["asp_images"][asp_drop_mask] = 0

    return updated_batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_experiment_logger(log_path=build_log_path(args), name="icl_vault.mcoa_multimodal")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)
    num_classes = len(train_dataset.label_to_index)
    model = MCOAMultimodalClassifier(
        num_classes=num_classes,
        use_imagenet_pretrain=args.use_imagenet_pretrain,
        mode=args.mode,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_dir = resolve_checkpoint_dir()
    latest_checkpoint_path = checkpoint_dir / f"mcoa_{args.mode}_latest.pth"
    best_checkpoint_path = checkpoint_dir / f"mcoa_{args.mode}_best.pth"
    best_val_accuracy = float("-inf")
    init_mode = "ImageNet pretrained" if args.use_imagenet_pretrain else "from scratch"

    logger.info("V2 training path: pretrain_mcoa_multimodal.py is running OCT + ASP multimodal MCOA pretraining.")
    logger.info("Mode: %s", args.mode)
    logger.info("Python executable: %s", sys.executable)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("Torchvision version: %s", torchvision.__version__)
    log_command(logger)
    logger.info("Manifest: %s", args.manifest_path)
    logger.info("Train split: %s | Val split: %s", args.train_split, args.val_split)
    logger.info(
        "Batch size: %s | Num workers: %s | Seed: %s | Epochs: %s | LR: %s",
        args.batch_size,
        args.num_workers,
        args.seed,
        args.epochs,
        args.lr,
    )
    logger.info("Max slices per eye: %s", args.max_slices)
    logger.info("Force missing OCT ratio: %.2f", args.force_missing_oct_ratio)
    logger.info("Force missing ASP ratio: %.2f", args.force_missing_asp_ratio)
    logger.info("Missing-modality strategy: zero-image placeholders with has_oct/has_asp flags.")
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
        logger.info("Preview train OCT images shape: %s", tuple(train_batch["oct_images"].shape))
        logger.info("Preview train ASP images shape: %s", tuple(train_batch["asp_images"].shape))
        logger.info("Preview train OCT slice mask shape: %s", tuple(train_batch["oct_slice_mask"].shape))
        logger.info("Preview train has_oct: %s", train_batch["has_oct"].tolist())
        logger.info("Preview train has_asp: %s", train_batch["has_asp"].tolist())
    if val_batch is not None:
        logger.info("Preview val OCT images shape: %s", tuple(val_batch["oct_images"].shape))
        logger.info("Preview val ASP images shape: %s", tuple(val_batch["asp_images"].shape))
        logger.info("Preview val OCT slice mask shape: %s", tuple(val_batch["oct_slice_mask"].shape))
        logger.info("Preview val has_oct: %s", val_batch["has_oct"].tolist())
        logger.info("Preview val has_asp: %s", val_batch["has_asp"].tolist())

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            force_missing_oct_ratio=args.force_missing_oct_ratio,
            force_missing_asp_ratio=args.force_missing_asp_ratio,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            force_missing_oct_ratio=args.force_missing_oct_ratio,
            force_missing_asp_ratio=args.force_missing_asp_ratio,
        )
        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | val_accuracy=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["val_accuracy"],
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
