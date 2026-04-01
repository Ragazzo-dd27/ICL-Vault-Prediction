"""V2 training entrypoint for MCOA backbone pretraining.

This is the first real trainable path in the V2 codebase. It keeps the
implementation intentionally minimal while using the new manifest-driven MCOA
dataset and a torchvision ResNet18 classifier.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icl_vault.data.datasets import MCOADataset
from icl_vault.engine.checkpoint import resolve_checkpoint_dir, save_checkpoint
from icl_vault.engine.evaluator import ClassificationEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for V2 MCOA pretraining."""
    parser = argparse.ArgumentParser(description="V2 entrypoint for MCOA backbone pretraining.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="data/manifests/mcoa_manifest_example.csv",
        help="Path to the MCOA manifest CSV file.",
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
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--use_imagenet_pretrain",
        action="store_true",
        help="Use ImageNet-pretrained torchvision ResNet18 weights.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set a basic random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Build minimal train/val transforms for ResNet18 pretraining."""
    train_transform = transforms.Compose(
        [
            # Conservative augmentation for small-sample MCOA pretraining.
            # TODO: Tune a stronger augmentation policy only after baseline
            # behavior is well understood on real experiments.
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


def build_dataloaders(args: argparse.Namespace) -> Tuple[MCOADataset, MCOADataset, DataLoader, DataLoader]:
    """Build train/val datasets and dataloaders from the MCOA manifest."""
    train_transform, val_transform = build_transforms()
    train_dataset = MCOADataset(
        manifest_path=args.manifest_path,
        split=args.train_split,
        transform=train_transform,
    )
    val_dataset = MCOADataset(
        manifest_path=args.manifest_path,
        split=args.val_split,
        transform=val_transform,
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
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(
    num_classes: int,
    device: torch.device,
    use_imagenet_pretrain: bool = False,
) -> nn.Module:
    """Build a minimal ResNet18 classifier."""
    weights = ResNet18_Weights.DEFAULT if use_imagenet_pretrain else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main() -> None:
    """Run minimal real training for V2 MCOA backbone pretraining."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)
    num_classes = len(train_dataset.label_to_index)
    model = build_model(
        num_classes=num_classes,
        device=device,
        use_imagenet_pretrain=args.use_imagenet_pretrain,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_dir = resolve_checkpoint_dir()
    latest_checkpoint_path = checkpoint_dir / "mcoa_latest.pth"
    best_checkpoint_path = checkpoint_dir / "mcoa_best.pth"
    best_val_accuracy = float("-inf")
    init_mode = "ImageNet pretrained" if args.use_imagenet_pretrain else "from scratch"

    print("V2 training path: pretrain_backbone.py is running minimal real MCOA pretraining.")
    print("Train transform: basic augmentation enabled.")
    print(f"Manifest: {args.manifest_path}")
    print(f"Train split: {args.train_split} | Val split: {args.val_split}")
    print(
        f"Batch size: {args.batch_size} | Num workers: {args.num_workers} | "
        f"Seed: {args.seed} | Epochs: {args.epochs} | LR: {args.lr}"
    )
    print(f"Device: {device}")
    print(f"Model init: {init_mode}")
    print(f"Num classes: {num_classes}")
    print(f"Label mapping: {train_dataset.label_to_index}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train summary: {train_dataset.describe()}")
    print(f"Val summary: {val_dataset.describe()}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    train_batch = next(iter(train_loader), None)
    val_batch = next(iter(val_loader), None)
    print(f"Preview train batch keys: {list(train_batch.keys()) if train_batch is not None else []}")
    print(f"Preview val batch keys: {list(val_batch.keys()) if val_batch is not None else []}")
    if train_batch is not None:
        print(f"Preview train image batch shape: {tuple(train_batch['image'].shape)}")
    if val_batch is not None:
        print(f"Preview val image batch shape: {tuple(val_batch['image'].shape)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        evaluator = ClassificationEvaluator(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            prefix="val",
        )
        val_metrics = evaluator.evaluate()
        print(
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
        print(f"Latest checkpoint saved to: {saved_latest_path}")

        if val_metrics["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["val_accuracy"]
            saved_best_path = save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=latest_metrics,
            )
            print(f"Best checkpoint updated: True -> {saved_best_path}")
        else:
            print("Best checkpoint updated: False")

    # TODO: Add resume support from saved checkpoints.
    # TODO: Save scheduler state and extra run config when needed.
    # TODO: Integrate with a more complete Trainer/Evaluator abstraction.
    # TODO: Add stronger transforms and augmentation policies for MCOA pretraining.


if __name__ == "__main__":
    main()
