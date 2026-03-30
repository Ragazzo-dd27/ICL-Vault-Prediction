"""Minimal evaluator scaffold for V2 classification tasks."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ClassificationEvaluator:
    """Minimal classification evaluator for V2.

    This evaluator currently focuses on validation for simple classification
    tasks such as MCOA backbone pretraining.

    TODO: Extend the evaluator family to support regression tasks.
    TODO: Add richer metrics beyond loss and accuracy.
    TODO: Integrate more tightly with the Trainer abstraction when the engine
    layer becomes more complete.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        prefix: str = "val",
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.prefix = prefix
        self.config = config or {}

    def evaluate(self) -> Dict[str, float]:
        """Run a minimal classification evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=1)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

        denom = max(total_samples, 1)
        return {
            f"{self.prefix}_loss": total_loss / denom,
            f"{self.prefix}_accuracy": total_correct / denom,
        }
