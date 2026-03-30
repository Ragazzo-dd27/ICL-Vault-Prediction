"""Minimal checkpoint helpers for V2 MCOA training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def resolve_checkpoint_dir(base_dir: str = "artifacts/checkpoints") -> Path:
    """Return the checkpoint directory path and ensure it exists."""
    checkpoint_dir = Path(base_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a minimal training checkpoint.

    TODO: Add resume/load helpers in a later step.
    TODO: Add scheduler state and extra config metadata when needed.
    TODO: Route checkpointing through a unified trainer abstraction later.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path
