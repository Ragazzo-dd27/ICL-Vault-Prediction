"""Minimal eye-level MCOA classifier with slice encoding and mean pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MCOAEyeClassifier(nn.Module):
    """Encode each slice with a 2D backbone, then mean-pool to eye level."""

    def __init__(self, num_classes: int, use_imagenet_pretrain: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if use_imagenet_pretrain else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor, slice_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_slices, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_slices, channels, height, width)
        slice_features = self.backbone(flat_images)
        slice_features = slice_features.view(batch_size, num_slices, -1)

        feature_mask = slice_mask.unsqueeze(-1).to(slice_features.dtype)
        masked_features = slice_features * feature_mask
        valid_counts = feature_mask.sum(dim=1).clamp_min(1.0)
        eye_features = masked_features.sum(dim=1) / valid_counts

        return self.classifier(eye_features)
