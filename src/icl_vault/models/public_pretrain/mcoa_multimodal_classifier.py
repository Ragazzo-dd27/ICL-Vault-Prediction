"""Minimal multimodal MCOA classifier with OCT and ASP branches."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def _build_resnet18_encoder(use_imagenet_pretrain: bool) -> tuple[nn.Module, int]:
    weights = ResNet18_Weights.DEFAULT if use_imagenet_pretrain else None
    backbone = models.resnet18(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class MCOAMultimodalClassifier(nn.Module):
    """Encode OCT slices and ASP image separately, then fuse with concatenation."""

    def __init__(
        self,
        num_classes: int,
        use_imagenet_pretrain: bool = False,
        mode: str = "oct_asp",
    ) -> None:
        super().__init__()
        if mode not in {"oct_only", "asp_only", "oct_asp"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.oct_backbone, oct_feature_dim = _build_resnet18_encoder(use_imagenet_pretrain)
        self.asp_backbone, asp_feature_dim = _build_resnet18_encoder(use_imagenet_pretrain)
        self.oct_feature_dim = oct_feature_dim
        self.asp_feature_dim = asp_feature_dim
        fusion_dim = oct_feature_dim + asp_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        oct_images: torch.Tensor,
        asp_images: torch.Tensor,
        oct_slice_mask: torch.Tensor,
        has_oct: torch.Tensor,
        has_asp: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_slices, channels, height, width = oct_images.shape

        if self.mode == "asp_only":
            oct_eye_features = torch.zeros(
                batch_size,
                self.oct_feature_dim,
                dtype=asp_images.dtype,
                device=asp_images.device,
            )
        else:
            flat_oct_images = oct_images.view(batch_size * num_slices, channels, height, width)
            oct_slice_features = self.oct_backbone(flat_oct_images).view(batch_size, num_slices, -1)
            oct_mask = oct_slice_mask.unsqueeze(-1).to(oct_slice_features.dtype)
            masked_oct_features = oct_slice_features * oct_mask
            valid_oct_counts = oct_mask.sum(dim=1).clamp_min(1.0)
            oct_eye_features = masked_oct_features.sum(dim=1) / valid_oct_counts
            oct_eye_features = oct_eye_features * has_oct.unsqueeze(-1).to(oct_eye_features.dtype)

        if self.mode == "oct_only":
            asp_features = torch.zeros(
                batch_size,
                self.asp_feature_dim,
                dtype=oct_images.dtype,
                device=oct_images.device,
            )
        else:
            asp_features = self.asp_backbone(asp_images)
            asp_features = asp_features * has_asp.unsqueeze(-1).to(asp_features.dtype)

        fused_features = torch.cat([oct_eye_features, asp_features], dim=1)
        return self.classifier(fused_features)
