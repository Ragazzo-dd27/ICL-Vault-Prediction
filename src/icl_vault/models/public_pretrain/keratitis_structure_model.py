"""Minimal U-Net style structure model for keratitis OCT pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class KeratitisStructureModel(nn.Module):
    """Small U-Net style binary segmentation model with an explicit encoder path."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder_stem = DoubleConv(in_channels, base_channels)
        self.encoder_down1 = DownBlock(base_channels, base_channels * 2)
        self.encoder_down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8)

        self.decoder_up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder_up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder_up0 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.segmentation_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.encoder_stem(x)
        skip1 = self.encoder_down1(skip0)
        skip2 = self.encoder_down2(skip1)
        bottleneck = self.bottleneck(skip2)

        x = self.decoder_up2(bottleneck, skip2)
        x = self.decoder_up1(x, skip1)
        x = self.decoder_up0(x, skip0)
        return self.segmentation_head(x)
