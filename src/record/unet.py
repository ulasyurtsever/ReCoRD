"""Compact U-Net for multispectral patch segmentation (MARIDA)."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Four-level U-Net with bilinear upsampling.

    Parameters
    ----------
    in_channels : int
        Number of input bands.
    n_classes : int
        Number of output classes.
    base_width : int
        Channel width of the first encoder level.
    """

    def __init__(self, in_channels: int, n_classes: int, base_width: int = 32):
        super().__init__()
        w = base_width
        self.enc1 = _ConvBlock(in_channels, w)
        self.enc2 = _ConvBlock(w, w * 2)
        self.enc3 = _ConvBlock(w * 2, w * 4)
        self.enc4 = _ConvBlock(w * 4, w * 8)
        self.bottleneck = _ConvBlock(w * 8, w * 16)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = _ConvBlock(w * 16 + w * 8, w * 8)
        self.dec3 = _ConvBlock(w * 8 + w * 4, w * 4)
        self.dec2 = _ConvBlock(w * 4 + w * 2, w * 2)
        self.dec1 = _ConvBlock(w * 2 + w, w)
        self.head = nn.Conv2d(w, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.head(d1)
