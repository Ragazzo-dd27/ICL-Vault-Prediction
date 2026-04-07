import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    两次3x3卷积（每次后接BN和ReLU）模块，常用于U-Net中的编码器和解码器阶段
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightweightUNet(nn.Module):
    """
    轻量级U-Net模型：
    - 输入通道: 1（OCT灰度图）
    - 输出通道: 1（二值分割掩码）
    - 每层特征通道数减半，显著降低计算量
    - 使用nn.Bilinear进行上采样
    """

    def __init__(self):
        super(LightweightUNet, self).__init__()
        # -------- Encoder（下采样路径） --------
        self.enc1 = DoubleConv(1, 32)          # 第一层，通道1->32
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = DoubleConv(32, 64)         # 通道32->64
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = DoubleConv(64, 128)        # 通道64->128
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc4 = DoubleConv(128, 256)       # 通道128->256

        # -------- Decoder（上采样路径） --------
        # 上采样采用nn.Upsample，模式为'bilinear'
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(256 + 128, 128)   # 拼接后输入通道256+128

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(128 + 64, 64)     # 拼接后输入通道128+64

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(64 + 32, 32)      # 拼接后输入通道64+32

        # 输出层（1x1卷积，将通道数变为1，用于二值分割）
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(x)              # e1: (B, 32, H, W)
        p1 = self.pool1(e1)            # p1: (B, 32, H/2, W/2)

        e2 = self.enc2(p1)             # e2: (B, 64, H/2, W/2)
        p2 = self.pool2(e2)            # p2: (B, 64, H/4, W/4)

        e3 = self.enc3(p2)             # e3: (B, 128, H/4, W/4)
        p3 = self.pool3(e3)            # p3: (B, 128, H/8, W/8)

        e4 = self.enc4(p3)             # e4: (B, 256, H/8, W/8)

        # -------- Decoder --------
        u3 = self.up3(e4)              # u3: (B, 256, H/4, W/4)
        # 与encoder对齐，拼接跳跃连接
        u3 = torch.cat([u3, e3], dim=1)   # (B, 256+128, H/4, W/4)
        d3 = self.dec3(u3)                # (B, 128, H/4, W/4)

        u2 = self.up2(d3)                 # (B, 128, H/2, W/2)
        u2 = torch.cat([u2, e2], dim=1)   # (B, 128+64, H/2, W/2)
        d2 = self.dec2(u2)                # (B, 64, H/2, W/2)

        u1 = self.up1(d2)                 # (B, 64, H, W)
        u1 = torch.cat([u1, e1], dim=1)   # (B, 64+32, H, W)
        d1 = self.dec1(u1)                # (B, 32, H, W)

        # 输出预测掩码
        out = self.out_conv(d1)           # (B, 1, H, W)
        return out