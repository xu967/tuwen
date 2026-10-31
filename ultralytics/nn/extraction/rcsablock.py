"""
RCSABlock - Residual Channel–Spatial Attention Block (2025 一区TOP)

论文题目: DRPCA-Net: Make Robust PCA Great Again for Infrared Small Target Detection
论文来源: 2025 一区TOP
论文链接: https://arxiv.org/abs/2501.03775
代码来源: /root/modules/198 RCSABlock（2025 一区TOP）.py
功能说明: 残差通道–空间注意力块。Conv-BN-SiLU → Conv-BN 后依次施加双池化
    通道注意力与动态卷积核空间注意力，再与输入残差相加。输入输出 C,H,W 一致。
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class _DualPoolChannelAttention(nn.Module):
    """AvgPool + MaxPool → MLP → 通道权重，仅供 RCSABlock 内部使用。"""

    def __init__(self, channels=32, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weights = self.activate(avg_out + max_out)
        return x * weights


class _DynamicKernelSpatialAttention(nn.Module):
    """为每个样本生成 k×k 卷积核，对空间位置加权，仅供内部使用。"""

    def __init__(self, channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, kernel_size * kernel_size, 1),
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        b, _, h, w = x.shape
        kernels = self.kernel_gen(x).view(b, 1, self.kernel_size, self.kernel_size)
        x_mean = x.mean(dim=1, keepdim=True).view(1, b, h, w)
        att = F.conv2d(x_mean, weight=kernels, padding=self.kernel_size // 2, groups=b)
        att = self.activate(att.view(b, 1, h, w))
        return x * att


class RCSABlock(nn.Module):
    """Residual Channel–Spatial Attention Block (DRPCA-Net).

    Conv-BN-SiLU → Conv-BN → 通道注意力 → 动态核空间注意力 → 残差。

    Args:
        dim (int): 输入/输出通道数（残差结构，通道不变）。
        use_bn (bool): 卷积后是否使用 BatchNorm，默认 True。

    Note:
        本模块不改变通道数，dim 同时为输入和输出通道。
    """

    def __init__(self, dim, use_bn=True):
        """Initialize RCSABlock with dim and use_bn."""
        super().__init__()
        layers = []
        for i in range(2):
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False))
            if use_bn:
                layers.append(nn.BatchNorm2d(dim))
            if i == 0:
                layers.append(nn.SiLU())
        layers.append(_DualPoolChannelAttention(channels=dim))
        layers.append(_DynamicKernelSpatialAttention(channels=dim))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        """Forward: conv body + dual attention + residual."""
        return x + self.body(x)


__all__ = ['RCSABlock']
