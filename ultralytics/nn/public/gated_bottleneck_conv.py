"""GatedBottleneckConv - 门控瓶颈卷积
论文: SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation in Structures
会议: CVPR 2025
链接: https://arxiv.org/pdf/2503.01113

门控瓶颈卷积 (Gated Bottleneck Convolution, GBC):
1. 瓶颈卷积 (BottConv) 通过低秩近似降低计算复杂度
2. 门控机制动态调整权重，精细化主分支特征
3. 残差连接保证训练稳定性
"""

import torch
from torch import nn as nn

__all__ = ['GatedBottleneckConv', 'BottConv']


class BottConv(nn.Module):
    """瓶颈卷积: 1x1 → depthwise k×k → 1x1，低秩近似降复杂度。"""

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        mid_channels = max(mid_channels, 2)
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class GatedBottleneckConv(nn.Module):
    """门控瓶颈卷积: 双分支瓶颈 + 门控相乘 + 残差, dim-based 通道不变。"""

    def __init__(self, in_channels, norm_type='GN') -> None:
        super().__init__()
        # 左侧主分支
        self.proj = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)
        self.norm = self._make_norm(in_channels, norm_type)
        self.nonliner = nn.ReLU()
        self.proj2 = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)
        self.norm2 = self._make_norm(in_channels, norm_type)
        self.nonliner2 = nn.ReLU()
        # 右下门控分支
        self.proj3 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)
        self.norm3 = self._make_norm(in_channels, norm_type)
        self.nonliner3 = nn.ReLU()
        # 右上融合分支
        self.proj4 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)
        self.norm4 = self._make_norm(in_channels, norm_type)
        self.nonliner4 = nn.ReLU()

    @staticmethod
    def _make_norm(in_channels, norm_type):
        if norm_type == 'GN':
            return nn.GroupNorm(max(in_channels // 16, 1), in_channels)
        return nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        x_residual = x
        # 左侧主分支
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)
        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)
        # 右下门控分支
        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)
        # 门控相乘
        x = x1 * x2
        # 右上融合
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        return x + x_residual


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
