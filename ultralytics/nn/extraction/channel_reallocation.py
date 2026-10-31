"""
Channel Reallocation FFN (ChannelAggregationFFN)
论文：MogaNet — Multi-Order Gated Aggregation Network (ICLR 2024)
论文地址：https://openreview.net/pdf?id=XhYWgjqCrV
讲解视频：https://www.bilibili.com/video/BV1VRtLe8EAg
功能：带通道聚合与特征分解的前馈网络（FFN）。
      - 升维 -> 深度卷积 -> 激活 -> 特征分解（feat_decompose）-> 降维
      - feat_decompose：将多通道压缩到单通道后做 GELU，再用可学习缩放 sigma
        重新注入到主路，强化主成分并抑制冗余分量，实现通道再分配。
      - 输入输出通道数保持不变（不改通道），仅内部经过 feedforward_channels 升维。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ElementScale(nn.Module):
    """可学习的逐元素缩放器。

    通过一个形状为 (1, C, 1, 1) 的可学习参数对输入张量逐通道加权，
    用于残差/分解路径上的强度调节。
    """

    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad,
        )

    def forward(self, x):
        return x * self.scale


class ChannelReallocationFFN(nn.Module):
    """带通道聚合与特征分解的前馈网络（不改通道）。

    输入输出通道数均为 `dim`，内部经 `ffn_ratio` 倍升维到 feedforward_channels
    后再降回 `dim`。

    Args:
        dim (int): 输入/输出特征维度。
        ffn_ratio (float): 隐藏层维度倍率，feedforward_channels = int(dim * ffn_ratio)。
        kernel_size (int): 深度卷积核大小。默认 3。
        ffn_drop (float): FFN 内 Dropout 概率。默认 0.0。
    """

    def __init__(self, dim, ffn_ratio=4.0, kernel_size=3, ffn_drop=0.0):
        super().__init__()

        self.embed_dims = dim
        feedforward_channels = int(dim * ffn_ratio)
        self.feedforward_channels = feedforward_channels

        # 升维投影：dim -> feedforward_channels
        self.fc1 = nn.Conv2d(dim, feedforward_channels, kernel_size=1)

        # 深度卷积：在升维后的特征上做空间聚合
        self.dwconv = nn.Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=feedforward_channels,
        )

        self.act = nn.GELU()

        # 降维投影：feedforward_channels -> dim
        self.fc2 = nn.Conv2d(feedforward_channels, dim, kernel_size=1)

        self.drop = nn.Dropout(ffn_drop)

        # 分解卷积：将 C 通道压缩到 1 通道，用于提取全局/主成分表示
        self.decompose = nn.Conv2d(feedforward_channels, 1, kernel_size=1)

        # 分解路径的可学习逐通道缩放
        self.sigma = ElementScale(feedforward_channels, init_value=1e-5, requires_grad=True)

        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # [B, C, H, W] -> [B, 1, H, W]：提取全局/主成分单通道表示
        temp = self.decompose(x)
        temp = self.decompose_act(temp)
        # 用原始特征减去主成分，保留细节/差异信息
        temp = x - temp
        # 可学习缩放后重新注入主路
        temp = self.sigma(temp)
        x = x + temp
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


__all__ = ["ChannelReallocationFFN"]
