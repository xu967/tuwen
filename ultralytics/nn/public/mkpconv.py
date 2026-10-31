"""Multi-Kernel Perception Unit
论文: FBRT-YOLO: Faster and Better for Real-Time Aerial Image Detection (https://arxiv.org/pdf/2504.20670)
会议: AAAI 2025
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ['ConvolutionLayer', 'MKPConv']


def calculate_padding(kernel_size, padding=None, dilation=1):
    """计算填充大小，使卷积输出保持相同尺寸"""
    if dilation > 1:
        # 计算膨胀后的实际核大小
        if isinstance(kernel_size, int):
            effective_kernel = dilation * (kernel_size - 1) + 1
        else:
            effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    else:
        effective_kernel = kernel_size

    # 自动计算填充大小，使输入输出尺寸相同
    if padding is None:
        if isinstance(effective_kernel, int):
            padding = effective_kernel // 2
        else:
            padding = [k // 2 for k in effective_kernel]
    return padding


class ConvolutionLayer(nn.Module):
    """标准卷积层，包含卷积、批归一化和激活函数"""
    default_activation = nn.SiLU()  # 默认激活函数

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
        """初始化卷积层"""
        super().__init__()
        # 创建卷积层，自动计算填充
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            calculate_padding(kernel_size, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # 设置激活函数
        if activation is True:
            self.activation = self.default_activation
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """前向传播：卷积 -> 批归一化 -> 激活"""
        return self.activation(self.batch_norm(self.convolution(x)))

    def forward_fused(self, x):
        """前向传播（融合版本）：卷积 -> 激活"""
        return self.activation(self.convolution(x))


class MKPConv(nn.Module):
    """多核感知卷积单元 (Multi-Kernel Perception Unit).

    通过级联 3x3 / 5x5 / 7x7 深度可分离卷积分支 + 逐点卷积，
    实现跨尺度特征的融合与交互，提升多尺度目标感知能力。
    """

    def __init__(self, channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation=True):
        """初始化多尺度并行卷积模块"""
        super().__init__()

        # 3x3深度可分离卷积分支
        self.depthwise_conv_3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.pointwise_conv_3x3 = ConvolutionLayer(channels, channels, kernel_size=1, stride=1)

        # 5x5深度可分离卷积分支
        self.depthwise_conv_5x5 = nn.Conv2d(
            channels, channels, kernel_size=5, stride=1, padding=2, groups=channels
        )
        self.pointwise_conv_5x5 = ConvolutionLayer(channels, channels, kernel_size=1, stride=1)

        # 7x7深度可分离卷积分支
        self.depthwise_conv_7x7 = nn.Conv2d(
            channels, channels, kernel_size=7, stride=1, padding=3, groups=channels
        )

        self.pointwise_conv_7x7 = ConvolutionLayer(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        """前向传播：多尺度卷积后残差连接"""
        x = self.depthwise_conv_3x3(x)
        x = self.pointwise_conv_3x3(x)
        x = self.depthwise_conv_5x5(x)
        x = self.pointwise_conv_5x5(x)
        x = self.depthwise_conv_7x7(x)
        x = self.pointwise_conv_7x7(x)
        return x


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
