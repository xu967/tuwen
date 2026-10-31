"""
LKAT - Large Kernel Attention Tail (CVPR 2024)

论文题目: Multi-scale Attention Network for Single Image Super-Resolution
论文来源: CVPR 2024
论文链接: https://arxiv.org/abs/2209.14145
代码来源: /root/modules/043 Large Kernel Attention Tail（CVPR 2024）.py
功能说明: 大核注意力尾部模块（MAN）。1x1+GELU 后，以 7x7 DWConv 与 9x9
    dilation=3 DWConv 再 1x1 生成空间注意力图，与特征逐元素相乘后 1x1 投影。
    通道与空间尺寸保持不变。
"""

from __future__ import annotations

import torch.nn as nn


class LKAT(nn.Module):
    """Large Kernel Attention Tail (MAN, CVPR 2024).

    大核深度卷积堆叠生成注意力图，与主干特征相乘后 1x1 投影，通道不变。

    Args:
        dim (int): 输入/输出通道数（通道不变）。

    Note:
        本模块不改变通道数，dim 同时为输入和输出通道。
    """

    def __init__(self, dim):
        """Initialize LKAT with dim."""
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
        )
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim),
            nn.Conv2d(dim, dim, 9, stride=1, padding=(9 // 2) * 3, groups=dim, dilation=3),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        """Forward: 1x1+GELU -> large-kernel attention multiply -> 1x1."""
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


__all__ = ['LKAT']
