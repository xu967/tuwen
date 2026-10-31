"""
GSAU - Gated Spatial Attention Unit (CVPR 2024)

论文题目: Multi-scale Attention Network for Single Image Super-Resolution
论文来源: CVPR 2024
论文链接: https://arxiv.org/abs/2209.14145
代码来源: /root/modules/042 Gated Spatial Attention Unit（CVPR 2024）.py
功能说明: 门控空间注意力单元（与 MLKA 同属 MAN 系列）。LayerNorm 后 1x1 升维并
    沿通道一分为二，一路经 7x7 深度卷积作为门控与另一路相乘，再 1x1 降维，
    以可学习 scale 缩放后残差相加。输入输出 C,H,W 一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerNorm(nn.Module):
    """支持 channels_first / channels_last 的 LayerNorm，仅供 GSAU 内部使用。"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GSAU(nn.Module):
    """Gated Spatial Attention Unit (MAN, CVPR 2024).

    归一化后通道对半切分，DWConv 门控与残差缩放保持输入输出通道一致。

    Args:
        dim (int): 输入/输出通道数（残差结构，通道不变）。

    Note:
        本模块不改变通道数，dim 同时为输入和输出通道。
    """

    def __init__(self, dim):
        """Initialize GSAU with dim."""
        super().__init__()
        i_feats = dim * 2

        self.norm = _LayerNorm(dim, data_format='channels_first')
        self.Conv1 = nn.Conv2d(dim, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.Conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.scale = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        """Forward: LN -> 1x1 split -> DWConv gate -> 1x1 -> scaled residual."""
        shortcut = x.clone()
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        return x * self.scale + shortcut


__all__ = ['GSAU']
