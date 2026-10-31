"""
SAFM - Spatially-Adaptive Feature Modulation (ICCV 2023)

论文题目: Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution
论文来源: ICCV 2023
论文链接: https://arxiv.org/abs/2302.13800
代码来源: /root/modules/086 087 Spatially-Adaptive Feature Modue（ICCV 2023）.py
功能说明: 空间自适应特征调制。将通道均分为 n_levels 个分块，除第 0 块外对其余
    分块做自适应最大池化降采样，经深度卷积空间加权后再插值回原分辨率，1x1 聚合后
    以 GELU 门控与输入逐元素相乘。残差调制保持输入输出 C,H,W 一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAFM(nn.Module):
    """Spatially-Adaptive Feature Modulation (ICCV 2023).

    多尺度通道分块 + 深度卷积空间加权 + GELU 门控调制，通道与空间尺寸不变。

    Args:
        dim (int): 输入/输出通道数（须能被 n_levels 整除）。
        n_levels (int): 通道分块层数，默认 4。

    Note:
        本模块不改变通道数与空间尺寸，dim 同时为输入和输出通道。
    """

    def __init__(self, dim, n_levels=4):
        """Initialize SAFM with dim and n_levels."""
        super().__init__()
        assert dim % n_levels == 0, f"SAFM: dim ({dim}) must be divisible by n_levels ({n_levels})"
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # 每个分块独立的 3x3 深度卷积，做空间加权
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(self.n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        """Forward: chunk -> multi-scale DWConv -> aggregate -> GELU gate * x."""
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)

        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (max(1, h // 2 ** i), max(1, w // 2 ** i))
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


__all__ = ['SAFM']
