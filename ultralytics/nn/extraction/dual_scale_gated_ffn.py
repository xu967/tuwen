"""
DualScaleGatedFFN - Dual-scale Gated Feed-Forward Network (ECCV 2024)

论文题目: Restoring Images in Adverse Weather Conditions via Histogram Transformer
论文来源: ECCV 2024
论文链接: https://ieeexplore.ieee.org/abstract/document/10445289/
代码来源: /root/modules/027 Dual-scale Gated Feed-Forward Network.py
功能说明: 双尺度门控前馈网络（Histoformer）。1x1 升维后 PixelShuffle 上采样，
    两路分别经 5x5 DWConv 与 dilation=2 的 3x3 DWConv，Mish 门控相乘后再
    PixelUnshuffle 还原空间尺寸并 1x1 投影回 dim。奇数 H/W 会 pad 后 crop。
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DualScaleGatedFFN(nn.Module):
    """Dual-scale Gated Feed-Forward Network (Histoformer, ECCV 2024).

    PixelShuffle 双尺度深度卷积 + Mish 门控，投影回原通道，空间尺寸不变。

    Args:
        dim (int): 输入/输出通道数。
        ffn_expansion_factor (float): 隐藏维度扩展因子，默认 2.0。
        bias (bool): 卷积是否使用偏置，默认 True。

    Note:
        本模块不改变通道数与空间尺寸。奇数 H/W 会 pad 到偶数再 crop 回原尺寸。
    """

    def __init__(self, dim, ffn_expansion_factor=2.0, bias=True):
        """Initialize DualScaleGatedFFN with dim, expansion factor and bias."""
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.p_shuffle = nn.PixelShuffle(2)
        self.dwconv_5 = nn.Conv2d(
            hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1,
            padding=2, groups=hidden_features // 4, bias=bias,
        )
        self.dwconv_dilated2_1 = nn.Conv2d(
            hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1,
            padding=2, groups=hidden_features // 4, bias=bias, dilation=2,
        )
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward: 1x1 -> PixelShuffle -> dual DWConv Mish-gate -> PixelUnshuffle -> 1x1."""
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x


__all__ = ['DualScaleGatedFFN']
