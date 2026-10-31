"""
MDCR - Multi-Dilated Channel Refiner (arxiv 2024)

论文题目: HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection
论文来源: arxiv 2024
论文链接: https://arxiv.org/abs/2403.10778
代码来源: /root/modules/073 Multi-Dilated Channel Refiner（Arxix 2024）.py
功能说明: 多膨胀率通道精炼。通道四等分后分别以 dilation=1/6/12/18 的深度卷积
    提取多感受野特征，再按通道堆叠经 1x1 (4→4) 混合并 1x1 投影。默认 out_features=dim，
    保持输入输出 C,H,W 一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv + 可选 Norm + 可选 SiLU，仅供 MDCR 内部使用。"""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        norm_type='bn',
        activation=True,
        use_bias=True,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=use_bias,
            groups=groups,
        )
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.act_layer = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.act_layer(x)
        return x


class MDCR(nn.Module):
    """Multi-Dilated Channel Refiner (HCF-Net, arxiv 2024).

    四路膨胀深度卷积 + 向量化通道混合，默认通道不变。

    Args:
        dim (int): 输入通道数（须能被 4 整除）。
        out_features (int | None): 输出通道数，默认等于 dim。
        norm_type (str): 归一化类型，'bn' 或 'gn'。
        activation (bool): 是否使用 SiLU 激活。
        rate (tuple): 四路膨胀率，默认 (1, 6, 12, 18)。

    Note:
        默认 out_features=dim，输入输出 C,H,W 一致。
    """

    def __init__(self, dim, out_features=None, norm_type='bn', activation=True, rate=(1, 6, 12, 18)):
        """Initialize MDCR with dim and optional out_features."""
        super().__init__()
        out_features = dim if out_features is None else out_features
        assert dim % 4 == 0, f"MDCR: dim ({dim}) must be divisible by 4"
        assert out_features % 4 == 0, f"MDCR: out_features ({out_features}) must be divisible by 4"

        in_q, out_q = dim // 4, out_features // 4
        self.block1 = _ConvBlock(
            in_features=in_q, out_features=out_q, padding=rate[0], dilation=rate[0],
            norm_type=norm_type, activation=activation, groups=in_q,
        )
        self.block2 = _ConvBlock(
            in_features=in_q, out_features=out_q, padding=rate[1], dilation=rate[1],
            norm_type=norm_type, activation=activation, groups=in_q,
        )
        self.block3 = _ConvBlock(
            in_features=in_q, out_features=out_q, padding=rate[2], dilation=rate[2],
            norm_type=norm_type, activation=activation, groups=in_q,
        )
        self.block4 = _ConvBlock(
            in_features=in_q, out_features=out_q, padding=rate[3], dilation=rate[3],
            norm_type=norm_type, activation=activation, groups=in_q,
        )
        self.out_s = _ConvBlock(
            in_features=4, out_features=4, kernel_size=(1, 1), padding=(0, 0),
            norm_type=norm_type, activation=activation,
        )
        self.out = _ConvBlock(
            in_features=out_features, out_features=out_features, kernel_size=(1, 1),
            padding=(0, 0), norm_type=norm_type, activation=activation,
        )

    def forward(self, x):
        """Forward: 4 dilated DWConv -> vectorized 4-ch mix -> 1x1 project."""
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x3 = self.block3(x3)
        x4 = self.block4(x4)

        stacked = torch.stack([x1, x2, x3, x4], dim=2)  # [B, Cq, 4, H, W]
        b, cq, _, h, w = stacked.shape
        y = stacked.reshape(b * cq, 4, h, w)
        y = self.out_s(y)
        y = y.reshape(b, cq, 4, h, w).reshape(b, cq * 4, h, w)
        return self.out(y)


__all__ = ['MDCR']
