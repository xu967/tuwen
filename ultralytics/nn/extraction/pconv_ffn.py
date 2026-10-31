"""
模块名: PCFNBlock (Partial Convolution-based Feed-forward Network Block)
论文: SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for
      Efficient Image Super-Resolution (ECCV 2024)
来源: ref/newmodule/015 Partial convolution-based feed-forward network (ECCV 2024)
链接: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
功能: 高效的基于部分卷积的前馈网络 (PCFN)。先用 1x1 卷积 + GELU 在扩展的
      隐藏空间内做跨通道交互，再沿通道切分为 {F1, F2} 两块，对 F1 用 3x3
      卷积 + GELU 编码局部上下文，最后拼接并经 1x1 卷积还原回原始维度。
      输入输出通道保持一致 (N, dim, H, W) -> (N, dim, H, W)。
"""

import torch
import torch.nn as nn

__all__ = ["PCFNBlock"]


class PCFNBlock(nn.Module):
    """Partial Convolution-based Feed-forward Network (PCFN).

    不改通道: 输入 (N, dim, H, W) -> 输出 (N, dim, H, W)。

    Args:
        dim (int): 输入输出通道数。
        growth_rate (float): 隐藏维度扩展倍数，hidden_dim = int(dim * growth_rate)。
        p_rate (float): 部分卷积分支通道比例，p_dim = int(hidden_dim * p_rate)。
    """

    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()

        # 扩展的隐藏维度
        hidden_dim = int(dim * growth_rate)
        # 部分卷积处理的通道数
        p_dim = int(hidden_dim * p_rate)

        # 1x1 卷积: dim -> hidden_dim，做跨通道交互
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        # 3x3 卷积: 仅作用于 p_dim 通道，编码局部上下文
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)
        # GELU 激活
        self.act = nn.GELU()
        # 1x1 卷积: hidden_dim -> dim，混合并还原通道
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # 1x1 扩展 + GELU 跨通道交互
        x = self.act(self.conv_0(x))

        # 沿通道切分为部分分支 F1 与直充分支 F2
        x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)

        # 对 F1 做 3x3 局部上下文编码
        x1 = self.act(self.conv_1(x1))

        # 拼接 F1、F2 并还原通道
        x = self.conv_2(torch.cat([x1, x2], dim=1))

        return x
