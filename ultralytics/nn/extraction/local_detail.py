"""
LDE - Local Detail Estimation (ECCV 2024)

论文题目: SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for
         Efficient Image Super-Resolution
论文来源: ECCV 2024
论文链接: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
代码来源: /home/zhizi/work/multimodel/ultralyticmm/ref/newmodule/014 Local Detail Estimation（ECCV 2024）.py
功能说明: 局部细节估计 (Local Detail Estimation, LDE) 分支。
    为了获得高频成分，首先使用一个 3x3 深度可分离卷积从输入特征中编码局部信息，
    再使用两个带 GELU 激活的 1x1 卷积（点卷积）生成增强的局部特征。
    LDE 分支能有效捕捉图像中的局部细节，输入输出通道数一致，可作为即插即用 block。
"""

from __future__ import annotations

import torch.nn as nn


class LDE(nn.Module):
    """Local Detail Estimation (LDE) block (SMFANet, ECCV 2024).

    通过 3x3 深度可分离卷积编码局部高频信息，配合 GELU 激活与 1x1 点卷积生成
    增强的局部细节特征，输入输出通道数一致。

    Args:
        dim (int): 输入/输出通道数（通道不变）。
        growth_rate (float): 隐藏维度扩张系数，hidden_dim = int(dim * growth_rate)。

    Note:
        本模块不改变通道数，dim 同时为输入和输出通道。
    """

    def __init__(self, dim, growth_rate=2.0):
        """Initialize LDE with dim and growth_rate."""
        super().__init__()
        hidden_dim = int(dim * growth_rate)  # 隐藏层维度，基于输入维度和增长因子
        self.conv_0 = nn.Sequential(  # 第一卷积序列：深度可分离卷积 + 点卷积
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度可分离卷积，分组卷积减少参数
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),  # 点卷积，通道融合
        )
        self.act = nn.GELU()  # GELU 激活函数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 最终点卷积，还原到输入维度

    def forward(self, x):
        """Forward pass: conv_0 -> act -> conv_1."""
        x = self.conv_0(x)  # 第一卷积序列
        x = self.act(x)  # GELU 激活
        x = self.conv_1(x)  # 还原到 dim
        return x


__all__ = ['LDE']
