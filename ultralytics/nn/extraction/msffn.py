"""
MSFFN: Multi-Scale Feed-Forward Network

论文: Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising
来源: IEEE Geoscience and Remote Sensing Letters (GRSL) 2024, Vol. 21
链接: https://doi.org/10.1109/LGRS.2024.3370299

功能:
    多尺度前馈网络模块 (Multi-Scale Feed-Forward Network)，作为 Transformer 中
    FFN 的替代模块。对 CAFM 的输出特征进行处理，聚合多尺度特征并增强非线性表达能力。
    原理:
        1. 通过元素乘法 Gate 机制增强非线性变换 (gelu(x1) * x2 * x3)。
        2. 多分支多尺度空洞卷积 (dilation=1/2/3) 用于多尺度特征提取，结合 3D 与 2D 深度可分离卷积。
    主类 Multi_Scale_Feed_Forward_Network 为通道不变模块 (forward 输入输出同通道数)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Multi_Scale_Feed_Forward_Network']


class Multi_Scale_Feed_Forward_Network(nn.Module):
    """多尺度前馈网络 (通道不变)。

    通过 project_in 将输入扩展到 3 倍隐藏维度，再切分成三路分支:
    分支一使用 3D 深度可分离卷积 (dilation=1)；分支二、三使用 2D 深度可分离卷积
    (dilation=2/3)。最后通过元素乘法门控融合，并由 project_out 还原回原始维度。

    Args:
        dim (int): 输入/输出通道数。
        ffn_expansion_factor (int | float): 隐藏层维度扩张因子，hidden_features = int(dim * factor)。
        bias (bool): 各卷积层是否使用偏置，默认 False。
    """

    def __init__(self, dim, ffn_expansion_factor=4, bias=False):
        super(Multi_Scale_Feed_Forward_Network, self).__init__()

        # 隐藏层特征数量，基于输入维度和扩张因子
        hidden_features = int(dim * ffn_expansion_factor)

        # 3D 卷积：将输入维度映射到隐藏层维度的 3 倍 (用于后续三路切分)
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)

        # 分支一：3D 深度可分离卷积，dilation=1，padding=1，输出尺寸不变
        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3), stride=1, dilation=1,
                                 padding=1, groups=hidden_features, bias=bias)

        # 分支二：2D 深度可分离卷积，dilation=2，padding=2
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, dilation=2, padding=2,
                                 groups=hidden_features, bias=bias)

        # 分支三：2D 深度可分离卷积，dilation=3，padding=3
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, dilation=3, padding=3,
                                 groups=hidden_features, bias=bias)

        # 3D 卷积：将隐藏层维度映射回原始输入维度
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)

    def forward(self, x):
        # 在第二维度上增加一个维度，使 4D (B,C,H,W) → 5D (B,C,1,H,W) 以适配 3D 卷积
        x = x.unsqueeze(2)

        # project_in 扩展并切分为三路分支
        x = self.project_in(x)
        x1, x2, x3 = x.chunk(3, dim=1)

        # 分支一：3D 深度可分离卷积后移除新增维度
        x1 = self.dwconv1(x1).squeeze(2)

        # 分支二、三：先移除维度再做 2D 深度可分离卷积 (空洞率不同)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))

        # Gate 门控融合：gelu(x1) * x2 * x3
        x = F.gelu(x1) * x2 * x3

        # 为输出 3D 卷积补回维度，再移除
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2)
        return x
