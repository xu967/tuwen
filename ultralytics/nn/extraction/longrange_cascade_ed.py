"""
Long-Range Cascade Encoder-Decoder (LR-CED) 模块
longrange_cascade_ed

论文：CEDNet: A Cascade Encoder-Decoder Network for Dense Prediction (ICLR 2024)
论文地址：https://arxiv.org/abs/2302.06052
来源：ref/newmodule/017 Long-Range Cascade Encoder-Decoder(ICLR 2024).py

功能：长依赖级联编码器-解码器块（LR-CED）。在 CED 基础上引入 7x7 扩张深度卷积，
并通过两个跳跃连接级联两次深度可分离卷积，捕获空间特征间的长距离依赖关系、
扩大感受野，而参数与计算开销仅略有增加。随后用两层 MLP（逐点线性层）做通道交互，
并以可学习缩放 gamma 与 DropPath 残差收尾。通道数保持不变（单输入）。
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

# 激活函数与 layer-scale 初始化值
act_layer = nn.ReLU
ls_init_value = 1e-6


class LRCED(nn.Module):
    """Long-Range Cascade Encoder-Decoder 块。

    参数：
        dim (int): 输入/输出通道数（不改变通道）。
        drop_path (float): DropPath 丢弃概率，默认 0.。
        dilation (int): 第二个 7x7 深度卷积的扩张率，默认 3。
    输入：x (Tensor)，形状 (N, dim, H, W)。
    输出：与输入同形状的 Tensor。
    """

    def __init__(self, dim, drop_path=0., dilation=3, **kwargs):
        super().__init__()

        # 第一个深度可分离卷积序列：7x7 标准深度卷积 + BN + 激活
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),
            nn.BatchNorm2d(dim),
            act_layer(),
        )

        # 第二个深度可分离卷积序列：7x7 扩张深度卷积 + BN + 激活
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            nn.BatchNorm2d(dim),
            act_layer(),
        )

        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 升维逐点卷积
        self.act = act_layer()  # 激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 降维逐点卷积

        # 可学习的通道缩放参数
        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        # DropPath 随机深度
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # 保存输入用于残差连接

        # 第一级深度卷积 + 残差跳跃
        x = self.dwconv1(x) + x
        # 第二级扩张深度卷积 + 残差跳跃（捕获长距离依赖）
        x = self.dwconv2(x) + x

        # 通道 MLP：全连接层作用于最后一维，故先转置通道到最后
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x  # 通道缩放
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # 残差连接 + DropPath
        x = input + self.drop_path(x)
        return x


__all__ = ['LRCED']
