"""
CED: Cascade Encoder-Decoder Block

论文: CEDNet: A Cascade Encoder-Decoder Network for Dense Prediction
会议: ICLR 2024
论文链接: https://arxiv.org/abs/2302.06052

来源: ref/newmodule/016 Cascade Encoder-Decoder(ICLR 2024).py

功能:
    级联编码器-解码器模块 (Cascade Encoder-Decoder, CED)。
    该块包括用于空间特征交互的 token 混合器 (默认轻量级 7x7 深度卷积)
    和用于通道特征交互的两层多层感知器 (MLP)，并配合层缩放 (Layer Scale)
    与随机深度 (DropPath) 进行残差连接。
    主类 CED 为通道不变模块 (forward 输入输出同通道数)。
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

__all__ = ['CED']

_LAYER_SCALE_INIT_VALUE = 1e-6


class CED(nn.Module):
    """级联编码器-解码器模块 (通道不变)。

    Args:
        dim (int): 输入/输出通道数。
        drop_path (float): 随机深度丢弃概率，默认 0.。
        act_layer (nn.Module): 激活函数类型，默认 nn.ReLU。
        ls_init_value (float): 层缩放 (Layer Scale) 初始值，<= 0 表示禁用，默认 1e-6。
    """

    def __init__(self, dim, drop_path=0., act_layer=nn.ReLU, ls_init_value=_LAYER_SCALE_INIT_VALUE, **kwargs):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim,)), requires_grad=True) if ls_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)

        # 全连接层作用于最后一维，因此需要 (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)

        return x
