"""
MLKA - Multi-Scale Large Kernel Attention

论文: Multi-scale Attention Network for Single Image Super-Resolution
来源: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 2024, NTIRE Workshop
链接: https://arxiv.org/abs/2209.14145

通过不同核大小 (3、5、7) 和膨胀率的卷积捕获多尺度特征，将大核注意力修改为
多尺度门控方案。在子分支上以 LKA* 的大核注意力结果乘以 X* 的深度可分离卷积
结果实现加权注意力，聚合全局与局部信息并避免潜在伪影，捕获多粒度级别的长距离
依赖关系。forward 输出与输入同形状 (B, C, H, W)。

通道自适应：dim 不必被 3 整除。若 dim 非 3 的倍数，内部自动 pad 到最近 3 的倍数
构建卷积（深度卷积分组需要 3 整除），forward 时将输入通道右侧 pad 后处理、输出裁回
原 dim；dim 为 3 的倍数时 need_pad=False，行为与原实现完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """支持 channels_first / channels_last 两种格式的 LayerNorm。"""

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
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLKA(nn.Module):
    """Multi-Scale Large Kernel Attention.

    Args:
        dim (int): 输入通道数，可为任意正整数。非 3 的倍数时自动 pad 适配。

    Inputs/Outputs:
        x: (B, dim, H, W) -> (B, dim, H, W)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        pad_dim = ((dim + 2) // 3) * 3  # 向上取整到 3 的倍数
        self.pad_dim = pad_dim
        self.need_pad = pad_dim != dim
        i_feats = 2 * pad_dim

        self.norm = LayerNorm(pad_dim, data_format='channels_first')

        self.proj_first = nn.Sequential(nn.Conv2d(pad_dim, i_feats, 1, 1, 0))

        # LKA3 分支: 3x3 dw -> 5x5 dilation=2 dw -> 1x1
        self.LKA3 = nn.Sequential(
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 3, 1, 1, groups=pad_dim // 3),
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 5, stride=1, padding=(5 // 2) * 2, groups=pad_dim // 3, dilation=2),
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 1, 1, 0))
        self.X3 = nn.Conv2d(pad_dim // 3, pad_dim // 3, 3, 1, 1, groups=pad_dim // 3)

        # LKA5 分支: 5x5 dw -> 7x7 dilation=3 dw -> 1x1
        self.LKA5 = nn.Sequential(
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 5, 1, 5 // 2, groups=pad_dim // 3),
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 7, stride=1, padding=(7 // 2) * 3, groups=pad_dim // 3, dilation=3),
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 1, 1, 0))
        self.X5 = nn.Conv2d(pad_dim // 3, pad_dim // 3, 5, 1, 5 // 2, groups=pad_dim // 3)

        # LKA7 分支: 7x7 dw -> 9x9 dilation=4 dw -> 1x1；X7 为 7x7 dw
        self.X7 = nn.Conv2d(pad_dim // 3, pad_dim // 3, 7, 1, 7 // 2, groups=pad_dim // 3)
        self.LKA7 = nn.Sequential(
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 7, 1, 7 // 2, groups=pad_dim // 3),
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 9, stride=1, padding=(9 // 2) * 4, groups=pad_dim // 3, dilation=4),
            nn.Conv2d(pad_dim // 3, pad_dim // 3, 1, 1, 0))

        self.proj_last = nn.Sequential(nn.Conv2d(pad_dim, pad_dim, 1, 1, 0))

        # 可学习缩放参数
        self.scale = nn.Parameter(torch.zeros((1, pad_dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        if self.need_pad:
            # 通道维 (倒数第3维) 右侧 pad 到 pad_dim
            x = F.pad(x, (0, 0, 0, 0, 0, self.pad_dim - self.dim))
        shortcut = x
        x = self.norm(x)
        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)

        # 大核注意力 × 深度可分离，多尺度门控加权
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1),
                       self.LKA5(a_2) * self.X5(a_2),
                       self.LKA7(a_3) * self.X7(a_3)], dim=1)

        out = self.proj_last(x * a) * self.scale + shortcut
        if self.need_pad:
            out = out[:, :self.dim, :, :]
        return out


__all__ = ['MLKA']
