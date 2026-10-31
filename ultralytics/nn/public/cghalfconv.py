"""CGHalfConv - 通道分组折半卷积模块
论文: Channel grouping vision transformer for lightweight fruit and vegetable recognition (CGViT)
会议/期刊: 2025 一区TOP
链接: https://www.sciencedirect.com/science/article/pii/S0957417425022559

通道分组折半卷积（Channel Grouping Half-convolution，CGHF）:
1. 将特征图通道划分为 3 组，每组独立 HalfConv 卷积，实现对判别性特征的差异化捕捉
2. 仅对每组一半通道进行 3x3 卷积，另一半保持不变，减少 FLOPs 与内存访问量
3. 残差连接增强训练稳定性
"""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['CGHalfConv', 'HalfConv']


class HalfConv(nn.Module):
    """折半卷积: 仅对一半通道做 3x3 卷积，另一半保持不变。"""

    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div  # 需要进行卷积的通道数
        self.dim_untouched = dim - self.dim_conv3  # 保持不变的通道数
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class CGHalfConv(nn.Module):
    """通道分组折半卷积: 通道分 3 组，各自 HalfConv + 残差。"""

    def __init__(self, dim):
        super().__init__()
        self.div_dim = int(dim / 3)  # 每份的通道数
        self.remainder_dim = dim % 3  # 余下的通道数
        self.p1 = HalfConv(self.div_dim, 2)
        self.p2 = HalfConv(self.div_dim, 2)
        self.p3 = HalfConv(self.div_dim + self.remainder_dim, 2)

    def forward(self, x):
        y = x
        x1, x2, x3 = torch.split(x, [self.div_dim, self.div_dim, self.div_dim + self.remainder_dim], dim=1)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x = torch.cat((x1, x2, x3), 1)
        return x + y


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
