"""
Star_Block - Rewrite the Stars (CVPR 2024)

论文题目: Rewrite the Stars
论文来源: CVPR 2024
论文链接: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Rewrite_the_Stars_CVPR_2024_paper.pdf
代码来源: /home/zhizi/work/multimodel/ultralyticmm/ref/newmodule/011 StarConv(CVPR2024).py
功能说明: Star 操作将低维输入映射到高维非线性特征空间。
    通过两个分支 f1/f2 的元素乘积（其中 f1 经 ReLU6 激活），隐式构建高维
    非线性表征（star operation），无需显式升维即可获得近似高维特征空间效果。
    残差结构（input + drop_path(x)）保持输入输出通道一致，可作为即插即用 block。
    依赖: timm.layers.DropPath
"""

from __future__ import annotations

import torch.nn as nn
from timm.layers import DropPath


class ConvBN(nn.Module):
    """Conv + BN (不带激活) 的轻量级卷积单元，供 Star_Block 内部使用，不对外导出。"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """Initialize ConvBN with kernel, stride, padding, groups, dilation.

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int | list): 卷积核大小，支持整数或列表。
            s (int): 步长。
            p (int | list | None): 填充，None 时自动取 k//2。
            g (int): 分组数（=c1 时为深度可分离卷积）。
            d (int): 膨胀率。
        """
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        """Apply convolution and batch normalization."""
        return self.bn(self.conv(x))


class StarBlock(nn.Module):
    """Star Block (Rewrite the Stars, CVPR 2024).

    通过 star operation（f1(x) * f2(x)，其中 f1 经 ReLU6）在低维输入下隐式获得
    高维非线性特征空间，残差结构保持输入输出通道一致。

    Args:
        dim (int): 输入/输出通道数（残差结构，通道不变）。
        mlp_ratio (int): 隐藏维度扩展比，中间维度为 mlp_ratio * dim。
        drop_path (float): DropPath 丢弃率。

    Note:
        本模块不改变通道数（残差连接），dim 同时为输入和输出通道。
    """

    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        """Initialize StarBlock with dim, mlp_ratio and drop_path."""
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, g=dim)  # 深度可分离卷积
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)  # 1x1 卷积分支 1
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)  # 1x1 卷积分支 2
        self.g = ConvBN(mlp_ratio * dim, dim, 1)  # 降回 dim（Star）
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)  # 多阶通道重分配
        self.act = nn.ReLU6()  # 仅 f1 分支激活
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass: dwconv -> f1/f2 -> star(act(f1)*f2) -> g -> dwconv2 -> residual."""
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # Star operation: 隐式高维非线性
        x = self.g(x)  # 通道降回 dim
        x = self.dwconv2(x)  # 自适应多阶通道重分配
        x = input + self.drop_path(x)  # 残差连接
        return x


__all__ = ['StarBlock']
