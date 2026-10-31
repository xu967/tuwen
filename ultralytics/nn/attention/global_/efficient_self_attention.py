"""
EASA - 高效近似自注意力 (Efficient Approximation of Self-Attention)

论文: SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
会议: ECCV 2024
论文链接: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf

通过下采样最大池化 + 深度可分离卷积生成非局部结构信息，并融合输入方差作为空间统计差异，
经调制后聚合回输入特征以获取代表性结构信息。单输入单输出，输出与输入同通道。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['EASA']


class EASA(nn.Module):
    """Efficient Approximation of Self-Attention."""

    def __init__(self, dim, down_scale=8):
        super(EASA, self).__init__()

        # 1x1 卷积，保持通道数不变，用于调制特征映射
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 1x1 卷积，保持通道数不变，用于最终输出投影
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        # 深度可分离卷积（Depth-wise Convolution），提取非局部结构信息
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        # 下采样因子，控制非局部信息提取的空间尺度
        self.down_scale = down_scale

        # 可学习参数 alpha，初始化为全 1
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        # 可学习参数 belt，初始化为全 0
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        _, _, h, w = x.shape

        # 自适应最大池化下采样后，经深度可分离卷积生成非局部结构信息 x_s
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))

        # 计算输入 x 的方差，作为空间信息的统计差异
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)

        # 融合非局部结构与方差统计，得到局部细节估计
        temp = x_s * self.alpha + x_v * self.belt

        # 调制特征上采样回原尺寸后与输入相乘聚合
        x_l = x * F.interpolate(self.gelu(self.linear_1(temp)), size=(h, w), mode='nearest')

        # 通过输出投影层得到最终结果
        return self.linear_2(x_l)
