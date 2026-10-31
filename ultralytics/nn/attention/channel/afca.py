"""
FCAttention - Adaptive Fine-Grained Channel Attention (FCA)

论文: Unsupervised Bidirectional Contrastive Reconstruction and Adaptive
      Fine-Grained Channel Attention Networks for image dehazing
会议: Neural Networks 2024 (一区 Top)
论文链接: https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387

自适应细粒度通道注意力。通过全局与局部两条支路分别建模通道依赖，再以可学习权重 Mix 融合，
克服 SE 通道注意力仅依赖全局信息而忽略局部信息的缺陷，实现通道权重的自适应分配。
forward 返回 input * W，输出与输入同通道同形状。
"""

import math

import torch
from torch import nn


class Mix(nn.Module):
    """可学习权重融合两个等形特征图，辅助类，不对外导出。"""

    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class FCAttention(nn.Module):
    """Adaptive Fine-Grained Channel Attention 主类。"""

    def __init__(self, dim, b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 根据 dim 自适应计算一维卷积核大小 k
        t = int(abs((math.log(dim, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        # 二维卷积（逐点）建模局部通道依赖
        self.fc = nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, x):
        U = self.avg_pool(x)

        # 全局通道依赖支路：squeeze + 1D conv
        Ugc = self.conv1(U.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (b, c, 1)
        # 局部通道依赖支路：1x1 conv
        Ulc = self.fc(U).squeeze(-1).transpose(-1, -2)  # (b, 1, c)

        out1 = torch.sum(torch.matmul(Ugc, Ulc), dim=1).unsqueeze(-1).unsqueeze(-1)  # (b, c, 1, 1)
        out1 = self.sigmoid(out1)

        out2 = torch.sum(torch.matmul(Ulc.transpose(-1, -2), Ugc.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)

        out = self.mix(out1, out2)

        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        W = self.sigmoid(out)

        return x * W


__all__ = ['FCAttention']
