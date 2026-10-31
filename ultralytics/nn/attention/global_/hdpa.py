"""
HDPA - Hierarchical Dual-Path Attention

论文: MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices
会议: ICCV 2025
论文链接: https://arxiv.org/abs/2507.01838

分层双路径注意力: 全局分支 (MBRConv1X1 + GAP + Sigmoid) 与局部分支
(通道 max 后的 MBRConv1X1 + Sigmoid) 相乘后调制输入。forward 输出与输入同形状。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MBRConv1X1(nn.Module):
    """MobileIE 重参数 1x1 卷积: 扩展通道后拼接 BN, 用固定核 + 可学习残差核做 1x1 投影。"""

    def __init__(self, in_channels, out_channels, rep_scale=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, kernel_size=1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, kernel_size=1)
        self.conv_out.weight.requires_grad = False
        self.weight1 = nn.Parameter(torch.zeros_like(self.conv_out.weight))
        nn.init.xavier_normal_(self.weight1)

    def forward(self, inp):
        x1 = self.conv(inp)
        x = torch.cat([x1, self.conv_bn(x1)], dim=1)
        final_weight = self.conv_out.weight + self.weight1
        return F.conv2d(x, final_weight, bias=self.conv_out.bias, stride=1, padding=0, dilation=1, groups=1)


class HDPA(nn.Module):
    """Hierarchical Dual-Path Attention (MobileIE, ICCV 2025).

    Args:
        dim (int): 输入/输出通道数。
        rep_scale (int): MBRConv1X1 通道扩展倍率。默认 4。
    """

    def __init__(self, dim, rep_scale=4):
        super().__init__()
        self.channels = dim

        self.globalatt = nn.Sequential(
            _MBRConv1X1(dim, dim, rep_scale=rep_scale),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )
        self.localatt = nn.Sequential(
            _MBRConv1X1(1, dim, rep_scale=rep_scale),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.globalatt(x)
        max_out, _ = torch.max(x1 * x, dim=1, keepdim=True)
        x2 = self.localatt(max_out)
        return torch.mul(x1, x2) * x


__all__ = ['HDPA']
