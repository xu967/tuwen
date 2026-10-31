"""
SCConv: Spatial and Channel Reconstruction Convolution

论文: SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy
会议: CVPR 2023
论文链接: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf

功能:
    SCConv 通过空间重构单元 (SRU) 和通道重构单元 (CRU) 两路子模块，
    利用特征之间的空间和通道冗余来进行 CNN 压缩，减少冗余计算并促进代表性特征学习。
    主类 ScConv 为通道不变模块 (forward 输入输出同通道数)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ScConv']


class GroupBatchnorm2d(nn.Module):
    """二维组归一化层 (辅助类，不导出)。

    Args:
        c_num (int): 输入通道数。
        group_num (int): 分组数量，默认 16。
        eps (float): 数值稳定小常数，默认 1e-10。
    """

    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super().__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    """Spatial Reconstruction Unit - 空间重构单元 (辅助类，不导出)。

    Args:
        oup_channels (int): 通道数。
        group_num (int): 分组数量。
        gate_treshold (float): 门限值。
    """

    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    """Channel Reconstruction Unit - 通道重构单元 (辅助类，不导出)。

    Args:
        op_channel (int): 通道数。
        alpha (float): 上下分割比例，0 < alpha < 1。
        squeeze_radio (int): 压缩率。
        group_size (int): 组卷积的组大小。
        group_kernel_size (int): 组卷积核大小。
    """

    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2,
                 group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size,
                             stride=1, padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio,
                              kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    """SCConv 主模块 (通道不变，forward 输入输出同通道数)。

    Args:
        dim (int): 输入输出通道数。
        group_num (int): SRU 的分组数量，默认 16。
        gate_treshold (float): SRU 门限值，默认 0.5。
        alpha (float): CRU 上下分割比例，默认 0.5。
        squeeze_radio (int): CRU 压缩率，默认 2。
        group_size (int): CRU 组卷积组大小，默认 2。
        group_kernel_size (int): CRU 组卷积核大小，默认 3。
    """

    def __init__(self, dim, group_num=16, gate_treshold=0.5, alpha=0.5,
                 squeeze_radio=2, group_size=2, group_kernel_size=3):
        super().__init__()
        self.SRU = SRU(dim, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(dim, alpha=alpha, squeeze_radio=squeeze_radio,
                       group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
