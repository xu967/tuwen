"""
BidomainNonlinearMapping - Spatial and Frequency Information Interaction (CVPR 2024)

论文题目: A Semi-supervised Nighttime Dehazing Baseline with Spatial-Frequency Aware
         and Realistic Brightness Constraint
论文来源: CVPR 2024
论文链接: https://openaccess.thecvf.com/content/CVPR2024/html/Cong_A_Semi-supervised_Nighttime_Dehazing_Baseline_with_Spatial-Frequency_Aware_and_Realistic_CVPR_2024_paper.html
代码来源: /root/modules/037 Spatial and Frequency Information Interaction （CVPR 2024）.py
功能说明: 双域非线性映射。空间支路 SpaBlock 残差卷积提取空间特征；频率支路对
    rFFT2 幅度/相位分别经 1x1+SE 动态聚合后再 iFFT。两路拼接后 1x1 融合回 dim，
    输入输出 C,H,W 一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _BasicConv(nn.Module):
    """Conv(+BN)+GELU 基础卷积单元，仅供双域模块内部使用。"""

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super().__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
            )
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class _SpaBlock(nn.Module):
    """空间残差卷积块（3x3 x3），仅供内部使用。"""

    def __init__(self, nc):
        super().__init__()
        self.conv1 = _BasicConv(nc, nc, kernel_size=3, stride=1, relu=True)
        self.trans_layer = _BasicConv(nc, nc, kernel_size=3, stride=1, relu=False)
        self.conv2 = _BasicConv(nc, nc, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans_layer(out)
        out = self.conv2(out)
        return out + x


class _SELayer(nn.Module):
    """Squeeze-and-Excitation 通道注意力，仅供内部使用。"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class _Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    """幅度/相位分别 1x1+SE 动态聚合后还原空间域，仅供内部使用。"""

    def __init__(self, nc):
        super().__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            _SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            _SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        orig_dtype = x.dtype
        conv_dtype = next(self.processmag.parameters()).dtype
        ctx = torch.autocast(device_type='cuda', enabled=False) if x.is_cuda else torch.autocast(device_type='cpu', enabled=False)
        with ctx:
            x_freq = torch.fft.rfft2(x.float(), norm='backward')
            ori_mag = torch.abs(x_freq)
            ori_pha = torch.angle(x_freq)
        mag = ori_mag.to(conv_dtype)
        pha = ori_pha.to(conv_dtype)
        mag = (ori_mag + self.processmag(mag).float())
        pha = (ori_pha + self.processpha(pha).float())
        with ctx:
            real = mag * torch.cos(pha)
            imag = mag * torch.sin(pha)
            x_out = torch.complex(real, imag)
            y = torch.fft.irfft2(x_out, s=(h, w), norm='backward')
        return y.to(orig_dtype)


class BidomainNonlinearMapping(nn.Module):
    """Bidomain Nonlinear Mapping (CVPR 2024).

    空间残差支路与频域幅度/相位动态聚合支路拼接后 1x1 融合，通道不变。

    Args:
        dim (int): 输入/输出通道数（通道不变）。

    Note:
        本模块不改变通道数与空间尺寸，dim 同时为输入和输出通道。
    """

    def __init__(self, dim):
        """Initialize BidomainNonlinearMapping with dim."""
        super().__init__()
        self.spatial_process = _SpaBlock(dim)
        self.frequency_process = _Frequency_Spectrum_Dynamic_Aggregation(dim)
        self.cat = nn.Conv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, x):
        """Forward: spatial residual || FFT mag/phase agg -> 1x1 fuse."""
        x_freq = self.frequency_process(x)
        x = self.spatial_process(x)
        xcat = torch.cat([x, x_freq], 1)
        return self.cat(xcat)


__all__ = ['BidomainNonlinearMapping']
