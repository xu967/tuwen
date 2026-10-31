"""
DFFN - Discriminative Frequency Filter Network (CVPR 2023)

论文题目: Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring
论文来源: CVPR 2023
论文链接: https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf
代码来源: /root/modules/034 Discriminative Frequency Filter Network(CVPR 2023).py
功能说明: 判别式频域滤波前馈网络。1x1 升维后按 8x8 patch 做 rFFT2，与可学习
    频域滤波器相乘再 iFFT，随后 DWConv 分路并以 GELU 门控融合，1x1 投影回 dim。
    若 H、W 不能被 8 整除，则右侧/底部零填充至 8 的倍数，滤波后再裁回原尺寸。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DFFN(nn.Module):
    """Discriminative Frequency Filter Network (CVPR 2023).

    8x8 patch 频域可学习滤波 + DWConv GELU 门控，输出通道与空间尺寸与输入一致。

    Args:
        dim (int): 输入/输出通道数。
        ffn_expansion_factor (float): 隐藏维度扩展因子，默认 2.66。
        bias (bool): 卷积是否使用偏置，默认 False。

    Note:
        本模块不改变通道数。空间尺寸若非 8 的倍数会先 pad 再 crop，不抛异常。
    """

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        """Initialize DFFN with dim, expansion factor and bias."""
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden_features * 2, bias=bias,
        )
        self.fft = nn.Parameter(
            torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward: pad-to-8 -> 1x1 -> patch rFFT filter -> iFFT -> DWConv GELU-gate -> 1x1 -> crop."""
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.project_in(x)
        x_patch = rearrange(
            x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size,
        )
        orig_dtype = x_patch.dtype
        if x_patch.is_cuda and x_patch.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type='cuda', enabled=False):
                x_patch_f = x_patch.float()
                x_patch_fft = torch.fft.rfft2(x_patch_f)
                x_patch_fft = x_patch_fft * self.fft.float()
                x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
            x_patch = x_patch.to(orig_dtype)
        else:
            x_patch_fft = torch.fft.rfft2(x_patch.float())
            x_patch_fft = x_patch_fft * self.fft
            x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=self.patch_size, patch2=self.patch_size,
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x[:, :, :h, :w]


__all__ = ['DFFN']
