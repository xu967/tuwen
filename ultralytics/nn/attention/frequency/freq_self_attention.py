"""
FSAS - Frequency Domain-Based Self-Attention Solver

论文: Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring
会议: CVPR 2023
论文链接: https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf

频域自注意力求解器。利用快速傅里叶变换（FFT）和卷积操作将注意力计算转移到频域，
通过 patch 级 rfft2/irfft2 在频域上计算 q*k 乘积，再回到空间域与 v 相乘，
大幅降低高分辨率图像上的注意力计算复杂度。
forward 接收单输入，输出与输入同通道同空间尺寸。
"""

import numbers

import torch
import torch.nn as nn
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """无偏置 LayerNorm，沿最后一维归一化。"""

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """带偏置 LayerNorm，沿最后一维归一化。"""

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """4D 张量上的 LayerNorm 封装，支持 BiasFree / WithBias 两种类型。"""

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FSAS(nn.Module):
    """
    Frequency Domain Self-Attention Solver (FSAS).

    通过 patch 级二维实数 FFT 在频域计算 q*k 注意力乘积，irfft2 回到空间域后
    与 v 相乘，从而以线性复杂度近似全局自注意力。
    """

    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        # 1x1 卷积将输入 dim 升维到 dim*6（后续 chunk 成 q/k/v 各 dim*2）
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)

        # 3x3 深度可分离卷积，保持 dim*6 维度
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        # 1x1 卷积将 dim*2 投影回 dim
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # LayerNorm，作用在 dim*2 通道上
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        # patch 尺寸，H/W 必须能被其整除才能直接做 patch FFT；否则 forward 会 pad
        self.patch_size = 8

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_size = self.patch_size

        # 自适应 patch：当 H/W 不是 patch_size 整数倍时，pad 到最近的整数倍
        # 保证 rearrange 与 rfft2/irfft2 对任意 H,W 均正常
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h != 0 or pad_w != 0:
            # 右下方向 pad
            x_padded = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x

        hidden = self.to_hidden(x_padded)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(
            q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=patch_size, patch2=patch_size,
        )
        k_patch = rearrange(
            k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=patch_size, patch2=patch_size,
        )

        # 频域上计算 q*k（rfft2 对任意 patch 尺寸自适应）
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft

        # 逆 FFT 回到空间域
        out = torch.fft.irfft2(out, s=(patch_size, patch_size))
        out = rearrange(
            out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=patch_size, patch2=patch_size,
        )

        out = self.norm(out)
        output = v * out
        output = self.project_out(output)

        # 去掉 pad，恢复原始空间尺寸
        if pad_h != 0 or pad_w != 0:
            output = output[:, :, :h, :w]

        return output


__all__ = ['FSAS']
