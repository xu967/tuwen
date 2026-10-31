"""
ACFM - Attention and Convolution Fusion Module

论文: CAF-YOLO: A Robust Framework for Multi-Scale Lesion Detection in Biomedical Imagery
会议: ICASSP / MICCAI 2025
论文链接: https://arxiv.org/pdf/2408.01897

注意力与卷积融合: 伪 3D Conv 生成 QKV, 一路多头通道注意力捕获全局依赖,
一路 9 方向深度卷积捕获局部结构, 再与残差相加。输出与输入同形状。

通道约束: dim 必须能被 num_heads 整除。
"""

import torch
import torch.nn as nn
from einops import rearrange


class ACFM(nn.Module):
    """Attention and Convolution Fusion Module (CAF-YOLO, ICASSP/MICCAI 2025).

    Args:
        dim (int): 输入/输出通道数, 须能被 num_heads 整除。
        num_heads (int): 多头数。默认 4。
        bias (bool): QKV/投影卷积是否使用偏置。默认 True。
    """

    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        assert dim % num_heads == 0, f'dim ({dim}) must be divisible by num_heads ({num_heads})'

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(
            9 * dim // self.num_heads,
            dim,
            kernel_size=(3, 3, 3),
            groups=dim // self.num_heads,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        temp = x
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)

        f_all = qkv.reshape(b, h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(b, 9 * c // self.num_heads, h, w)
        out_conv = self.dep_conv(f_conv.unsqueeze(2)).squeeze(2)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        out = self.project_out(out.unsqueeze(2)).squeeze(2)
        out = temp + out
        return out + out_conv


__all__ = ['ACFM']
