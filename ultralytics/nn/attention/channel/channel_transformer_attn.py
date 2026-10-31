"""
通道注意力 (Transformer 版) — 通道维多头自注意力
论文: Restormer: Efficient Transformer for High-Resolution Image Restoration
来源: CVPR 2022 (Oral)
链接: https://arxiv.org/abs/2111.09881
功能: 在通道维度上做多头自注意力，配合可学习温度参数调节注意力分布；
      通过 1x1 卷积生成 QKV 并经深度可分离卷积增强，捕获通道间全局依赖。
约束: dim 须能被 num_heads 整除（rearrange 将通道拆分到多头）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['ChannelTransformerAttention']


class ChannelTransformerAttention(nn.Module):
    """通道维多头自注意力 (Channel Attention - Transformer 版)
    Args:
        dim (int): 输入通道数，须能被 num_heads 整除
        num_heads (int): 注意力头数，默认 8
        bias (bool): 卷积是否启用偏置，默认 False
    """
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
