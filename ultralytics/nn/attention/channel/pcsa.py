"""
PCSA - Progressive Channel-wise Self-Attention

论文: SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
期刊: 2025 JCR 一区 TOP
论文链接: https://arxiv.org/pdf/2407.05128

渐进式通道自注意力: 对全局均值描述子做 GroupNorm 与深度 1x1 QKV,
多头通道注意力后经门控得到通道权重并乘回输入。输出与输入同形状。

通道约束: dim 必须能被 4 整除, 且能被 head_num 整除。
dim < 8 时自动将 head_num 调整为 dim 的一个约数。
"""

import torch
import torch.nn as nn
from einops import rearrange


class PCSA(nn.Module):
    """Progressive Channel-wise Self-Attention (SCSA, 2025).

    Args:
        dim (int): 输入/输出通道数, 须能被 4 整除。
        head_num (int): 注意力头数。默认 8。dim < 8 时自动改为 dim 的约数。
        qkv_bias (bool): QKV 1x1 卷积是否使用偏置。默认 False。
        attn_drop_ratio (float): 注意力 Dropout。默认 0。
        gate_layer (str): 通道门控, 'sigmoid' 或 'softmax'。默认 'sigmoid'。
    """

    def __init__(self, dim, head_num=8, qkv_bias=False, attn_drop_ratio=0., gate_layer='sigmoid'):
        super().__init__()
        assert dim % 4 == 0, 'dim must be divisible by 4'
        if dim < 8:
            head_num = next(h for h in range(min(head_num, dim), 0, -1) if dim % h == 0)
        assert dim % head_num == 0, f'dim ({dim}) must be divisible by head_num ({head_num})'

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

    def forward(self, y):
        temp = y
        y = y.mean((2, 3), keepdim=True)
        _, _, h_, w_ = y.size()
        y = self.norm(y)

        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=int(self.head_num), head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=int(self.head_num), head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=int(self.head_num), head_dim=int(self.head_dim))

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        attn = self.ca_gate(attn.mean((2, 3), keepdim=True))
        return attn * temp


__all__ = ['PCSA']
