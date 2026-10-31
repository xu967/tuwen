"""
ESSA - Efficient SCC-kernel-based Self-Attention

论文: ESSAformer: Efficient Transformer for Hyperspectral Image Super-resolution
会议: ECCV 2023
论文链接: https://arxiv.org/pdf/2307.14010

高效 SCC 核自注意力: 线性层生成 QKV, 对 Q/K 中心化后用平方归一化构造
二阶通道相关核, 以 (Q^2)(K^2)^T V / sqrt(N) 近似注意力并与 V 相加。
输入 BCHW, 内部按 BN C 计算后还原为 BCHW, 输出与输入同形状。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESSA(nn.Module):
    """Efficient SCC-kernel-based Self-Attention (ESSAformer, ECCV 2023).

    Args:
        dim (int): 输入/输出通道数。
    """

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        b, N, C = x.shape
        qkv = torch.split(self.lnqkv(x), C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q - torch.mean(q, dim=2, keepdim=True)
        k = k - torch.mean(k, dim=2, keepdim=True)
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)

        t1 = v
        k2 = F.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = F.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = self.ln(t1 + t2)
        return attn.reshape(b, h, w, c).permute(0, 3, 1, 2)


__all__ = ['ESSA']
