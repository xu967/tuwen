"""
SMSSA - Shareable Multi-Semantic Spatial Attention

论文: SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
期刊: 2025 JCR 一区 TOP
论文链接: https://arxiv.org/pdf/2407.05128

可共享多语义空间注意力: 沿 H/W 均值投影后按 4 组通道分别做 1D 深度卷积
(核 3/5/7/9), GroupNorm + 门控得到轴向注意力图, 再乘回输入。输出与输入同形状。

通道约束: dim 必须能被 4 整除。
"""

import torch
import torch.nn as nn


class SMSSA(nn.Module):
    """Shareable Multi-Semantic Spatial Attention (SCSA, 2025).

    Args:
        dim (int): 输入/输出通道数, 须能被 4 整除。
        group_kernel_sizes (tuple): 四组 1D 深度卷积核大小。默认 (3, 5, 7, 9)。
        gate_layer (str): 门控, 'sigmoid' 或 'softmax'。默认 'sigmoid'。
    """

    def __init__(self, dim, group_kernel_sizes=(3, 5, 7, 9), gate_layer='sigmoid'):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, 'dim must be divisible by 4'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2, groups=group_chans,
        )
        self.global_dwc_s = nn.Conv1d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2, groups=group_chans,
        )
        self.global_dwc_m = nn.Conv1d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2, groups=group_chans,
        )
        self.global_dwc_l = nn.Conv1d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2, groups=group_chans,
        )
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

    def forward(self, x):
        b, c, h_, w_ = x.size()
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l),
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)
        return x * x_h_attn * x_w_attn


__all__ = ['SMSSA']
