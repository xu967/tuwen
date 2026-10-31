"""StripeAttentionBlock - 条带卷积注意力
论文: Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
会议: AAAI 2026
链接: https://arxiv.org/abs/2501.03775
"""

import torch
import torch.nn as nn

__all__ = ['StripeAttentionBlock', 'DirectionalStripeAttention']


class DirectionalStripeAttention(nn.Module):
    """方向性条带注意力 (Directional Stripe Attention)

    通过纵向和横向条带卷积捕获各向异性上下文，
    结合局部空间特征实现方向感知的特征调制。

    Args:
        dim: 输入特征通道数
        k1: 条带卷积短边核大小 (默认 1)
        k2: 条带卷积长边核大小 (默认 19)
    """

    def __init__(self, dim, k1, k2):
        super().__init__()

        # 基础局部空间特征（5×5深度卷积）
        self.local_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 纵向条纹卷积：1×k2
        self.stripe_vertical = nn.Conv2d(
            dim, dim,
            kernel_size=(k1, k2),
            padding=(k1 // 2, k2 // 2),
            groups=dim
        )

        # 横向条纹卷积：k2×1
        self.stripe_horizontal = nn.Conv2d(
            dim, dim,
            kernel_size=(k2, k1),
            padding=(k2 // 2, k1 // 2),
            groups=dim
        )

        # 通道融合
        self.channel_mix = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.local_conv(x)
        attn = self.stripe_vertical(attn)
        attn = self.stripe_horizontal(attn)
        attn = self.channel_mix(attn)
        return x * attn


class StripeAttentionBlock(nn.Module):
    """条带卷积注意力块 (Stripe Attention Block)

    通过条带方向注意力机制捕获遥感图像中的细长目标上下文，
    利用纵横各异的条带卷积核高效建模各向异性特征。

    Args:
        d_model: 输入特征通道数
        k1: 条带卷积短边核大小 (默认 1)
        k2: 条带卷积长边核大小 (默认 19)
    """

    def __init__(self, d_model, k1=1, k2=19):
        super().__init__()

        self.proj_in = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()

        # 核心：条状方向注意力
        self.stripe_attn = DirectionalStripeAttention(d_model, k1, k2)

        self.proj_out = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.stripe_attn(x)
        x = self.proj_out(x)
        return x + shortcut


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
