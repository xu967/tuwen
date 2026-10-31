"""
SPRS - 空间像素精化自注意力
论文: Spatial Pixel Refinement Self-attention
会议: ACM MM 2025
"""

import torch.nn.functional as F
import torch
import torch.nn as nn

__all__ = ['SPRS']


class SPRS(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x1 = F.adaptive_avg_pool2d(x, (1, 1))  # 全局平均池化
        x1 = F.softmax(x1, dim=1)             # softmax归一化
        x = x1 * x                            # 加权特征
        x = self.act(x)
        x = self.conv_1(x)
        return x


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
