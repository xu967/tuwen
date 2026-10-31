"""
HFFE - High-Frequency Feature Extraction Block

论文: Bracketing Image Restoration and Enhancement with High-Low Frequency
      Decomposition (CVPR 2024)
链接: https://arxiv.org/abs/2404.13537
来源: https://www.bilibili.com/video/BV1BetCemEyy/

功能:
    高频信息代表图像的细节，需要较小的感受野聚焦局部信息以恢复细节。
    本模块为论文中的局部特征提取块（Local Feature Extraction Block for
    high-frequency），由多个小卷积核 (3x3) 的卷积层与密集连接 (dense
    connection) + 残差连接 (residual connection) 组合而成。
    - 小卷积核: 聚焦细节区域，捕捉细微结构和纹理（高频信息关键）。
    - 密集连接: 在不同层间建立连接，将低级与高级特征结合，增强细节理解。
    - 残差连接: 直接学习输入与输出的差异，专注需增强/修复的细节部分。

通道约束:
    本模块不改通道（输入输出通道相同），按 (dim, ) 单通道参数风格设计。
    forward 输入输出张量形状完全一致 [B, C, H, W]。
"""

import torch
import torch.nn as nn

__all__ = ["HFFE"]


class HFFE(nn.Module):
    """High-Frequency Feature Extraction Block (CVPR 2024).

    通过 6 个 3x3 卷积层与累积密集残差连接提取高频细节特征。

    Args:
        dim (int): 输入/输出通道数（不改通道）。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # 小卷积核 (3x3)：聚焦细节区域，捕捉细微结构与纹理
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 密集连接 + 残差连接：累积融合低级与高级特征，强化高频细节
        x1 = self.conv1(x)
        x1 = self.gelu(x1 + x)

        x2 = self.conv2(x1)
        x2 = self.gelu(x2 + x1 + x)

        x3 = self.conv3(x2)
        x3 = self.gelu(x3 + x2 + x1 + x)

        x4 = self.conv4(x3)
        x4 = self.gelu(x4 + x3 + x2 + x1 + x)

        x5 = self.conv5(x4)
        x5 = self.gelu(x5 + x4 + x3 + x2 + x1 + x)

        x6 = self.conv6(x5)
        x6 = self.gelu(x6 + x5 + x4 + x3 + x2 + x1 + x)

        return x6
