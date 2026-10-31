"""
DynamicSpatialAttention - 动态空间注意力

论文: DRPCA-Net: Make Robust PCA Great Again for Infrared Small Target Detection
会议: 2025 一区TOP
论文链接: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11079687

动态空间注意力 (DSA):
    ① 异质特征建模: 通过动态生成样本专属卷积核，适配不同区域的背景特征差异
    ② 低对比度目标定位: 生成自适应空间注意力图，聚焦低对比度目标区域
    ③ 场景针对性融合: 根据目标大小、位置动态调整特征权重
    实现: 全局平均池化 → 动态卷积核生成 → 通道平均空间描述符 → 卷积 → Sigmoid 注意力图 → 特征加权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DynamicSpatialAttention']


class DynamicSpatialAttention(nn.Module):
    """动态空间注意力模块

    通过动态生成每个样本专属的卷积核，对通道平均后的空间描述符进行卷积，
    产生自适应空间注意力图，实现目标区域特征强化与背景干扰抑制。
    """

    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # 卷积核生成器: 自适应平均池化 → 1×1 Conv → ReLU → 1×1 Conv → k×k 动态卷积核
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成每个样本的动态卷积核 → [B, 1, k, k]
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)

        # 通道维度求平均，得到空间描述符 → [1, B, H, W]
        x_mean = x.mean(dim=1, keepdim=True).view(1, B, H, W)

        # 分组卷积: 每个样本使用自己生成的卷积核
        att = F.conv2d(x_mean, weight=kernels, padding=self.kernel_size // 2, groups=B)

        # 归一化注意力图并加权
        att = self.sigmoid(att.view(B, 1, H, W))
        return x * att


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
