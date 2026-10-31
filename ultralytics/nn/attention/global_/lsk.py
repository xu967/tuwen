"""
LSKNetAttention - Large Selective Kernel Network Attention (LSKNet)

论文: LSKNet: Large Selective Kernel Network for Remote Sensing Object Detection
会议: ICCV 2023
论文链接: https://arxiv.org/pdf/2303.09030.pdf

功能: 大核空间选择性注意力模块。先用大核深度可分离卷积显式生成具有不同大感受野的特征
(5x5 标准核 + 7x7 膨胀核 dilation=3), 再通过空间选择机制 (avg+max 聚合 + 7x7 squeeze +
sigmoid) 在不同尺度大核之间做空间选择, 增强对检测目标最相关空间上下文的关注能力。
forward(x): 单输入, 输出与输入同通道同空间尺寸, 返回 x * attn (残差注意力乘法)。

通道约束: dim 为输入通道数, 内部使用 dim // 2 做分支降维后再拼接恢复, 要求 dim >= 2
(否则 dim // 2 == 0 导致 Conv2d 出错)。
"""

import torch
import torch.nn as nn


class LSKNetAttention(nn.Module):
    """Large Selective Kernel Network 注意力块 (ICCV 2023, LSKNet)。

    Args:
        dim (int): 输入/输出通道数。内部按 dim//2 分两路降维, 要求 dim >= 2。
    """

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 5x5 深度可分离卷积
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # 7x7 膨胀(dilation=3)空间卷积

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)  # 1x1 降维: 分支1
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)  # 1x1 降维: 分支2

        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)  # 7x7 空间选择 squeeze
        self.conv = nn.Conv2d(dim // 2, dim, 1)  # 1x1 恢复到原始 dim 通道

    def forward(self, x):
        attn1 = self.conv0(x)  # 5x5 大核分支
        attn2 = self.conv_spatial(attn1)  # 7x7 膨胀大核分支
        attn1 = self.conv1(attn1)  # 降维到 dim//2
        attn2 = self.conv2(attn2)  # 降维到 dim//2

        attn = torch.cat([attn1, attn2], dim=1)  # 通道拼接回 dim

        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 2 通道聚合

        sig = self.conv_squeeze(agg).sigmoid()  # 空间选择权重

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)  # 加权融合
        attn = self.conv(attn)  # 恢复到 dim
        return x * attn  # 残差注意力


__all__ = ['LSKNetAttention']
