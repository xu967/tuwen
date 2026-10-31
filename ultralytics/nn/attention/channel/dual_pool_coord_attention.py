"""DualPoolCoordAttention - 双坐标注意力 (均值+最大池化并行)
论文: Flora-NET: Integrating dual coordinate attention with adaptive kernel based convolution network for medicinal flower identification
会议/期刊: 2025 一区TOP
链接: https://www.sciencedirect.com/science/article/pii/S0168169924012250

双坐标注意力 (Dual Coordinate Attention):
1. 沿水平/垂直方向分别做均值池化和最大池化，捕获长程位置依赖
2. 均值分支: 捕获全局上下文平滑信息
3. 最大分支: 保留显著判别特征锐度
4. 两个分支 Sigmoid 加权后相加, 增强特征表示
"""

import torch
import torch.nn as nn

__all__ = ['DualPoolCoordAttention']


class DualPoolCoordAttention(nn.Module):
    """双坐标注意力: 均值/最大池化并行分支 + Sigmoid 加权。dim-based 通道不变。"""

    def __init__(self, inp, oup=None, groups=32):
        super().__init__()
        oup = oup or inp  # 默认通道不变

        # 均值池化分支
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        # 最大池化分支
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        # 均值分支: 压缩 + 恢复
        self.conv1_mean = nn.Conv2d(inp, mip, 1)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, 1)
        # 最大分支: 压缩 + 恢复
        self.conv1_max = nn.Conv2d(inp, mip, 1)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # ---- 均值分支 ----
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.relu(self.bn1_mean(self.conv1_mean(y_mean)))
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid().expand(-1, -1, h, w)
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid().expand(-1, -1, h, w)

        # ---- 最大分支 ----
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.relu(self.bn1_max(self.conv1_max(y_max)))
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        x_h_max = self.conv2_max(x_h_max).sigmoid().expand(-1, -1, h, w)
        x_w_max = self.conv2_max(x_w_max).sigmoid().expand(-1, -1, h, w)

        # ---- 融合: 均值加权 + 最大加权 ----
        return identity * x_w_mean * x_h_mean + identity * x_w_max * x_h_max


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
