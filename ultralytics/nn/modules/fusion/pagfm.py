import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Pixel-attention-guided Fusion Module (PagFM)

论文：PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers (CVPR 2023)
链接：https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 像素注意力门控融合，双线性插值对齐两路空间尺寸
- mid_channels 默认 max(in_channels // 2, 1)
"""

__all__ = ['PagFM']


class PagFM(nn.Module):
    """Pixel-attention-guided Fusion Module.

    Args:
        in_channels (int): 输入通道数（parse_model 注入）。
        mid_channels (int | None): 中间通道数，默认 max(in_channels // 2, 1)。
        after_relu (bool): 融合前是否对输入做激活。
        with_channel (bool): 是否使用通道注意力生成相似度图。
    """

    def __init__(self, in_channels, mid_channels=None, after_relu=False, with_channel=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(in_channels // 2, 1)

        self.with_channel = with_channel
        self.after_relu = after_relu

        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels)
            )

        if after_relu:
            self.relu = nn.SiLU()

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        x, y = x1, x2
        input_size = x.size()

        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y
        return x
