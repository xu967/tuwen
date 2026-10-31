import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Detail-Preserving Contextual Fusion (DPCF)

论文：SAMamba: Adaptive State Space Modeling with Hierarchical Vision for Infrared Small Target Detection (2025 一区TOP)
链接：https://arxiv.org/pdf/2505.23214

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]，按 (x_low, x_high) 解包
- 通道四等分后自适应融合，1×1 卷积整合
- 约束：通道数须能被 4 整除
"""

__all__ = ['DPCF']


class AdaptiveCombiner(nn.Module):
    """Learn a scalar gate to blend two feature maps."""

    def __init__(self):
        super().__init__()
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        batch_size, channel, w, h = p.shape
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class conv_block(nn.Module):
    """Conv + norm + SiLU block."""

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)

        if self.act:
            self.relu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class DPCF(nn.Module):
    """Detail-Preserving Contextual Fusion.

    Args:
        in_features (int): 输入通道数（parse_model 注入）。
        out_features (int | None): 输出通道数，默认与 in_features 相同。
    """

    def __init__(self, in_features, out_features=None):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.ac = AdaptiveCombiner()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0))
        )

    def forward(self, x_low, x_high=None):
        if x_high is None and isinstance(x_low, (list, tuple)):
            x_low, x_high = x_low

        assert x_low.shape[1] % 4 == 0 and x_high.shape[1] % 4 == 0, \
            f"DPCF: 通道数须能被 4 整除, got {x_low.shape[1]} vs {x_high.shape[1]}"

        _, _, h, w = x_low.shape
        x_low = torch.chunk(x_low, 4, dim=1)
        x_high = F.interpolate(x_high, size=[h, w], mode='bilinear', align_corners=True)
        x_high = torch.chunk(x_high, 4, dim=1)

        x0 = self.ac(x_low[0], x_high[0])
        x1 = self.ac(x_low[1], x_high[1])
        x2 = self.ac(x_low[2], x_high[2])
        x3 = self.ac(x_low[3], x_high[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x
