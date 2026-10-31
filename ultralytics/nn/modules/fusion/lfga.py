import torch
import torch.nn as nn

"""
Local Feature Guidance Attention (LFGA)

论文：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection (CCF-A 2023)
链接：https://arxiv.org/pdf/2309.11131

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 用第二路特征生成 HW×HW 注意力，引导第一路局部特征
- 复杂度 O(HW^2)，仅适合 P5 等低分辨率特征
"""

__all__ = ['LFGA']


class LFGA(nn.Module):
    """Local Feature Guidance Attention.

    Args:
        in_channel (int): 输入通道数（parse_model 注入）。
        out_channel (int | None): Q/K 通道数，默认 in_channel // ratio（至少为 1）。
        ratio (int): Q/K 通道压缩比，默认 4。
    """

    def __init__(self, in_channel, out_channel=None, ratio=4):
        super().__init__()
        self.chanel_in = in_channel

        if out_channel is None:
            out_channel = in_channel // ratio if in_channel // ratio > 0 else 1

        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.SiLU()
        self.bn = nn.BatchNorm2d(self.chanel_in)

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        fa, fb = x1, x2
        B, C, H, W = fa.size()

        proj_query = self.query_conv(fb).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
        proj_key = self.key_conv(fb).view(B, -1, H * W)  # B, C', HW
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)

        proj_value = self.value_conv(fa).view(B, -1, H * W)  # B, C, HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(B, C, H, W)
        out = self.gamma * out + fa
        return self.relu(out)
