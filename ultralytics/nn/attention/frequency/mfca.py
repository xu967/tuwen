"""
MFCA - Multi Frequency Channel Attention

论文: Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention
会议: CVPR 2024
论文链接: https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf

多频率通道注意力: 按预定义 DCT 频率分量对池化特征加权, 聚合 avg/max/min
通道统计后经 1x1 瓶颈生成通道门控, 再乘回输入。输出与输入同形状。
H,W 与 dct_h/dct_w 不一致时先自适应池化到 DCT 网格。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_freq_indices(method):
    """返回预定义 DCT 频率坐标 (mapper_x, mapper_y)。"""
    assert method in [
        'top1', 'top2', 'top4', 'top8', 'top16', 'top32',
        'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
        'low1', 'low2', 'low4', 'low8', 'low16', 'low32',
    ]
    num_freq = int(method[3:])
    if 'top' in method:
        all_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
    elif 'low' in method:
        all_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
    elif 'bot' in method:
        all_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
    else:
        raise NotImplementedError
    return all_indices_x[:num_freq], all_indices_y[:num_freq]


class MFCA(nn.Module):
    """Multi Frequency Channel Attention (CVPR 2024).

    Args:
        dim (int): 输入/输出通道数。
        dct_h (int): DCT 网格高。默认 7。
        dct_w (int): DCT 网格宽。默认 7。
        frequency_branches (int): 频率分支数, 须为 1/2/4/8/16/32。默认 16。
        frequency_selection (str): 频率选取策略 'top'/'low'/'bot'。默认 'top'。
        reduction (int): 通道瓶颈缩减比。默认 16。中间通道为 max(dim // reduction, 1)。
    """

    def __init__(self, dim, dct_h=7, dct_w=7, frequency_branches=16, frequency_selection='top', reduction=16):
        super().__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = _get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        assert len(mapper_x) == len(mapper_y)

        for freq_idx in range(frequency_branches):
            self.register_buffer(
                f'dct_weight_{freq_idx}',
                self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], dim),
            )

        mid = max(dim // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(
                    t_y, mapper_y, tile_size_y
                )
        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        return result * math.sqrt(2)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x_pooled = x
        if H != self.dct_h or W != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for freq_idx in range(self.num_freq):
            params = getattr(self, f'dct_weight_{freq_idx}')
            x_pooled_spectral = x_pooled * params
            multi_spectral_feature_avg = multi_spectral_feature_avg + self.average_channel_pooling(x_pooled_spectral)
            multi_spectral_feature_max = multi_spectral_feature_max + self.max_channel_pooling(x_pooled_spectral)
            multi_spectral_feature_min = multi_spectral_feature_min + (-self.max_channel_pooling(-x_pooled_spectral))

        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)
        multi_spectral_attention_map = F.sigmoid(
            multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map
        )
        return x * multi_spectral_attention_map.expand_as(x)


__all__ = ['MFCA']
