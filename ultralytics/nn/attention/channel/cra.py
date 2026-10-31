"""
ChannelReductionAttention (CRA) - Channel Reduction Self-Attention

论文: MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation
会议: WACV 2024
论文链接: https://openaccess.thecvf.com/content/WACV2024/papers/Kang_MetaSeg_MetaFormer-Based_Global_Contexts-Aware_Network_for_Efficient_Semantic_Segmentation_WACV_2024_paper.pdf

通道缩减自注意力。设计理念是将 Q 和 K 的通道维度缩减到一维，从而有效地考虑全局性，
实现自注意力操作的计算减少，兼顾全局上下文提取与自注意力在语义分割中的计算效率。
forward 返回与输入同 shape 的注意力增强特征。
"""

import torch
import torch.nn as nn


class ChannelReductionAttention(nn.Module):
    """通道缩减自注意力模块（Channel Reduction Self-Attention, CRA）。

    将 Q/K 投影到缩减后的通道维度，V 保持原通道数，通过全局池化与逐位置注意力
    实现兼顾全局上下文与计算效率的自注意力。
    """

    def __init__(self, dim, reduction_ratio=16):
        super(ChannelReductionAttention, self).__init__()
        # 计算缩减后的通道数（dim 必须能被 reduction_ratio 整除）
        reduced_channels = dim // reduction_ratio

        # Q 投影：dim -> reduced_channels
        self.query_projection = nn.Linear(dim, reduced_channels)
        # K 投影：dim -> reduced_channels
        self.key_projection = nn.Linear(dim, reduced_channels)
        # V 投影：dim -> dim（保持通道数不变）
        self.value_projection = nn.Linear(dim, dim)

    @staticmethod
    def get_module_name():
        return "cra"

    def forward(self, x):
        # 输入 x: (B, C, H, W)
        batch_size, channels, height, width = x.size()

        # 展平空间维度
        input_flat = x.view(batch_size, channels, -1)  # (B, C, H*W)
        # 通道维全局平均，得到每个通道的均值
        avg_pool = torch.mean(input_flat, dim=-1, keepdim=True)  # (B, C, 1)

        # Q: (B, H*W, reduced_channels)
        query = self.query_projection(input_flat.permute(0, 2, 1))
        # K: (B, 1, reduced_channels)
        key = self.key_projection(avg_pool.permute(0, 2, 1))
        # V: (B, 1, C)
        value = self.value_projection(avg_pool.permute(0, 2, 1))

        # 注意力图: (B, H*W, 1)
        attention_map = torch.softmax(torch.bmm(query, key.permute(0, 2, 1)), dim=1)
        # 注意力加权输出: (B, H*W, C)
        out = torch.bmm(attention_map, value)

        # 还原为原始输入形状 (B, C, H, W)
        out = out.view(batch_size, channels, height, width)
        return out


__all__ = ['ChannelReductionAttention']
