"""
Multi-Order Gated Aggregation (MOGA) - ICLR 2024

模块名:
    MultiOrderGatedAggregation (空间多阶门控聚合 block)

论文:
    MogaNet: Multi-Order Gated Aggregation Network
    https://openreview.net/pdf?id=XhYWgjqCrV

源参考:
    https://www.bilibili.com/video/BV1RwtReMEzR/

功能:
    A 类单输入特征提取 block。通过多阶膨胀深度卷积获取不同感受野特征，
    结合门控聚合 (gated aggregation) 与特征分解 (feat_decompose) 调整，
    实现通道级空间特征融合。不改通道 (输入输出通道数一致)。

主类:
    - MultiOrderGatedAggregation

辅助类 (不导出，留同文件):
    - ElementScale: 可学习的逐元素缩放器，用于残差/分解路径的权重分配
    - MultiOrderDWConv: 多阶膨胀深度卷积，按通道比例切分三段并施加不同膨胀率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiOrderGatedAggregation"]


class ElementScale(nn.Module):
    """可学习的逐元素缩放器。

    类似自注意力中的权重分配，通过学习合适的缩放参数为不同元素分配不同权重，
    自动决定哪些特征应被放大或抑制。

    Args:
        embed_dims (int): 输入通道数。
        init_value (float): 初始缩放值。
        requires_grad (bool): 是否需要计算梯度。
    """

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()

        # 形状 (1, embed_dims, 1, 1) 的可学习缩放因子
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad,
        )

    def forward(self, x):
        return x * self.scale


class MultiOrderDWConv(nn.Module):
    """使用膨胀深度卷积核的多阶特征。

    Args:
        embed_dims (int): 输入通道数。
        dw_dilation (list): 三个深度卷积层的膨胀率。
        channel_split (list): 三个分割通道的比例。
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, ],
                 channel_split=[1, 3, 4, ],
                 ):
        super(MultiOrderDWConv, self).__init__()

        # 各部分通道比例: 1/8, 3/8, 4/8
        self.split_ratio = [i / sum(channel_split) for i in channel_split]

        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2

        self.embed_dims = embed_dims

        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # 基础深度卷积 (膨胀率 1)
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1,
            dilation=dw_dilation[0],
        )
        # 第二个深度卷积 (膨胀率 2)
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],
        )
        # 第三个深度卷积 (膨胀率 3)
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],
        )
        # 逐点卷积用于融合不同分支
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)  # 第一个 5x5 深度卷积

        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])

        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])

        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """具有多阶门控聚合的空间块 (MOGA block, ICLR 2024)。

    不改通道：输入输出通道数一致 (均为 embed_dims)。

    Args:
        dim (int): 输入通道数 (与输出通道数相同)。
        attn_dw_dilation (list): 三个深度卷积层的膨胀率。
        attn_channel_split (list): 分割通道的比例。
        attn_force_fp32 (bool): 是否强制使用 FP32 计算 (本实现保留接口位)。
    """

    def __init__(self,
                 dim,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = dim
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=dim,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        self.proj_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

        # 门控和值激活
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # 分解操作的可学习缩放
        self.sigma = ElementScale(dim, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)

        # 对 x 进行全局平均池化
        x_d = F.adaptive_avg_pool2d(x, output_size=1)

        x = x + self.sigma(x - x_d)  # 加上分解后的调整

        x = self.act_value(x)  # 应用激活函数

        return x

    def forward(self, x):
        shortcut = x.clone()

        # 图中蓝色框: 特征分解
        x = self.feat_decompose(x)

        # 图中灰色框: 左 F 右 G
        F = self.gate(x)
        G = self.value(x)

        # 两个 SiLU 和点乘
        x = self.proj_2(self.act_gate(F) * self.act_gate(G))

        # 残差连接
        x = x + shortcut

        return x
