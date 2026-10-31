"""
CAFM - 卷积和注意力融合模块 (Convolution and Attention Fusion Module)

论文: Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising
来源: IEEE Geoscience and Remote Sensing Letters (GRSL), 2024
链接: https://doi.org/10.1109/LGRS.2024.3370299

混合注意力模块，同时建模全局和局部特征：局部通过深度卷积分支提取邻域相关性，
全局通过多头自注意力分支捕获长距离依赖。forward 返回 out_attn + out_conv，
输出与输入同形状同通道。输入 (b, c, h, w)，内部用 Conv3d 把 2D 视作 depth=1 的伪 3D。
"""

import torch
import torch.nn as nn
from einops import rearrange


class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 查询、键和值的卷积层
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # 深度可分离卷积层，用于 qkv
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1,
                                    groups=dim * 3, bias=bias)

        # 输出投影层
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

        # 全连接层，用于调整通道数
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        # 深度卷积层，用于局部特征提取
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.unsqueeze(2)  # (b, c, 1, h, w) 用于 3D 卷积
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)  # (b, 3*c, h, w)

        # ====================== 局部特征 ======================
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)

        # ====================== 全局特征 ======================
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)

        output = out + out_conv  # 全局自注意力 + 局部卷积
        return output


__all__ = ['CAFM']
