"""
SHSA - Single-Head Self-Attention

论文: SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design
会议: CVPR 2024
论文链接: https://arxiv.org/pdf/2401.16456

单头自注意力模块。在注意力操作之前缩小了 K 和 V 的空间尺度（仅对前 pdim 通道做注意力，
剩余通道直接旁路），大幅减少计算/内存开销；最后通过投影层聚合两路特征。
单输入单输出，输出与输入同通道同空间形状。
"""

import torch
import torch.nn as nn

__all__ = ['SHSA']


class _GroupNorm(nn.GroupNorm):
    """使用 1 个组进行的 Group Normalization。

    输入：形状为 [B, C, H, W] 的张量。
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class _Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm2d 顺序组合，可融合卷积与 BN。"""

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups, w.size(0), w.shape[2:],
            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
            groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SHSA(nn.Module):
    """Single-Head Self-Attention.

    仅对输入前 pdim 个通道执行单头注意力，剩余通道直接旁路拼接，再经投影聚合。
    """

    def __init__(self, dim, qk_dim=16, pdim=32):
        super(SHSA, self).__init__()

        assert pdim < dim, f'pdim ({pdim}) 必须小于 dim ({dim})，否则 split 失败'
        assert qk_dim <= pdim, f'qk_dim ({qk_dim}) 必须不大于 pdim ({pdim})'

        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = _GroupNorm(pdim)
        self.qkv = _Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.ReLU(), _Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape

        x1, x_id = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)

        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)

        x = self.proj(torch.cat([x1, x_id], dim=1))
        return x
