"""
DynamicRangeHistogramSelfAttention - 动态范围直方图自注意力

论文: Restoring Images in Adverse Weather Conditions via Histogram Transformer
来源: ECCV 2024
链接: https://arxiv.org/abs/2407.10172

基于直方图排序 + 双分支（局部框/全局序列）自注意力，捕获长距离依赖与邻域光谱相关性，
有效融合全局与局部信息。forward 输入 (B,C,H,W) 输出同形状同通道。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['DynamicRangeHistogramSelfAttention']


class DynamicRangeHistogramSelfAttention(nn.Module):
    """动态范围直方图自注意力（Histoformer, ECCV 2024）。"""

    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(DynamicRangeHistogramSelfAttention, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 卷积将 dim -> dim*5（双 q/k + v）
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        # 深度可分离卷积，提取局部特征
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        # 输出投影回 dim
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        """对最后一维填充至可被 factor 整除。"""
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        """移除 pad 添加的填充。"""
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        """带 +1 平滑项的归一化（避免极端值溢出）。"""
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        """对排序后的序列做框/全局双模式自注意力。"""
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor

        # ifBox=True: 块状局部关注；False: 全局长距离依赖
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"

        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = attn @ v

        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        # 克隆输入，避免原 ref 对 x[:, :c//2] 的 in-place 写入污染调用方张量
        x = x.clone()

        # 对前半通道沿 H、W 排序（直方图化），保留逆序索引
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = torch.cat([x_sort, x[:, c // 2:]], dim=1)

        # q/k/v 生成
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        # v 排序并据此 gather q1/k1/q2/k2，使所有序列按相同动态范围顺序对齐
        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 双分支：ifBox=True 局部块注意力，False 全局长距离注意力
        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        # 用排序索引将结果逆序回原始空间位置
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)

        out = out1 * out2
        out = self.project_out(out)

        # 反向应用 H/W 排序索引到前半通道
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out = torch.cat([out_replace, out[:, c // 2:]], dim=1)

        return out
