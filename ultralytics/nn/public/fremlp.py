"""FreMLP - 频域多层感知机
论文: DarkIR: Robust Low-Light Image Restoration
会议: CVPR 2025
论文地址: https://arxiv.org/abs/2412.13443

频率域前馈网络（Feed-Forward Network in the Frequency domain，FreMLP）：
    实际意义：①增强低光图像光照：在低光环境中，FreMLP 能够在不改变图像相位的情况下，
                仅对幅度进行处理，校正图像的光照效果，让图像更加清晰、明亮。
            ②多分辨率下的一致性：在不同分辨率下均能保持良好增强效果。
    实现方式：傅里叶域转换 -> 幅度增强操作 -> 逆变换(空间域)
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ['FreMLP', 'LayerNormFunction', 'LayerNorm2d', 'Frequency_Domain']


class LayerNormFunction(torch.autograd.Function):
    """自定义 LayerNorm 的 autograd Function，支持通道维度归一化。"""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """通道维度 Layer Normalization（自定义 autograd 实现）。"""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FreMLP(nn.Module):
    """频域多层感知机 —— 在傅里叶幅度谱上做 MLP 增强。

    Args:
        nc (int): 输入通道数。
        expand (int): 通道扩展倍率，默认 2。
    """

    def __init__(self, nc, expand=2):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        # FFT 必须在 float32 下 (半精度会产生 ComplexHalf, 不支持 rfft2/irfft2)
        # 但模型参数 (Conv2d weight/bias) 在验证时为 Half, 因此 process1 的输入
        # 必须转回原 dtype 以匹配参数 dtype, 复数重建时再转回 float32 做 FFT
        orig_dtype = x.dtype
        x_f32 = x.float()
        # 二维实值快速傅里叶变换 (float32)
        x_freq = torch.fft.rfft2(x_f32, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        # 幅度谱 MLP 处理 - 转回原 dtype 匹配 Conv2d 权重 (训练时 float32, 验证时 Half)
        mag = self.process1(mag.to(orig_dtype))
        # 重建复数频域表示 (复数运算需 float32)
        mag_f32 = mag.float()
        real = mag_f32 * torch.cos(pha)
        imag = mag_f32 * torch.sin(pha)
        x_out = torch.complex(real, imag)
        # 逆傅里叶变换回空间域
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out.to(orig_dtype)


class Frequency_Domain(nn.Module):
    """频域处理模块：LayerNorm + FreMLP + 残差连接。

    Args:
        channels (int): 输入通道数。
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.freq = FreMLP(nc=channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        A = inp
        x_step2 = self.norm(inp)
        x_freq = self.freq(x_step2)
        x = A * x_freq
        x = A + x * self.gamma
        return x


if __name__ == '__main__':
    model = Frequency_Domain(channels=32)
    input_tensor = torch.rand(1, 32, 50, 50)
    output = model(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')


# ─────────────────────────────────────────────
# 关注汤导B站空间，了解更多项目进展
# https://space.bilibili.com/3546905915165116
# ─────────────────────────────────────────────
