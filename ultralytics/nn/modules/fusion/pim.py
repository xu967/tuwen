import torch
import torch.nn as nn

"""
Phase Integration Module (PIM)

论文：Guided Real Image Dehazing Using YCbCr Color Space (AAAI 2025)
链接：https://ojs.aaai.org/index.php/AAAI/article/download/32297/34452

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- FFT 幅值/相位双分支，跨模态相位叠加后 IFFT 重建
- 双输出改造为单输出：out_rgb + out_ycbcr
"""

__all__ = ['PIM']


class PIM(nn.Module):
    """Phase Integration Module.

    Args:
        channel (int): 输入通道数（parse_model 注入）。
    """

    def __init__(self, channel):
        super().__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0)
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0)
        )

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        rgb_x, ycbcr_x = x1, x2
        orig_dtype = rgb_x.dtype
        conv_dtype = next(self.processmag.parameters()).dtype
        spatial_size = rgb_x.shape[-2:]
        device_type = 'cuda' if rgb_x.is_cuda else 'cpu'
        ctx = torch.autocast(device_type=device_type, enabled=False) if rgb_x.is_cuda else torch.autocast(device_type='cpu', enabled=False)

        with ctx:
            rgb_fft = torch.fft.rfft2(rgb_x.float(), norm='backward')
            ycbcr_fft = torch.fft.rfft2(ycbcr_x.float(), norm='backward')
            rgb_amp = torch.abs(rgb_fft)
            rgb_phase = torch.angle(rgb_fft)
            ycbcr_amp = torch.abs(ycbcr_fft)
            ycbcr_phase = torch.angle(ycbcr_fft)

        rgb_amp = self.processmag(rgb_amp.to(conv_dtype)).float()
        rgb_phase = self.processpha(rgb_phase.to(conv_dtype)).float()
        ycbcr_amp = self.processmag(ycbcr_amp.to(conv_dtype)).float()
        ycbcr_phase = self.processpha(ycbcr_phase.to(conv_dtype)).float()
        mix_phase = rgb_phase + ycbcr_phase

        with ctx:
            out_rgb = torch.fft.irfft2(
                rgb_amp * torch.exp(1j * mix_phase), s=spatial_size, norm='backward'
            )
            out_ycbcr = torch.fft.irfft2(
                ycbcr_amp * torch.exp(1j * mix_phase), s=spatial_size, norm='backward'
            )
        return (out_rgb + out_ycbcr).to(orig_dtype)
