import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Gated Fusion Module (GFM)

论文：DNF: Decouple and Feedback Network for Seeing in the Dark (CVPR 2023)
链接：https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]；若首参为 list/tuple 则作为 inp_feats
- 门控逐点/深度卷积融合（GELU 门控），残差接到第一路特征
- 输出通道与 in_channels 一致
"""

__all__ = ['GFM']


class GFM(nn.Module):
    """Gated Fusion Module.

    Args:
        in_channels (int): 输入通道数（parse_model 注入）。
        feature_num (int): 融合特征路数，默认 2。
        bias (bool): 卷积偏置。
        padding_mode (str): 深度卷积填充模式，默认 reflect。
    """

    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect'):
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 3, 1, 1,
            bias=bias, padding_mode=padding_mode, groups=hidden_features * 2
        )
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def forward(self, x1, x2=None):
        if isinstance(x1, (list, tuple)):
            inp_feats = tuple(x1)
        elif x2 is not None:
            inp_feats = (x1, x2)
        else:
            inp_feats = (x1,)

        assert len(inp_feats) == self.feature_num, \
            f"GFM: 期望 {self.feature_num} 路输入, got {len(inp_feats)}"

        shortcut = inp_feats[0]
        x = torch.cat(inp_feats, dim=1)
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)
