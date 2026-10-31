"""
nn/mamba/ - Mamba / SSM 模块族

按依赖与机制分为三类:
- 纯 PyTorch 实现（无 mamba_ssm 依赖，开箱即用）
    - gated_cnn_block.GatedCNNBlock_BCHW  : MambaOut 系 GatedCNN 块（BCHW 兼容版）
    - mobilemamba.MobileMambaModule       : 基于 SS2D + 小波的轻量 Mamba 块（自带 torch fallback）
- mamba_ssm 依赖实现（缺包时容错：模块本身可 import，实例化时按需 lazy 导入）
    - mamba_vision.MambaVisionBlock       : MambaVision 混合块（CNN+SSM）
    - trans_mamba.TransMambaBlock         : Transformer + DRMamba 融合块
    - mamba_vss.VSSBlock / Mamba2Block    : VMamba 视觉状态空间块
    - evssm.EVSSM                          : Enhanced Visual SSM

注意：mamba_ssm 缺失时，以下 5 类不会被导出（占位为 None），但 import 不会崩溃；
gated_cnn_block 与 mobilemamba 始终可用。
"""

# ---- 纯 PyTorch（始终导出） ----
from .gated_cnn_block import GatedCNNBlock_BCHW

# ---- mobilemamba（自带 torch fallback；pywt 缺失时仅实例化失败） ----
try:
    from .mobilemamba.mobilemamba import MobileMambaModule
except ImportError:
    pass

# ---- mamba_ssm 依赖（容错导入；缺包则不导出该类） ----
try:
    from .mamba_vision import MambaVisionBlock
    from .trans_mamba import TransMambaBlock
    from .mamba_vss import VSSBlock, Mamba2Block
    from .evssm import EVSSM
except ImportError:
    pass
