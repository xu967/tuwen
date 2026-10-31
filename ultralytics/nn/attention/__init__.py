"""
nn/attention/ - 注意力模块族

按机制分为4个子族:
- channel/   : 通道注意力
- frequency/ : 频率注意力
- window/    : 窗口注意力
- global_/   : 全局注意力
"""

# ---- channel ----
try:
    from .channel import ACA, CASAB, MCA, SimAM
    from .channel import BinaryAttention, CascadedGroupAtt, CascadedGroupAttention
    from .channel import CBSA, MaskUnitAttention, DHPF
    from .channel import ChannelTransformerAttention, FCAttention, ChannelReductionAttention
    from .channel import SMSSA, PCSA
except ImportError:
    pass

# ---- frequency ----
try:
    from .frequency import ContrastDrivenFeatureAggregation, HaarWaveletConv
    from .frequency import CTA, ChannelProjection, SpatialProjection
    from .frequency import SFA, FrequencyProjection
    from .frequency import FSSA, CPIA_SA, KSFA
    from .frequency import FSA, Adaptive_global_filter, FSAS
    from .frequency import MFCA
except ImportError:
    pass

# ---- window ----
try:
    from .window import BiLevelRoutingAttention, DilatedGCSA, DilatedMWSA
    from .window import DPWA, DWM_MSA, DHOGSA
    from .window import PatchSA, Token_Selective_Attention
except ImportError:
    pass

# ---- global ----
try:
    from .global_ import EfficientGlobalSA, LRSA, MALA, RoPE, TAB
    from .global_ import GLSA, ContextBlock, LWGA
    from .global_ import CFBlock, ConvolutionalAttention
    from .global_ import EASA, CAFM, DynamicRangeHistogramSelfAttention
    from .global_ import SHSA, LSKNetAttention, MLKA
    from .global_ import HDPA, ACFM, ESSA
except ImportError:
    pass
