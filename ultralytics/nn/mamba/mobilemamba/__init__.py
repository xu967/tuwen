"""
nn/mamba/mobilemamba/ - MobileMamba 子族

基于 SS2D (vmambanew) 与小波卷积 (MBWTConv2d, 依赖 pywt) 的轻量 Mamba 模块。
- vmambanew.py : SS2D 核心（selective_scan 双路径：triton/torch fallback）
- csms6s.py   : selective_scan 接口（cuda_oflex/core/mamba 三选一 + torch fallback）
- csm_triton.py / csm_tritonk2.py : Triton cross_scan/cross_merge kernel
- mobilemamba.py : MobileMambaModule / MobileMambaBlock 及系列模型
"""

# 默认导出 MobileMambaModule（注册入口）；其余模型类按需懒加载
try:
    from .mobilemamba import MobileMambaModule
except ImportError:
    pass
