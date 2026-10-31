# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Forward-driven complexity analysis for YOLOMM / YOLOVO multimodal models.

Shapes and FLOPs come from one real eval() forward under FlopCounterMode.
Stage breakdown is structural (backbone / neck / head / YAML text:).
"""

from .engine import (
    build_complexity_input_spec,
    compute_default_multimodal_complexity_report,
    compute_multimodal_complexity_report,
    compute_pruning_complexity_report,
)
from .report import (
    build_default_complexity_summary,
    format_default_complexity_lines,
    get_model_param_count,
    log_default_complexity,
)
from .schema import (
    ComplexityInputSpec,
    ComplexityReport,
    NodeComplexity,
    TensorShapeSpec,
)

__all__ = [
    "ComplexityInputSpec",
    "ComplexityReport",
    "NodeComplexity",
    "TensorShapeSpec",
    "build_complexity_input_spec",
    "build_default_complexity_summary",
    "compute_default_multimodal_complexity_report",
    "compute_multimodal_complexity_report",
    "compute_pruning_complexity_report",
    "format_default_complexity_lines",
    "get_model_param_count",
    "log_default_complexity",
]
