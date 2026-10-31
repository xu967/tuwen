# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Automatic FLOPs calculation for YOLOMM complexity engine.

Replaces the manual RULES registry with per-node automatic FLOPs counting.
Two strategies:
1. thop per-node profiling (when thop is available)
2. Recursive Conv2d + Linear sub-module counting (always-available fallback)

Any module registered in tasks.py parse_model is automatically supported.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .schema import TensorShapeSpec

# ------------------------------------------------------------------
# Constants (mirrored from graph.py to avoid circular imports)
# ------------------------------------------------------------------

_ROUTE_ONLY_TYPES = frozenset({"Concat", "Upsample", "Index"})
_HEAD_TYPES = frozenset({"Detect", "Segment", "Pose", "OBB", "Classification", "YOLOVODetect", "YOLOVOSegment"})

# Check thop availability
_THOP_AVAILABLE = False
try:
    import thop
    _THOP_AVAILABLE = True
except ImportError:
    pass


# ------------------------------------------------------------------
# Core: automatic FLOPs computation
# ------------------------------------------------------------------

def auto_compute_flops(node, input_shapes, output_shapes, device=None):
    """Compute FLOPs for a single node automatically.

    Strategy (in order of priority):
    1. route_only -> 0.0
    2. thop available -> per-node thop profiling
    3. head fallback -> multi-scale branch traversal
    4. fallback -> recursive Conv2d + Linear sub-module counting

    Args:
        node: PruneNode instance.
        input_shapes: Tuple of input TensorShapeSpec.
        output_shapes: Tuple of output TensorShapeSpec.
        device: Optional torch.device override.

    Returns:
        FLOPs as float.
    """
    ltype = node.type_name

    # 1. Zero-FLOPs modules
    if node.is_route_only or ltype in _ROUTE_ONLY_TYPES:
        return 0.0

    if not input_shapes or not output_shapes:
        return 0.0

    if device is None:
        try:
            device = next(node.module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # 2. Try thop per-node profiling
    if _THOP_AVAILABLE:
        result = _thop_node_flops(node, input_shapes, device)
        if result is not None:
            return result

    # 3. Head fallback: keep pruning "non-prunable" semantics separate from FLOPs accounting.
    if node.is_head or ltype in _HEAD_TYPES:
        return _recursive_head_flops(node, input_shapes, output_shapes)

    # 4. Fallback: recursive sub-module counting
    return _recursive_flops(node.module, output_shapes[0])


# ------------------------------------------------------------------
# thop per-node profiling
# ------------------------------------------------------------------

def _thop_node_flops(node, input_shapes, device):
    """Use thop to profile a single node's FLOPs.

    Returns:
        FLOPs as float on success, None on failure.
    """
    src = input_shapes[0]
    c, h, w = src.channels, src.height, src.width

    dummy = torch.zeros(1, c, h, w, device=device)

    try:
        m_copy = copy.deepcopy(node.module)
        m_copy.eval()

        # Single-input modules
        if len(input_shapes) == 1:
            flops = thop.profile(m_copy, inputs=(dummy,), verbose=False)[0]
        else:
            # Multi-input: wrap as list input
            dummies = [
                torch.zeros(1, s.channels, s.height, s.width, device=device)
                for s in input_shapes
            ]
            wrapper = _ListInputWrapper(m_copy)
            flops = thop.profile(wrapper, inputs=tuple(dummies), verbose=False)[0]

        # thop returns MACs (multiply-accumulate), convert to arithmetic FLOPs
        return float(flops * 2)

    except Exception:
        return None


class _ListInputWrapper(nn.Module):
    """Wrapper to convert multi-tensor args to list input for modules like Concat."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, *inputs):
        return self.base(list(inputs))


# ------------------------------------------------------------------
# Head fallback FLOPs counting
# ------------------------------------------------------------------

def _as_pair(value):
    """Return a scalar or pair-like value as a 2-tuple."""
    if isinstance(value, (tuple, list)):
        return value
    return value, value


def _conv_output_shape(in_shape, conv):
    """Compute Conv2d output shape for fallback branch traversal."""
    kh, kw = _as_pair(conv.kernel_size)
    sh, sw = _as_pair(conv.stride)
    ph, pw = _as_pair(conv.padding)
    dh, dw = _as_pair(conv.dilation)

    h_out = (in_shape.height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (in_shape.width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return TensorShapeSpec(channels=conv.out_channels, height=int(h_out), width=int(w_out))


def _conv2d_flops(conv, in_shape, out_shape):
    """Count arithmetic FLOPs for a Conv2d layer."""
    groups = max(int(conv.groups), 1)
    kh, kw = _as_pair(conv.kernel_size)
    return float(
        2
        * out_shape.height
        * out_shape.width
        * conv.out_channels
        * (conv.in_channels // groups)
        * kh
        * kw
    )


def _sequential_flops(module, in_shape):
    """Traverse sequential head branches while updating spatial/channel shape per child."""
    total = 0.0
    current_shape = in_shape

    if isinstance(module, (tuple, list, nn.Sequential, nn.ModuleList)):
        for sub_module in module:
            sub_flops, current_shape = _sequential_flops(sub_module, current_shape)
            total += sub_flops
        return total, current_shape

    conv = getattr(module, "conv", None)
    if isinstance(conv, nn.Conv2d):
        out_shape = _conv_output_shape(current_shape, conv)
        return _conv2d_flops(conv, current_shape, out_shape), out_shape

    if isinstance(module, nn.Conv2d):
        out_shape = _conv_output_shape(current_shape, module)
        return _conv2d_flops(module, current_shape, out_shape), out_shape

    if isinstance(module, nn.Linear):
        tokens = current_shape.height * current_shape.width
        return float(2 * tokens * module.in_features * module.out_features), current_shape

    children = list(module.children())
    if children:
        for child in children:
            sub_flops, current_shape = _sequential_flops(child, current_shape)
            total += sub_flops
        return total, current_shape

    return 0.0, current_shape


def _recursive_head_flops(node, input_shapes, output_shapes):
    """Fallback count for head modules when thop is unavailable or cannot profile the head."""
    module = node.module
    total = 0.0

    # Common YOLO heads store per-scale branches in ModuleLists.
    for attr in ("cv2", "cv3", "cv4", "cv5", "cv6", "kpt_conv"):
        branches = getattr(module, attr, None)
        if isinstance(branches, (tuple, list, nn.ModuleList)):
            for i, src in enumerate(input_shapes):
                if i < len(branches):
                    flops, _ = _sequential_flops(branches[i], src)
                    total += flops
        elif branches is not None and input_shapes:
            flops, _ = _sequential_flops(branches, input_shapes[0])
            total += flops

    # Segmentation proto or classifier-like heads are not always per-scale.
    proto = getattr(module, "proto", None)
    if proto is not None and input_shapes:
        flops, _ = _sequential_flops(proto, input_shapes[0])
        total += flops

    if total > 0:
        return total

    return _recursive_flops(module, output_shapes[0])


# ------------------------------------------------------------------
# Recursive sub-module FLOPs counting (always-available fallback)
# ------------------------------------------------------------------

def _recursive_flops(module, out_shape):
    """Count FLOPs by traversing all Conv2d and Linear sub-modules.

    This is a generalization of the per-type rules in the old rules.py.
    Instead of matching module type names, it directly inspects sub-modules.

    Args:
        module: The nn.Module to analyze.
        out_shape: TensorShapeSpec of the module's output.

    Returns:
        Total FLOPs as float.
    """
    total = 0.0
    h_out, w_out = out_shape.height, out_shape.width

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            groups = max(int(m.groups), 1)
            total += float(
                2
                * h_out
                * w_out
                * m.out_channels
                * (m.in_channels // groups)
                * m.kernel_size[0]
                * m.kernel_size[1]
            )
        elif isinstance(m, nn.Linear):
            # For Linear layers in conv models, spatial dims = number of tokens
            tokens = h_out * w_out
            total += float(2 * tokens * m.in_features * m.out_features)

    return total
