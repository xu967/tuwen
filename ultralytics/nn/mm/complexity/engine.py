# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Forward-driven complexity engine for YOLOMM / YOLOVO multimodal models.

This engine derives everything from the model itself — no manual type tables:

1. Input semantics come from the multimodal router (INPUT_SOURCES) or entry
   layer markers, never inferred from first-layer parameters.
2. Tensor shapes are captured by forward hooks during one real eval() forward
   pass, so any module registered in tasks.py parse_model is supported
   automatically (arbitrary forward signatures included).
3. FLOPs are counted at the aten operator level via
   torch.utils.flop_counter.FlopCounterMode (zero per-module registration).
4. Stage classification is structural:
   - backbone: layer index < len(yaml['backbone'])
   - head:     graph sinks (layers whose output no other layer consumes)
   - neck:     everything else
"""

from __future__ import annotations

import torch

from ultralytics.utils import LOGGER

from .schema import (
    ComplexityInputSpec,
    ComplexityReport,
    NodeComplexity,
    RouteMode,
    TensorShapeSpec,
)


# ------------------------------------------------------------------
# Debug mode — reads from global default.yaml 'debug' parameter
# ------------------------------------------------------------------
def _is_debug() -> bool:
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT
        return DEFAULT_CFG_DICT.get("debug", False)
    except Exception:
        return False


def _debug_log(msg: str):
    """Print debug message if global debug mode is enabled."""
    if _is_debug():
        LOGGER.info(f"[ComplexityDebug] {msg}")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compute_multimodal_complexity_report(
    model,
    imgsz: int = 640,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityReport:
    """Compute forward-driven complexity report with explicit multimodal input semantics."""
    input_spec = build_complexity_input_spec(
        model=model,
        imgsz=imgsz,
        route_mode=route_mode,
        modality=modality,
    )
    return _run_forward_engine(model=model, input_spec=input_spec)


def compute_default_multimodal_complexity_report(model, imgsz: int = 640) -> ComplexityReport:
    """Compute the default-structure complexity report for YOLOMM mainline flows.

    This intentionally ignores runtime modality switches and reports the model's
    structural complexity truth source using the shared forward-driven engine.
    """
    return compute_multimodal_complexity_report(
        model=model,
        imgsz=imgsz,
        route_mode="dual",
        modality=None,
    )


def compute_pruning_complexity_report(model, imgsz: int = 640) -> ComplexityReport:
    """Compute pruning-stage complexity report using the shared multimodal engine."""
    return compute_default_multimodal_complexity_report(model=model, imgsz=imgsz)


# ------------------------------------------------------------------
# Input spec construction
# ------------------------------------------------------------------

def build_complexity_input_spec(
    model,
    graph=None,
    imgsz: int = 640,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityInputSpec:
    """Build explicit input spec from router or entry-layer metadata.

    Priority order:
    1. Read from multimodal_router.INPUT_SOURCES (most reliable)
    2. Fall back to entry-layer analysis (layers marked _mm_input_source == 'X')
    3. Default to single-modal RGB if no multimodal info

    Args:
        model: The model to analyze.
        graph: Deprecated, ignored (kept for signature compatibility).
        imgsz: Input image size.
        route_mode: Requested route mode, recorded into the spec.
        modality: Reserved for future routing-sensitive branching.
    """
    _ = graph, modality

    # Try to read from the real router
    router = getattr(model, "multimodal_router", None) or getattr(model, "mm_router", None)
    if router is not None and hasattr(router, "INPUT_SOURCES"):
        rgb_channels = int(router.INPUT_SOURCES.get("PRIMARY", 3))
        x_channels = int(router.INPUT_SOURCES.get("SECONDARY", 3))
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode=route_mode,
            rgb_channels=rgb_channels,
            x_channels=x_channels,
        )

    # Fall back to entry-layer analysis
    layers = _layer_list(model)
    dual_entry = None
    for layer in layers:
        source = getattr(layer, "_mm_input_source", None)
        if source in ("X", "SECONDARY"):
            x_channels = int(getattr(layer, "in_channels", 3) or 3)
            return ComplexityInputSpec(
                imgsz=(imgsz, imgsz),
                route_mode=route_mode,
                rgb_channels=3,
                x_channels=x_channels,
            )
        if source in ("Dual", "FUSED") and dual_entry is None:
            dual_entry = layer

    # Early fusion: single Dual entry consuming RGB+X concatenated
    if dual_entry is not None:
        total = int(getattr(dual_entry, "in_channels", 6) or 6)
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode=route_mode,
            rgb_channels=3,
            x_channels=max(total - 3, 0),
        )

    # YOLOVO: YAML `text:` is a cached embedding, not extra image channels
    yaml_cfg = getattr(model, "yaml", None)
    if isinstance(yaml_cfg, dict) and yaml_cfg.get("text") is not None:
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode="rgb",
            rgb_channels=3,
            x_channels=0,
        )

    # No X modality detected
    return ComplexityInputSpec(
        imgsz=(imgsz, imgsz),
        route_mode="rgb" if route_mode != "dual" else "dual",
        rgb_channels=3,
        x_channels=0,
    )


# ------------------------------------------------------------------
# Structural topology (no type tables)
# ------------------------------------------------------------------

def _layer_list(model) -> list:
    """Return the ordered top-level layer list (model.model or the model itself)."""
    layers_module = getattr(model, "model", model)
    return list(layers_module)


def _sink_indices(layers) -> set[int]:
    """Return indices of graph sinks: layers whose output no other layer consumes.

    Sinks are the model's heads by construction — every backbone/neck layer is
    referenced downstream, while heads terminate the graph. Multimodal fresh
    input entries (X branch starts) reuse f=-1 in YAML but semantically consume
    the original image, so their pseudo-reference is skipped (same rule as the
    pruning graph builder).
    """
    referenced: set[int] = set()
    for i, layer in enumerate(layers):
        if getattr(layer, "_mm_new_input_start", False):
            continue  # fresh image input: f=-1 is not a real edge
        f = getattr(layer, "f", None)
        if f is None:
            continue
        from_list = list(f) if isinstance(f, (list, tuple)) else [f]
        for rel in from_list:
            abs_idx = rel if rel >= 0 else i + rel
            if 0 <= abs_idx < len(layers) and abs_idx != i:
                referenced.add(abs_idx)
    return {i for i in range(len(layers)) if i not in referenced}


def _get_backbone_len(model) -> int:
    """Return the number of layers in the YAML backbone section.

    Uses model.yaml which preserves the original backbone/head split.
    Returns 0 with a warning if yaml is unavailable — Stages output
    will be unreliable in that case.
    """
    yaml_cfg = getattr(model, "yaml", None)
    if yaml_cfg and isinstance(yaml_cfg, dict) and "backbone" in yaml_cfg:
        return len(yaml_cfg["backbone"])
    LOGGER.warning("model.yaml missing or has no 'backbone' key — Stages breakdown skipped")
    return 0


def _yaml_text_indices(model) -> set[int]:
    """Return Sequential indices owned by YAML ``text:`` (section fence, not type names)."""
    yaml_cfg = getattr(model, "yaml", None)
    if isinstance(yaml_cfg, dict) and yaml_cfg.get("text"):
        bb = len(yaml_cfg.get("backbone") or [])
        return set(range(bb, bb + len(yaml_cfg["text"])))
    extra = getattr(model, "_text_layer_indices", None)
    return set(extra) if extra else set()


def _classify_stage(idx: int, sinks: set[int], backbone_len: int, text_indices: set[int] | None = None) -> str:
    """Classify a layer into its structural stage: backbone, neck, head, or text."""
    if text_indices and idx in text_indices:
        return "text"
    if idx in sinks:
        return "head"
    if idx < backbone_len:
        return "backbone"
    return "neck"


# ------------------------------------------------------------------
# Shape capture (forward hooks)
# ------------------------------------------------------------------

def _extract_shapes(obj) -> tuple[TensorShapeSpec, ...]:
    """Normalize a forward input/output object into TensorShapeSpec tuples.

    Handles Tensor / list / tuple / dict values. Only tensor-like entries are
    recorded; 4D tensors map to (C, H, W), 2D to (C, 1, 1), 3D token tensors
    [B, N, C] to (C, 1, N).
    """
    shapes: list[TensorShapeSpec] = []

    def _add(t):
        if not isinstance(t, torch.Tensor):
            return
        if t.dim() == 4:
            shapes.append(TensorShapeSpec(channels=int(t.shape[1]), height=int(t.shape[2]), width=int(t.shape[3])))
        elif t.dim() == 3:
            shapes.append(TensorShapeSpec(channels=int(t.shape[-1]), height=1, width=int(t.shape[1])))
        elif t.dim() == 2:
            shapes.append(TensorShapeSpec(channels=int(t.shape[1]), height=1, width=1))

    def _walk(o):
        if isinstance(o, torch.Tensor):
            _add(o)
        elif isinstance(o, (list, tuple)):
            for item in o:
                _walk(item)
        elif isinstance(o, dict):
            for item in o.values():
                _walk(item)

    _walk(obj)
    return tuple(shapes)


# ------------------------------------------------------------------
# FLOPs aggregation (FlopCounterMode)
# ------------------------------------------------------------------

def _aggregate_layer_flops(flop_counts: dict, model, layers) -> dict[int, float]:
    """Aggregate FlopCounterMode results per top-level layer index.

    FlopCounterMode keys modules as '<RootClassName>.<fqn>' (root itself is
    '<RootClassName>'). Values for a module already include its whole subtree.
    Both prefixed and bare fqn lookups are attempted for robustness.
    """
    root_name = type(model).__name__

    # Build fqn for each top-level layer via identity match
    layer_fqn: dict[int, str] = {}
    id_to_idx = {id(layer): i for i, layer in enumerate(layers)}
    for name, module in model.named_modules():
        idx = id_to_idx.get(id(module))
        if idx is not None and idx not in layer_fqn:
            layer_fqn[idx] = name

    result: dict[int, float] = {}
    for idx, fqn in layer_fqn.items():
        for key in (f"{root_name}.{fqn}", fqn):
            ops = flop_counts.get(key)
            if ops:
                result[idx] = float(sum(ops.values()))
                break
        else:
            result[idx] = 0.0
    return result


# ------------------------------------------------------------------
# Main engine
# ------------------------------------------------------------------

def _run_forward_engine(model, input_spec: ComplexityInputSpec) -> ComplexityReport:
    """Run one eval() forward pass under FlopCounterMode with shape hooks.

    Shapes and FLOPs are captured from the real execution — no per-module-type
    registration and no per-node input reconstruction are needed.
    """
    from torch.utils.flop_counter import FlopCounterMode

    layers = _layer_list(model)
    sinks = _sink_indices(layers)
    backbone_len = _get_backbone_len(model)
    text_indices = _yaml_text_indices(model)

    # Dummy input: the model's external contract is one concatenated tensor
    # [B, rgb+x, H, W]; the router distributes modalities internally.
    # YOLOVO uses RGB-only (x_channels=0) plus a cached text embedding.
    h, w = input_spec.imgsz
    total_ch = max(input_spec.rgb_channels + input_spec.x_channels, 1)
    try:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    if not torch.is_floating_point(torch.empty(0, dtype=dtype)):
        dtype = torch.float32

    # Register shape-capture hooks on top-level layers
    captured: dict[int, dict] = {}

    def _make_hook(idx: int):
        def _hook(module, args, output):
            entry = captured.setdefault(idx, {})
            entry["in"] = _extract_shapes(args)
            entry["out"] = _extract_shapes(output)
        return _hook

    handles = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(layers)]

    was_training = model.training
    counter = FlopCounterMode(display=False, depth=None)
    try:
        model.eval()
        # Validator/predictor entry points run under torch.inference_mode
        # (smart_inference_mode), which suppresses TorchDispatchMode dispatch
        # and would silently zero out every FLOPs count. Locally disable it;
        # this context is a no-op when inference mode is not active.
        with torch.inference_mode(mode=False):
            dummy = torch.zeros(1, total_ch, h, w, device=device, dtype=dtype)
            if hasattr(model, "ensure_dummy_text"):
                model.ensure_dummy_text()
            with counter, torch.no_grad():
                model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    layer_flops = _aggregate_layer_flops(counter.get_flop_counts(), model, layers)

    nodes: list[NodeComplexity] = []
    for i, layer in enumerate(layers):
        stage = _classify_stage(i, sinks, backbone_len, text_indices)
        shapes = captured.get(i, {})
        flops = layer_flops.get(i, 0.0)

        if _is_debug():
            _debug_log(
                f"Node {i} ({type(layer).__name__}, stage={stage}): "
                f"input={[(s.channels, s.height, s.width) for s in shapes.get('in', ())]} -> "
                f"output={[(s.channels, s.height, s.width) for s in shapes.get('out', ())]} "
                f"FLOPs={flops / 1e6:.2f}M"
            )

        nodes.append(
            NodeComplexity(
                node_idx=i,
                type_name=type(layer).__name__,
                stage=stage,
                input_shapes=shapes.get("in", ()),
                output_shapes=shapes.get("out", ()),
                flops=flops,
            )
        )

    return ComplexityReport(input_spec=input_spec, nodes=nodes)
