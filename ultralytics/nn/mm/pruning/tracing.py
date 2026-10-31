# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Operator-level tracing for the zero-registration pruning engine.

Records one (or two) real forward passes at the aten dispatch level via
TorchDispatchMode. Every aten call becomes an OpRecord whose tensor arguments
are resolved to provenance sources: parameters, buffers, external inputs, or
outputs of earlier records (at (record, slot) granularity).

Hard requirements baked in (from the red-team review):
- Strong references to every intermediate output (без keepalive, 21% of the
  records showed tensor id reuse corrupting provenance).
- (record, slot) provenance for multi-output ops (split, native_batch_norm).
- Buffers participate in the id map (BN running stats follow channels).
- Warmup forward before building the parameter map (lazily-built layers such
  as SequenceShuffleAttention / FCMFeatureFusion materialize on first call).
- eval + fp32 + no_grad + inference_mode(mode=False): inference mode changes
  the dispatched op set (conv2d vs convolution) and would break the rule table.
- Train-mode supplementary trace when eval leaves parameters untouched
  (training-only branches like SemSegHead.aux_head consume prunable features
  that the eval graph cannot see).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from ultralytics.utils import LOGGER

# nn layers whose parameters the executor can physically rewrite. Everything
# else is a "raw" parameter: it may appear in the graph but taints the groups
# it touches instead of registering prunable axes.
BASE_LAYER_TYPES = (
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.BatchNorm2d,
    nn.BatchNorm1d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.Linear,
)


@dataclass(frozen=True)
class Source:
    """Provenance of one tensor argument."""

    kind: str  # "record" | "param" | "buffer" | "external" | "unknown"
    record: int = -1
    slot: int = 0
    name: str = ""

    def __repr__(self):
        if self.kind == "record":
            return f"r{self.record}[{self.slot}]"
        if self.kind in ("param", "buffer"):
            return f"{self.kind}:{self.name}"
        return self.kind.upper()


@dataclass
class OpRecord:
    """One aten call captured during tracing."""

    idx: int
    op: str  # canonical, e.g. "convolution" (overload + trailing _ stripped)
    raw_op: str
    module: str  # innermost nn.Module path active at call time
    inputs: list  # list of (arg_pos, Source | list[Source], shape | list[shape])
    n_outputs: int
    output_shapes: list
    meta: list  # non-tensor positional args (json-safe best effort)
    inplace: bool = False
    mode: str = "eval"  # which forward pass produced it ("eval" | "train")


@dataclass
class TraceResult:
    """Everything downstream stages need."""

    records: list = field(default_factory=list)
    # base-layer axis registry: param/buffer fqn -> module fqn
    param_owner: dict = field(default_factory=dict)
    # fqn -> nn.Module type name for base layers ("Conv2d", ...)
    base_layer_type: dict = field(default_factory=dict)
    # module fqn -> layer object id (for executor lookup)
    base_layers: dict = field(default_factory=dict)
    # names of params/buffers seen in at least one record
    touched: set = field(default_factory=set)
    all_params: dict = field(default_factory=dict)  # fqn -> shape tuple
    all_buffers: dict = field(default_factory=dict)
    # modules where unknown-origin tensors appeared (pybind/CUDA ext fencing)
    fenced_modules: dict = field(default_factory=dict)  # module fqn -> reason
    # params never touched by any record even after the train pass
    untraced_params: set = field(default_factory=set)
    notes: list = field(default_factory=list)


def _canonical_op(func) -> tuple[str, bool]:
    """Return (canonical op name, is_inplace). 'aten.add_.Tensor' -> ('add', True)."""
    name = func._schema.name  # e.g. "aten::add_"
    name = name.split("::")[-1]
    inplace = name.endswith("_") and not name.endswith("__")
    if inplace:
        name = name[:-1]
    return name, inplace


def _jsonable(v):
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    return repr(v)


class _ModuleStack:
    """Tracks the innermost active nn.Module via pre/post forward hooks."""

    def __init__(self, model: nn.Module):
        self.stack: list[str] = []
        self.handles = []
        for name, mod in model.named_modules():
            self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self.handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(module, args):
            self.stack.append(name)
        return hook

    def _post(self, name):
        def hook(module, args, output):
            if self.stack and self.stack[-1] == name:
                self.stack.pop()
            elif name in self.stack:  # exception-safety: unwind to name
                while self.stack and self.stack.pop() != name:
                    pass
        return hook

    @property
    def current(self) -> str:
        return self.stack[-1] if self.stack else ""

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class ChannelTracer(TorchDispatchMode):
    """Records aten ops with full tensor provenance."""

    def __init__(self, id_map: dict, module_stack: _ModuleStack, mode_tag: str, idx_offset: int = 0):
        super().__init__()
        self.id_map = id_map  # id(tensor) -> Source
        self.module_stack = module_stack
        self.mode_tag = mode_tag
        self.idx_offset = idx_offset  # records get globally unique indices
        self.records: list[OpRecord] = []
        self.keepalive: list = []  # strong refs: id reuse is real (21% measured)
        self.touched: set = set()
        self.unknown_events: list = []  # (module, op)

    def _resolve(self, t: torch.Tensor) -> Source:
        src = self.id_map.get(id(t))
        if src is None:
            src = Source(kind="unknown")
            self.unknown_events.append((self.module_stack.current, "input"))
        elif src.kind in ("param", "buffer"):
            self.touched.add(src.name)
        return src

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        op, inplace = _canonical_op(func)

        inputs, meta = [], []
        for pos, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                inputs.append((pos, self._resolve(a), tuple(a.shape)))
            elif isinstance(a, (list, tuple)) and a and all(isinstance(x, torch.Tensor) for x in a):
                inputs.append((pos, [self._resolve(x) for x in a], [tuple(x.shape) for x in a]))
            else:
                meta.append((pos, _jsonable(a)))

        outs = out if isinstance(out, (list, tuple)) else (out,)
        out_shapes = [tuple(o.shape) if isinstance(o, torch.Tensor) else None for o in outs]

        rec = OpRecord(
            idx=self.idx_offset + len(self.records),
            op=op,
            raw_op=str(func),
            module=self.module_stack.current,
            inputs=inputs,
            n_outputs=len(outs),
            output_shapes=out_shapes,
            meta=meta,
            inplace=inplace,
            mode=self.mode_tag,
        )
        self.records.append(rec)

        for slot, o in enumerate(outs):
            if isinstance(o, torch.Tensor):
                # in-place ops overwrite provenance for the same id (ordered)
                self.id_map[id(o)] = Source(kind="record", record=rec.idx, slot=slot)
                self.keepalive.append(o)

        return out


def _build_id_map(model: nn.Module) -> tuple[dict, TraceResult]:
    """Register parameters and buffers into the id map; classify base layers."""
    result = TraceResult()
    id_map: dict[int, Source] = {}

    base_param_names: set = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, BASE_LAYER_TYPES):
            result.base_layer_type[mod_name] = type(mod).__name__
            result.base_layers[mod_name] = mod
            for p_name, _ in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                base_param_names.add(f"{mod_name}.{p_name}" if mod_name else p_name)

    for name, p in model.named_parameters():
        id_map[id(p)] = Source(kind="param", name=name)
        result.all_params[name] = tuple(p.shape)
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        result.param_owner[name] = owner
    for name, b in model.named_buffers():
        id_map[id(b)] = Source(kind="buffer", name=name)
        result.all_buffers[name] = tuple(b.shape)
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        result.param_owner[name] = owner

    result.notes.append(f"base_layers={len(result.base_layer_type)} params={len(result.all_params)}")
    return id_map, result


def _snapshot_buffers(model: nn.Module) -> dict:
    return {name: b.detach().clone() for name, b in model.named_buffers()}


def _restore_buffers(model: nn.Module, snap: dict):
    for name, b in model.named_buffers():
        if name in snap:
            b.detach().copy_(snap[name])


def trace_model(model: nn.Module, example: torch.Tensor) -> TraceResult:
    """Run the full dual-mode trace and return a TraceResult.

    The model is temporarily set to eval/fp32; original train/eval mode is
    restored. Buffers are snapshotted around the train-mode pass so BN running
    stats and warm-up counters are unaffected.
    """
    was_training = model.training
    model.eval()
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("Pruning trace requires an fp32 model; call model.float() first.")

    # 1. Warmup: materialize lazily-built layers before building the id map.
    with torch.inference_mode(mode=False), torch.no_grad():
        model(example)

    # 2. Build param/buffer registry AFTER warmup.
    id_map, result = _build_id_map(model)
    id_map[id(example)] = Source(kind="external", name="input")

    stack = _ModuleStack(model)
    tracer = ChannelTracer(id_map, stack, mode_tag="eval")
    try:
        # 3. Eval trace. inference_mode must stay disabled: it changes the
        #    dispatched op set and would evade the rule table entirely.
        with torch.inference_mode(mode=False), torch.no_grad():
            with tracer:
                model(example)

        result.records = list(tracer.records)
        result.touched = set(tracer.touched)

        # 4. Train-mode supplementary trace if params were left untouched
        #    (training-only branches consume prunable channels invisibly).
        untouched = {
            n for n in list(result.all_params) + list(result.all_buffers)
            if n not in result.touched and not n.endswith("num_batches_tracked")
        }
        if untouched:
            snap = _snapshot_buffers(model)
            model.train()
            # idx_offset makes record indices globally unique so Source.record
            # references written into the shared id_map stay unambiguous.
            train_tracer = ChannelTracer(id_map, stack, mode_tag="train", idx_offset=len(result.records))
            try:
                with torch.inference_mode(mode=False), torch.no_grad():
                    with train_tracer:
                        model(example)
                result.records.extend(train_tracer.records)
                result.touched |= train_tracer.touched
                result.notes.append(f"train_pass=+{len(train_tracer.records)} records")
                tracer.unknown_events.extend(train_tracer.unknown_events)
                # keep the train keepalive too until grouping is done
                tracer.keepalive.extend(train_tracer.keepalive)
            except Exception as e:
                for name in untouched:
                    owner = result.param_owner.get(name, "")
                    result.fenced_modules.setdefault(owner, f"train-pass failed: {type(e).__name__}")
                result.notes.append(f"train_pass_failed: {e}")
            finally:
                model.eval()
                _restore_buffers(model, snap)

        # 5. Final untraced set. Module fencing triggers ONLY on parameters the
        #    trace never touched (pybind/CUDA extensions consume params outside
        #    the dispatcher). Unknown-origin tensors consumed by known ops are
        #    handled by the rule layer (no-provenance => taint the consuming
        #    axis), which is sufficiently conservative without locking the
        #    whole module (e.g. Detect's cached anchors tensors).
        result.untraced_params = {
            n for n in list(result.all_params) + list(result.all_buffers)
            if n not in result.touched and not n.endswith("num_batches_tracked")
        }
        for name in result.untraced_params:
            owner = result.param_owner.get(name, "")
            result.fenced_modules.setdefault(owner, "parameter never traced")
        if tracer.unknown_events:
            mods = {m for m, _ in tracer.unknown_events}
            result.notes.append(f"unknown_origin_modules={sorted(mods)[:8]} (handled by rule-level taint)")

        # keepalive must survive until semantics replay: hand it over.
        result.notes.append(f"records={len(result.records)} fenced={len(result.fenced_modules)}")
        result._keepalive = tracer.keepalive  # noqa: SLF001 - intentional lifetime extension
        return result
    finally:
        stack.remove()
        if was_training:
            model.train()
