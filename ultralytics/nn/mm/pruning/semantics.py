# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Per-channel index-map propagation over the traced aten op stream.

Each tensor in the replay carries a dims tuple: one entry per tensor dim,
either None (no channel semantics: batch/spatial/token) or an int64 ndarray of
global channel-element ids (length == dim size). Rules translate every aten op
into element unions (coupling), provenance slicing/concatenation (index
transformation) or taint (conservative refusal).

Safety semantics (from the boundary-hunter review):
- Unknown op => taint ALL provenance touched by its inputs; outputs carry no
  provenance ("skip" would silently mis-prune, taint is the dividing line).
- Folding reshape (a provenance dim merged/split with другой dim) => taint.
- Raw (non-base-layer) parameter or buffer entering an op => taint the other
  inputs' provenance (A2C2f.gamma, tgt_embed, TAB.means, wavelet buffers...).
- A None provenance consumed by a parameter axis (conv input on external or
  unknown tensors) => taint that axis; this also protects stem in-channels.
"""

from __future__ import annotations

import numpy as np

from .grouping import RAW, ElementRegistry
from .tracing import OpRecord, Source, TraceResult

_MAXINT = 2**63 - 1


# ------------------------------------------------------------------
# Op sets
# ------------------------------------------------------------------

# unary / shape-preserving (per-dim size-aware passthrough)
PASSTHROUGH = {
    "silu", "relu", "relu6", "gelu", "sigmoid", "hardsigmoid", "tanh", "hardswish",
    "hardtanh", "leaky_relu", "elu", "mish", "softplus", "erf", "exp", "log", "log2",
    "sqrt", "rsqrt", "abs", "neg", "reciprocal", "clamp", "clamp_min", "clamp_max",
    "clone", "detach", "contiguous", "_to_copy", "lift_fresh", "alias", "pow",
    "dropout", "native_dropout", "_softmax", "softmax", "_log_softmax", "log_softmax",
    "sin", "cos", "floor", "round", "ceil", "sign", "sigmoid_backward", "tril", "triu",
    "masked_fill", "index_put", "bitwise_not", "logical_not", "nan_to_num",
    "max_pool2d", "max_pool2d_with_indices", "avg_pool2d", "adaptive_avg_pool2d",
    "adaptive_max_pool2d", "adaptive_max_pool2d_with_indices",
    "upsample_nearest2d", "upsample_bilinear2d", "upsample_bicubic2d",
    "constant_pad_nd", "reflection_pad2d", "replication_pad2d", "pixel_unshuffle",
    "flip", "roll", "type_as", "to",
}

# binary elementwise (broadcast-aware union)
ELEMENTWISE = {
    "add", "sub", "mul", "div", "maximum", "minimum", "atan2", "fmod", "remainder",
    "ge", "gt", "le", "lt", "eq", "ne", "logical_and", "logical_or", "rsub",
    "floor_divide", "true_divide",
}

# reductions with a standard dim argument at position 1
REDUCE = {"mean", "sum", "amax", "amin", "prod", "max", "min"}

# ops whose outputs are index-like / provenance-free but harmless
NO_PROV_OUT = {
    "arange", "zeros", "ones", "full", "empty", "empty_like", "zeros_like",
    "ones_like", "full_like", "rand", "randn", "randperm", "scalar_tensor",
    "argmax", "argmin", "topk", "sort", "argsort", "nonzero", "unique",
    "cumsum", "meshgrid", "repeat_interleave", "one_hot", "embedding",
    "uniform", "normal", "bernoulli", "fill", "zero", "new_zeros", "new_ones",
    "new_full", "new_empty", "index", "gather", "im2col", "col2im", "grid_sampler_2d",
    # statistics whose arg layout differs from the standard reduce signature
    "norm", "var", "std", "linalg_vector_norm", "var_mean", "std_mean",
}


# ------------------------------------------------------------------
# Propagator
# ------------------------------------------------------------------

class Propagator:
    """Replays the record stream, driving unions/taints on the registry."""

    def __init__(self, trace: TraceResult, registry: ElementRegistry):
        self.trace = trace
        self.reg = registry
        self.prov: dict[tuple, tuple] = {}  # ("r", idx, slot) -> dims tuple
        self.conservative_log: list[dict] = []

    # -------------------------------------------------- provenance lookup
    def dims_of(self, src: Source, shape: tuple) -> object:
        """Return dims tuple, RAW, or all-None tuple."""
        if src.kind == "record":
            return self.prov.get(("r", src.record, src.slot), tuple(None for _ in shape))
        if src.kind in ("param", "buffer"):
            d = self.reg.param_dims.get(src.name, RAW)
            if d == RAW:
                return RAW
            if len(d) != len(shape):  # shape drift guard
                return tuple(None for _ in shape)
            return d
        return tuple(None for _ in shape)  # external / unknown

    def set_out(self, rec: OpRecord, slot: int, dims):
        if dims is not None:
            self.prov[("r", rec.idx, slot)] = tuple(dims)

    def none_dims(self, shape):
        return tuple(None for _ in shape)

    # -------------------------------------------------- taint helpers
    def taint_dims(self, dims, reason: str):
        if dims == RAW or dims is None:
            return
        for d in dims:
            if isinstance(d, np.ndarray) and len(d):
                self.reg.taint(d, reason)

    def conservative(self, rec: OpRecord, reason: str):
        """Unknown/unsafe op: taint every input provenance, no output prov."""
        touched = False
        for _, src, shape in _iter_tensor_inputs(rec):
            dims = self.dims_of(src, shape)
            if dims == RAW:
                continue
            for d in dims:
                if isinstance(d, np.ndarray) and len(d):
                    self.reg.taint(d, f"{reason} @ {rec.module or 'top'}:{rec.op}")
                    touched = True
        if touched:
            self.conservative_log.append(
                {"op": rec.op, "module": rec.module, "reason": reason, "record": rec.idx}
            )

    # -------------------------------------------------- main loop
    def run(self):
        for rec in self.trace.records:
            try:
                self._dispatch(rec)
            except Exception as e:  # rule bug => conservative, never crash
                self.conservative(rec, f"rule error {type(e).__name__}: {e}")
                for slot, shp in enumerate(rec.output_shapes):
                    if shp is not None:
                        self.set_out(rec, slot, self.none_dims(shp))

        # module fencing: taint every param of fenced modules
        for mod, why in self.trace.fenced_modules.items():
            self._taint_module_params(mod, f"module fenced: {why}")

        # Folding fence: modules where a folding reshape or unknown op occurred
        # read shapes dynamically in ways the trace records as constants
        # (Attention's `view(B, C, H, W)` reads the runtime input width). The
        # only safe automatic answer is to freeze every parameter of such a
        # module — its channel groups become untouchable, which matches the
        # legacy engine's "never self-prune C2PSA/SSA" semantics.
        folding_modules = {
            item["module"] for item in self.conservative_log
            if item["module"]  # top-level ("") events cannot be attributed; skip
        }
        for mod in folding_modules:
            self._taint_module_params(mod, "folding/unknown op inside module (dynamic-shape coupling guard)")
        return self

    def _taint_module_params(self, mod: str, reason: str):
        for name in list(self.trace.all_params) + list(self.trace.all_buffers):
            owner = self.trace.param_owner.get(name, "")
            if mod == "":
                match = owner == ""
            else:
                match = owner == mod or owner.startswith(mod + ".")
            if match:
                self.reg.taint_param(name, reason)

    # -------------------------------------------------- dispatch
    def _dispatch(self, rec: OpRecord):
        tensor_inputs = list(_iter_tensor_inputs(rec))

        # RAW parameter / buffer touching an op => taint siblings, no output prov
        raw_present = any(
            self.dims_of(src, shape) == RAW
            for _, src, shape in tensor_inputs
            if src.kind in ("param", "buffer")
        )
        if raw_present and rec.op not in ("_to_copy", "clone", "detach", "lift_fresh", "alias", "t", "view", "expand", "unsqueeze", "reshape", "embedding"):
            for _, src, shape in tensor_inputs:
                dims = self.dims_of(src, shape)
                if dims != RAW:
                    self.taint_dims(dims, f"raw parameter in {rec.op} @ {rec.module}")
            for slot, shp in enumerate(rec.output_shapes):
                if shp is not None:
                    self.set_out(rec, slot, self.none_dims(shp))
            return

        fn = OP_RULES.get(rec.op)
        if fn is not None:
            fn(self, rec)
        elif rec.op in PASSTHROUGH:
            _rule_passthrough(self, rec)
        elif rec.op in ELEMENTWISE:
            _rule_elementwise(self, rec)
        elif rec.op in REDUCE:
            _rule_reduce(self, rec)
        elif rec.op in NO_PROV_OUT:
            for slot, shp in enumerate(rec.output_shapes):
                if shp is not None:
                    self.set_out(rec, slot, self.none_dims(shp))
        else:
            self.conservative(rec, "no rule for op")
            for slot, shp in enumerate(rec.output_shapes):
                if shp is not None:
                    self.set_out(rec, slot, self.none_dims(shp))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _iter_tensor_inputs(rec: OpRecord):
    """Yield (pos, Source, shape) flattening tensor-list args."""
    for pos, src, shape in rec.inputs:
        if isinstance(src, list):
            for s, sh in zip(src, shape):
                yield pos, s, sh
        else:
            yield pos, src, shape


def _arg(rec: OpRecord, pos: int, default=None):
    """Positional lookup across tensor inputs and meta."""
    for p, src, shape in rec.inputs:
        if p == pos:
            return ("tensor", src, shape)
    for p, v in rec.meta:
        if p == pos:
            return ("meta", v, None)
    return ("missing", default, None)


def _tensor_at(rec: OpRecord, pos: int):
    kind, a, b = _arg(rec, pos)
    if kind == "tensor":
        return a, b
    return None, None


def _meta_at(rec: OpRecord, pos: int, default=None):
    kind, v, _ = _arg(rec, pos, default)
    return v if kind == "meta" else default


def _norm_dim(d: int, rank: int) -> int:
    return d + rank if d < 0 else d


# ------------------------------------------------------------------
# Generic rules
# ------------------------------------------------------------------

def _rule_passthrough(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        for slot, shp in enumerate(rec.output_shapes):
            if shp is not None:
                ctx.set_out(rec, slot, ctx.none_dims(shp))
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    out_shape = rec.output_shapes[0]
    if out_shape is None:
        return
    if len(out_shape) == len(shape):
        out = tuple(
            d if isinstance(d, np.ndarray) and len(d) == out_shape[i] else None
            for i, d in enumerate(dims)
        )
    else:
        out = ctx.none_dims(out_shape)
    ctx.set_out(rec, 0, out)
    for slot in range(1, rec.n_outputs):
        shp = rec.output_shapes[slot]
        if shp is not None:
            ctx.set_out(rec, slot, ctx.none_dims(shp))


def _rule_elementwise(ctx: Propagator, rec: OpRecord):
    tensors = [(src, shape) for _, src, shape in _iter_tensor_inputs(rec)]
    out_shape = rec.output_shapes[0]
    if out_shape is None:
        return
    if len(tensors) == 1:  # scalar variant
        _rule_passthrough(ctx, rec)
        return

    (sa, sha), (sb, shb) = tensors[0], tensors[1]
    da, db = ctx.dims_of(sa, sha), ctx.dims_of(sb, shb)
    if da == RAW or db == RAW:  # handled earlier normally; be safe
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        return

    rank = len(out_shape)
    pa = _broadcast_dims(da, sha, rank)
    pb = _broadcast_dims(db, shb, rank)
    out = []
    for i in range(rank):
        ea, eb = pa[i], pb[i]
        size = out_shape[i]
        if isinstance(ea, np.ndarray) and isinstance(eb, np.ndarray):
            if len(ea) == len(eb) == size:
                ctx.reg.union_arrays(ea, eb)
                out.append(ea)
            else:  # broadcast of provenance dims of different size
                ctx.reg.taint(ea, f"broadcast width clash in {rec.op}")
                ctx.reg.taint(eb, f"broadcast width clash in {rec.op}")
                out.append(None)
        elif isinstance(ea, np.ndarray):
            if len(ea) == size and (pb[i] is None and _bdim_size(shb, i, rank) == size):
                # equal-width unknown side flowing in => conservative
                ctx.reg.taint(ea, f"equal-width no-provenance operand in {rec.op}")
                out.append(None)
            elif len(ea) == size:
                out.append(ea)  # broadcast against size-1
            else:
                out.append(None)
        elif isinstance(eb, np.ndarray):
            if len(eb) == size and (pa[i] is None and _bdim_size(sha, i, rank) == size):
                ctx.reg.taint(eb, f"equal-width no-provenance operand in {rec.op}")
                out.append(None)
            elif len(eb) == size:
                out.append(eb)
            else:
                out.append(None)
        else:
            out.append(None)
    ctx.set_out(rec, 0, tuple(out))


def _bdim_size(shape, i, rank):
    """Size of shape's dim aligned at broadcast position i (right-aligned)."""
    off = rank - len(shape)
    j = i - off
    return shape[j] if 0 <= j < len(shape) else 1


def _broadcast_dims(dims, shape, rank):
    """Right-align dims to the broadcast rank."""
    off = rank - len(shape)
    out = [None] * rank
    for j, d in enumerate(dims):
        out[off + j] = d if isinstance(d, np.ndarray) else None
    return out


def _rule_reduce(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    out_shape = rec.output_shapes[0]
    if src is None or out_shape is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    red = _meta_at(rec, 1, None)
    if isinstance(red, int):
        red = [red]
    if not isinstance(red, (list, tuple)):
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        for slot in range(1, rec.n_outputs):
            if rec.output_shapes[slot] is not None:
                ctx.set_out(rec, slot, ctx.none_dims(rec.output_shapes[slot]))
        return
    red = {_norm_dim(int(d), len(shape)) for d in red}
    keepdim = len(out_shape) == len(shape)
    out = []
    for i, d in enumerate(dims):
        if i in red:
            if keepdim:
                out.append(None)
        else:
            out.append(d if isinstance(d, np.ndarray) else None)
    out = out[: len(out_shape)] if len(out) >= len(out_shape) else ctx.none_dims(out_shape)
    ctx.set_out(rec, 0, tuple(out) if len(out) == len(out_shape) else ctx.none_dims(out_shape))
    for slot in range(1, rec.n_outputs):
        if rec.output_shapes[slot] is not None:
            ctx.set_out(rec, slot, ctx.none_dims(rec.output_shapes[slot]))


# ------------------------------------------------------------------
# Specific rules
# ------------------------------------------------------------------

def _rule_convolution(ctx: Propagator, rec: OpRecord):
    """aten.convolution(input, weight, bias, stride, padding, dilation, transposed, out_pad, groups)."""
    in_src, in_shape = _tensor_at(rec, 0)
    w_src, w_shape = _tensor_at(rec, 1)
    transposed = bool(_meta_at(rec, 6, False))
    groups = int(_meta_at(rec, 8, 1) or 1)
    out_shape = rec.output_shapes[0]

    in_dims = ctx.dims_of(in_src, in_shape)
    w_dims = ctx.dims_of(w_src, w_shape)

    if w_dims == RAW or not isinstance(w_dims, tuple):
        # functional conv with a non-registered weight (HaarWavelet etc.)
        if isinstance(in_dims, tuple):
            ctx.taint_dims(in_dims, f"functional conv with raw weight @ {rec.module}")
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        return

    in_ch_prov = in_dims[1] if isinstance(in_dims, tuple) and len(in_dims) > 1 else None
    in_ch_size = in_shape[1]

    if not transposed:
        w_out, w_in = w_dims[0], w_dims[1]
        out_ch_prov = w_out
        if groups == 1:
            if isinstance(in_ch_prov, np.ndarray) and isinstance(w_in, np.ndarray) and len(in_ch_prov) == len(w_in):
                ctx.reg.union_arrays(w_in, in_ch_prov)
            else:
                # external / unknown / tainted input => the in-axis must not
                # be pruned independently of its (invisible) producer
                if isinstance(w_in, np.ndarray):
                    ctx.reg.taint(w_in, f"conv input without provenance @ {rec.module}")
        elif groups == in_ch_size and w_shape[1] == 1 and out_shape[1] == in_ch_size:
            # depthwise: in/out coupled elementwise
            if isinstance(in_ch_prov, np.ndarray) and isinstance(w_out, np.ndarray) and len(in_ch_prov) == len(w_out):
                ctx.reg.union_arrays(w_out, in_ch_prov)
            elif isinstance(w_out, np.ndarray):
                ctx.reg.taint(w_out, f"depthwise conv input without provenance @ {rec.module}")
            if isinstance(w_dims[1], np.ndarray):
                pass  # single-position axis, no coupling needed
        else:
            # general grouped conv: conservative (equal-per-group quota not yet modeled)
            if isinstance(in_ch_prov, np.ndarray):
                ctx.reg.taint(in_ch_prov, f"grouped conv (g={groups}) @ {rec.module}")
            if isinstance(w_out, np.ndarray):
                ctx.reg.taint(w_out, f"grouped conv (g={groups}) @ {rec.module}")
            if isinstance(w_in, np.ndarray):
                ctx.reg.taint(w_in, f"grouped conv (g={groups}) @ {rec.module}")
            out_ch_prov = None
    else:
        # ConvTranspose: weight (in, out/g, kH, kW)
        w_in, w_out = w_dims[0], w_dims[1]
        if groups == 1:
            if isinstance(in_ch_prov, np.ndarray) and isinstance(w_in, np.ndarray) and len(in_ch_prov) == len(w_in):
                ctx.reg.union_arrays(w_in, in_ch_prov)
            elif isinstance(w_in, np.ndarray):
                ctx.reg.taint(w_in, f"convT input without provenance @ {rec.module}")
            out_ch_prov = w_out
        else:
            for d in (w_in, w_out, in_ch_prov):
                if isinstance(d, np.ndarray):
                    ctx.reg.taint(d, f"grouped convT (g={groups}) @ {rec.module}")
            out_ch_prov = None

    out = [None] * len(out_shape)
    if isinstance(out_ch_prov, np.ndarray) and len(out_ch_prov) == out_shape[1]:
        out[1] = out_ch_prov
    ctx.set_out(rec, 0, tuple(out))


def _rule_batch_norm(ctx: Propagator, rec: OpRecord):
    """native_batch_norm(input, weight, bias, running_mean, running_var, ...)."""
    in_src, in_shape = _tensor_at(rec, 0)
    in_dims = ctx.dims_of(in_src, in_shape)
    ch = in_dims[1] if isinstance(in_dims, tuple) and len(in_dims) > 1 else None

    for pos in (1, 2, 3, 4):
        p_src, p_shape = _tensor_at(rec, pos)
        if p_src is None:
            continue
        p_dims = ctx.dims_of(p_src, p_shape)
        if p_dims == RAW or not isinstance(p_dims, tuple) or not p_dims:
            continue
        p = p_dims[0]
        if isinstance(p, np.ndarray):
            if isinstance(ch, np.ndarray) and len(ch) == len(p):
                ctx.reg.union_arrays(p, ch)
            else:
                ctx.reg.taint(p, f"norm over no-provenance input @ {rec.module}")

    out = list(in_dims) if isinstance(in_dims, tuple) else ctx.none_dims(in_shape)
    ctx.set_out(rec, 0, tuple(out))
    for slot in range(1, rec.n_outputs):
        if rec.output_shapes[slot] is not None:
            ctx.set_out(rec, slot, ctx.none_dims(rec.output_shapes[slot]))


def _rule_group_norm(ctx: Propagator, rec: OpRecord):
    """native_group_norm(input, weight, bias, N, C, HxW, group, eps)."""
    _rule_batch_norm_like(ctx, rec, param_positions=(1, 2))


def _rule_layer_norm(ctx: Propagator, rec: OpRecord):
    """native_layer_norm(input, normalized_shape, weight, bias, eps) — channel on last dim."""
    in_src, in_shape = _tensor_at(rec, 0)
    in_dims = ctx.dims_of(in_src, in_shape)
    ch = in_dims[-1] if isinstance(in_dims, tuple) and in_dims else None
    for pos in (2, 3):
        p_src, p_shape = _tensor_at(rec, pos)
        if p_src is None:
            continue
        p_dims = ctx.dims_of(p_src, p_shape)
        if p_dims == RAW or not isinstance(p_dims, tuple) or not p_dims:
            continue
        p = p_dims[0]
        if isinstance(p, np.ndarray):
            if isinstance(ch, np.ndarray) and len(ch) == len(p):
                ctx.reg.union_arrays(p, ch)
            else:
                ctx.reg.taint(p, f"layer_norm over no-provenance input @ {rec.module}")
    ctx.set_out(rec, 0, tuple(in_dims) if isinstance(in_dims, tuple) else ctx.none_dims(in_shape))
    for slot in range(1, rec.n_outputs):
        if rec.output_shapes[slot] is not None:
            ctx.set_out(rec, slot, ctx.none_dims(rec.output_shapes[slot]))


def _rule_batch_norm_like(ctx: Propagator, rec: OpRecord, param_positions):
    in_src, in_shape = _tensor_at(rec, 0)
    in_dims = ctx.dims_of(in_src, in_shape)
    ch = in_dims[1] if isinstance(in_dims, tuple) and len(in_dims) > 1 else None
    for pos in param_positions:
        p_src, p_shape = _tensor_at(rec, pos)
        if p_src is None:
            continue
        p_dims = ctx.dims_of(p_src, p_shape)
        if p_dims == RAW or not isinstance(p_dims, tuple) or not p_dims:
            continue
        p = p_dims[0]
        if isinstance(p, np.ndarray):
            if isinstance(ch, np.ndarray) and len(ch) == len(p):
                ctx.reg.union_arrays(p, ch)
            else:
                ctx.reg.taint(p, f"group_norm over no-provenance input @ {rec.module}")
    ctx.set_out(rec, 0, tuple(in_dims) if isinstance(in_dims, tuple) else ctx.none_dims(in_shape))
    for slot in range(1, rec.n_outputs):
        if rec.output_shapes[slot] is not None:
            ctx.set_out(rec, slot, ctx.none_dims(rec.output_shapes[slot]))


def _rule_cat(ctx: Propagator, rec: OpRecord):
    entry = rec.inputs[0]
    _, sources, shapes = entry
    if not isinstance(sources, list):
        ctx.conservative(rec, "cat with non-list input")
        return
    dim = int(_meta_at(rec, 1, 0) or 0)
    out_shape = rec.output_shapes[0]
    rank = len(out_shape)
    dim = _norm_dim(dim, rank)

    all_dims = [ctx.dims_of(s, sh) for s, sh in zip(sources, shapes)]
    all_dims = [d if isinstance(d, tuple) else ctx.none_dims(sh) for d, sh in zip(all_dims, shapes)]

    out = []
    for i in range(rank):
        if i == dim:
            parts, ok = [], True
            for d, sh in zip(all_dims, shapes):
                if isinstance(d[i], np.ndarray) and len(d[i]) == sh[i]:
                    parts.append(d[i])
                else:
                    ok = False
                    break
            if ok and parts:
                out.append(np.concatenate(parts))
            else:
                # mixed provenance: taint known parts (an unknown-width sibling
                # would silently shift offsets otherwise)
                for d in all_dims:
                    if isinstance(d[i], np.ndarray):
                        ctx.reg.taint(d[i], f"cat with no-provenance sibling @ {rec.module}")
                out.append(None)
        else:
            provs = [d[i] for d in all_dims if isinstance(d[i], np.ndarray)]
            if provs and all(len(p) == out_shape[i] for p in provs):
                base = provs[0]
                for p in provs[1:]:
                    ctx.reg.union_arrays(base, p)
                out.append(base)
            else:
                out.append(None)
    ctx.set_out(rec, 0, tuple(out))


def _rule_stack(ctx: Propagator, rec: OpRecord):
    entry = rec.inputs[0]
    _, sources, shapes = entry
    if not isinstance(sources, list):
        ctx.conservative(rec, "stack with non-list input")
        return
    dim = int(_meta_at(rec, 1, 0) or 0)
    out_shape = rec.output_shapes[0]
    rank_in = len(shapes[0])
    dim = _norm_dim(dim, rank_in + 1)

    all_dims = [ctx.dims_of(s, sh) for s, sh in zip(sources, shapes)]
    all_dims = [d if isinstance(d, tuple) else ctx.none_dims(sh) for d, sh in zip(all_dims, shapes)]

    merged = []
    for i in range(rank_in):
        provs = [d[i] for d in all_dims if isinstance(d[i], np.ndarray)]
        if provs and all(len(p) == shapes[0][i] for p in provs):
            base = provs[0]
            for p in provs[1:]:
                ctx.reg.union_arrays(base, p)  # stack aligns siblings (CBFuse sum)
            merged.append(base)
        else:
            merged.append(None)
    out = merged[:dim] + [None] + merged[dim:]
    ctx.set_out(rec, 0, tuple(out) if len(out) == len(out_shape) else ctx.none_dims(out_shape))


def _rule_split(ctx: Propagator, rec: OpRecord):
    """split.Tensor(t, split_size, dim) / split_with_sizes(t, sizes, dim)."""
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    sizes = _meta_at(rec, 1)
    dim = int(_meta_at(rec, 2, 0) or 0)
    dim = _norm_dim(dim, len(shape))
    if isinstance(sizes, int):
        n = shape[dim]
        sizes = [sizes] * (n // sizes) + ([n % sizes] if n % sizes else [])
    if not isinstance(sizes, (list, tuple)):
        ctx.conservative(rec, "split with unknown sizes")
        return
    off = 0
    seg_provs = []
    for slot, sz in enumerate(sizes):
        out = list(dims)
        d = dims[dim]
        seg = d[off:off + sz] if isinstance(d, np.ndarray) else None
        out[dim] = seg
        seg_provs.append(seg)
        off += sz
        if slot < rec.n_outputs and rec.output_shapes[slot] is not None:
            ctx.set_out(rec, slot, tuple(out))
    # sibling-equal constraint: module code typically re-slices with one shared
    # scalar (C2f.c / C2PSA.c), so all sibling segments must keep equal counts
    live = [s for s in seg_provs if isinstance(s, np.ndarray) and len(s)]
    if len(live) > 1 and len({len(s) for s in live}) == 1:
        for a, b in zip(live, live[1:]):
            ctx.reg.add_equal(int(a[0]), int(b[0]))


def _rule_slice(ctx: Propagator, rec: OpRecord):
    """slice.Tensor(t, dim, start, end, step)."""
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    dim = _norm_dim(int(_meta_at(rec, 1, 0) or 0), len(shape))
    start = _meta_at(rec, 2, 0) or 0
    end = _meta_at(rec, 3, _MAXINT)
    end = shape[dim] if end is None or end > shape[dim] else end
    step = _meta_at(rec, 4, 1) or 1
    out = list(dims)
    d = dims[dim]
    out[dim] = d[int(start):int(end):int(step)] if isinstance(d, np.ndarray) else None
    ctx.set_out(rec, 0, tuple(out))


def _rule_select(ctx: Propagator, rec: OpRecord):
    """select.int(t, dim, index) — dimension is removed."""
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    dim = _norm_dim(int(_meta_at(rec, 1, 0) or 0), len(shape))
    out = [d for i, d in enumerate(dims) if i != dim]
    ctx.set_out(rec, 0, tuple(out))


def _rule_unbind(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    dim = _norm_dim(int(_meta_at(rec, 1, 0) or 0), len(shape))
    for slot in range(rec.n_outputs):
        out = [d for i, d in enumerate(dims) if i != dim]
        if rec.output_shapes[slot] is not None:
            ctx.set_out(rec, slot, tuple(out))


def _rule_view(ctx: Propagator, rec: OpRecord):
    """view/_unsafe_view/reshape: safe only when provenance dims map 1:1."""
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    out_shape = rec.output_shapes[0]
    mapped = _match_reshape(dims, shape, out_shape)
    if mapped is None:
        # folding reshape: channel dim merged/split => conservative taint
        ctx.taint_dims(dims, f"folding reshape {tuple(shape)}->{tuple(out_shape)} @ {rec.module}")
        ctx.conservative_log.append(
            {"op": rec.op, "module": rec.module, "reason": "folding reshape", "record": rec.idx}
        )
        mapped = ctx.none_dims(out_shape)
    ctx.set_out(rec, 0, mapped)


def _match_reshape(dims, in_shape, out_shape):
    """Map provenance across reshape; None => unsafe (a prov dim was folded)."""
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    out = [None] * len(out_shape)
    i = j = 0
    while i < len(in_shape) or j < len(out_shape):
        if i < len(in_shape) and j < len(out_shape) and in_shape[i] == out_shape[j]:
            out[j] = dims[i] if isinstance(dims[i], np.ndarray) else None
            i += 1
            j += 1
            continue
        # accumulate a factor block until products match
        pi, pj = 1, 1
        bi, bj = [], []
        ii, jj = i, j
        while True:
            if pi == pj and pi != 1 and bi and bj:
                break
            if pi <= pj and ii < len(in_shape):
                pi *= in_shape[ii]
                bi.append(ii)
                ii += 1
            elif jj < len(out_shape):
                pj *= out_shape[jj]
                bj.append(jj)
                jj += 1
            else:
                break
        if pi != pj or not bi or not bj:
            return None if any(isinstance(dims[k], np.ndarray) for k in range(i, len(in_shape))) else tuple(out)
        # block with any provenance dim => unsafe fold (unless trivial 1-maps)
        if any(isinstance(dims[k], np.ndarray) and in_shape[k] > 1 for k in bi):
            # exception: single non-1 in-dim mapping to single equal out-dim
            nz_in = [k for k in bi if in_shape[k] > 1]
            nz_out = [k for k in bj if out_shape[k] > 1]
            if len(nz_in) == 1 and len(nz_out) == 1 and in_shape[nz_in[0]] == out_shape[nz_out[0]]:
                out[nz_out[0]] = dims[nz_in[0]]
            else:
                return None
        i, j = ii, jj
    return tuple(out)


def _rule_permute(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    perm = _meta_at(rec, 1)
    if not isinstance(perm, (list, tuple)):
        ctx.set_out(rec, 0, ctx.none_dims(rec.output_shapes[0]))
        return
    out = tuple(dims[_norm_dim(int(p), len(shape))] for p in perm)
    ctx.set_out(rec, 0, out)


def _rule_transpose(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    dims = list(dims) if isinstance(dims, tuple) else list(ctx.none_dims(shape))
    d0 = _norm_dim(int(_meta_at(rec, 1, 0) or 0), len(shape))
    d1 = _norm_dim(int(_meta_at(rec, 2, 1) or 1), len(shape))
    dims[d0], dims[d1] = dims[d1], dims[d0]
    ctx.set_out(rec, 0, tuple(dims))


def _rule_t(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    ctx.set_out(rec, 0, tuple(reversed(dims)) if len(shape) == 2 else tuple(dims))


def _rule_unsqueeze(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    dims = list(dims) if isinstance(dims, tuple) else list(ctx.none_dims(shape))
    dim = _norm_dim(int(_meta_at(rec, 1, 0) or 0), len(shape) + 1)
    dims.insert(dim, None)
    ctx.set_out(rec, 0, tuple(dims))


def _rule_squeeze(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    out_shape = rec.output_shapes[0]
    d = _meta_at(rec, 1, None)
    if d is None:
        out = [dims[i] for i in range(len(shape)) if shape[i] != 1]
    else:
        ds = {_norm_dim(int(x), len(shape)) for x in (d if isinstance(d, (list, tuple)) else [d])}
        out = [dims[i] for i in range(len(shape)) if not (i in ds and shape[i] == 1)]
    ctx.set_out(rec, 0, tuple(out) if len(out) == len(out_shape) else ctx.none_dims(out_shape))


def _rule_expand(ctx: Propagator, rec: OpRecord):
    src, shape = _tensor_at(rec, 0)
    if src is None:
        return
    dims = ctx.dims_of(src, shape)
    if dims == RAW:
        dims = ctx.none_dims(shape)
    out_shape = rec.output_shapes[0]
    off = len(out_shape) - len(shape)
    out = [None] * len(out_shape)
    for i, d in enumerate(dims):
        oi = off + i
        if isinstance(d, np.ndarray):
            if len(d) == out_shape[oi]:
                out[oi] = d
            elif len(d) == 1:
                out[oi] = np.repeat(d, out_shape[oi])
    ctx.set_out(rec, 0, tuple(out))


def _rule_copy(ctx: Propagator, rec: OpRecord):
    """copy_(dst, src): output takes src provenance."""
    s_src, s_shape = _tensor_at(rec, 1)
    out_shape = rec.output_shapes[0]
    if s_src is None or out_shape is None:
        return
    dims = ctx.dims_of(s_src, s_shape)
    if dims == RAW or len(dims) != len(out_shape):
        dims = ctx.none_dims(out_shape)
    ctx.set_out(rec, 0, tuple(dims))


def _rule_addmm(ctx: Propagator, rec: OpRecord):
    """addmm(bias, mat1, mat2)."""
    b_src, b_shape = _tensor_at(rec, 0)
    a_src, a_shape = _tensor_at(rec, 1)
    m_src, m_shape = _tensor_at(rec, 2)
    out_shape = rec.output_shapes[0]
    if a_src is None or m_src is None:
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        return
    da = ctx.dims_of(a_src, a_shape)
    dm = ctx.dims_of(m_src, m_shape)
    da = da if isinstance(da, tuple) else ctx.none_dims(a_shape)
    dm = dm if isinstance(dm, tuple) else ctx.none_dims(m_shape)
    # contraction: mat1 last dim с mat2 first dim
    if isinstance(da[-1], np.ndarray) and isinstance(dm[0], np.ndarray) and len(da[-1]) == len(dm[0]):
        ctx.reg.union_arrays(da[-1], dm[0])
    elif isinstance(dm[0], np.ndarray) and not isinstance(da[-1], np.ndarray):
        ctx.reg.taint(dm[0], f"matmul contraction without provenance @ {rec.module}")
    elif isinstance(da[-1], np.ndarray) and not isinstance(dm[0], np.ndarray):
        ctx.reg.taint(da[-1], f"matmul contraction without provenance @ {rec.module}")
    out = list(ctx.none_dims(out_shape))
    if isinstance(dm[1], np.ndarray) and len(dm[1]) == out_shape[-1]:
        out[-1] = dm[1]
    if b_src is not None:
        db = ctx.dims_of(b_src, b_shape)
        if isinstance(db, tuple) and db and isinstance(db[-1], np.ndarray) and isinstance(dm[1], np.ndarray) and len(db[-1]) == len(dm[1]):
            ctx.reg.union_arrays(db[-1], dm[1])
    ctx.set_out(rec, 0, tuple(out))


def _rule_mm(ctx: Propagator, rec: OpRecord):
    a_src, a_shape = _tensor_at(rec, 0)
    m_src, m_shape = _tensor_at(rec, 1)
    out_shape = rec.output_shapes[0]
    if a_src is None or m_src is None:
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        return
    da = ctx.dims_of(a_src, a_shape)
    dm = ctx.dims_of(m_src, m_shape)
    da = da if isinstance(da, tuple) else ctx.none_dims(a_shape)
    dm = dm if isinstance(dm, tuple) else ctx.none_dims(m_shape)
    if isinstance(da[-1], np.ndarray) and isinstance(dm[-2], np.ndarray) and len(da[-1]) == len(dm[-2]):
        ctx.reg.union_arrays(da[-1], dm[-2])
    else:
        for d in (da[-1], dm[-2]):
            if isinstance(d, np.ndarray):
                ctx.reg.taint(d, f"matmul contraction without provenance @ {rec.module}")
    out = list(ctx.none_dims(out_shape))
    if isinstance(dm[-1], np.ndarray) and len(dm[-1]) == out_shape[-1]:
        out[-1] = dm[-1]
    if isinstance(da[-2], np.ndarray) and len(da[-2]) == out_shape[-2]:
        out[-2] = da[-2]
    ctx.set_out(rec, 0, tuple(out))


def _rule_linear(ctx: Propagator, rec: OpRecord):
    """aten.linear(input, weight, bias) — weight (out, in)."""
    in_src, in_shape = _tensor_at(rec, 0)
    w_src, w_shape = _tensor_at(rec, 1)
    b_src, b_shape = _tensor_at(rec, 2)
    out_shape = rec.output_shapes[0]
    di = ctx.dims_of(in_src, in_shape) if in_src else None
    dw = ctx.dims_of(w_src, w_shape) if w_src else None
    di = di if isinstance(di, tuple) else ctx.none_dims(in_shape or ())
    if not isinstance(dw, tuple):
        if isinstance(di, tuple):
            ctx.taint_dims(di, f"functional linear with raw weight @ {rec.module}")
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        return
    if isinstance(di[-1], np.ndarray) and isinstance(dw[1], np.ndarray) and len(di[-1]) == len(dw[1]):
        ctx.reg.union_arrays(dw[1], di[-1])
    elif isinstance(dw[1], np.ndarray):
        ctx.reg.taint(dw[1], f"linear input without provenance @ {rec.module}")
    out = list(ctx.none_dims(out_shape))
    if isinstance(dw[0], np.ndarray) and len(dw[0]) == out_shape[-1]:
        out[-1] = dw[0]
    ctx.set_out(rec, 0, tuple(out))


def _rule_where(ctx: Propagator, rec: OpRecord):
    # where(cond, a, b): treat a/b like elementwise
    tensors = [(src, shape) for _, src, shape in _iter_tensor_inputs(rec)]
    out_shape = rec.output_shapes[0]
    if len(tensors) < 3:
        ctx.set_out(rec, 0, ctx.none_dims(out_shape))
        return
    (sa, sha), (sb, shb) = tensors[1], tensors[2]
    da, db = ctx.dims_of(sa, sha), ctx.dims_of(sb, shb)
    da = da if isinstance(da, tuple) else ctx.none_dims(sha)
    db = db if isinstance(db, tuple) else ctx.none_dims(shb)
    rank = len(out_shape)
    pa, pb = _broadcast_dims(da, sha, rank), _broadcast_dims(db, shb, rank)
    out = []
    for i in range(rank):
        ea, eb = pa[i], pb[i]
        if isinstance(ea, np.ndarray) and isinstance(eb, np.ndarray) and len(ea) == len(eb) == out_shape[i]:
            ctx.reg.union_arrays(ea, eb)
            out.append(ea)
        elif isinstance(ea, np.ndarray) and len(ea) == out_shape[i] and eb is None:
            out.append(ea)
        elif isinstance(eb, np.ndarray) and len(eb) == out_shape[i] and ea is None:
            out.append(eb)
        else:
            out.append(None)
    ctx.set_out(rec, 0, tuple(out))


def _rule_pixel_shuffle(ctx: Propagator, rec: OpRecord):
    # channel dim is factorized by r^2 — conservative
    src, shape = _tensor_at(rec, 0)
    if src is not None:
        dims = ctx.dims_of(src, shape)
        if isinstance(dims, tuple):
            ctx.taint_dims(dims, f"pixel_shuffle folds channels @ {rec.module}")
    ctx.set_out(rec, 0, ctx.none_dims(rec.output_shapes[0]))


OP_RULES = {
    "convolution": _rule_convolution,
    "_convolution": _rule_convolution,
    "conv2d": _rule_convolution,
    "native_batch_norm": _rule_batch_norm,
    "_native_batch_norm_legit": _rule_batch_norm,
    "_native_batch_norm_legit_no_training": _rule_batch_norm,
    "batch_norm": _rule_batch_norm,
    "cudnn_batch_norm": _rule_batch_norm,
    "miopen_batch_norm": _rule_batch_norm,
    "native_group_norm": _rule_group_norm,
    "group_norm": _rule_group_norm,
    "native_layer_norm": _rule_layer_norm,
    "layer_norm": _rule_layer_norm,
    "cat": _rule_cat,
    "stack": _rule_stack,
    "split": _rule_split,
    "split_with_sizes": _rule_split,
    "slice": _rule_slice,
    "select": _rule_select,
    "unbind": _rule_unbind,
    "view": _rule_view,
    "_unsafe_view": _rule_view,
    "reshape": _rule_view,
    "permute": _rule_permute,
    "transpose": _rule_transpose,
    "t": _rule_t,
    "unsqueeze": _rule_unsqueeze,
    "squeeze": _rule_squeeze,
    "expand": _rule_expand,
    "copy": _rule_copy,
    "addmm": _rule_addmm,
    "mm": _rule_mm,
    "bmm": _rule_mm,
    "matmul": _rule_mm,
    "linear": _rule_linear,
    "where": _rule_where,
    "pixel_shuffle": _rule_pixel_shuffle,
}


def propagate(trace: TraceResult, registry: ElementRegistry) -> Propagator:
    """Run semantics replay; returns the Propagator (with conservative log)."""
    return Propagator(trace, registry).run()
