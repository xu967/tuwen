# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Per-channel element registry, union-find and group construction.

Every channel position of every base-layer parameter axis is a global element.
The semantics replay unions elements that must be pruned together (residual
adds, follower BN stats, consumer input axes). cat/split/slice are index
transformation edges: they are expressed by slicing/concatenating provenance
arrays, never by unioning across the concat boundary (the red-team review
showed axis-level union corrupts cross-modal Concat and C2f interval coupling).

A group is prunable iff it carries no taint and contains at least one
initiator axis (an output-semantic axis of a Conv/ConvTranspose/Linear owned
by a non-protected module). Protected (head/sink) modules participate in
groups — their input axes follow upstream pruning — but never initiate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch.nn as nn

RAW = "RAW"  # marker dims for parameters that cannot be rewritten


@dataclass
class AxisInfo:
    """One registered parameter axis."""

    axis_id: int
    param: str  # fqn
    module: str  # owner module fqn
    layer_type: str  # Conv2d / BatchNorm2d / ...
    axis: int  # tensor dim index
    size: int
    start: int  # first element id
    semantic: str  # "out" | "in" | "follow"


@dataclass
class Slot:
    """One channel position group: everything that prunes together when this
    single channel slot is removed (e.g. conv.out[k] + bn[k] + consumer.in[k],
    possibly multiple positions per axis after SPPF-style same-source concat)."""

    root: int
    members: dict = field(default_factory=dict)  # axis_id -> np.ndarray(local idxs)
    tainted: bool = False


@dataclass
class PruneUnit:
    """A pruning decision unit: all slots sharing the same axis signature.

    Selecting keep channels happens per unit; each kept slot keeps all its
    member positions on every axis. Tainted slots form separate (unprunable)
    units so a partially tainted axis can still be pruned on its clean span.
    """

    key: tuple  # (frozenset(axis_ids), tainted)
    slots: list = field(default_factory=list)
    tainted: bool = False
    taint_reasons: list = field(default_factory=list)
    divisors: list = field(default_factory=list)
    equal_partners: list = field(default_factory=list)  # unit indices (filled by engine)
    initiators: list = field(default_factory=list)  # axis_ids allowed to score

    @property
    def size(self) -> int:
        return len(self.slots)

    @property
    def prunable(self) -> bool:
        return not self.tainted and bool(self.initiators) and self.size > 1

    def axis_ids(self):
        return self.key[0]

    def keep_local_indices(self, axis_id: int, keep_slots) -> np.ndarray:
        """Local indices to KEEP on the given axis for the kept slot set."""
        parts = [self.slots[s].members[axis_id] for s in keep_slots if axis_id in self.slots[s].members]
        return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)

    def all_local_indices(self, axis_id: int) -> np.ndarray:
        parts = [s.members[axis_id] for s in self.slots if axis_id in s.members]
        return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


class ElementRegistry:
    """Global channel-element table + union-find + constraints."""

    def __init__(self):
        self.n = 0
        self._parent = np.empty(0, dtype=np.int64)
        self._taint = np.empty(0, dtype=bool)
        self.taint_reasons: dict[int, str] = {}  # element -> first reason
        self.axes: list[AxisInfo] = []
        self.elem_axis: np.ndarray = np.empty(0, dtype=np.int64)  # element -> axis_id
        self.param_dims: dict[str, object] = {}  # fqn -> dims tuple | RAW
        self.divisor_constraints: list[tuple[int, int]] = []  # (element, divisor)
        self.equal_constraints: list[tuple[int, int]] = []  # (elemA, elemB) sibling-equal

    # -------------------------------------------------- allocation
    def _alloc(self, size: int) -> np.ndarray:
        start = self.n
        self.n += size
        self._parent = np.concatenate([self._parent, np.arange(start, self.n, dtype=np.int64)])
        self._taint = np.concatenate([self._taint, np.zeros(size, dtype=bool)])
        return np.arange(start, self.n, dtype=np.int64)

    def register_axis(self, param: str, module: str, layer_type: str, axis: int, size: int, semantic: str) -> np.ndarray:
        elems = self._alloc(size)
        info = AxisInfo(
            axis_id=len(self.axes), param=param, module=module, layer_type=layer_type,
            axis=axis, size=size, start=int(elems[0]) if size else self.n, semantic=semantic,
        )
        self.axes.append(info)
        if len(self.elem_axis) < self.n:
            self.elem_axis = np.concatenate(
                [self.elem_axis, np.full(self.n - len(self.elem_axis), info.axis_id, dtype=np.int64)]
            )
        return elems

    # -------------------------------------------------- union-find
    def find(self, x: int) -> int:
        parent = self._parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return int(root)

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self._parent[rb] = ra
        t = self._taint[ra] or self._taint[rb]
        self._taint[ra] = t
        self._taint[rb] = t

    def union_arrays(self, a: np.ndarray, b: np.ndarray):
        if len(a) != len(b):
            raise ValueError(f"union_arrays length mismatch: {len(a)} vs {len(b)}")
        for x, y in zip(a.tolist(), b.tolist()):
            self.union(x, y)

    # -------------------------------------------------- taint
    def taint(self, elems, reason: str):
        arr = np.asarray(elems, dtype=np.int64).ravel()
        for e in arr.tolist():
            r = self.find(e)
            if not self._taint[r]:
                self._taint[r] = True
                self.taint_reasons.setdefault(r, reason)
            self._taint[e] = True

    def taint_param(self, param: str, reason: str):
        dims = self.param_dims.get(param)
        if dims is None or dims == RAW:
            return
        for d in dims:
            if isinstance(d, np.ndarray):
                self.taint(d, reason)

    def is_tainted(self, elem: int) -> bool:
        return bool(self._taint[self.find(elem)])

    # -------------------------------------------------- constraints
    def add_divisor(self, elem: int, divisor: int):
        if divisor > 1:
            self.divisor_constraints.append((int(elem), int(divisor)))

    def add_equal(self, elem_a: int, elem_b: int):
        self.equal_constraints.append((int(elem_a), int(elem_b)))

    # -------------------------------------------------- model registration
    def register_model(self, trace_result) -> None:
        """Register base-layer parameter axes; mark everything else RAW."""
        base_owned: dict[str, tuple[str, str]] = {}  # param fqn -> (module, type)
        for mod_name, ltype in trace_result.base_layer_type.items():
            mod = trace_result.base_layers[mod_name]
            for p_name, _ in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                fqn = f"{mod_name}.{p_name}" if mod_name else p_name
                base_owned[fqn] = (mod_name, ltype)

        for mod_name, ltype in trace_result.base_layer_type.items():
            mod = trace_result.base_layers[mod_name]
            self._register_base_layer(mod_name, ltype, mod)

        for name, shape in list(trace_result.all_params.items()) + list(trace_result.all_buffers.items()):
            if name not in self.param_dims:
                self.param_dims[name] = RAW

    def _register_base_layer(self, mod_name: str, ltype: str, mod: nn.Module):
        def fqn(p):
            return f"{mod_name}.{p}" if mod_name else p

        if isinstance(mod, nn.Conv2d):
            out_e = self.register_axis(fqn("weight"), mod_name, ltype, 0, mod.out_channels, "out")
            in_size = mod.weight.shape[1]  # in_channels // groups
            in_e = self.register_axis(fqn("weight"), mod_name, ltype, 1, in_size, "in")
            self.param_dims[fqn("weight")] = (out_e, in_e, None, None)
            if mod.bias is not None:
                b_e = self.register_axis(fqn("bias"), mod_name, ltype, 0, mod.out_channels, "follow")
                self.union_arrays(b_e, out_e)
                self.param_dims[fqn("bias")] = (b_e,)
        elif isinstance(mod, nn.ConvTranspose2d):
            # weight layout: (in, out // groups, kH, kW) — axes are flipped vs Conv2d
            in_e = self.register_axis(fqn("weight"), mod_name, ltype, 0, mod.in_channels, "in")
            out_size = mod.weight.shape[1]
            out_e = self.register_axis(fqn("weight"), mod_name, ltype, 1, out_size, "out")
            self.param_dims[fqn("weight")] = (in_e, out_e, None, None)
            if mod.bias is not None:
                b_e = self.register_axis(fqn("bias"), mod_name, ltype, 0, mod.out_channels, "follow")
                if mod.groups == 1:
                    self.union_arrays(b_e, out_e)
                else:
                    self.taint(b_e, f"grouped ConvTranspose2d at {mod_name}")
                    self.taint(out_e, f"grouped ConvTranspose2d at {mod_name}")
                self.param_dims[fqn("bias")] = (b_e,)
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            c = mod.num_features
            anchor = None
            for p in ("weight", "bias", "running_mean", "running_var"):
                t = getattr(mod, p, None)
                if t is None:
                    continue
                e = self.register_axis(fqn(p), mod_name, ltype, 0, c, "follow")
                self.param_dims[fqn(p)] = (e,)
                if anchor is None:
                    anchor = e
                else:
                    self.union_arrays(e, anchor)
        elif isinstance(mod, nn.GroupNorm):
            c = mod.num_channels
            anchor = None
            for p in ("weight", "bias"):
                t = getattr(mod, p, None)
                if t is None:
                    continue
                e = self.register_axis(fqn(p), mod_name, ltype, 0, c, "follow")
                self.param_dims[fqn(p)] = (e,)
                if anchor is None:
                    anchor = e
                else:
                    self.union_arrays(e, anchor)
            if anchor is not None:
                self.add_divisor(int(anchor[0]), mod.num_groups)
        elif isinstance(mod, nn.Linear):
            out_e = self.register_axis(fqn("weight"), mod_name, ltype, 0, mod.out_features, "out")
            in_e = self.register_axis(fqn("weight"), mod_name, ltype, 1, mod.in_features, "in")
            self.param_dims[fqn("weight")] = (out_e, in_e)
            if mod.bias is not None:
                b_e = self.register_axis(fqn("bias"), mod_name, ltype, 0, mod.out_features, "follow")
                self.union_arrays(b_e, out_e)
                self.param_dims[fqn("bias")] = (b_e,)

    # -------------------------------------------------- unit construction
    def build_units(self, protected_modules: set[str], frozen_modules: set[str] | None = None):
        """Partition elements into per-channel slots, then aggregate slots
        sharing the same axis signature into pruning decision units.

        protected_modules (heads): participate as followers, never initiate.
        frozen_modules (YAML ``text:`` / text tower): any slot that touches
        these axes is tainted so coupled vision channels stay intact too.
        """
        frozen_modules = frozen_modules or set()

        def is_protected(module: str) -> bool:
            return any(module == p or module.startswith(p + ".") for p in protected_modules)

        def is_frozen(module: str) -> bool:
            return any(module == p or module.startswith(p + ".") for p in frozen_modules)

        # 1. per-channel slots (position groups) keyed by union-find root
        slots: dict[int, Slot] = {}
        for info in self.axes:
            elems = np.arange(info.start, info.start + info.size, dtype=np.int64)
            roots = np.fromiter((self.find(int(e)) for e in elems), dtype=np.int64, count=info.size)
            for root in np.unique(roots):
                s = slots.setdefault(int(root), Slot(root=int(root)))
                local = np.nonzero(roots == root)[0]
                if info.axis_id in s.members:
                    s.members[info.axis_id] = np.concatenate([s.members[info.axis_id], local])
                else:
                    s.members[info.axis_id] = local
                if self._taint[root] or is_frozen(info.module):
                    s.tainted = True

        # 2. aggregate slots into units by (axis signature, taint state)
        units: dict[tuple, PruneUnit] = {}
        for s in slots.values():
            key = (frozenset(s.members.keys()), s.tainted)
            u = units.setdefault(key, PruneUnit(key=key, tainted=s.tainted))
            u.slots.append(s)
            if s.tainted and s.root in self.taint_reasons:
                reason = self.taint_reasons[s.root]
                if reason not in u.taint_reasons:
                    u.taint_reasons.append(reason)

        # deterministic slot order: sort by the smallest local index on the
        # lexicographically-first axis (keeps channel semantics stable)
        for u in units.values():
            anchor = min(u.key[0])
            u.slots.sort(key=lambda s: int(s.members[anchor].min()) if anchor in s.members else 0)

        # 3. initiators, constraints
        for u in units.values():
            for axis_id in u.key[0]:
                info = self.axes[axis_id]
                if (
                    info.semantic == "out"
                    and info.layer_type in ("Conv2d", "ConvTranspose2d", "Linear")
                    and not is_protected(info.module)
                    and not is_frozen(info.module)
                ):
                    u.initiators.append(axis_id)

        root_to_unit: dict[int, PruneUnit] = {}
        for u in units.values():
            for s in u.slots:
                root_to_unit[s.root] = u
        for elem, d in self.divisor_constraints:
            u = root_to_unit.get(self.find(elem))
            if u is not None and d not in u.divisors:
                u.divisors.append(d)

        # sibling-equal constraints (split/chunk halves must keep equal counts:
        # module code slices with a single shared scalar such as C2f.c)
        unit_list = list(units.values())
        unit_index = {id(u): i for i, u in enumerate(unit_list)}
        for ea, eb in self.equal_constraints:
            ua = root_to_unit.get(self.find(ea))
            ub = root_to_unit.get(self.find(eb))
            if ua is None or ub is None or ua is ub:
                continue
            ia, ib = unit_index[id(ua)], unit_index[id(ub)]
            if ib not in ua.equal_partners:
                ua.equal_partners.append(ib)
            if ia not in ub.equal_partners:
                ub.equal_partners.append(ia)

        return unit_list, root_to_unit
