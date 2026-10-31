# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Zero-registration structured pruning engine (v2).

Orchestrates: trace -> semantics -> units -> plan (importance x constraints x
scope) -> structured rewrite -> metadata sync -> triple verification.

No module-type tables anywhere: any module that parse_model can build and the
model can forward is either pruned correctly, skipped conservatively with a
reported reason, or the whole run is refused with an explanation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from .criteria import IMPORTANCE_REGISTRY, UnitAccessor
from .executors.structured import StructuredExecutor
from .fencing import summarize_conservative, summarize_fencing
from .grouping import ElementRegistry
from .metadata import sync_module_metadata
from .semantics import propagate
from .tracing import trace_model
from .verify import check_entry_preconditions, verify_pruned_model


def _lcm(values) -> int:
    out = 1
    for v in values:
        out = out * v // math.gcd(out, v)
    return out


def sink_modules(model: nn.Module) -> set[str]:
    """Graph sinks = heads. Same structural rule as the complexity engine."""
    layers = list(getattr(model, "model", model))
    referenced = set()
    for i, layer in enumerate(layers):
        if getattr(layer, "_mm_new_input_start", False):
            continue
        f = getattr(layer, "f", None)
        for rel in ([] if f is None else (f if isinstance(f, (list, tuple)) else [f])):
            a = rel if rel >= 0 else i + rel
            if 0 <= a < len(layers) and a != i:
                referenced.add(a)
    return {f"model.{i}" for i in range(len(layers)) if i not in referenced}


def yaml_text_layer_indices(model: nn.Module) -> set[int]:
    """Sequential indices owned by YAML ``text:``. Fence by section, not module type."""
    indices = getattr(model, "_text_layer_indices", None)
    if indices is not None:
        return set(indices)
    yaml_cfg = getattr(model, "yaml", None)
    if isinstance(yaml_cfg, dict) and yaml_cfg.get("text"):
        bb = len(yaml_cfg.get("backbone") or [])
        return set(range(bb, bb + len(yaml_cfg["text"])))
    return set()


def text_fence_prefixes(model: nn.Module) -> set[str]:
    """Modules that must not be pruned: YAML ``text:`` layers + text_encoder."""
    prefixes = {f"model.{i}" for i in yaml_text_layer_indices(model)}
    if hasattr(model, "text_encoder"):
        prefixes.add("text_encoder")
    return prefixes


def ensure_dummy_text(model: nn.Module, k: int = 1) -> None:
    """Inject a cached [K, D] text embedding for trace/verify/complexity.

    Does not call encode_text / MobileCLIP. Unit-normalized dummy is enough
    for channel provenance; cosine matching only needs a finite vector.
    """
    if hasattr(model, "ensure_dummy_text"):
        model.ensure_dummy_text(k=k)
        return
    if not hasattr(model, "set_input"):
        return
    if hasattr(model, "has_input") and model.has_input("text"):
        return
    dim = int(getattr(model, "text_dim", 0) or 0)
    if dim <= 0:
        yaml_cfg = getattr(model, "yaml", None) or {}
        dim = int((yaml_cfg.get("text_encoder") or {}).get("dim", 512) or 512)
    try:
        param = next(model.parameters())
        device, dtype = param.device, param.dtype
    except StopIteration:
        return
    if not torch.is_floating_point(torch.empty(0, dtype=dtype)):
        dtype = torch.float32
    t = torch.zeros(k, dim, device=device, dtype=dtype)
    t[:, 0] = 1.0
    model.set_input("text", t, cache=True)


@dataclass
class PrunePlanEntry:
    unit_index: int
    size: int
    n_keep: int
    keep_slots: list
    initiator_modules: list = field(default_factory=list)


class ZeroRegPruneEngine:
    """Structured pruning without module-type registration.

    Args:
        model: nn.Module (DetectionModel-family or raw module list owner).
        method: importance criterion registered in criteria.IMPORTANCE_REGISTRY.
        ratio: global pruning ratio in (0, 1).
        imgsz: trace/verify input size (channel semantics are size-invariant;
            256 keeps keepalive memory low).
        round_to: keep counts rounded to multiples of this value.
        min_ch: minimum kept channels per unit.
        scope: optional module-path prefix filter, e.g. "model.0-26" style list
            of layer indices, or None for the full graph.
        skip_types / prune_types: legacy user filters, matched against the
            top-level layer type name of initiator modules (scope preference,
            not a correctness mechanism).
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "l1",
        ratio: float = 0.3,
        imgsz: int = 256,
        round_to: int = 8,
        min_ch: int = 8,
        scope: list | None = None,
        skip_types: list | None = None,
        prune_types: list | None = None,
        input_channels: int | None = None,
    ):
        if method not in IMPORTANCE_REGISTRY:
            raise ValueError(f"Unknown importance '{method}'. Registered: {sorted(IMPORTANCE_REGISTRY)}")
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0,1), got {ratio}")
        self.model = model
        self.method = method
        self.ratio = ratio
        self.imgsz = imgsz
        self.round_to = max(int(round_to), 1)
        self.min_ch = max(int(min_ch), 1)
        self.scope = scope
        self.skip_types = set(skip_types or [])
        self.prune_types = set(prune_types) if prune_types else None
        self.input_channels = input_channels

        self.trace = None
        self.registry: ElementRegistry | None = None
        self.units = None
        self.sinks: set[str] = set()
        self.text_fence: set[str] = set()
        self._frozen_shapes: dict[str, tuple] = {}
        self.plan_entries: list[PrunePlanEntry] = []
        self.report: dict = {}

    # -------------------------------------------------- helpers
    def _detect_input_channels(self) -> int:
        if self.input_channels:
            return int(self.input_channels)
        # router contract first (multimodal), then first-conv reflection
        router = getattr(self.model, "multimodal_router", None) or getattr(self.model, "mm_router", None)
        if router is not None and hasattr(router, "INPUT_SOURCES"):
            return int(router.INPUT_SOURCES.get("PRIMARY", 3)) + int(router.INPUT_SOURCES.get("SECONDARY", 3))
        yaml_cfg = getattr(self.model, "yaml", None)
        if isinstance(yaml_cfg, dict) and yaml_cfg.get("channels"):
            return int(yaml_cfg["channels"])
        # YOLOVO: RGB only; text lives in YAML `text:` as a cached embedding
        if isinstance(yaml_cfg, dict) and yaml_cfg.get("text") is not None:
            return 3
        return 6  # multimodal default (RGB + X)

    def _layer_index_of_module(self, module_fqn: str) -> int:
        # "model.<idx>..." -> idx
        parts = module_fqn.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            return int(parts[1])
        return -1

    def _top_layer_type(self, module_fqn: str) -> str:
        idx = self._layer_index_of_module(module_fqn)
        layers = list(getattr(self.model, "model", self.model))
        if 0 <= idx < len(layers):
            return type(layers[idx]).__name__
        return ""

    # -------------------------------------------------- pipeline
    def analyze(self):
        check_entry_preconditions(self.model)
        ch = self._detect_input_channels()
        ensure_dummy_text(self.model)
        example = torch.zeros(
            1, ch, self.imgsz, self.imgsz,
            device=next(self.model.parameters()).device,
        )
        self._example = example
        self.trace = trace_model(self.model, example)
        self.registry = ElementRegistry()
        self.registry.register_model(self.trace)
        self.propagator = propagate(self.trace, self.registry)
        self.sinks = sink_modules(self.model)
        self.text_fence = text_fence_prefixes(self.model)
        # Heads follow upstream pruning; YAML `text:` is frozen (not rewritten).
        self.units, self._root_to_unit = self.registry.build_units(
            protected_modules=self.sinks,
            frozen_modules=self.text_fence,
        )
        self._frozen_shapes = {
            n: tuple(p.shape)
            for n, p in self.model.named_parameters()
            if any(n == pfx or n.startswith(pfx + ".") for pfx in self.text_fence)
        }
        return self

    def plan(self):
        acc = UnitAccessor(self.model, self.registry)
        score_fn = IMPORTANCE_REGISTRY[self.method]
        unit_index = {id(u): i for i, u in enumerate(self.units)}

        # user scope filters act on initiator modules (preference, not safety)
        def in_scope(unit) -> bool:
            types = {self._top_layer_type(self.registry.axes[a].module) for a in unit.initiators}
            if self.skip_types and types & self.skip_types:
                return False
            if self.prune_types is not None and not (types & self.prune_types):
                return False
            if self.scope is not None:
                idxs = {self._layer_index_of_module(self.registry.axes[a].module) for a in unit.initiators}
                if not any(i in self.scope for i in idxs):
                    return False
            return True

        # sibling-equal closure: partner units must keep equal counts
        def equal_closure(i, seen):
            seen.add(i)
            for j in self.units[i].equal_partners:
                if j not in seen:
                    equal_closure(j, seen)
            return seen

        n_keep_by_unit: dict[int, int] = {}
        for i, u in enumerate(self.units):
            if not u.prunable or not in_scope(u):
                continue
            if u.divisors and u.size % _lcm(u.divisors) != 0:
                continue  # current structure violates its own divisor: refuse
            lcm = _lcm([self.round_to] + u.divisors)
            target = max(self.min_ch, round(u.size * (1.0 - self.ratio)))
            n_keep = min(u.size, max(lcm, int(math.ceil(target / lcm)) * lcm))
            n_keep_by_unit[i] = n_keep

        # harmonize sibling-equal partners. Module code re-slices siblings with
        # one shared scalar (C2f.c / C2PSA.c), so:
        # - if ANY sibling cannot be pruned (tainted / out of scope), the whole
        #   family must stay intact (a lone pruned sibling breaks split((c,c)))
        # - otherwise all siblings keep the same (most conservative) count.
        for i in list(n_keep_by_unit):
            if i not in n_keep_by_unit:
                continue
            group = equal_closure(i, set())
            if len(group) <= 1:
                continue
            all_plannable = all(g in n_keep_by_unit for g in group)
            same_size = len({self.units[g].size for g in group}) == 1
            if not all_plannable or not same_size:
                for g in group:
                    n_keep_by_unit.pop(g, None)
                continue
            keep = max(n_keep_by_unit[g] for g in group)
            for g in group:
                n_keep_by_unit[g] = keep

        self.plan_entries = []
        for i, n_keep in sorted(n_keep_by_unit.items()):
            u = self.units[i]
            if n_keep >= u.size:
                continue
            scores = score_fn(u, acc)
            keep_slots = torch.argsort(scores, descending=True)[:n_keep].sort().values.tolist()
            mods = sorted({self.registry.axes[a].module for a in u.initiators})[:3]
            self.plan_entries.append(
                PrunePlanEntry(unit_index=i, size=u.size, n_keep=n_keep, keep_slots=keep_slots, initiator_modules=mods)
            )
        return self

    def apply(self):
        executor = StructuredExecutor(self.model, self.registry, self.units)
        keep_by_unit = {e.unit_index: e.keep_slots for e in self.plan_entries}
        rewrite_log = executor.apply(keep_by_unit)
        frozen_now = dict(self.model.named_parameters())
        broken = {
            n: (shape, tuple(frozen_now[n].shape))
            for n, shape in getattr(self, "_frozen_shapes", {}).items()
            if n in frozen_now and tuple(frozen_now[n].shape) != shape
        }
        if broken:
            sample = dict(list(broken.items())[:5])
            raise RuntimeError(f"YAML text: fence violated (params rewritten): {sample}")
        metadata_updates = sync_module_metadata(self.model)

        expected = {}
        for item in rewrite_log:
            expected[item["param"]] = item["new_shape"]
        verify_pruned_model(self.model, self._example, expected)

        self.report = {
            "n_units": len(self.units),
            "n_pruned_units": len(self.plan_entries),
            "rewrites": rewrite_log,
            "metadata_updates": metadata_updates,
            "fencing": [f.__dict__ for f in summarize_fencing(self.trace, self.registry)],
            "conservative": summarize_conservative(self.propagator),
            "sinks": sorted(self.sinks),
            "text_fence": sorted(getattr(self, "text_fence", set())),
        }
        return self

    def run(self) -> nn.Module:
        params_before = sum(p.numel() for p in self.model.parameters())
        self.analyze()
        LOGGER.info(
            f"[PruneV2] traced {len(self.trace.records)} ops, "
            f"{len(self.units)} units ({sum(1 for u in self.units if u.prunable)} prunable), "
            f"sinks={sorted(self.sinks)}, text_fence={sorted(self.text_fence)}"
        )
        self.plan()
        LOGGER.info(f"[PruneV2] plan: {len(self.plan_entries)} units to prune (method={self.method}, ratio={self.ratio})")
        self.apply()
        params_after = sum(p.numel() for p in self.model.parameters())
        self.report["params_before"] = params_before
        self.report["params_after"] = params_after
        LOGGER.info(
            f"[PruneV2] params {params_before / 1e6:.2f}M -> {params_after / 1e6:.2f}M "
            f"(-{(1 - params_after / params_before) * 100:.1f}%)"
        )
        for item in self.report["conservative"][:8]:
            LOGGER.info(f"[PruneV2] conservative x{item['count']}: {item['module']} {item['op']} ({item['reason']})")
        for f in self.report["fencing"][:4]:
            LOGGER.info(f"[PruneV2] fenced: {f['module']} ({f['reason']})")
        return self.model
