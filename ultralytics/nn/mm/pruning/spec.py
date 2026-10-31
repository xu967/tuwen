# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Prune spec, shape-adaptive reconstruction and factory self-check.

Reconstruction protocol (hybrid, per the integration review):
- pruned.pt (pickled module) stays the runtime source of truth.
- prune_spec records per-parameter shape transitions + keep indices (audit).
- Rebuilding from config: build the FULL skeleton from the original yaml
  (stride inference runs on the intact structure), then adapt_model_to_
  state_dict slices every base layer down to the checkpoint's shapes and
  strict-loads. Because strict load overwrites all values, shape equivalence
  suffices — no derivation formulas (C2f e-ratio, Detect head widths, SPPF
  c1//2) ever need to be re-expressed in YAML.
- Factory self-check: prune() refuses to ship a checkpoint whose skeleton
  cannot be rebuilt this way (kills the dual-truth divergence risk at source).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from .metadata import sync_module_metadata

SPEC_VERSION = 2


def _yaml_safe(obj):
    """Make model.yaml dumpable: Path / tuple / numpy scalars appear at runtime."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    return obj


# ------------------------------------------------------------------
# Spec construction
# ------------------------------------------------------------------

def build_prune_spec(engine, source_weights: str = "") -> dict:
    """Serialize the engine's rewrite plan into an auditable spec."""
    rewrites = engine.report.get("rewrites", [])
    return {
        "schema_version": SPEC_VERSION,
        "engine": "zero-registration-v2",
        "date": datetime.now().isoformat(),
        "source_weights": source_weights,
        "method": engine.method,
        "ratio": engine.ratio,
        "params_before": engine.report.get("params_before"),
        "params_after": engine.report.get("params_after"),
        "rewrites": rewrites,
        "pruned_units": [
            {
                "unit": e.unit_index,
                "size": e.size,
                "n_keep": e.n_keep,
                "keep_slots": e.keep_slots,
                "initiators": e.initiator_modules,
            }
            for e in engine.plan_entries
        ],
        "conservative": engine.report.get("conservative", []),
        "fencing": engine.report.get("fencing", []),
        "sinks": engine.report.get("sinks", []),
    }


# ------------------------------------------------------------------
# Shape-adaptive reconstruction
# ------------------------------------------------------------------

@torch.no_grad()
def adapt_model_to_state_dict(model: nn.Module, state_dict: dict) -> list[str]:
    """Slice base layers of a freshly built skeleton to match target shapes.

    Keep-index choice is irrelevant (strict load overwrites values); we take
    the leading span per axis. Base-layer metadata and composite-module channel
    scalars are re-synced afterwards. Returns the list of adapted parameters.
    """
    adapted = []
    modules = dict(model.named_modules())

    for owner_name, mod in modules.items():
        prefix = owner_name + "." if owner_name else ""
        entries = list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False))
        for pname, tensor in entries:
            fqn = prefix + pname
            target = state_dict.get(fqn)
            if target is None or tuple(target.shape) == tuple(tensor.shape):
                continue
            if len(target.shape) != len(tensor.shape):
                raise RuntimeError(f"rank mismatch at {fqn}: {tuple(tensor.shape)} vs {tuple(target.shape)}")
            new_t = tensor
            for dim, (cur, tgt) in enumerate(zip(tensor.shape, target.shape)):
                if tgt > cur:
                    raise RuntimeError(
                        f"target dim larger than skeleton at {fqn} dim{dim}: {cur} -> {tgt} "
                        "(checkpoint does not belong to this yaml)"
                    )
                if tgt < cur:
                    new_t = new_t.narrow(dim, 0, tgt)
            new_t = new_t.contiguous()
            if pname in mod._parameters and mod._parameters[pname] is not None:
                old = mod._parameters[pname]
                mod._parameters[pname] = nn.Parameter(new_t, requires_grad=old.requires_grad)
            elif pname in mod._buffers and mod._buffers[pname] is not None:
                mod._buffers[pname] = new_t
            adapted.append(fqn)

    # refresh base-layer metadata from actual shapes
    from .executors.structured import StructuredExecutor

    owners = {fqn.rsplit(".", 1)[0] for fqn in adapted}
    for owner in owners:
        mod = modules.get(owner)
        if mod is not None:
            StructuredExecutor._sync_base_layer(mod, owner)

    sync_module_metadata(model)
    return adapted


def rebuild_pruned_model(yaml_cfg, state_dict: dict, model_class, **build_kwargs) -> nn.Module:
    """Reconstruct a pruned model from its original yaml + pruned state_dict."""
    skeleton = model_class(yaml_cfg, verbose=False, **build_kwargs)
    adapt_model_to_state_dict(skeleton, state_dict)
    skeleton.load_state_dict(state_dict, strict=True)
    return skeleton.eval()


def factory_self_check(model: nn.Module, model_class=None) -> bool:
    """Rebuild the skeleton from the model's own yaml and strict-load.

    Raises on divergence — a checkpoint that cannot round-trip must not ship.
    """
    yaml_cfg = getattr(model, "yaml", None)
    if not isinstance(yaml_cfg, dict):
        LOGGER.warning("[PruneV2] factory self-check skipped: model has no yaml dict")
        return False
    cls = model_class or type(model)
    sd = model.state_dict()
    build_kwargs = {}
    ch = yaml_cfg.get("channels")
    if ch:
        build_kwargs["ch"] = int(ch)
    rebuilt = rebuild_pruned_model(dict(yaml_cfg), sd, cls, **build_kwargs)
    for (na, pa), (nb, pb) in zip(model.named_parameters(), rebuilt.named_parameters()):
        if na != nb or pa.shape != pb.shape:
            raise RuntimeError(f"factory self-check mismatch at {na} vs {nb}: {pa.shape} vs {pb.shape}")
    LOGGER.info("[PruneV2] factory self-check passed: yaml + state_dict rebuild is strict-loadable")
    return True


# ------------------------------------------------------------------
# Artifact saving
# ------------------------------------------------------------------

def save_pruned_artifacts(
    nn_model: nn.Module,
    engine,
    save_dir: str | Path,
    source_weights: str = "",
    extra_info: dict | None = None,
    train_args: dict | None = None,
) -> dict:
    """Write pruned.pt / pruned.yaml / prune_spec.json; returns prune_info."""
    import yaml as pyyaml
    from copy import deepcopy

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    spec = build_prune_spec(engine, source_weights)
    spec_path = save_dir / "prune_spec.json"
    spec_path.write_text(json.dumps(spec, indent=1, default=str))

    prune_info = {
        "is_pruned": True,
        "mode": "structured",
        "engine": "zero-registration-v2",
        "source_weights": str(source_weights),
        "prune_method": engine.method,
        "prune_ratio": engine.ratio,
        "spec_file": spec_path.name,
        **(extra_info or {}),
    }

    # pruned.yaml: the FULL original yaml dict (loadable) + prune marker.
    # Reconstruction = build skeleton from this yaml + adapt to state_dict.
    yaml_cfg = getattr(nn_model, "yaml", None)
    if isinstance(yaml_cfg, dict):
        out_yaml = _yaml_safe({k: v for k, v in yaml_cfg.items() if k != "yaml_file"})
        out_yaml["prune_info"] = {k: v for k, v in prune_info.items() if k != "spec_file"}
        (save_dir / "pruned.yaml").write_text(
            "# Pruned model config — rebuild: skeleton from this yaml, then\n"
            "# adapt_model_to_state_dict(pruned.pt state_dict) + strict load.\n"
            + pyyaml.safe_dump(out_yaml, sort_keys=False, allow_unicode=True)
        )

    model_copy = deepcopy(nn_model).half()
    model_copy.prune_info = dict(prune_info)
    ckpt = {
        "model": model_copy,
        "prune_info": prune_info,
        "train_args": dict(train_args or {}),
        "date": datetime.now().isoformat(),
    }
    torch.save(ckpt, save_dir / "pruned.pt")
    LOGGER.info(f"[PruneV2] artifacts saved to {save_dir} (pruned.pt / pruned.yaml / {spec_path.name})")
    return prune_info
