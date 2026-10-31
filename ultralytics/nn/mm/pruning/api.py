# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Shared prune pipeline used by YOLOVO.prune() (and leftover YOLOMM APIs).

Owns the lifecycle contract:
- run the zero-registration engine,
- synthesize ``self.ckpt`` in memory (finetrain admission without a disk
  round-trip; ema/optimizer keys are stripped so Model.save cannot leak an
  unpruned EMA),
- attach ``prune_info`` to the nn.Module itself (pickle-borne marker used by
  train() interception and BaseModel.load fail-fast),
- save artifacts + factory self-check when save_dir is given.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch

from ultralytics.utils import LOGGER

from .engine_v2 import ZeroRegPruneEngine, text_fence_prefixes
from .spec import factory_self_check, save_pruned_artifacts
from .trainability import restore_parameter_trainability


def _normalize_for_finetrain(nn_model) -> None:
    """Drop pickled loss objects and unfreeze leftover EMA params.

    v8DetectionLoss.proj is a raw tensor (not a buffer), so a CPU-pickled
    criterion will not follow model.to(cuda) and crashes on the first step.
    Finetrain re-inits criterion after the model is on the training device.
    """
    restore_parameter_trainability(nn_model)
    if hasattr(nn_model, "criterion"):
        delattr(nn_model, "criterion")


def _mm_meta(model_obj) -> dict:
    """Best-effort multimodal metadata for prune_info."""
    meta = {}
    nn_model = model_obj.model
    router = getattr(nn_model, "mm_router", None) or getattr(nn_model, "multimodal_router", None)
    if router is not None and hasattr(router, "INPUT_SOURCES"):
        meta["x_channels"] = int(router.INPUT_SOURCES.get("SECONDARY", 3))
    yaml_cfg = getattr(nn_model, "yaml", None)
    if isinstance(yaml_cfg, dict):
        ds = yaml_cfg.get("dataset_config") or {}
        if isinstance(ds, dict):
            if ds.get("x_modality"):
                meta["x_modality"] = ds["x_modality"]
            if ds.get("Xch"):
                meta.setdefault("x_channels", int(ds["Xch"]))
    return meta


def run_structured_prune(
    model_obj,
    *,
    method: str,
    ratio: float,
    save_dir=None,
    imgsz: int = 256,
    round_to: int = 8,
    min_ch: int = 8,
    prune_types=None,
    skip_types=None,
    scope=None,
    save_report=False,
) -> ZeroRegPruneEngine:
    """Full structured-prune lifecycle on a Model-API object (YOLOVO)."""
    if save_report is True and save_dir is None:
        raise ValueError("save_report=True requires save_dir.")

    nn_model = model_obj.model
    if next(nn_model.parameters()).dtype != torch.float32:
        nn_model.float()  # half checkpoints must be traced/rewritten in fp32

    engine = ZeroRegPruneEngine(
        nn_model, method=method, ratio=ratio, imgsz=imgsz,
        round_to=round_to, min_ch=min_ch, scope=scope,
        skip_types=skip_types, prune_types=prune_types,
    )
    engine.run()
    _normalize_for_finetrain(nn_model)

    source_weights = str(model_obj.overrides.get("model", "")) if isinstance(model_obj.overrides, dict) else ""
    extra = _mm_meta(model_obj)

    prune_info = {
        "is_pruned": True,
        "mode": "structured",
        "engine": "zero-registration-v2",
        "source_weights": source_weights,
        "prune_method": method,
        "prune_ratio": ratio,
        **extra,
    }

    if save_dir is not None:
        prune_info = save_pruned_artifacts(
            nn_model, engine, save_dir,
            source_weights=source_weights, extra_info=extra,
            train_args=dict(model_obj.overrides) if isinstance(model_obj.overrides, dict) else {},
        )
        if save_report:
            report_path = Path(save_dir) / "prune_report.json"
            report_path.write_text(json.dumps(engine.report, indent=1, default=str))
        # factory gate: a checkpoint that cannot round-trip must not ship
        factory_self_check(nn_model)
    else:
        try:
            factory_self_check(nn_model)
        except Exception as e:
            LOGGER.warning(f"[PruneV2] in-memory factory self-check failed (non-fatal): {e}")

    # pickle-borne marker + in-memory finetrain admission
    _attach_prune_state(model_obj, prune_info)
    return engine


def _attach_prune_state(model_obj, prune_info: dict) -> None:
    """Attach prune_info to the nn.Module and synthesize a clean self.ckpt."""
    model_obj.model.prune_info = dict(prune_info)
    old_ckpt = model_obj.ckpt if isinstance(model_obj.ckpt, dict) else {}
    model_obj.ckpt = {
        k: v for k, v in old_ckpt.items()
        if k not in ("ema", "optimizer", "updates", "best_fitness", "model")
    }
    model_obj.ckpt.update({
        "prune_info": dict(prune_info),
        "date": datetime.now().isoformat(),
    })


def run_sparse_prune(
    model_obj,
    *,
    sparsity: float = 0.5,
    pattern: str = "unstructured",
    save_dir=None,
) -> dict:
    """Unstructured / 2:4 sparsity lifecycle on a Model-API object.

    Masks are registered as module buffers (persist through pickle); weights
    are zeroed in place. Structure is untouched so the model yaml stays
    accurate. Finetrain re-applies masks EMA-safely inside optimizer_step.
    """
    from .executors.unstructured import UnstructuredExecutor, measured_sparsity

    nn_model = model_obj.model
    if next(nn_model.parameters()).dtype != torch.float32:
        nn_model.float()

    prefixes = text_fence_prefixes(nn_model)

    def _vision_scope(name: str) -> bool:
        return not any(name == p or name.startswith(p + ".") for p in prefixes)

    executor = UnstructuredExecutor(
        nn_model,
        sparsity=sparsity,
        pattern=pattern,
        scope_check=_vision_scope if prefixes else None,
    )
    stats = executor.apply()
    _normalize_for_finetrain(nn_model)
    actual = measured_sparsity(nn_model)

    source_weights = str(model_obj.overrides.get("model", "")) if isinstance(model_obj.overrides, dict) else ""
    prune_info = {
        "is_pruned": True,
        "mode": pattern,  # "unstructured" | "semi24"
        "engine": "zero-registration-v2",
        "source_weights": source_weights,
        "target_sparsity": sparsity,
        "measured_sparsity": round(actual, 4),
        **_mm_meta(model_obj),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        from copy import deepcopy

        model_copy = deepcopy(nn_model).half()
        model_copy.prune_info = dict(prune_info)
        torch.save(
            {
                "model": model_copy,
                "prune_info": prune_info,
                "train_args": dict(model_obj.overrides) if isinstance(model_obj.overrides, dict) else {},
                "date": datetime.now().isoformat(),
            },
            save_dir / "pruned.pt",
        )
        (save_dir / "prune_spec.json").write_text(json.dumps({**prune_info, **stats}, indent=1, default=str))
        LOGGER.info(f"[PruneV2] sparse artifacts saved to {save_dir}")

    _attach_prune_state(model_obj, prune_info)
    LOGGER.info(f"[PruneV2] {pattern} sparsity: target={sparsity:.0%} measured={actual:.2%}")
    return prune_info
