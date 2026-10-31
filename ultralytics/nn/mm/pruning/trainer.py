# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Task-agnostic pruned-finetune trainer mixin.

YOLOVO.finetrain() wraps the task's normal trainer (detect/obb/seg/pose)
with this mixin so prune_info, trainability restore, criterion rebuild, sparse
mask re-application and checkpoint tagging stay identical across tasks.
"""

from __future__ import annotations

import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.patches import torch_load

from .trainability import find_frozen_floating_parameters, restore_parameter_trainability


def require_pruned_checkpoint(ckpt: dict) -> dict:
    """Validate that a checkpoint carries the pruned marker."""
    prune_info = ckpt.get("prune_info") if isinstance(ckpt, dict) else None
    if not isinstance(prune_info, dict) or prune_info.get("is_pruned") is not True:
        raise ValueError(
            'finetrain 仅接受带 `prune_info={"is_pruned": True}` 的剪枝权重。'
        )
    return prune_info


class PrunedTrainerMixin:
    """Overlay for any multimodal trainer: keep the pruned graph, do not rebuild YAML."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.prune_info = None
        if getattr(self.args, "model_scale", None):
            raise ValueError(
                "finetrain 不支持 model_scale/scale; 剪枝后结构必须直接来自 checkpoint。"
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Use the already-injected pruned model; skip YAML-based rebuild."""
        if isinstance(self.model, torch.nn.Module):
            if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
                raise ValueError("剪枝后训练器收到内存模型，但缺少合法 prune_info。")
            return self.model

        if not str(self.model).endswith(".pt"):
            raise ValueError("finetrain 仅接受带 prune_info 的剪枝后 .pt 权重。")

        from ultralytics.nn.tasks import attempt_load_one_weight

        weights_obj, ckpt = attempt_load_one_weight(self.model)
        self.prune_info = require_pruned_checkpoint(ckpt)
        self.model = weights_obj
        return self.model

    def setup_model(self):
        """Skip YAML rebuild; restore trainability; drop pickled criterion."""
        if not isinstance(self.model, torch.nn.Module):
            self.model = self.get_model()

        if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
            raise ValueError("剪枝后训练器收到模型，但缺少合法 prune_info。")

        restored = restore_parameter_trainability(self.model)
        remaining = find_frozen_floating_parameters(self.model)
        if remaining:
            raise RuntimeError(
                "Legacy pruned checkpoint still contains frozen floating parameters "
                f"after finetrain normalization: {remaining[:20]}"
            )
        if restored:
            LOGGER.info(
                f"[PrunedFinetrain] Restored trainability for {len(restored)} parameters "
                "from legacy pruned checkpoint before trainer freeze policy is applied."
            )

        # Drop a pickled criterion without leaving criterion=None in place.
        # DetectionModel.loss uses getattr(..., None); RTDETR uses hasattr and
        # would skip re-init if the attribute exists as None.
        if hasattr(self.model, "criterion"):
            delattr(self.model, "criterion")

        try:
            from ultralytics.nn.mm.complexity import (
                compute_default_multimodal_complexity_report,
                log_default_complexity,
            )

            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(self.model, imgsz=imgsz)
            log_default_complexity(self.model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"剪枝后训练模型复杂度统计失败（可忽略，不影响训练）：{e}")

        from ultralytics.nn.mm.pruning.executors.unstructured import has_sparsity_masks

        self._sparse_finetune = has_sparsity_masks(self.model)
        if self._sparse_finetune:
            LOGGER.info(
                "[PrunedFinetrain] sparsity masks detected — masks will be re-applied inside "
                "optimizer_step (before EMA update) so best.pt stays sparse."
            )

    def optimizer_step(self):
        if not getattr(self, "_sparse_finetune", False):
            return super().optimizer_step()

        from ultralytics.nn.mm.pruning.executors.unstructured import reapply_masks, zero_masked_gradients

        self.scaler.unscale_(self.optimizer)
        zero_masked_gradients(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        reapply_masks(self.model)
        if self.ema:
            self.ema.update(self.model)

    def save_model(self):
        super().save_model()
        if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
            raise ValueError(f"{type(self).__name__}.save_model() 缺少合法 prune_info。")
        targets = [self.last]
        if self.best.exists():
            targets.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            epoch_pt = self.wdir / f"epoch{self.epoch}.pt"
            if epoch_pt.exists():
                targets.append(epoch_pt)
        for path in targets:
            if path.exists():
                ckpt = torch_load(path, map_location="cpu")
                ckpt["prune_info"] = self.prune_info
                torch.save(ckpt, path)


_PRUNED_TRAINER_CACHE: dict[type, type] = {}


def make_pruned_trainer_class(base_cls):
    """Compose PrunedTrainerMixin in front of the task's normal trainer class.

    Cache is keyed by the exact class object. Storing the wrapper on the base
    class would leak to subclasses (RTDETRMMOBBTrainer inherits RTDETRMMTrainer).
    """
    cached = _PRUNED_TRAINER_CACHE.get(base_cls)
    if cached is not None:
        return cached
    cls = type(f"Pruned{base_cls.__name__}", (PrunedTrainerMixin, base_cls), {})
    _PRUNED_TRAINER_CACHE[base_cls] = cls
    return cls
