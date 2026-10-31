# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Channel importance scoring — the algorithm extension point.

Adding a pruning criterion = writing one function over (unit, accessor) and
registering it. Scores are computed on initiator axes (output-semantic axes of
Conv/ConvTranspose/Linear) and summed group-wise, so residual-coupled layers
vote together (group importance). Follower BN gammas are available for the
'bn' criterion. No module-type knowledge is required anywhere here.
"""

from __future__ import annotations

import numpy as np
import torch

IMPORTANCE_REGISTRY: dict[str, callable] = {}


def register_importance(name: str):
    def deco(fn):
        IMPORTANCE_REGISTRY[name] = fn
        return fn
    return deco


class UnitAccessor:
    """Resolves unit slots to actual weight slices for scoring."""

    def __init__(self, model, registry):
        self.tensors = dict(model.named_parameters())
        self.tensors.update(dict(model.named_buffers()))
        self.reg = registry

    def initiator_scores(self, unit, reducer) -> torch.Tensor:
        """Sum per-slot scores over all initiator axes of the unit.

        reducer(weight2d) -> per-row score tensor; weight2d is the axis-first
        flattened weight (rows = channel positions on that axis).
        """
        total = torch.zeros(unit.size, dtype=torch.float32)
        for axis_id in unit.initiators:
            info = self.reg.axes[axis_id]
            w = self.tensors.get(info.param)
            if w is None:
                continue
            w2d = w.detach().float().movedim(info.axis, 0).flatten(1)
            row_scores = reducer(w2d)
            for si, slot in enumerate(unit.slots):
                locals_ = slot.members.get(axis_id)
                if locals_ is not None and len(locals_):
                    total[si] += row_scores[torch.as_tensor(np.asarray(locals_), dtype=torch.long)].sum()
        return total

    def bn_scores(self, unit) -> torch.Tensor | None:
        """Sum |gamma| over follower BN axes, if any."""
        total = torch.zeros(unit.size, dtype=torch.float32)
        n = 0
        for axis_id in unit.axis_ids():
            info = self.reg.axes[axis_id]
            if info.layer_type not in ("BatchNorm2d", "BatchNorm1d", "SyncBatchNorm"):
                continue
            if not info.param.endswith(".weight"):
                continue
            gamma = self.tensors.get(info.param)
            if gamma is None:
                continue
            g = gamma.detach().float().abs()
            for si, slot in enumerate(unit.slots):
                locals_ = slot.members.get(axis_id)
                if locals_ is not None and len(locals_):
                    total[si] += g[torch.as_tensor(np.asarray(locals_), dtype=torch.long)].sum()
            n += 1
        return total if n else None


@register_importance("l1")
def _l1(unit, acc: UnitAccessor) -> torch.Tensor:
    return acc.initiator_scores(unit, lambda w: w.abs().sum(dim=1))


@register_importance("l2")
def _l2(unit, acc: UnitAccessor) -> torch.Tensor:
    return acc.initiator_scores(unit, lambda w: (w ** 2).sum(dim=1).sqrt())


@register_importance("lamp")
def _lamp(unit, acc: UnitAccessor) -> torch.Tensor:
    scores = acc.initiator_scores(unit, lambda w: w.abs().sum(dim=1))
    order = torch.argsort(scores)
    sq = scores[order] ** 2
    denom = torch.flip(torch.cumsum(torch.flip(sq, [0]), 0), [0]).clamp_min(1e-12)
    lamp = torch.empty_like(scores)
    lamp[order] = sq / denom
    return lamp


@register_importance("bn")
def _bn(unit, acc: UnitAccessor) -> torch.Tensor:
    bn = acc.bn_scores(unit)
    if bn is not None and bn.abs().sum() > 0:
        return bn
    return acc.initiator_scores(unit, lambda w: w.abs().sum(dim=1))


@register_importance("random")
def _random(unit, acc: UnitAccessor) -> torch.Tensor:
    return torch.rand(unit.size)
