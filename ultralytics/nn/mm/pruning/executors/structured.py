# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Structured executor: physically rewrite base-layer parameters group-wise.

Given per-unit keep decisions, aggregates keep indices per parameter axis
(a partially tainted axis keeps its tainted span in full), index_selects
weights/biases/BN statistics, and updates base-layer metadata
(out_channels/in_channels/groups/num_features/num_channels/features).
Composite-module attribute sync (C2f.c etc.) lives in metadata.py.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER


class StructuredExecutor:
    """Applies keep decisions produced by the planner."""

    def __init__(self, model: nn.Module, registry, units):
        self.model = model
        self.reg = registry
        self.units = units
        self.rewrite_log: list[dict] = []

    # -------------------------------------------------- keep aggregation
    def build_axis_keeps(self, keep_slots_by_unit: dict) -> dict[int, np.ndarray]:
        """axis_id -> sorted local KEEP indices, merged across all units."""
        axis_keep: dict[int, list] = defaultdict(list)
        for ui, unit in enumerate(self.units):
            kept = keep_slots_by_unit.get(ui)
            for axis_id in unit.axis_ids():
                if kept is None:  # unit untouched: keep everything it owns
                    axis_keep[axis_id].append(unit.all_local_indices(axis_id))
                else:
                    axis_keep[axis_id].append(unit.keep_local_indices(axis_id, kept))
        return {
            aid: np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
            for aid, parts in axis_keep.items()
        }

    # -------------------------------------------------- rewrite
    @torch.no_grad()
    def apply(self, keep_slots_by_unit: dict) -> list[dict]:
        axis_keeps = self.build_axis_keeps(keep_slots_by_unit)

        # group per-parameter: fqn -> {tensor_dim: keep ndarray}
        per_param: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        for aid, keep in axis_keeps.items():
            info = self.reg.axes[aid]
            if len(keep) == info.size:  # nothing pruned on this axis
                continue
            per_param[info.param][info.axis] = keep

        tensors = dict(self.model.named_parameters())
        tensors.update(dict(self.model.named_buffers()))
        modules = dict(self.model.named_modules())

        touched_modules: set[str] = set()
        for fqn, axis_map in per_param.items():
            t = tensors.get(fqn)
            if t is None:
                LOGGER.warning(f"[PruneV2] parameter {fqn} not found at rewrite time")
                continue
            new_t = t
            for dim, keep in sorted(axis_map.items()):
                idx = torch.as_tensor(keep, device=t.device, dtype=torch.long)
                new_t = new_t.index_select(dim, idx)
            owner, _, pname = fqn.rpartition(".")
            mod = modules.get(owner)
            if mod is None:
                continue
            if pname in mod._parameters and mod._parameters[pname] is not None:
                old = mod._parameters[pname]
                mod._parameters[pname] = nn.Parameter(new_t.contiguous(), requires_grad=old.requires_grad)
            elif pname in mod._buffers and mod._buffers[pname] is not None:
                mod._buffers[pname] = new_t.contiguous()
            touched_modules.add(owner)
            self.rewrite_log.append({"param": fqn, "old_shape": tuple(t.shape), "new_shape": tuple(new_t.shape)})

        for owner in touched_modules:
            self._sync_base_layer(modules[owner], owner)

        return self.rewrite_log

    @staticmethod
    def _sync_base_layer(mod: nn.Module, name: str):
        """Refresh base-layer metadata from actual tensor shapes."""
        if isinstance(mod, nn.Conv2d) and not isinstance(mod, nn.ConvTranspose2d):
            out_c, in_per_group = mod.weight.shape[0], mod.weight.shape[1]
            if mod.groups == 1:
                mod.in_channels = in_per_group
            elif in_per_group == 1:  # depthwise: groups follow the channel count
                mod.groups = out_c
                mod.in_channels = out_c
            mod.out_channels = out_c
        elif isinstance(mod, nn.ConvTranspose2d):
            in_c, out_per_group = mod.weight.shape[0], mod.weight.shape[1]
            mod.in_channels = in_c
            if mod.groups == 1:
                mod.out_channels = out_per_group
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            if mod.weight is not None:
                mod.num_features = mod.weight.shape[0]
            elif mod.running_mean is not None:
                mod.num_features = mod.running_mean.shape[0]
        elif isinstance(mod, nn.GroupNorm):
            if mod.weight is not None:
                mod.num_channels = mod.weight.shape[0]
        elif isinstance(mod, nn.Linear):
            mod.out_features = mod.weight.shape[0]
            mod.in_features = mod.weight.shape[1]
