# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Unstructured & 2:4 semi-structured sparsity executor.

Weight-level masks stored as module buffers ("weight_mask") so they persist
through pickle checkpoints automatically. Structure is untouched — yaml stays
accurate, dependency analysis is not needed.

EMA safety (the integration review's hardest finding): masks must be re-applied
inside optimizer_step AFTER scaler.step and BEFORE ema.update. Re-applying at
on_train_batch_end is too late — the EMA blends non-zero values first, and
best.pt prefers EMA weights, silently shipping a dense model. The Pruned
trainers override optimizer_step accordingly and call reapply_masks(model).

GMP (gradual magnitude pruning): sparsity ramps from 0 to target with the
cubic schedule; the trainer calls maybe_update_masks(model, step) from the same
override point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

MASK_NAME = "weight_mask"

_MASKABLE = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


def _maskable_weights(model: nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, _MASKABLE) and mod.weight is not None and mod.weight.requires_grad:
            yield name, mod


def _semi24_mask(weight: torch.Tensor) -> torch.Tensor:
    """2:4 pattern along the input dimension: keep top-2 |w| per group of 4."""
    w = weight.detach().abs().flatten(1)  # (out, in*k*k)
    n = w.shape[1]
    pad = (-n) % 4
    if pad:
        w = torch.cat([w, torch.zeros(w.shape[0], pad, device=w.device)], dim=1)
    g = w.view(w.shape[0], -1, 4)
    idx = g.topk(2, dim=2).indices
    mask = torch.zeros_like(g)
    mask.scatter_(2, idx, 1.0)
    mask = mask.view(w.shape[0], -1)[:, :n]
    return mask.view_as(weight)


class UnstructuredExecutor:
    """One-shot / GMP magnitude masking with buffer persistence."""

    def __init__(self, model: nn.Module, sparsity: float = 0.5, pattern: str = "unstructured",
                 scope_check=None, gmp_steps: int = 0):
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0,1), got {sparsity}")
        if pattern not in ("unstructured", "semi24"):
            raise ValueError(f"pattern must be 'unstructured' or 'semi24', got {pattern}")
        self.model = model
        self.sparsity = sparsity
        self.pattern = pattern
        self.scope_check = scope_check  # optional fn(module_name) -> bool
        self.gmp_steps = int(gmp_steps)

    # -------------------------------------------------- mask computation
    @torch.no_grad()
    def compute_masks(self, sparsity: float | None = None) -> dict:
        """(Re)compute masks at the given sparsity; registers/updates buffers."""
        sparsity = self.sparsity if sparsity is None else sparsity
        stats = {"masked_params": 0, "zeroed": 0}

        if self.pattern == "semi24":
            for name, mod in _maskable_weights(self.model):
                if self.scope_check and not self.scope_check(name):
                    continue
                mask = _semi24_mask(mod.weight)
                self._set_mask(mod, mask)
                stats["masked_params"] += mask.numel()
                stats["zeroed"] += int((mask == 0).sum())
            return stats

        # global magnitude threshold across all maskable weights
        all_scores = []
        entries = []
        for name, mod in _maskable_weights(self.model):
            if self.scope_check and not self.scope_check(name):
                continue
            w = mod.weight.detach().abs().flatten()
            all_scores.append(w)
            entries.append((name, mod))
        if not entries:
            raise RuntimeError("no maskable weights found")
        flat = torch.cat(all_scores)
        k = int(flat.numel() * sparsity)
        threshold = flat.kthvalue(max(k, 1)).values if k > 0 else torch.tensor(-1.0)

        for name, mod in entries:
            mask = (mod.weight.detach().abs() > threshold).to(mod.weight.dtype)
            self._set_mask(mod, mask)
            stats["masked_params"] += mask.numel()
            stats["zeroed"] += int((mask == 0).sum())
        return stats

    @staticmethod
    def _set_mask(mod: nn.Module, mask: torch.Tensor):
        if MASK_NAME in mod._buffers:
            mod._buffers[MASK_NAME] = mask
        else:
            mod.register_buffer(MASK_NAME, mask)

    # -------------------------------------------------- application
    @torch.no_grad()
    def apply(self) -> dict:
        stats = self.compute_masks()
        reapply_masks(self.model)
        density = 1 - stats["zeroed"] / max(stats["masked_params"], 1)
        LOGGER.info(
            f"[PruneV2] {self.pattern} masks applied: {stats['zeroed']:,}/{stats['masked_params']:,} "
            f"zeroed (density {density:.2%})"
        )
        return stats

    # -------------------------------------------------- GMP schedule
    def gmp_sparsity_at(self, step: int) -> float:
        """Cubic ramp: s(t) = s_final * (1 - (1 - t/T)^3)."""
        if self.gmp_steps <= 0 or step >= self.gmp_steps:
            return self.sparsity
        t = max(step, 0) / self.gmp_steps
        return self.sparsity * (1 - (1 - t) ** 3)


# ------------------------------------------------------------------
# Trainer-facing helpers (no executor instance needed)
# ------------------------------------------------------------------

@torch.no_grad()
def reapply_masks(model: nn.Module) -> int:
    """Multiply every masked weight by its buffer mask; returns #masked layers."""
    n = 0
    for mod in model.modules():
        mask = mod._buffers.get(MASK_NAME) if hasattr(mod, "_buffers") else None
        if mask is not None and isinstance(mod, _MASKABLE) and mod.weight is not None:
            if mask.shape == mod.weight.shape:
                mod.weight.mul_(mask.to(mod.weight.dtype))
                n += 1
    return n


@torch.no_grad()
def zero_masked_gradients(model: nn.Module) -> None:
    """Zero gradients at masked positions (keeps clip_grad_norm budget honest)."""
    for mod in model.modules():
        mask = mod._buffers.get(MASK_NAME) if hasattr(mod, "_buffers") else None
        if mask is not None and isinstance(mod, _MASKABLE) and mod.weight is not None and mod.weight.grad is not None:
            if mask.shape == mod.weight.grad.shape:
                mod.weight.grad.mul_(mask.to(mod.weight.grad.dtype))


def has_sparsity_masks(model: nn.Module) -> bool:
    return any(
        hasattr(m, "_buffers") and m._buffers.get(MASK_NAME) is not None for m in model.modules()
    )


def measured_sparsity(model: nn.Module) -> float:
    """Actual zero fraction over masked weights (for assertions/reports)."""
    zeros = total = 0
    for mod in model.modules():
        mask = mod._buffers.get(MASK_NAME) if hasattr(mod, "_buffers") else None
        if mask is not None and isinstance(mod, _MASKABLE) and mod.weight is not None:
            w = mod.weight.detach()
            zeros += int((w == 0).sum())
            total += w.numel()
    return zeros / total if total else 0.0
