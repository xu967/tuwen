# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Composite-module channel metadata synchronization.

The executor rewrites base layers (Conv/BN/Linear/GN); composite ultralytics
modules additionally cache construction-time channel scalars that their
forward paths read (C2f.forward_split / C2PSA.forward split on self.c, the
exporter switches C2f to forward_split). This is the small closed table the
review allowed: attribute assignment only, ~8 entries, derived from actual
tensor shapes after rewrite. Modules whose channel-coupled internals are
tainted (SSA._c, EDFFN._c, Attention heads) never change and need no sync.
"""

from __future__ import annotations

import torch.nn as nn

from ultralytics.utils import LOGGER


def _cv1_out(mod) -> int | None:
    cv1 = getattr(mod, "cv1", None)
    conv = getattr(cv1, "conv", None)
    return getattr(conv, "out_channels", None)


def _sync_half_of_cv1(mod) -> bool:
    out = _cv1_out(mod)
    if out is None:
        return False
    if out % 2:
        raise RuntimeError(
            f"{type(mod).__name__}: cv1 out_channels={out} is odd after pruning — "
            "sibling-equal constraint violated"
        )
    mod.c = out // 2
    return True


# type name -> sync fn(module) -> bool (True if attribute was updated)
METADATA_SYNC = {
    "C2f": _sync_half_of_cv1,
    "C3k2": _sync_half_of_cv1,
    "C2": _sync_half_of_cv1,
    "C2a": _sync_half_of_cv1,
    "C2fAttn": _sync_half_of_cv1,
    "C2fCIB": _sync_half_of_cv1,
    "C2PSA": _sync_half_of_cv1,
    "PSA": _sync_half_of_cv1,
    "A2C2f": _sync_half_of_cv1,
}


def sync_module_metadata(model: nn.Module) -> list[str]:
    """Walk the model and refresh cached channel scalars; returns updated fqns."""
    updated = []
    for name, mod in model.named_modules():
        fn = METADATA_SYNC.get(type(mod).__name__)
        if fn is None:
            continue
        old = getattr(mod, "c", None)
        try:
            if fn(mod) and getattr(mod, "c", None) != old:
                updated.append(f"{name}: c {old} -> {mod.c}")
        except RuntimeError as e:
            raise RuntimeError(f"metadata sync failed at {name}: {e}") from e
    if updated:
        LOGGER.info(f"[PruneV2] metadata sync: {len(updated)} modules updated")
    return updated
