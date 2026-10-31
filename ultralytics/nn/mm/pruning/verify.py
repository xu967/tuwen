# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Post-prune verification: the last line of defense against silent failure.

Three checks (boundary-hunter TOP-7):
1. Dual-mode dry-run: eval() and train() forward must both succeed (training
   only branches like SemSegHead.aux_head crash here if mishandled).
2. Parameter-set fingerprint: names/shapes must match the rewrite plan exactly
   and must not change across a forward — catches SSA/EDFFN-style silent
   rebuild-with-random-weights, the only failure mode that never raises.
3. Entry preconditions: refuse fused models and heads needing extra inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER


class PruneVerificationError(RuntimeError):
    pass


def check_entry_preconditions(model: nn.Module):
    """Refuse models the engine cannot safely trace or rewrite."""
    is_fused = getattr(model, "is_fused", None)
    if callable(is_fused):
        try:
            if model.is_fused():
                raise PruneVerificationError(
                    "model is fused (Conv+BN merged); prune before fuse()/export"
                )
        except TypeError:
            pass
    for m in model.modules():
        tname = type(m).__name__
        if tname in ("WorldDetect", "YOLOEDetect", "YOLOESegment"):
            raise PruneVerificationError(
                f"{tname} requires extra (text) inputs; this head family is not supported for pruning"
            )
        if tname in ("v10Detect", "v26Detect") and getattr(m, "one2one_cv2", None) is None:
            raise PruneVerificationError(f"{tname} was end2end-fused; prune before fuse()")


def _param_fingerprint(model: nn.Module) -> dict[str, tuple]:
    return {n: tuple(p.shape) for n, p in model.named_parameters()}


@torch.no_grad()
def verify_pruned_model(model: nn.Module, example: torch.Tensor, expected_shapes: dict[str, tuple]) -> dict:
    """Run the triple check; raises PruneVerificationError on any failure."""
    report = {}

    # 1a. fingerprint vs rewrite plan
    actual = _param_fingerprint(model)
    mismatched = {
        n: (expected_shapes[n], actual[n])
        for n in expected_shapes
        if n in actual and actual[n] != expected_shapes[n]
    }
    if mismatched:
        sample = dict(list(mismatched.items())[:5])
        raise PruneVerificationError(f"parameter shapes diverge from rewrite plan: {sample}")

    # 1b. eval dry-run
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(mode=False):
            out = model(example)
        report["eval_forward"] = "ok"
    except Exception as e:
        raise PruneVerificationError(f"eval forward failed after pruning: {type(e).__name__}: {e}") from e

    # 2. train dry-run (training-only branches) + buffer snapshot/restore
    buf_snap = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.train()
    try:
        with torch.inference_mode(mode=False):
            model(example)
        report["train_forward"] = "ok"
    except Exception as e:
        raise PruneVerificationError(f"train forward failed after pruning: {type(e).__name__}: {e}") from e
    finally:
        model.eval()
        for n, b in model.named_buffers():
            if n in buf_snap and b.shape == buf_snap[n].shape:
                b.copy_(buf_snap[n])
        if was_training:
            model.train()

    # 3. no-silent-rebuild fingerprint: identical parameter set after forwards
    after = _param_fingerprint(model)
    if set(after) != set(actual):
        added = set(after) - set(actual)
        removed = set(actual) - set(after)
        raise PruneVerificationError(
            f"parameter set mutated during forward (silent rebuild!): +{sorted(added)[:5]} -{sorted(removed)[:5]}"
        )
    changed = {n for n in after if after[n] != actual[n]}
    if changed:
        raise PruneVerificationError(f"parameter shapes mutated during forward: {sorted(changed)[:5]}")

    LOGGER.info("[PruneV2] verification passed: eval+train dry-run, fingerprint stable")
    return report
