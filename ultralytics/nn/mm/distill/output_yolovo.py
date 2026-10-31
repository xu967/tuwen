# Ultralytics YOLOVO Output Distillation
# Computes objectness, dynamic cls, decoded box, and DFL KD losses.

"""
YOLOVO output-level knowledge distillation.

YOLOVO detection head outputs differ from YOLOMM:
- objectness (per-anchor binary confidence)
- dynamic cls logits via einsum(embeddings, t_enhanced)
- decoded boxes + DFL distribution

This module:
1. Standardises YOLOVODetect training dict into YOLOVOOutputBundle.
2. Computes foreground mask using objectness * max(dynamic cls).
3. Applies per-component KD losses: obj/cls/loc/dfl.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.head import YOLOVODetect
from ultralytics.utils.metrics import bbox_iou

_OUTPUT_DISTILL_WEIGHT = 0.5
_OUTPUT_OBJ_WEIGHT = 1.0
_OUTPUT_CLS_WEIGHT = 1.0
_OUTPUT_LOC_WEIGHT = 1.0
_OUTPUT_DFL_WEIGHT = 0.5
_OUTPUT_WARMUP_EPOCHS = 5
_FG_CONF_THR = 0.20
_FG_TOPK_PER_LEVEL = 100
_CLS_TEMPERATURE = 2.0
_DFL_TEMPERATURE = 2.0


@dataclass
class YOLOVOOutputBundle:
    """YOLOVO output KD standardised dense prediction bundle."""

    cls_logits: torch.Tensor          # [B, N, K]
    obj_logits: torch.Tensor          # [B, N, 1]
    box_decoded: torch.Tensor         # [B, N, 4], xyxy pixels
    dfl_logits: torch.Tensor          # [B, N, 4 * reg_max]
    final_scores: torch.Tensor        # [B, N], sigmoid(obj) * max sigmoid(cls)
    reg_max: int
    splits: List[int]


def _find_yolovo_head(model) -> YOLOVODetect:
    """Find YOLOVODetect head in model."""
    for module in model.modules():
        if isinstance(module, YOLOVODetect):
            return module
    raise RuntimeError("YOLOVO output distillation requires YOLOVODetect head")


def _compute_cls_logits(preds: dict, head: YOLOVODetect) -> torch.Tensor:
    """Compute dynamic class logits per YOLOVOLoss semantics."""
    embeddings = preds["embeddings"]                         # [B, D, N]
    t_enhanced = preds["text_condition"][0]                   # [B, K, D]
    emb_norm = F.normalize(embeddings.permute(0, 2, 1), dim=-1)
    t_norm = F.normalize(t_enhanced, dim=-1)
    cls_logits = torch.einsum("bnd,bkd->bnk", emb_norm, t_norm)
    scale = head.logit_scale.clamp(max=math.log(100)).exp()
    return cls_logits * scale + head.logit_bias


def build_yolovo_output_bundle(preds: dict, model) -> YOLOVOOutputBundle:
    """Convert YOLOVODetect training dict to KD bundle."""
    required = {"boxes", "objectness", "embeddings", "feats", "text_condition"}
    missing = required - set(preds)
    if missing:
        raise KeyError(f"YOLOVO preds missing keys: {sorted(missing)}")

    head = _find_yolovo_head(model)
    raw_boxes = preds["boxes"]                                # [B, 4*R, N]
    obj_logits = preds["objectness"].permute(0, 2, 1)          # [B, N, 1]
    feats = preds["feats"]
    splits = [int(f.shape[-2] * f.shape[-1]) for f in feats]

    decoded = head.decode_bboxes(head.dfl(raw_boxes), head.anchors.unsqueeze(0), xywh=False) * head.strides
    box_decoded = decoded.permute(0, 2, 1).contiguous()        # [B, N, 4]
    dfl_logits = raw_boxes.permute(0, 2, 1).contiguous()       # [B, N, 4*R]
    cls_logits = _compute_cls_logits(preds, head)              # [B, N, K]
    final_scores = obj_logits.sigmoid().squeeze(-1) * cls_logits.sigmoid().max(dim=-1).values

    return YOLOVOOutputBundle(
        cls_logits=cls_logits,
        obj_logits=obj_logits,
        box_decoded=box_decoded,
        dfl_logits=dfl_logits,
        final_scores=final_scores,
        reg_max=int(head.reg_max),
        splits=splits,
    )


def _assert_pair(student: YOLOVOOutputBundle, teacher: YOLOVOOutputBundle):
    """Check v1 same-structure KD prerequisites."""
    if student.cls_logits.shape[:2] != teacher.cls_logits.shape[:2]:
        raise ValueError(
            "YOLOVO output KD requires same B and N. "
            f"student={tuple(student.cls_logits.shape)}, teacher={tuple(teacher.cls_logits.shape)}"
        )
    if student.cls_logits.shape[-1] != teacher.cls_logits.shape[-1]:
        raise ValueError(
            "YOLOVO output KD requires same dynamic K prompts. "
            f"student K={student.cls_logits.shape[-1]}, teacher K={teacher.cls_logits.shape[-1]}"
        )
    if student.reg_max != teacher.reg_max:
        raise ValueError(
            "YOLOVO output KD v1 requires same reg_max for DFL KD. "
            f"student={student.reg_max}, teacher={teacher.reg_max}"
        )


def build_yolovo_fg_mask(
    teacher: YOLOVOOutputBundle,
    conf_thr: float = _FG_CONF_THR,
    topk_per_level: int = _FG_TOPK_PER_LEVEL,
) -> torch.Tensor:
    """Select foreground anchors by YOLOVO final score (obj * max(cls))."""
    scores = teacher.final_scores.detach()
    fg_mask = scores >= conf_thr

    if topk_per_level > 0:
        topk_mask = torch.zeros_like(fg_mask, dtype=torch.bool)
        start = 0
        for n_level in teacher.splits:
            end = start + n_level
            level_scores = scores[:, start:end]
            k = min(topk_per_level, level_scores.shape[1])
            if k > 0:
                _, idx = level_scores.topk(k, dim=1)
                topk_mask[:, start:end].scatter_(1, idx, True)
            start = end
        fg_mask = fg_mask | topk_mask

    return fg_mask


def compute_yolovo_obj_kd(student: YOLOVOOutputBundle, teacher: YOLOVOOutputBundle) -> torch.Tensor:
    """Objectness KD: teacher obj probability as soft target."""
    target = teacher.obj_logits.detach().sigmoid()
    return F.binary_cross_entropy_with_logits(student.obj_logits, target, reduction="mean")


def compute_yolovo_cls_kd(
    student: YOLOVOOutputBundle,
    teacher: YOLOVOOutputBundle,
    fg_mask: torch.Tensor,
    temperature: float = _CLS_TEMPERATURE,
) -> torch.Tensor:
    """Dynamic class KD: only on teacher foreground anchors."""
    if fg_mask.sum() == 0:
        return student.cls_logits.sum() * 0.0
    s_logits = student.cls_logits[fg_mask] / temperature
    t_probs = (teacher.cls_logits[fg_mask].detach() / temperature).sigmoid()
    return F.binary_cross_entropy_with_logits(s_logits, t_probs, reduction="mean") * (temperature ** 2)


def _compute_dfl_kl(
    student: YOLOVOOutputBundle,
    teacher: YOLOVOOutputBundle,
    fg_mask: torch.Tensor,
    temperature: float = _DFL_TEMPERATURE,
) -> torch.Tensor:
    """Raw DFL distribution KL."""
    if fg_mask.sum() == 0:
        return student.dfl_logits.new_zeros(())
    reg_max = student.reg_max
    s = student.dfl_logits[fg_mask].view(-1, 4, reg_max) / temperature
    t = teacher.dfl_logits[fg_mask].detach().view(-1, 4, reg_max) / temperature
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean") * (temperature ** 2)


def _valid_yolovo_box_mask(
    student: YOLOVOOutputBundle,
    teacher: YOLOVOOutputBundle,
    fg_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Return foreground positions with finite, non-degenerate xyxy boxes and DFL logits."""
    s_box, t_box = student.box_decoded, teacher.box_decoded
    s_wh = s_box[..., 2:] - s_box[..., :2]
    t_wh = t_box[..., 2:] - t_box[..., :2]

    return (
        fg_mask
        & torch.isfinite(s_box).all(dim=-1)
        & torch.isfinite(t_box).all(dim=-1)
        & torch.isfinite(student.dfl_logits).all(dim=-1)
        & torch.isfinite(teacher.dfl_logits).all(dim=-1)
        & (s_wh[..., 0] > eps)
        & (s_wh[..., 1] > eps)
        & (t_wh[..., 0] > eps)
        & (t_wh[..., 1] > eps)
    )


def compute_yolovo_loc_kd(
    student: YOLOVOOutputBundle,
    teacher: YOLOVOOutputBundle,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decoded box L1 + GIoU, and DFL KL."""
    valid_mask = _valid_yolovo_box_mask(student, teacher, fg_mask)
    if valid_mask.sum() == 0:
        zero = student.box_decoded.new_zeros(())
        return zero, zero.detach()

    s_box = student.box_decoded[valid_mask]
    t_box = teacher.box_decoded[valid_mask].detach()
    giou = bbox_iou(s_box, t_box, xywh=False, GIoU=True).squeeze(-1)
    giou_finite = torch.isfinite(giou)
    if not giou_finite.any():
        zero = student.box_decoded.new_zeros(())
        return zero, _compute_dfl_kl(student, teacher, valid_mask)
    s_box = s_box[giou_finite]
    t_box = t_box[giou_finite]
    giou = giou[giou_finite]
    l1_loss = F.l1_loss(s_box, t_box, reduction="mean")
    giou_loss = (1.0 - giou).mean()
    dfl_loss = _compute_dfl_kl(student, teacher, valid_mask)
    return l1_loss + giou_loss, dfl_loss


def _compute_warmup_factor(current_epoch: int, warmup_epochs: int) -> float:
    """Linear warmup, epoch from 0."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(current_epoch + 1) / float(warmup_epochs)))


def compute_yolovo_output_kd(
    student_preds: dict,
    teacher_preds: dict,
    student_model,
    teacher_model,
    current_epoch: int = 0,
    total_epochs: int = 100,
    obj_weight: float = _OUTPUT_OBJ_WEIGHT,
    cls_weight: float = _OUTPUT_CLS_WEIGHT,
    loc_weight: float = _OUTPUT_LOC_WEIGHT,
    dfl_weight: float = _OUTPUT_DFL_WEIGHT,
    overall_weight: float = _OUTPUT_DISTILL_WEIGHT,
    warmup_epochs: int = _OUTPUT_WARMUP_EPOCHS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute YOLOVO output distillation loss."""
    student = build_yolovo_output_bundle(student_preds, student_model)
    teacher = build_yolovo_output_bundle(teacher_preds, teacher_model)
    _assert_pair(student, teacher)

    fg_mask = build_yolovo_fg_mask(teacher)
    obj_kd = compute_yolovo_obj_kd(student, teacher)
    cls_kd = compute_yolovo_cls_kd(student, teacher, fg_mask)
    loc_kd, dfl_kd = compute_yolovo_loc_kd(student, teacher, fg_mask)

    raw_total = obj_weight * obj_kd + cls_weight * cls_kd + loc_weight * (loc_kd + dfl_weight * dfl_kd)
    warmup = _compute_warmup_factor(current_epoch, warmup_epochs)
    total = overall_weight * warmup * raw_total

    items = {
        "distill_output_obj": obj_kd.detach(),
        "distill_output_cls": cls_kd.detach(),
        "distill_output_loc": loc_kd.detach(),
        "distill_output_dfl": dfl_kd.detach(),
        "distill_output_total": total.detach(),
    }
    return total, items
