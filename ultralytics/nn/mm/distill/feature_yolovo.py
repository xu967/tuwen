# Ultralytics YOLOVO Feature Distillation Guidance
# Foreground mask generation using YOLOVO final scores.

"""
YOLOVO feature-level distillation foreground guidance.

YOLOVO foreground score is computed as:
    final_score = sigmoid(objectness) * max(sigmoid(dynamic_cls))

This differs from YOLOMM which uses cls_logits directly.
We reuse YOLOMM's box rasterization for the soft feature mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch

from .feature_yolomm import build_yolomm_feature_mask as _build_box_feature_mask
from .output_yolovo import YOLOVOOutputBundle, build_yolovo_output_bundle

_YOLOVO_FEATURE_TOPK = 100
_FEATURE_CONF_THR = 0.20


@dataclass
class YOLOVOFeatureGuidance:
    """YOLOVO teacher foreground guidance for feature distillation."""

    boxes_xyxy: List[torch.Tensor]
    scores: List[torch.Tensor]


def build_yolovo_feature_guidance(
    teacher_preds,
    teacher_model,
    conf_thr: float = _FEATURE_CONF_THR,
    topk: int = _YOLOVO_FEATURE_TOPK,
) -> YOLOVOFeatureGuidance:
    """Generate YOLOVO foreground guidance using final score (obj * max(cls))."""
    bundle: YOLOVOOutputBundle = build_yolovo_output_bundle(teacher_preds, teacher_model)
    scores = bundle.final_scores.detach()
    boxes = bundle.box_decoded.detach()

    boxes_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    for b in range(scores.shape[0]):
        img_scores = scores[b]
        img_boxes = boxes[b]
        keep = img_scores >= conf_thr

        if keep.sum().item() == 0:
            boxes_list.append(torch.zeros(0, 4, device=img_boxes.device, dtype=img_boxes.dtype))
            scores_list.append(torch.zeros(0, device=img_scores.device, dtype=img_scores.dtype))
            continue

        valid_scores = img_scores[keep]
        valid_boxes = img_boxes[keep]
        if valid_scores.shape[0] > topk:
            valid_scores, idx = valid_scores.topk(topk)
            valid_boxes = valid_boxes[idx]

        boxes_list.append(valid_boxes)
        scores_list.append(valid_scores)

    return YOLOVOFeatureGuidance(boxes_xyxy=boxes_list, scores=scores_list)


def build_yolovo_feature_mask(
    guidance: YOLOVOFeatureGuidance,
    feature_shape: Union[Tuple[int, int, int, int], torch.Size],
    input_size: Union[Tuple[int, int], None] = None,
) -> torch.Tensor:
    """Reuse generic box -> soft feature mask implementation."""
    return _build_box_feature_mask(guidance=guidance, feature_shape=feature_shape, input_size=input_size)