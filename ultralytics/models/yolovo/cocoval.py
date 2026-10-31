# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOVO COCO validators: detect (text-aware).

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from ultralytics.models.yolovo.val import YOLOVOValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM
from ultralytics.utils.coco_metrics import COCOMetrics


def _xyxy01_to_xywh_px(box_xyxy01, shape_hw):
    """Convert normalized xyxy -> pixel xywh for COCO format."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x1 = float(box_xyxy01[0]) * w
    y1 = float(box_xyxy01[1]) * h
    x2 = float(box_xyxy01[2]) * w
    y2 = float(box_xyxy01[3]) * h
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return [x1, y1, bw, bh]


class YOLOVOCOCOValidator(YOLOVOValidator):
    """YOLOVO COCO validator for detection (text-aware).

    Inherits YOLOVOValidator's text-cache pipeline (set_input_text +
    MobileCLIP embeddings). Adds COCO metric collection via the
    ported COCOevalBBoxMM evaluator.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.coco_metrics = None
        self._coco_computed = False
        self.image_ori_shapes = []
        self.all_pred_boxes = []
        self.all_target_boxes = []
        self.all_pred_cls = []
        self.all_target_cls = []
        self.pred_to_img = []
        self.target_to_img = []
        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}

    def init_metrics(self, model):
        super().init_metrics(model)
        self.image_ori_shapes = []
        self.all_pred_boxes = []
        self.all_target_boxes = []
        self.all_pred_cls = []
        self.all_target_cls = []
        self.pred_to_img = []
        self.target_to_img = []
        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}
        self._coco_computed = False
        self.coco_metrics = COCOMetrics(
            save_dir=self.save_dir,
            names=getattr(model, "names", {}),
            plot=getattr(self.args, "plots", False),
            on_plot=getattr(self, "on_plot", None),
        )

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)
            ori_shape = pbatch["ori_shape"]
            self.image_ori_shapes.append(ori_shape)
            img_idx = len(self.image_ori_shapes) - 1
            h, w = int(ori_shape[0]), int(ori_shape[1])
            if w == 0 or h == 0:
                continue
            scale = torch.tensor([w, h, w, h], device=predn["bboxes"].device, dtype=predn["bboxes"].dtype)
            if predn["bboxes"] is not None and predn["bboxes"].numel() > 0:
                bboxes_norm = (predn["bboxes"] / scale).clamp_(0, 1)
                for i in range(bboxes_norm.shape[0]):
                    self.all_pred_boxes.append(bboxes_norm[i].detach().cpu().numpy())
                    self.all_pred_cls.append(predn["cls"][i].detach().cpu().numpy())
                    self.pred_to_img.append(img_idx)
            if pbatch["bboxes"] is not None and pbatch["bboxes"].numel() > 0:
                gt_norm = (pbatch["bboxes"] / scale).clamp_(0, 1)
                gt_cls = pbatch["cls"]
                for i in range(gt_norm.shape[0]):
                    self.all_target_boxes.append(gt_norm[i].detach().cpu().numpy())
                    self.all_target_cls.append(gt_cls[i].detach().cpu().numpy())
                    self.target_to_img.append(img_idx)

    def get_stats(self):
        if self.coco_metrics is None:
            return {}
        if not self._coco_computed:
            if hasattr(self, "metrics") and hasattr(self.metrics, "stats") and self.metrics.stats:
                self._compute_coco_metrics()
            else:
                self._set_default_coco_stats()
            self._coco_computed = True
        base_stats = super().get_stats()
        stats = {
            "metrics/coco/AP": getattr(self.coco_metrics, "AP", 0.0),
            "metrics/coco/AP50": getattr(self.coco_metrics, "AP50", 0.0),
            "metrics/coco/AP75": getattr(self.coco_metrics, "AP75", 0.0),
            "metrics/coco/APsmall": getattr(self.coco_metrics, "APsmall", 0.0),
            "metrics/coco/APmedium": getattr(self.coco_metrics, "APmedium", 0.0),
            "metrics/coco/APlarge": getattr(self.coco_metrics, "APlarge", 0.0),
            "metrics/coco/AR1": getattr(self.coco_metrics, "AR1", 0.0),
            "metrics/coco/AR10": getattr(self.coco_metrics, "AR10", 0.0),
            "metrics/coco/AR100": getattr(self.coco_metrics, "AR100", 0.0),
            "metrics/coco/ARsmall": getattr(self.coco_metrics, "ARsmall", 0.0),
            "metrics/coco/ARmedium": getattr(self.coco_metrics, "ARmedium", 0.0),
            "metrics/coco/ARlarge": getattr(self.coco_metrics, "ARlarge", 0.0),
            "fitness": getattr(self.coco_metrics, "AP", 0.0),
        }
        try:
            if isinstance(base_stats, dict):
                base_stats.update(stats)
                return base_stats
        except Exception:
            pass
        return stats

    def print_results(self):
        """COCO 模式下跳过 YOLO 口径表格输出，避免 conf=0.05 下的 YOLO 指标误导。

        COCO 指标统一由 eval_json 后的 COCO 汇总输出；此处仅保留警告检查。
        """
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

    def eval_json(self, stats):
        """标准 JSON 评估后，输出 COCO 指标汇总（替代被屏蔽的 YOLO 口径表格）。"""
        stats = super().eval_json(stats)
        if self.coco_metrics is not None:
            cm = self.coco_metrics
            LOGGER.info(
                "COCO: AP=%.4f AP50=%.4f AP75=%.4f | APsmall=%.4f APmedium=%.4f APlarge=%.4f | "
                "AR1=%.4f AR10=%.4f AR100=%.4f",
                getattr(cm, "AP", 0.0), getattr(cm, "AP50", 0.0), getattr(cm, "AP75", 0.0),
                getattr(cm, "APsmall", 0.0), getattr(cm, "APmedium", 0.0), getattr(cm, "APlarge", 0.0),
                getattr(cm, "AR1", 0.0), getattr(cm, "AR10", 0.0), getattr(cm, "AR100", 0.0),
            )
        return stats

    def _compute_coco_metrics(self):
        try:
            stats = self.metrics.stats
            tp = np.concatenate(stats["tp"], axis=0) if stats["tp"] else np.array([])
            conf = np.concatenate(stats["conf"], axis=0) if stats["conf"] else np.array([])
            pred_cls = np.concatenate(stats["pred_cls"], axis=0) if stats["pred_cls"] else np.array([])
            target_cls = np.concatenate(stats["target_cls"], axis=0) if stats["target_cls"] else np.array([])
            if conf.size == 0 or len(target_cls) == 0:
                self._set_default_coco_stats()
                return
            if not self.all_pred_boxes or not self.all_target_boxes or not self.image_ori_shapes:
                self._set_default_coco_stats()
                return
            pred_boxes = np.array(self.all_pred_boxes)
            target_boxes = np.array(self.all_target_boxes)
            pred_to_img = np.array(self.pred_to_img)
            target_to_img = np.array(self.target_to_img)
            ori_shapes = self.image_ori_shapes
            img_count = len(ori_shapes) if isinstance(ori_shapes, (list, tuple)) else int(np.max(pred_to_img)) + 1
            imgIds = list(range(img_count))
            dts = []
            for i in range(len(conf)):
                img_id = int(pred_to_img[i])
                bbox_xywh = _xyxy01_to_xywh_px(pred_boxes[i], ori_shapes[img_id])
                dts.append({
                    "image_id": img_id,
                    "category_id": int(pred_cls[i]),
                    "bbox": bbox_xywh,
                    "score": float(conf[i]),
                    "id": i + 1,
                    "area": max(0.0, bbox_xywh[2]) * max(0.0, bbox_xywh[3]),
                })
            gts = []
            for j in range(len(target_cls)):
                img_id = int(target_to_img[j])
                bbox_xywh = _xyxy01_to_xywh_px(target_boxes[j], ori_shapes[img_id])
                gts.append({
                    "image_id": img_id,
                    "category_id": int(target_cls[j]),
                    "bbox": bbox_xywh,
                    "iscrowd": 0,
                    "ignore": 0,
                    "id": j + 1,
                    "area": max(0.0, bbox_xywh[2]) * max(0.0, bbox_xywh[3]),
                })
            catIds = sorted(list({d["category_id"] for d in dts} | {g["category_id"] for g in gts}))
            evaluator = COCOevalBBoxMM()
            evaluator.set_data(gts=gts, dts=dts, imgIds=imgIds, catIds=catIds)
            evaluator.evaluate()
            evaluator.accumulate()
            coco_stats = evaluator.summarize()
            for k, v in list(coco_stats.items()):
                if v == -1:
                    coco_stats[k] = 0.0
            self.coco_metrics.update({
                "AP": coco_stats.get("AP", 0.0),
                "AP50": coco_stats.get("AP50", 0.0),
                "AP75": coco_stats.get("AP75", 0.0),
                "APsmall": coco_stats.get("APsmall", 0.0),
                "APmedium": coco_stats.get("APmedium", 0.0),
                "APlarge": coco_stats.get("APlarge", 0.0),
                "APsmall50": coco_stats.get("APsmall50", 0.0),
                "APsmall75": coco_stats.get("APsmall75", 0.0),
                "APmedium50": coco_stats.get("APmedium50", 0.0),
                "APmedium75": coco_stats.get("APmedium75", 0.0),
                "APlarge50": coco_stats.get("APlarge50", 0.0),
                "APlarge75": coco_stats.get("APlarge75", 0.0),
                "AR1": coco_stats.get("AR1", 0.0),
                "AR10": coco_stats.get("AR10", 0.0),
                "AR100": coco_stats.get("AR100", 0.0),
                "ARsmall": coco_stats.get("ARsmall", 0.0),
                "ARmedium": coco_stats.get("ARmedium", 0.0),
                "ARlarge": coco_stats.get("ARlarge", 0.0),
            })
        except Exception as e:
            LOGGER.error(f"YOLOVO COCO metrics error: {e}")
            self._set_default_coco_stats()

    def _set_default_coco_stats(self):
        default_stats = {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "APsmall": 0.0, "APmedium": 0.0, "APlarge": 0.0,
            "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
            "ARsmall": 0.0, "ARmedium": 0.0, "ARlarge": 0.0,
        }
        self.coco_metrics.update(default_stats)
