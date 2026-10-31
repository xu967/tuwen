# Ultralytics YOLOVO 🚀, AGPL-3.0 license
"""YOLOVO pose COCO validator.

Inherits YOLOVOPoseValidator (text-cache + pose pipeline).
Adds pose-level COCO (COCOevalPoseMM, OKS-based) + box-level COCO
(COCOevalBBoxMM) on top of the standard YOLO pose metrics.

Outputs:
  metrics/coco_pose/AP|AP50|AP75|APsmall|APmedium|APlarge|AR1|AR10|AR100|ARsmall|ARmedium|ARlarge
  metrics/coco_box/AP|AP50|AP75|...
  metrics/coco/AP = pose AP (primary)
  fitness = pose AP
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.models.yolovo.pose.val import YOLOVOPoseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM
from ultralytics.utils.coco_eval_pose_mm import COCOevalPoseMM
from ultralytics.utils.coco_metrics import COCOMetrics, COCO_AREA_MEDIUM, COCO_AREA_SMALL


class YOLOVOPoseCOCOValidator(YOLOVOPoseValidator):
    """YOLOVO pose COCO validator (OKS + box)."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

        self.coco_metrics_pose = None
        self.coco_metrics_box = None

        self._img_shapes = []
        self._dt_img = []
        self._gt_img = []
        self._dt_cls = []
        self._dt_conf = []
        self._dt_bbox_xyxy = []
        self._dt_kpts = []
        self._gt_cls = []
        self._gt_bbox_xyxy = []
        self._gt_kpts = []
        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}
        self._coco_computed = False

    def init_metrics(self, model):
        super().init_metrics(model)
        self.coco_metrics_pose = COCOMetrics(
            save_dir=self.save_dir, names=getattr(model, "names", {}), plot=False
        )
        self.coco_metrics_box = COCOMetrics(
            save_dir=self.save_dir, names=getattr(model, "names", {}), plot=False
        )
        self._img_shapes = []
        self._dt_img, self._gt_img = [], []
        self._dt_cls, self._dt_conf, self._dt_bbox_xyxy, self._dt_kpts = [], [], [], []
        self._gt_cls, self._gt_bbox_xyxy, self._gt_kpts = [], [], []
        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}
        self._coco_computed = False

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)

        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)

            ori_shape = pbatch["ori_shape"]
            self._img_shapes.append((int(ori_shape[0]), int(ori_shape[1])))
            img_id = len(self._img_shapes) - 1

            # ---- GT ----
            gt_cls = pbatch.get("cls", None)
            gt_bboxes = pbatch.get("bboxes", None)
            gt_kpts = pbatch.get("keypoints", None)

            if gt_cls is not None and gt_cls.numel() > 0 and gt_kpts is not None:
                gt_cls_np = gt_cls.view(-1).detach().cpu().numpy()
                gt_bboxes_np = gt_bboxes.detach().cpu().numpy()
                gt_kpts_np = gt_kpts.detach().cpu().numpy()

                for i in range(len(gt_cls_np)):
                    self._gt_img.append(img_id)
                    self._gt_cls.append(int(gt_cls_np[i]))
                    self._gt_bbox_xyxy.append(gt_bboxes_np[i])
                    self._gt_kpts.append(gt_kpts_np[i].reshape(-1).tolist())

                    x1, y1, x2, y2 = gt_bboxes_np[i]
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area < COCO_AREA_SMALL:
                        self.gt_size_counts["small"] += 1
                    elif area < COCO_AREA_MEDIUM:
                        self.gt_size_counts["medium"] += 1
                    else:
                        self.gt_size_counts["large"] += 1

            # ---- DT ----
            dt_bboxes = predn.get("bboxes", None)
            dt_conf = predn.get("conf", None)
            dt_cls = predn.get("cls", None)
            dt_kpts = predn.get("keypoints", None)

            if dt_bboxes is not None and dt_bboxes.numel() > 0 and dt_kpts is not None:
                dt_bboxes_np = dt_bboxes.detach().cpu().numpy()
                dt_cls_np = dt_cls.detach().cpu().numpy()
                dt_conf_np = dt_conf.detach().cpu().numpy()
                dt_kpts_np = dt_kpts.detach().cpu().numpy()

                for i in range(len(dt_conf_np)):
                    self._dt_img.append(img_id)
                    self._dt_cls.append(int(dt_cls_np[i]))
                    self._dt_conf.append(float(dt_conf_np[i]))
                    self._dt_bbox_xyxy.append(dt_bboxes_np[i])
                    self._dt_kpts.append(dt_kpts_np[i].reshape(-1).tolist())

    def get_stats(self):
        if not self._coco_computed:
            self._compute_coco_metrics()
            self._coco_computed = True

        base_stats = super().get_stats()
        out = dict(base_stats) if isinstance(base_stats, dict) else {}

        if self.coco_metrics_pose:
            for k in ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                      'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'):
                out[f'metrics/coco_pose/{k}'] = getattr(self.coco_metrics_pose, k, 0.0)
        if self.coco_metrics_box:
            for k in ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                      'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'):
                out[f'metrics/coco_box/{k}'] = getattr(self.coco_metrics_box, k, 0.0)

        # Primary COCO metric = pose AP (OKS)
        out['metrics/coco/AP'] = getattr(self.coco_metrics_pose, 'AP', 0.0) if self.coco_metrics_pose else 0.0
        out['metrics/coco/AP50'] = getattr(self.coco_metrics_pose, 'AP50', 0.0) if self.coco_metrics_pose else 0.0
        out['metrics/coco/AP75'] = getattr(self.coco_metrics_pose, 'AP75', 0.0) if self.coco_metrics_pose else 0.0
        out['metrics/coco/AR100'] = getattr(self.coco_metrics_pose, 'AR100', 0.0) if self.coco_metrics_pose else 0.0
        out['fitness'] = float(out['metrics/coco/AP'])

        return out

    def print_results(self):
        """Skip YOLO table (conf=0.05 misleading), emit COCO summary in eval_json."""
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

    def eval_json(self, stats):
        """Standard JSON eval, then print COCO pose summary."""
        stats = super().eval_json(stats)
        if self.coco_metrics_pose and self.coco_metrics_box:
            cm = self.coco_metrics_pose
            cb = self.coco_metrics_box
            LOGGER.info(
                "COCO(pose): pose AP=%.4f AP50=%.4f AP75=%.4f | box AP=%.4f AP50=%.4f AP75=%.4f",
                getattr(cm, "AP", 0.0), getattr(cm, "AP50", 0.0), getattr(cm, "AP75", 0.0),
                getattr(cb, "AP", 0.0), getattr(cb, "AP50", 0.0), getattr(cb, "AP75", 0.0),
            )
        return stats

    def _compute_coco_metrics(self):
        if len(self._img_shapes) == 0:
            self._set_default_stats()
            return

        imgIds = list(range(len(self._img_shapes)))
        catIds = sorted({c for c in self._dt_cls} | {c for c in self._gt_cls})
        if not catIds:
            self._set_default_stats()
            return

        # ---- Pose (OKS) ----
        pose_gts, pose_dts = [], []
        for j in range(len(self._gt_cls)):
            bbox_xywh = self._xyxy_to_xywh(self._gt_bbox_xyxy[j])
            area = bbox_xywh[2] * bbox_xywh[3]
            pose_gts.append({
                "image_id": int(self._gt_img[j]),
                "category_id": int(self._gt_cls[j]),
                "keypoints": self._gt_kpts[j],
                "bbox": bbox_xywh,
                "area": float(area),
                "iscrowd": 0, "ignore": 0,
                "id": j + 1,
            })
        for i in range(len(self._dt_conf)):
            bbox_xywh = self._xyxy_to_xywh(self._dt_bbox_xyxy[i])
            area = bbox_xywh[2] * bbox_xywh[3]
            pose_dts.append({
                "image_id": int(self._dt_img[i]),
                "category_id": int(self._dt_cls[i]),
                "keypoints": self._dt_kpts[i],
                "bbox": bbox_xywh,
                "score": float(self._dt_conf[i]),
                "area": float(area),
                "id": i + 1,
            })

        pose_eval = COCOevalPoseMM()
        pose_eval.set_data(gts=pose_gts, dts=pose_dts, imgIds=imgIds, catIds=catIds)
        pose_eval.evaluate()
        pose_eval.accumulate()
        pose_stats = pose_eval.summarize()
        pose_per_class = pose_eval.compute_per_class_metrics()
        for k, v in list(pose_stats.items()):
            if v == -1:
                pose_stats[k] = 0.0
        self.coco_metrics_pose.update(pose_stats)
        self.coco_metrics_pose.per_class_metrics = pose_per_class

        # ---- Box ----
        box_gts, box_dts = [], []
        for j in range(len(self._gt_cls)):
            bbox_xywh = self._xyxy_to_xywh(self._gt_bbox_xyxy[j])
            box_gts.append({
                "image_id": int(self._gt_img[j]),
                "category_id": int(self._gt_cls[j]),
                "bbox": bbox_xywh,
                "area": float(bbox_xywh[2] * bbox_xywh[3]),
                "iscrowd": 0, "ignore": 0,
                "id": j + 1,
            })
        for i in range(len(self._dt_conf)):
            bbox_xywh = self._xyxy_to_xywh(self._dt_bbox_xyxy[i])
            box_dts.append({
                "image_id": int(self._dt_img[i]),
                "category_id": int(self._dt_cls[i]),
                "bbox": bbox_xywh,
                "score": float(self._dt_conf[i]),
                "area": float(bbox_xywh[2] * bbox_xywh[3]),
                "id": i + 1,
            })

        box_eval = COCOevalBBoxMM()
        box_eval.set_data(gts=box_gts, dts=box_dts, imgIds=imgIds, catIds=catIds)
        box_eval.evaluate()
        box_eval.accumulate()
        box_stats = box_eval.summarize()
        box_per_class = box_eval.compute_per_class_metrics()
        for k, v in list(box_stats.items()):
            if v == -1:
                box_stats[k] = 0.0
        self.coco_metrics_box.update(box_stats)
        self.coco_metrics_box.per_class_metrics = box_per_class

    def _set_default_stats(self):
        defaults = {
            'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
            'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
            'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
            'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0,
        }
        if self.coco_metrics_pose:
            self.coco_metrics_pose.update(defaults)
        if self.coco_metrics_box:
            self.coco_metrics_box.update(defaults)

    @staticmethod
    def _xyxy_to_xywh(box_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        return [x1, y1, x2 - x1, y2 - y1]
