# Ultralytics YOLOVO 🚀, AGPL-3.0 license
"""YOLOVO segment COCO validator.

Inherits YOLOVOSegValidator (text-cache + segment mask plotting).
Adds mask-level COCO (COCOevalSegmMM) + box-level COCO (COCOevalBBoxMM)
on top of the standard YOLO segment metrics.

Outputs:
  metrics/coco_mask/AP|AP50|AP75|APsmall|APmedium|APlarge|AR1|AR10|AR100|ARsmall|ARmedium|ARlarge
  metrics/coco_box/AP|AP50|AP75|...
  metrics/coco/AP = mask AP (primary)
  fitness = mask AP
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.models.yolovo.segment.val import YOLOVOSegValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM
from ultralytics.utils.coco_eval_segm_mm import COCOevalSegmMM, rle_area, rle_encode
from ultralytics.utils.coco_metrics import COCOMetrics, COCO_AREA_MEDIUM, COCO_AREA_SMALL


class YOLOVOSegCOCOValidator(YOLOVOSegValidator):
    """YOLOVO instance-segmentation COCO validator."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

        self.coco_metrics_mask = None
        self.coco_metrics_box = None

        self._img_shapes = []
        self._dt_img = []
        self._gt_img = []

        self._dt_cls = []
        self._dt_conf = []
        self._dt_bbox_xyxy = []
        self._dt_mask_rle = []
        self._dt_mask_area = []

        self._gt_cls = []
        self._gt_bbox_xyxy = []
        self._gt_mask_rle = []
        self._gt_mask_area = []

        self.gt_size_counts_mask = {"small": 0, "medium": 0, "large": 0}
        self.gt_size_counts_box = {"small": 0, "medium": 0, "large": 0}

        self._coco_computed = False

    def init_metrics(self, model):
        super().init_metrics(model)
        self.coco_metrics_mask = COCOMetrics(
            save_dir=self.save_dir, names=getattr(model, "names", {}), plot=False
        )
        self.coco_metrics_box = COCOMetrics(
            save_dir=self.save_dir, names=getattr(model, "names", {}), plot=False
        )
        self._img_shapes = []
        self._dt_img, self._gt_img = [], []
        self._dt_cls, self._dt_conf, self._dt_bbox_xyxy = [], [], []
        self._dt_mask_rle, self._dt_mask_area = [], []
        self._gt_cls, self._gt_bbox_xyxy = [], []
        self._gt_mask_rle, self._gt_mask_area = [], []
        self.gt_size_counts_mask = {"small": 0, "medium": 0, "large": 0}
        self.gt_size_counts_box = {"small": 0, "medium": 0, "large": 0}
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
            gt_masks = pbatch.get("masks", None)
            if gt_cls is None or gt_bboxes is None or gt_masks is None:
                continue

            gt_cls_flat = gt_cls.view(-1)
            if gt_cls_flat.numel() > 0:
                gt_masks_bin = self._get_instance_masks(
                    gt_cls_flat, gt_masks, overlap_mask=bool(self.args.overlap_mask)
                )
                gt_masks_rle, gt_mask_area = self._scale_and_encode_masks(
                    gt_masks_bin, ori_shape, pbatch.get("ratio_pad", None)
                )
                gt_bboxes_xyxy = gt_bboxes.detach().cpu().numpy()
                gt_cls_np = gt_cls_flat.detach().cpu().numpy()

                for i in range(len(gt_cls_np)):
                    self._gt_img.append(img_id)
                    self._gt_cls.append(int(gt_cls_np[i]))
                    self._gt_bbox_xyxy.append(gt_bboxes_xyxy[i])
                    self._gt_mask_rle.append(gt_masks_rle[i])
                    self._gt_mask_area.append(float(gt_mask_area[i]))

                for a in gt_mask_area.tolist():
                    if a < COCO_AREA_SMALL:
                        self.gt_size_counts_mask["small"] += 1
                    elif a < COCO_AREA_MEDIUM:
                        self.gt_size_counts_mask["medium"] += 1
                    else:
                        self.gt_size_counts_mask["large"] += 1

                gt_box_area = self._xyxy_area_np(gt_bboxes_xyxy)
                for a in gt_box_area.tolist():
                    if a < COCO_AREA_SMALL:
                        self.gt_size_counts_box["small"] += 1
                    elif a < COCO_AREA_MEDIUM:
                        self.gt_size_counts_box["medium"] += 1
                    else:
                        self.gt_size_counts_box["large"] += 1

            # ---- DT ----
            dt_bboxes = predn.get("bboxes", None)
            dt_conf = predn.get("conf", None)
            dt_cls = predn.get("cls", None)
            dt_masks = predn.get("masks", None)
            if dt_masks is None:
                continue

            if dt_bboxes is not None and dt_bboxes.numel() > 0:
                dt_masks_rle, dt_mask_area = self._scale_and_encode_masks(
                    dt_masks, ori_shape, pbatch.get("ratio_pad", None)
                )
                dt_bboxes_xyxy = dt_bboxes.detach().cpu().numpy()
                dt_cls_np = dt_cls.detach().cpu().numpy()
                dt_conf_np = dt_conf.detach().cpu().numpy()

                for i in range(len(dt_conf_np)):
                    self._dt_img.append(img_id)
                    self._dt_cls.append(int(dt_cls_np[i]))
                    self._dt_conf.append(float(dt_conf_np[i]))
                    self._dt_bbox_xyxy.append(dt_bboxes_xyxy[i])
                    self._dt_mask_rle.append(dt_masks_rle[i])
                    self._dt_mask_area.append(float(dt_mask_area[i]))

    def get_stats(self):
        base_stats = super().get_stats()

        if not self._coco_computed:
            self._compute_coco_metrics()
            self._coco_computed = True

        out = dict(base_stats) if isinstance(base_stats, dict) else {}

        if self.coco_metrics_mask:
            cm = self.coco_metrics_mask
            out["metrics/coco_mask/AP"] = getattr(cm, "AP", 0.0)
            out["metrics/coco_mask/AP50"] = getattr(cm, "AP50", 0.0)
            out["metrics/coco_mask/AP75"] = getattr(cm, "AP75", 0.0)
            out["metrics/coco_mask/APsmall"] = getattr(cm, "APsmall", 0.0)
            out["metrics/coco_mask/APmedium"] = getattr(cm, "APmedium", 0.0)
            out["metrics/coco_mask/APlarge"] = getattr(cm, "APlarge", 0.0)
            out["metrics/coco_mask/AR1"] = getattr(cm, "AR1", 0.0)
            out["metrics/coco_mask/AR10"] = getattr(cm, "AR10", 0.0)
            out["metrics/coco_mask/AR100"] = getattr(cm, "AR100", 0.0)
            out["metrics/coco_mask/ARsmall"] = getattr(cm, "ARsmall", 0.0)
            out["metrics/coco_mask/ARmedium"] = getattr(cm, "ARmedium", 0.0)
            out["metrics/coco_mask/ARlarge"] = getattr(cm, "ARlarge", 0.0)

        if self.coco_metrics_box:
            cm = self.coco_metrics_box
            out["metrics/coco_box/AP"] = getattr(cm, "AP", 0.0)
            out["metrics/coco_box/AP50"] = getattr(cm, "AP50", 0.0)
            out["metrics/coco_box/AP75"] = getattr(cm, "AP75", 0.0)
            out["metrics/coco_box/APsmall"] = getattr(cm, "APsmall", 0.0)
            out["metrics/coco_box/APmedium"] = getattr(cm, "APmedium", 0.0)
            out["metrics/coco_box/APlarge"] = getattr(cm, "APlarge", 0.0)
            out["metrics/coco_box/AR1"] = getattr(cm, "AR1", 0.0)
            out["metrics/coco_box/AR10"] = getattr(cm, "AR10", 0.0)
            out["metrics/coco_box/AR100"] = getattr(cm, "AR100", 0.0)
            out["metrics/coco_box/ARsmall"] = getattr(cm, "ARsmall", 0.0)
            out["metrics/coco_box/ARmedium"] = getattr(cm, "ARmedium", 0.0)
            out["metrics/coco_box/ARlarge"] = getattr(cm, "ARlarge", 0.0)

        # Primary COCO metric = mask AP
        out["metrics/coco/AP"] = getattr(self.coco_metrics_mask, "AP", 0.0) if self.coco_metrics_mask else 0.0
        out["metrics/coco/AP50"] = getattr(self.coco_metrics_mask, "AP50", 0.0) if self.coco_metrics_mask else 0.0
        out["metrics/coco/AP75"] = getattr(self.coco_metrics_mask, "AP75", 0.0) if self.coco_metrics_mask else 0.0
        out["metrics/coco/AR100"] = getattr(self.coco_metrics_mask, "AR100", 0.0) if self.coco_metrics_mask else 0.0
        out["fitness"] = float(out["metrics/coco/AP"])

        return out

    def print_results(self):
        """Skip YOLO table (conf=0.05 misleading), emit COCO summary in eval_json."""
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

    def eval_json(self, stats):
        """Standard JSON eval, then print COCO segment summary."""
        stats = super().eval_json(stats)
        if self.coco_metrics_mask and self.coco_metrics_box:
            cm = self.coco_metrics_mask
            cb = self.coco_metrics_box
            LOGGER.info(
                "COCO(seg): mask AP=%.4f AP50=%.4f AP75=%.4f | box AP=%.4f AP50=%.4f AP75=%.4f",
                getattr(cm, "AP", 0.0), getattr(cm, "AP50", 0.0), getattr(cm, "AP75", 0.0),
                getattr(cb, "AP", 0.0), getattr(cb, "AP50", 0.0), getattr(cb, "AP75", 0.0),
            )
        return stats

    def _compute_coco_metrics(self):
        if len(self._img_shapes) == 0:
            self._set_default_stats()
            return

        imgIds = list(range(len(self._img_shapes)))

        # ---- Mask (segm) ----
        segm_dts = []
        for i in range(len(self._dt_conf)):
            segm_dts.append({
                "image_id": int(self._dt_img[i]),
                "category_id": int(self._dt_cls[i]),
                "score": float(self._dt_conf[i]),
                "segmentation": self._dt_mask_rle[i],
                "area": float(self._dt_mask_area[i]),
                "id": i + 1,
            })
        segm_gts = []
        for j in range(len(self._gt_cls)):
            segm_gts.append({
                "image_id": int(self._gt_img[j]),
                "category_id": int(self._gt_cls[j]),
                "segmentation": self._gt_mask_rle[j],
                "area": float(self._gt_mask_area[j]),
                "iscrowd": 0,
                "ignore": 0,
                "id": j + 1,
            })

        catIds = sorted({d["category_id"] for d in segm_dts} | {g["category_id"] for g in segm_gts})
        if not catIds:
            self._set_default_stats()
            return

        segm_eval = COCOevalSegmMM()
        segm_eval.set_data(gts=segm_gts, dts=segm_dts, imgIds=imgIds, catIds=catIds)
        segm_eval.evaluate()
        segm_eval.accumulate()
        segm_stats = segm_eval.summarize()
        segm_per_class = segm_eval.compute_per_class_metrics()

        for k, v in list(segm_stats.items()):
            if v == -1:
                segm_stats[k] = 0.0
        self.coco_metrics_mask.update(segm_stats)
        self.coco_metrics_mask.per_class_metrics = segm_per_class

        # ---- Box (bbox) ----
        box_dts = []
        for i in range(len(self._dt_conf)):
            xywh = self._xyxy_to_xywh(self._dt_bbox_xyxy[i])
            box_dts.append({
                "image_id": int(self._dt_img[i]),
                "category_id": int(self._dt_cls[i]),
                "bbox": xywh,
                "score": float(self._dt_conf[i]),
                "area": float(xywh[2] * xywh[3]),
                "id": i + 1,
            })
        box_gts = []
        for j in range(len(self._gt_cls)):
            xywh = self._xyxy_to_xywh(self._gt_bbox_xyxy[j])
            box_gts.append({
                "image_id": int(self._gt_img[j]),
                "category_id": int(self._gt_cls[j]),
                "bbox": xywh,
                "iscrowd": 0,
                "ignore": 0,
                "area": float(xywh[2] * xywh[3]),
                "id": j + 1,
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
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "APsmall": 0.0, "APmedium": 0.0, "APlarge": 0.0,
            "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
            "ARsmall": 0.0, "ARmedium": 0.0, "ARlarge": 0.0,
        }
        if self.coco_metrics_mask:
            self.coco_metrics_mask.update(defaults)
        if self.coco_metrics_box:
            self.coco_metrics_box.update(defaults)

    # ---- Helpers ----
    @staticmethod
    def _get_instance_masks(gt_cls: torch.Tensor, gt_masks: torch.Tensor, overlap_mask: bool) -> torch.Tensor:
        """Return shape [N, H, W] per-instance binary GT masks."""
        if gt_cls.numel() == 0:
            return gt_masks.new_zeros((0, gt_masks.shape[-2], gt_masks.shape[-1]))
        if not overlap_mask:
            if gt_masks.ndim != 3:
                raise ValueError(f"Expected GT masks shape [N,H,W] when overlap_mask=False, got {gt_masks.shape}")
            return gt_masks.gt(0.5).to(torch.uint8)
        # overlap_mask=True: single instance-id mask
        if gt_masks.ndim == 3 and gt_masks.shape[0] == 1:
            m = gt_masks[0]
        elif gt_masks.ndim == 2:
            m = gt_masks
        else:
            raise ValueError(f"Unexpected GT masks shape for overlap_mask=True: {gt_masks.shape}")
        nl = int(gt_cls.numel())
        index = torch.arange(nl, device=m.device).view(nl, 1, 1) + 1
        mm = m.repeat(nl, 1, 1)
        return (mm == index).to(torch.uint8)

    @staticmethod
    def _scale_and_encode_masks(masks: torch.Tensor, ori_shape, ratio_pad):
        """Scale masks to ori_shape and encode to RLE. Returns (rles, areas)."""
        if masks is None:
            return [], np.zeros((0,), dtype=np.float32)
        if isinstance(masks, torch.Tensor):
            if masks.numel() == 0:
                return [], np.zeros((0,), dtype=np.float32)
            if masks.dtype in (torch.float16, torch.float32, torch.float64):
                masks_u8 = masks.gt(0.5).to(torch.uint8).detach().cpu().numpy()
            else:
                masks_u8 = masks.gt(0).to(torch.uint8).detach().cpu().numpy()
        else:
            masks_u8 = np.asarray(masks, dtype=np.uint8)
            if masks_u8.size == 0:
                return [], np.zeros((0,), dtype=np.float32)
        if masks_u8.ndim != 3:
            raise ValueError(f"Expected masks shape [N,H,W], got {masks_u8.shape}")
        hwc = np.transpose(masks_u8, (1, 2, 0))
        scaled = ops.scale_image(hwc, ori_shape, ratio_pad=ratio_pad)
        if scaled.ndim == 2:
            scaled = scaled[:, :, None]
        nchw = np.transpose(scaled, (2, 0, 1))
        nchw = (nchw > 0).astype(np.uint8)
        rles = []
        areas = np.zeros((nchw.shape[0],), dtype=np.float32)
        for i in range(nchw.shape[0]):
            rle = rle_encode(nchw[i])
            rles.append(rle)
            areas[i] = float(rle_area(rle))
        return rles, areas

    @staticmethod
    def _xyxy_area_np(xyxy: np.ndarray) -> np.ndarray:
        xyxy = np.asarray(xyxy, dtype=np.float32)
        if xyxy.ndim == 1:
            xyxy = xyxy.reshape(1, -1)
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        return np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray):
        x1, y1, x2, y2 = xyxy.tolist()
        w = x2 - x1
        h = y2 - y1
        return [float(x1), float(y1), float(w), float(h)]
