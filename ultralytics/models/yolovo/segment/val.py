# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO instance segmentation validator."""

import torch

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolovo.val import YOLOVOValidator
from ultralytics.utils.plotting import plot_images


class YOLOVOSegValidator(YOLOVOValidator, SegmentationValidator):
    """Segmentation validator with YOLOVO text-cache lifecycle + instance segmentation mask plotting."""

    def plot_val_samples(self, batch, ni):
        """Plot validation samples with instance segmentation masks."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1) if batch["cls"].ndim > 1 else batch["cls"],
            bboxes=batch["bboxes"],
            masks=batch.get("masks"),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plot predictions with instance segmentation masks."""
        from ultralytics.utils import ops

        # 合并所有 batch 的 preds（postprocess 后每个 pred dict 有 bboxes/cls/conf/masks）
        batch_idx_list = []
        cls_list = []
        bboxes_list = []
        confs_list = []
        masks_list = []

        for i, pred in enumerate(preds):
            n = len(pred["conf"])
            if n == 0:
                continue
            batch_idx_list.append(torch.ones(n, device=pred["conf"].device, dtype=torch.long) * i)
            cls_list.append(pred["cls"])
            bboxes_list.append(pred["bboxes"])
            confs_list.append(pred["conf"])
            m = pred.get("masks")
            if m is not None and m.numel() > 0:
                masks_list.append(m.cpu())

        if not batch_idx_list:
            plot_images(
                images=batch["img"],
                batch_idx=torch.zeros(0, dtype=torch.long),
                cls=torch.zeros(0, dtype=torch.long),
                bboxes=torch.zeros(0, 4),
                paths=batch["im_file"],
                fname=self.save_dir / f"val_batch{ni}_pred.jpg",
                names=self.names,
                on_plot=self.on_plot,
            )
            return

        batch_idx = torch.cat(batch_idx_list)
        cls = torch.cat(cls_list)
        bboxes = torch.cat(bboxes_list)
        confs = torch.cat(confs_list)

        # postprocess 返回 xyxy，plot_images 需要 xywh
        bboxes_xywh = ops.xyxy2xywh(bboxes[:, :4]) if bboxes.shape[-1] >= 4 else bboxes
        masks = torch.cat(masks_list, dim=0) if masks_list else None

        plot_images(
            images=batch["img"],
            batch_idx=batch_idx,
            cls=cls,
            bboxes=bboxes_xywh,
            confs=confs,
            masks=masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
