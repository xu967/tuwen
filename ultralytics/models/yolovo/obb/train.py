# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO oriented bounding box trainer."""

import os
from copy import copy
from typing import Optional

import torch

from ultralytics.data.dataset import YOLOVODataset
from ultralytics.models.yolovo.train import YOLOVOTrainer
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.nn.tasks import YOLOVOOBBModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel


class YOLOVOOBBTrainer(YOLOVOTrainer):
    """Trainer for YOLOVO RGB+Text oriented bounding box detection."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        overrides["task"] = "obb"
        overrides["afss_task_name"] = "obb"
        super().__init__(cfg, overrides, _callbacks)
        setattr(self.args, "afss_task_name", "obb")
        self.afss_runtime = None
        self._warn_afss_noop()

    def _parse_distill_arg(self):
        """Keep distill interface stable but no-op for YOLOVO obb."""
        distill = getattr(self.args, "distill", None)
        if distill:
            LOGGER.warning("YOLOVO obb distill interface is reserved; current implementation is a no-op.")
        return None

    def _warn_afss_noop(self):
        """Warn when AFSS is requested for the reserved obb interface."""
        try:
            from ultralytics.engine.afss import AFSSConfig

            afss_config = AFSSConfig.from_args(self.args)
            if afss_config.enabled:
                LOGGER.warning("YOLOVO obb AFSS interface is reserved; current implementation is a no-op.")
        except Exception:
            for name in ("afss", "afss_enable", "enable_afss"):
                if bool(getattr(self.args, name, False)):
                    LOGGER.warning("YOLOVO obb AFSS interface is reserved; current implementation is a no-op.")
                    break

    def _ensure_afss_runtime(self, dataset=None):
        """Reserved AFSS hook; returns no runtime."""
        LOGGER.warning("YOLOVO obb AFSS runtime is not implemented; using no-op.")
        return None

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """Build YOLOVODataset with oriented bounding boxes enabled."""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        debug_enabled = bool(getattr(self.args, "yolovo_debug", False)) and RANK in {-1, 0}
        debug_dir = self.save_dir / "object_paste_debug" if debug_enabled else None
        return YOLOVODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="obb",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            yolovo_debug=debug_enabled and mode == "train",
            yolovo_object_paste_debug_dir=debug_dir,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOVOOBBModel instance."""
        model = YOLOVOOBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        model._debug_logger = self.dbg
        self._log_model_contract(model, cfg)
        if weights:
            ckpt = torch.load(weights, map_location="cpu") if isinstance(weights, (str, os.PathLike)) else weights
            ckpt_model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = ckpt_model.state_dict() if hasattr(ckpt_model, "state_dict") else ckpt_model
            if self._is_yolo11_ckpt(sd):
                LOGGER.info("Remapping YOLO11 checkpoint to YOLOVO obb layout")
                sd = self._remap_yolo11_to_vcte(sd)
                result = model.load_state_dict(sd, strict=False)
                self.dbg.log(
                    "",
                    "weight_load",
                    source="yolo11_remap",
                    missing=len(result.missing_keys),
                    unexpected=len(result.unexpected_keys),
                    missing_preview=",".join(result.missing_keys[:8]),
                    unexpected_preview=",".join(result.unexpected_keys[:8]),
                )
            else:
                model.load(ckpt)
                self.dbg.log("", "weight_load", source="yolovo_obb_checkpoint", path=str(weights))
        if verbose and RANK in {-1, 0}:
            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"YOLOVO obb complexity summary skipped: {e}")
        return model

    def get_validator(self):
        """Return a YOLOVO OBB validator."""
        from ultralytics.models.yolovo.obb.val import YOLOVOOBBValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss", "obj_loss"
        return YOLOVOOBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plot OBB-specific training metrics."""
        from ultralytics.utils.plotting import plot_results

        plot_results(file=self.csv, obb=True, on_plot=self.on_plot)
