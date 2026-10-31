# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO pose estimation trainer."""

import os
from copy import copy
from typing import Optional

import torch

from ultralytics.data.dataset import YOLOVODataset
from ultralytics.models.yolovo.train import YOLOVOTrainer
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.nn.tasks import YOLOVOPoseModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel


class YOLOVOPoseTrainer(YOLOVOTrainer):
    """Trainer for YOLOVO RGB+Text pose estimation."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        overrides["task"] = "pose"
        overrides["afss_task_name"] = "pose"
        super().__init__(cfg, overrides, _callbacks)
        setattr(self.args, "afss_task_name", "pose")
        self.afss_runtime = None
        self._warn_afss_noop()

    def _parse_distill_arg(self):
        """Keep distill interface stable but no-op for YOLOVO pose."""
        distill = getattr(self.args, "distill", None)
        if distill:
            LOGGER.warning("YOLOVO pose distill interface is reserved; current implementation is a no-op.")
        return None

    def _warn_afss_noop(self):
        """Warn when AFSS is requested for the reserved pose interface."""
        try:
            from ultralytics.engine.afss import AFSSConfig

            afss_config = AFSSConfig.from_args(self.args)
            if afss_config.enabled:
                LOGGER.warning("YOLOVO pose AFSS interface is reserved; current implementation is a no-op.")
        except Exception:
            for name in ("afss", "afss_enable", "enable_afss"):
                if bool(getattr(self.args, name, False)):
                    LOGGER.warning("YOLOVO pose AFSS interface is reserved; current implementation is a no-op.")
                    break

    def _ensure_afss_runtime(self, dataset=None):
        """Reserved AFSS hook; returns no runtime."""
        LOGGER.warning("YOLOVO pose AFSS runtime is not implemented; using no-op.")
        return None

    def get_dataset(self):
        """Retrieve the dataset and require `kpt_shape` in data.yaml."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data

    def set_model_attributes(self):
        """Set keypoints shape attribute on the pose model."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """Build YOLOVODataset with keypoints enabled."""
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
            task="pose",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            yolovo_debug=debug_enabled and mode == "train",
            yolovo_object_paste_debug_dir=debug_dir,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOVOPoseModel instance."""
        model = YOLOVOPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        model._debug_logger = self.dbg
        self._log_model_contract(model, cfg)
        if weights:
            ckpt = torch.load(weights, map_location="cpu") if isinstance(weights, (str, os.PathLike)) else weights
            ckpt_model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = ckpt_model.state_dict() if hasattr(ckpt_model, "state_dict") else ckpt_model
            if self._is_yolo11_ckpt(sd):
                LOGGER.info("Remapping YOLO11 checkpoint to YOLOVO pose layout")
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
                self.dbg.log("", "weight_load", source="yolovo_pose_checkpoint", path=str(weights))
        if verbose and RANK in {-1, 0}:
            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"YOLOVO pose complexity summary skipped: {e}")
        return model

    def get_validator(self):
        """Return a YOLOVO pose validator."""
        from ultralytics.models.yolovo.pose.val import YOLOVOPoseValidator

        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "obj_loss"
        return YOLOVOPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch: dict, ni: int) -> None:
        """Plot training samples with keypoints."""
        from ultralytics.utils.plotting import plot_images

        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1) if batch["cls"].ndim > 1 else batch["cls"],
            bboxes=batch["bboxes"],
            kpts=batch.get("keypoints"),
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plot pose-specific training metrics."""
        from ultralytics.utils.plotting import plot_results

        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)
