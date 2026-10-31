# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.nn.tasks import SemSegModel
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_results
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.utils.torch_utils import de_parallel, sanitize_checkpoint_train_args
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    set_single_modal_build_metadata_in_ckpt,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)
from .val import MultiModalSemSegValidator


class MultiModalSemSegTrainer(MultiModalDetectionTrainer):
    """Trainer for YOLOMM semantic segmentation.

    Inherits MultiModalDetectionTrainer for modality ablation + AFSS + distill.
    Overrides only the task-specific methods below.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = overrides or {}
        overrides["task"] = "semseg"
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        self.args.ignore_index = int(self.data.get("ignore_index", getattr(self.args, "ignore_index", 255)))
        x_modality = self._determine_x_modality_from_data()
        x_dir = self._get_x_modality_path(x_modality)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,  # SemSeg 不需要矩形批处理，mask 像素严格对齐
            stride=gs,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
            precision_report_dir=self.save_dir,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        x_channels = self.data.get("Xch", 3)
        channels = 3 + x_channels
        from ultralytics.nn.mm.utils import prepare_multimodal_build_cfg

        cfg_dict = prepare_multimodal_build_cfg(cfg, self.data, self.args, LOGGER)
        model = SemSegModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        if model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )
        if weights:
            model.load(weights)
        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(self.data))
        if verbose and RANK in {-1, 0}:
            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")
        return model

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["sem_mask"] = batch["sem_mask"].to(self.device, non_blocking=True).long()
        return batch

    def get_validator(self):
        criterion = getattr(self.model, "criterion", None) or self.model.init_criterion()
        self.loss_names = getattr(criterion, "names", ("semseg_ce_loss", "semseg_dice_loss", "semseg_aux_loss"))
        return MultiModalSemSegValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def save_model(self):
        import torch
        from datetime import datetime

        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": de_parallel(self.model).half(),
            "ema": de_parallel(self.ema.ema).half() if self.ema else None,
            "optimizer": self.optimizer.state_dict(),
            "train_args": sanitize_checkpoint_train_args(vars(self.args)),
            "yaml": self.model.yaml,
            "nc": self.data["nc"],
            "names": self.data["names"],
            "x_modality_name": getattr(self.model, "x_modality_name", None),
            "Xch": self.data.get("Xch", 3),
            "ignore_index": self.args.ignore_index,
            "modality": self.modality,
            "is_multimodal": True,
            "date": datetime.now().isoformat(),
        }
        set_x_modality_name_in_ckpt_metadata(ckpt, ckpt["x_modality_name"])
        set_single_modal_build_metadata_in_ckpt(ckpt, model=self.model, args=self.args)
        torch.save(ckpt, self.last)
        if self.best_fitness == self.last_fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def plot_metrics(self):
        from ultralytics.utils.plotting import plot_distill_results
        plot_results(file=self.csv, semseg=True, on_plot=self.on_plot)
        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="yolomm", on_plot=self.on_plot)

    def _sync_afss_task_name(self):
        self.args.afss_task_name = "semseg"

    def plot_training_samples(self, batch, ni):
        """Plot multimodal semseg training samples — 薄包装，委托 MultiModalPlotter。

        SemSeg 无 bbox：传占位 batch_idx/cls/bboxes，GT 由 sem_masks=batch["sem_mask"] 承载。
        支持双模态（RGB / X / 并排）与模态消融（单模态）。
        """
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
        )

        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3
        B = batch["img"].shape[0]
        device = batch["img"].device

        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="train", source="labels", images=batch["img"],
            batch_idx=torch.zeros(B, dtype=torch.long, device=device),
            cls=torch.zeros(0, dtype=torch.long, device=device),
            bboxes=torch.zeros((0, 4), device=device),
            xch=xch,
            x_modality_name=MultiModalPlotter.resolve_x_modality_name(
                getattr(self, "modality", None),
                getattr(self, "data", None),
                getattr(self, "x_modality_name", None),
            ),
            precision_contract=extract_precision_contract_from_batch(batch),
            modality_mode=MultiModalPlotter.determine_modality_mode(
                getattr(self, "modality", None)
            ),
            sem_masks=batch["sem_mask"],
            paths=batch["im_file"], names=getattr(self, "names", None), batch_ni=ni,
            threaded=True,  # 异步绘图，恢复旧版 @threaded 行为
        )
