# Ultralytics YOLO, AGPL-3.0 license

"""
Multi-Modal Pose Validator.

Provides MultiModalPoseValidator for RGB+X pose estimation validation.
"""

import torch
from pathlib import Path
import numpy as np

from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import LOGGER, callbacks
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import (
    de_parallel,
    maybe_log_multimodal_canonical_summary,
    select_device,
    set_multimodal_complexity_owner_state,
    smart_inference_mode,
)
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import Profile
from ultralytics.utils import TQDM
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    get_runtime_x_modality_name,
    normalize_modality_token,
    resolve_rgb_x_pair,
)
from ultralytics.utils.metrics import OKS_SIGMA


class MultiModalPoseValidator(PoseValidator):
    """
    Multi-modal pose validator with 6+ channel warmup and runtime modality injection.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize MultiModalPoseValidator with modality configuration."""
        # PoseValidator 不接收 pbar 参数，保留形参仅兼容外部调用
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

        # Multi-modal flags
        if args:
            if isinstance(args, dict):
                self.modality = args.get("modality", None)
            else:
                self.modality = getattr(args, "modality", None)
        else:
            self.modality = None

        # 仅对 rgb/x token 做归一化：rgb/RGB→RGB、x/X→X（其它模态名保持原样）
        self.modality = normalize_modality_token(self.modality)
        # 回写 args/self.args，确保训练内 copy(args) 与后续读取一致
        if args is not None:
            if isinstance(args, dict):
                args["modality"] = self.modality
            else:
                setattr(args, "modality", self.modality)
        if hasattr(self, "args") and self.args is not None:
            if isinstance(self.args, dict):
                self.args["modality"] = self.modality
            else:
                setattr(self.args, "modality", self.modality)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"MultiModalPoseValidator initialized - single modal: {self.modality}-only")
        else:
            LOGGER.info("MultiModalPoseValidator initialized - dual modal validation")

    def preprocess(self, batch):
        """多模态 Pose 感知的预处理。

        与 MultiModalDetectionValidator.preprocess 保持一致的 contract 归一化策略，
        并保留 Pose 任务所需的 keypoints 处理。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        img = batch["img"].float()
        # contract-driven preprocessing
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img

        if self.args.half:
            batch["img"] = batch["img"].half()

        for k in ("batch_idx", "cls", "bboxes"):
            batch[k] = batch[k].to(self.device)

        # keypoints 处理（Pose 任务特有）
        if "keypoints" in batch:
            batch["keypoints"] = batch["keypoints"].to(self.device).float()

        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建多模态 Pose 数据集，与 MultiModalPoseTrainer.build_dataset 对齐。"""
        stride = int(getattr(self, "stride", 32) or 32)
        x_modality = self._determine_x_modality_from_data()
        x_modality_dir = self._get_x_modality_path(x_modality)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
            precision_report_dir=self.save_dir,
        )

    def _determine_x_modality_from_data(self) -> str:
        """解析 data.yaml 中的 X 模态名称。"""
        data = getattr(self, "data", {}) or {}
        x_modality = extract_x_modality_name_from_data(data)
        return x_modality if x_modality != "unknown" else "unknown"

    def _get_x_modality_path(self, x_modality: str) -> str:
        """根据 data.yaml modalities 映射获取 X 模态目录。"""
        data = getattr(self, "data", {}) or {}
        return get_modality_path_from_data(data, x_modality)

    def init_metrics(self, model):  # type: ignore[override]
        """Initialize metrics and capture runtime X modality name from loaded weights."""
        super().init_metrics(model)
        self.x_modality_name = get_runtime_x_modality_name(model)

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation with 6+ channel multi-modal input support."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
            if hasattr(model, "mm_router") and model.mm_router and self.modality:
                model.mm_router.set_runtime_params(
                    self.modality,
                    strategy=getattr(self.args, "ablation_strategy", None),
                    seed=getattr(self.args, "seed", None),
                )
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 6, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(f"Dataset '{self.args.data}' for task={self.args.task} not found")

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not pt:
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()

            # Runtime modality injection
            try:
                if (hasattr(model, "pt") and model.pt and
                    hasattr(model, "model") and hasattr(model.model, "mm_router") and
                    model.model.mm_router and self.modality):
                    model.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, "ablation_strategy", None),
                        seed=getattr(self.args, "seed", None),
                    )
            except Exception:
                pass

            # Multi-modal warmup
            if hasattr(self, "data") and self.data and "Xch" in self.data:
                x_channels = self.data.get("Xch", 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"Multi-modal Pose warmup: {total_channels}ch (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:
                LOGGER.info("Multi-modal Pose warmup: 6ch (default)")
                model.warmup(imgsz=(1 if pt else self.args.batch, 6, imgsz, imgsz))

            set_multimodal_complexity_owner_state(model, direct_ingress=True)
            maybe_log_multimodal_canonical_summary(model, imgsz=imgsz, logger=LOGGER, direct_ingress=True)

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds = model(batch["img"], augment=augment)
            with dt[2]:
                if self.training:
                    orig_mode = model.training
                    try:
                        model.train()
                        self.loss += model.loss(batch, preds)[1]
                    finally:
                        if not orig_mode:
                            model.eval()
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
        return self.metrics.results_dict

    # ------------------------------------------------------------------
    # Visualization (GT samples + predictions)
    # ------------------------------------------------------------------

    def plot_val_samples(self, batch, ni):
        """
        绘制 Pose 验证样本 — 改为薄包装，全部委托 MultiModalPlotter。

        Pose 特有：额外传 kpts=batch.get("keypoints")（[N, 17, 3]，已归一化 [0,1]）。
        并排图坐标变换（含 kpts 的 x 半幅）在 MultiModalPlotter._transform_for_side_by_side 内部完成。
        """
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            ensure_batch_idx_long,
        )

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3

        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="val", source="labels", images=batch["img"],
            batch_idx=ensure_batch_idx_long(batch["batch_idx"]),
            cls=batch["cls"].squeeze(-1), bboxes=batch["bboxes"],
            xch=xch,
            x_modality_name=MultiModalPlotter.resolve_x_modality_name(
                getattr(self, 'modality', None),
                getattr(self, 'data', None),
                getattr(self, 'x_modality_name', None),
            ),
            precision_contract=extract_precision_contract_from_batch(batch),
            modality_mode=MultiModalPlotter.determine_modality_mode(
                getattr(self, 'modality', None)
            ),
            kpts=batch.get("keypoints"),
            paths=batch["im_file"], names=self.names, batch_ni=ni,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        绘制 Pose 预测结果 — 改为薄包装，坐标转换在入口完成后委托 MultiModalPlotter。

        Pose 特有：
        - 从 preds[1] 提取 pred_kpts（[B, num_kpts, 3] 格式）
        - pred_kpts 已归一化 [0,1]，无需额外变换

        统一坐标制：output_to_target → 像素 xywh → 归一化 xywh → 绘图。
        """
        from ultralytics.utils.plotting import output_to_target
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            to_norm_xywh_for_plot,
            ensure_batch_idx_long,
            clip_boxes_norm_xywh,
        )

        # 多模态张量与 X 通道数
        multimodal_images = batch["img"]
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3

        # 统一坐标制：output_to_target → 像素 xywh → 归一化 xywh
        batch_ids, cls_ids, boxes_xywh_px, confs = output_to_target(preds, max_det=self.args.max_det)
        _, _, H, W = multimodal_images.shape
        batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm = to_norm_xywh_for_plot(
            batch_ids, cls_ids, boxes_xywh_px, confs, (H, W)
        )
        batch_ids_norm = ensure_batch_idx_long(batch_ids_norm)
        # 几何裁剪到单图域 [0,1]，防越界
        if (isinstance(boxes_norm, torch.Tensor) and boxes_norm.numel() > 0) or (
            isinstance(boxes_norm, np.ndarray) and boxes_norm.size > 0
        ):
            boxes_norm = clip_boxes_norm_xywh(boxes_norm, 0.0, 1.0, 0.0, 1.0)

        # Pose predictions: 提取预测的 keypoints
        # preds 通常是 (det_output, kpts_output)，preds[1] 为 [B, num_kpts, 3]
        pred_kpts = None
        if isinstance(preds, (list, tuple)) and len(preds) > 1:
            try:
                pred_kpts = preds[1]  # [B, num_kpts, 3]，已归一化
            except (IndexError, TypeError):
                pred_kpts = None

        plotter = MultiModalPlotter(self.save_dir, on_plot=self.on_plot)
        plotter.plot_batch(
            stage="val", source="pred", images=multimodal_images,
            batch_idx=batch_ids_norm, cls=cls_ids_norm, bboxes=boxes_norm, confs=confs_norm,
            xch=xch,
            x_modality_name=MultiModalPlotter.resolve_x_modality_name(
                getattr(self, 'modality', None),
                getattr(self, 'data', None),
                getattr(self, 'x_modality_name', None),
            ),
            precision_contract=extract_precision_contract_from_batch(batch),
            modality_mode=MultiModalPlotter.determine_modality_mode(
                getattr(self, 'modality', None)
            ),
            kpts=pred_kpts,
            paths=batch["im_file"], names=self.names, batch_ni=ni,
        )

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _afss_pr_from_match(matched: int, pred_count: int, gt_count: int) -> tuple[float, float]:
        """Compute precision/recall with empty-label handling consistent with detect AFSS."""
        if gt_count == 0 and pred_count == 0:
            return 1.0, 1.0
        if gt_count == 0:
            return 0.0, 1.0
        if pred_count == 0:
            return 0.0, 0.0
        return matched / pred_count, matched / gt_count

    def _afss_ensure_pose_sigma(self):
        """Ensure PoseValidator sigma/kpt_shape exists for _process_batch during AFSS scoring."""
        if self.sigma is not None:
            return
        kpt_shape = None
        if getattr(self, "data", None):
            kpt_shape = self.data.get("kpt_shape")
        if not kpt_shape:
            kpt_shape = [17, 3]
        self.kpt_shape = list(kpt_shape)
        is_pose_17 = self.kpt_shape == [17, 3]
        nkpt = int(self.kpt_shape[0]) if len(self.kpt_shape) else 17
        self.sigma = OKS_SIGMA if is_pose_17 else np.ones(nkpt) / nkpt

    def afss_score_sample(self, pred, batch, si):
        """Build one AFSS score row for pose task using box and keypoint branches."""
        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")
        self._afss_ensure_pose_sigma()
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)
        tp_box = result["tp"]
        tp_pose = result["tp_p"]
        matched_box = int(tp_box[:, 0].sum()) if len(tp_box) else 0
        matched_pose = int(tp_pose[:, 0].sum()) if len(tp_pose) else 0
        pred_count = int(len(predn["cls"]))
        gt_count = int(len(pbatch["cls"]))

        box_precision, box_recall = self._afss_pr_from_match(matched_box, pred_count, gt_count)
        pose_precision, pose_recall = self._afss_pr_from_match(matched_pose, pred_count, gt_count)
        box_sufficiency = min(float(box_precision), float(box_recall))
        pose_sufficiency = min(float(pose_precision), float(pose_recall))
        # joint_min: either branch failing must block easy classification.
        sufficiency_raw = min(box_sufficiency, pose_sufficiency)

        im_file = str(batch["im_file"][si])
        kpt_dim = int(self.kpt_shape[1]) if self.kpt_shape and len(self.kpt_shape) > 1 else 2
        return {
            "sample_key": str(Path(im_file).resolve()),
            "im_file": im_file,
            "task_name": "pose",
            "precision": box_precision,
            "recall": box_recall,
            "precision_op": box_precision,
            "recall_op": box_recall,
            "sufficiency_raw": sufficiency_raw,
            "valid_for_afss": True,
            "task_metrics": {
                "box_precision": float(box_precision),
                "box_recall": float(box_recall),
                "pose_precision": float(pose_precision),
                "pose_recall": float(pose_recall),
                "box_sufficiency": float(box_sufficiency),
                "pose_sufficiency": float(pose_sufficiency),
                "matched_box_count": int(matched_box),
                "matched_pose_count": int(matched_pose),
                "pred_count": int(pred_count),
                "gt_count": int(gt_count),
                "kpt_dim": int(kpt_dim),
                "kobj_branch_enabled": int(kpt_dim == 3),
                "pose_branch_source": "tp_p",
                "sufficiency_mode": "joint_min",
            },
            "matched_count": int(matched_box),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        }

    def afss_score_batch(self, preds, batch):
        """Score all samples in a batch for AFSS pose adapter reuse."""
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
