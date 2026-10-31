from ultralytics.models.yolo.segment.val import SegmentationValidator
import numpy as np
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import callbacks
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    get_runtime_x_modality_name,
    normalize_modality_token,
    resolve_rgb_x_pair,
)


class MultiModalSegmentationValidator(SegmentationValidator):
    """
    Multimodal segmentation validator for RGB+X inputs.

    Provides channel-aware warmup and accepts a runtime modality parameter,
    mirroring the detection-side multimodal validator behavior.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

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

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):  # type: ignore[override]
        self.training = trainer is not None
        if self.training:
            # Align with training setup
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            model.eval()
            # Inject runtime modality to router during train-time val
            try:
                if hasattr(model, "mm_router") and model.mm_router and self.modality:
                    model.mm_router.set_runtime_params(self.modality)
            except Exception:
                pass
        else:
            # 方案A：不在子类中手动创建 AutoBackend，避免二次包裹与错误 fuse 调用。
            # 委托父类创建并管理 AutoBackend（包括设备、数据、warmup 等）。
            pass

        # Defer to parent implementation for the main loop
        return super().__call__(trainer=trainer, model=model)

    def init_metrics(self, model):  # type: ignore[override]
        """Initialize metrics and capture runtime X modality name from loaded weights."""
        super().init_metrics(model)
        self.x_modality_name = get_runtime_x_modality_name(model)

    def preprocess(self, batch):
        """多模态分割感知的预处理。

        与 MultiModalDetectionValidator.preprocess 保持一致的 contract 归一化策略，
        并保留分割任务所需的 masks 处理。
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

        # masks 处理（分割任务特有）
        if "masks" in batch:
            batch["masks"] = batch["masks"].to(self.device).float()

        return batch

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode: str = "val", batch=None):
        """
        构建多模态分割验证数据集。

        通过传递 multi_modal_image=True 启用 YOLOMultiModalImageDataset，
        并从 data.yaml/配置中解析 x_modality 与目录映射。严格遵循分割任务，
        使用 task='segment' 使 use_segments=True。
        """
        # 解析模态组合
        data = getattr(self, "data", {}) or {}
        modalities = None
        mod_map = None
        if "modality_used" in data and isinstance(data["modality_used"], list):
            modalities = data["modality_used"]
        elif "models" in data and isinstance(data["models"], list):
            modalities = data["models"]
        mod_map = data.get("modality") or data.get("modalities")

        # 提取 X 模态
        _rgb_modality, x_modality = resolve_rgb_x_pair(data, default_x=extract_x_modality_name_from_data(data))
        x_modality_dir = get_modality_path_from_data(data, x_modality)

        # stride
        stride = getattr(self, "stride", 32) or 32

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=True,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=False,  # 验证默认关闭自体生成
            precision_report_dir=self.save_dir,
        )

    # -----------------
    # Visualization (GT samples)
    # -----------------
    def plot_val_samples(self, batch, ni):
        """
        绘制验证样本 — 改为薄包装，全部委托 MultiModalPlotter。

        支持：
        - 双模态：输出 RGB、X 与并排对比图
        - 单模态（modality="RGB"）：仅输出 RGB
        - 单模态（modality="X" / 具体模态名）：仅输出 X
        - 分割任务：传 masks（实例分割掩码 [N, H, W]）
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
            masks=batch.get("masks"),
            paths=batch["im_file"], names=self.names, batch_ni=ni,
        )

    # -----------------
    # Visualization (Preds)
    # -----------------
    def plot_predictions(self, batch, preds, ni):
        """
        绘制预测结果 — 改为薄包装，坐标转换在入口完成后委托 MultiModalPlotter。

        统一坐标制：output_to_target → 像素 xywh → 归一化 xywh → 绘图。
        分割任务：传 masks（实例分割掩码 [N, H, W]），并排视图自动处理 mask 坐标系。
        """
        import torch
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

        # 合并所有 batch 的实例分割 masks
        # preds[i]["masks"]: [N_i, H, W] → 拼接为 [N, H, W]
        masks = None
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            masks_list = []
            for p in preds:
                m = p.get("masks") if isinstance(p, dict) else None
                if m is not None and isinstance(m, torch.Tensor) and m.numel() > 0:
                    masks_list.append(m.to(torch.uint8).cpu())
            if masks_list:
                masks = torch.cat(masks_list, dim=0)

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
            masks=masks,
            paths=batch["im_file"], names=self.names, batch_ni=ni,
        )

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers
    # ------------------------------------------------------------------
    def afss_score_sample(self, pred, batch, si):
        """Build one AFSS score row for segment task using validator-native primitives."""
        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")

        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)

        tp_box = result.get("tp")
        tp_mask = result.get("tp_m")
        matched_box = int(tp_box[:, 0].sum()) if tp_box is not None and len(tp_box) else 0
        matched_mask = int(tp_mask[:, 0].sum()) if tp_mask is not None and len(tp_mask) else 0
        pred_count = int(len(predn["cls"]))
        gt_count = int(len(pbatch["cls"]))
        pred_mask_count = int(predn["masks"].shape[0]) if "masks" in predn else 0
        gt_mask_count = int(pbatch["masks"].shape[0]) if "masks" in pbatch else 0

        if gt_count == 0 and pred_count == 0:
            box_precision = 1.0
            box_recall = 1.0
            mask_precision = 1.0
            mask_recall = 1.0
            empty_case = "no_pred_no_label"
        elif gt_count == 0:
            box_precision = 0.0
            box_recall = 1.0
            mask_precision = 0.0
            mask_recall = 1.0
            empty_case = "pred_without_label"
        elif pred_count == 0:
            box_precision = 0.0
            box_recall = 0.0
            mask_precision = 0.0
            mask_recall = 0.0
            empty_case = "label_without_pred"
        else:
            box_precision = matched_box / pred_count
            box_recall = matched_box / gt_count
            mask_precision = matched_mask / pred_count
            mask_recall = matched_mask / gt_count
            empty_case = "normal"

        box_sufficiency = min(box_precision, box_recall)
        mask_sufficiency = min(mask_precision, mask_recall)
        sufficiency_raw = min(box_sufficiency, mask_sufficiency)

        return {
            "im_file": batch["im_file"][si],
            "task_name": "segment",
            "precision_op": box_precision,
            "recall_op": box_recall,
            "sufficiency_raw": sufficiency_raw,
            "valid_for_afss": True,
            "task_metrics": {
                "box_precision": float(box_precision),
                "box_recall": float(box_recall),
                "mask_precision": float(mask_precision),
                "mask_recall": float(mask_recall),
                "box_sufficiency": float(box_sufficiency),
                "mask_sufficiency": float(mask_sufficiency),
                "matched_box_count": int(matched_box),
                "matched_mask_count": int(matched_mask),
                "pred_count": int(pred_count),
                "gt_count": int(gt_count),
                "pred_mask_count": int(pred_mask_count),
                "gt_mask_count": int(gt_mask_count),
                "overlap_mask": int(bool(getattr(self.args, "overlap_mask", False))),
                "empty_case": empty_case,
            },
            # Keep legacy compatibility fields for state store and older tooling.
            "precision": float(box_precision),
            "recall": float(box_recall),
            "matched_count": int(matched_box),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        }

    def afss_score_batch(self, preds, batch):
        """Score all samples in a batch for segment AFSS adapter reuse."""
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
