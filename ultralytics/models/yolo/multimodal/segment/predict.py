from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.multimodal.predict import MultiModalDetectionPredictor


class MultiModalSegmentationPredictor(MultiModalDetectionPredictor, SegmentationPredictor):
    """
    Multimodal segmentation predictor.

    Inherits multimodal input handling, preprocessing and visualization layout
    from the detection-side predictor, while leveraging segmentation plotting
    from the segmentation base where applicable.

    Further mask-specific visual overlays can be extended later while keeping
    the multimodal input semantics unchanged.
    """

    def write_results(self, i, p, im, s) -> str:
        """
        多模态分割推理可视化 — 委托 MultiModalPlotter，并传入 masks。

        - 单模态（RGB）：仅输出 RGB + mask 叠加
        - 单模态（X）：仅输出 X + mask 叠加
        - 双模态：输出 RGB + X + 并排对比图，均叠加 mask
        """
        # 1) Run parent for txt/crops but suppress image saving to avoid legacy visuals
        orig_save = getattr(self.args, 'save', False)
        try:
            self.args.save = False
            string = super().write_results(i, p, im, s)
        finally:
            self.args.save = orig_save

        # 2) Generate unified visualization outputs
        if not orig_save:
            return string

        import torch
        import cv2
        from ultralytics.models.utils.multimodal.plotter import MultiModalPlotter
        from ultralytics.models.utils.multimodal.vis import ensure_batch_idx_long

        # 将原始 numpy 图像转换为张量 (1,3,H,W)，display-ready [0,1]
        def _np_to_tensor3ch(img_np):
            if img_np is None:
                raise RuntimeError("缺少原始图像用于可视化背景")
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError(f"不支持的原始图像形状: {img_np.shape}")
            return torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        # 从缓存中提取原始 RGB/X 图像
        def _get_orig_modal_tensors():
            if not hasattr(self, '_orig_imgs_cache') or self._orig_imgs_cache is None:
                raise RuntimeError("未找到原始图像缓存，无法生成以原图为背景的可视化")
            oi = self._orig_imgs_cache
            rgb_t, x_t = None, None
            if isinstance(oi, (list, tuple)):
                if len(oi) == 2:  # 双模态
                    rgb_t = _np_to_tensor3ch(oi[0])
                    x_t = _np_to_tensor3ch(oi[1])
                elif len(oi) == 1:  # 单模态
                    if self.modality and str(self.modality).lower() == 'rgb':
                        rgb_t = _np_to_tensor3ch(oi[0])
                    else:
                        x_t = _np_to_tensor3ch(oi[0])
                else:
                    raise RuntimeError(f"原始图像数量异常: {len(oi)}")
            else:
                if self.modality and str(self.modality).lower() == 'rgb':
                    rgb_t = _np_to_tensor3ch(oi)
                else:
                    x_t = _np_to_tensor3ch(oi)
            return rgb_t, x_t

        # 当前结果对象与基础命名
        result = self.results[i]
        base = p.stem

        # 使用原始 RGB/X 图像作为背景
        rgb_tensor, x_tensor = _get_orig_modal_tensors()

        # 从 Results 提取归一化坐标
        n_boxes = 0 if result.boxes is None else len(result.boxes)
        if n_boxes:
            cls_ids = result.boxes.cls.long()
            boxes_norm = result.boxes.xywhn  # 归一化 xywh
            confs = getattr(result.boxes, 'conf', None)
            if not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, dtype=torch.long)
            if not isinstance(boxes_norm, torch.Tensor):
                boxes_norm = torch.as_tensor(boxes_norm, dtype=torch.float32)
            if confs is not None and not isinstance(confs, torch.Tensor):
                confs = torch.as_tensor(confs, dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros(cls_ids.shape[0]))
        else:
            cls_ids = torch.zeros((0,), dtype=torch.long)
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
            confs = torch.zeros((0,), dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros((0,), dtype=torch.long))

        # masks: 从 Results 提取实例分割掩码
        if (getattr(result, 'masks', None) is not None
                and getattr(result.masks, 'data', None) is not None):
            masks_data = result.masks.data
            if masks_data.dtype == torch.float32 or masks_data.dtype == torch.float64:
                masks = (masks_data > 0.5).to(torch.uint8) * 255
            else:
                masks = torch.as_tensor(masks_data, dtype=torch.uint8)
        else:
            # 推导图像尺寸
            Hr = rgb_tensor.shape[-2] if rgb_tensor is not None else x_tensor.shape[-2]
            Wr = rgb_tensor.shape[-1] if rgb_tensor is not None else x_tensor.shape[-1]
            masks = torch.zeros((0, Hr, Wr), dtype=torch.uint8)

        names = getattr(self.model, 'names', {})
        x_modality = MultiModalPlotter.resolve_x_modality_name(self.modality, None, None)

        # 单模态
        if self.is_single_modal:
            if self.modality.lower() == 'rgb':
                if rgb_tensor is None:
                    raise RuntimeError("期望 RGB 原图用于可视化，但缓存缺失")
                img = rgb_tensor
            else:
                if x_tensor is None:
                    raise RuntimeError("期望 X 原图用于可视化，但缓存缺失")
                img = x_tensor
            plotter = MultiModalPlotter(self.save_dir)
            plotter.plot_batch(
                stage="predict", source="labels", images=img,
                batch_idx=batch_idx, cls=cls_ids, bboxes=boxes_norm, confs=confs,
                xch=3, x_modality_name=x_modality,
                precision_contract={"rgb": "legacy", "x": "legacy"},
                modality_mode=MultiModalPlotter.determine_modality_mode(self.modality),
                masks=masks,
                names=names, pred_base=base,
            )
            return string

        # 双模态：对齐尺寸后拼接成 [1, 6, H, W]
        if rgb_tensor is None or x_tensor is None:
            raise RuntimeError("双模态可视化需要 RGB 与 X 原图，但缓存缺失")
        Hr, Wr = rgb_tensor.shape[-2:]
        Hx, Wx = x_tensor.shape[-2:]
        if (Hr, Wr) != (Hx, Wx):
            x_tensor = torch.nn.functional.interpolate(
                x_tensor, size=(Hr, Wr), mode='bilinear', align_corners=False
            )
        dual_img = torch.cat([rgb_tensor, x_tensor], dim=1)

        plotter = MultiModalPlotter(self.save_dir)
        plotter.plot_batch(
            stage="predict", source="labels", images=dual_img,
            batch_idx=batch_idx, cls=cls_ids, bboxes=boxes_norm, confs=confs,
            xch=3, x_modality_name=x_modality,
            precision_contract={"rgb": "legacy", "x": "legacy"},
            modality_mode="dual",
            masks=masks,
            names=names, pred_base=base,
        )

        return string
