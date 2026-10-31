"""
MultiModalPlotter: 多模态统一绘图编排器。

职责
----
统一编排 YOLOMM / RTDETRMM 在 train / val / predict 三阶段、detect / segment / pose / obb /
semseg / classify 六任务的可视化输出，消灭约 20 个入口里重复的"拆模态→坐标处理→拼并排→落图"逻辑。

架构
----
    MultiModalPlotter（编排层 — 本文件）
      ├─ split_modalities / visualize_x_to_3ch / concat_side_by_side / prepare_rgb_for_display
      │     （vis.py 工具层 — 复用现有）
      ├─ _transform_for_side_by_side（坐标变换层 — 本文件，并排图标注复制）
      └─ plot_images_mm（绘图层 — plotting_mm.py）
            └─ Annotator（原语层 — plotting.py 复用，不改）

设计原则
--------
- 单一入口：所有多模态绘图经 plot_batch 进入。
- 明确契约：所有坐标输入必须归一化 [0,1]（predict 阶段用 result.boxes.xywhn）。
- 模态判定集中：modality_mode 枚举（dual / rgb_only / x_only），不再各入口自行 if-else。
- 命名统一：resolve_x_modality_name + build_filename，三阶段文件名一致。
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.plotting_mm import plot_images_mm

# vis.py 工具函数（复用，不重写）
from ultralytics.models.utils.multimodal.vis import (
    concat_side_by_side,
    ensure_batch_idx_long,
    extract_precision_contract_from_batch,
    prepare_rgb_for_display,
    split_modalities,
    visualize_x_to_3ch,
)


ModalityMode = Literal["dual", "rgb_only", "x_only"]
Stage = Literal["train", "val", "predict"]
Source = Literal["labels", "pred"]


class MultiModalPlotter:
    """多模态统一绘图编排器。

    所有多模态可视化（train_batch / val_batch / pred）经此类进入，保证：
      1. 单模态/多模态行为一致
      2. 各任务（bbox/mask/kpts/obb/sem_mask）并排图标注完整
      3. train/val/predict 文件命名一致
      4. 坐标处理唯一正确实现

    Examples:
        >>> plotter = MultiModalPlotter(save_dir, on_plot=trainer.on_plot)
        >>> plotter.plot_batch(
        ...     stage="val", source="labels", images=batch["img"],
        ...     batch_idx=batch["batch_idx"], cls=batch["cls"].squeeze(-1),
        ...     bboxes=batch["bboxes"], xch=3, x_modality_name="thermal",
        ...     precision_contract=extract_precision_contract_from_batch(batch),
        ...     modality_mode="dual", paths=batch["im_file"],
        ...     names=self.names, batch_ni=ni,
        ... )
    """

    def __init__(self, save_dir: Union[str, Path], on_plot: Optional[Callable] = None):
        """初始化绘图器。

        Args:
            save_dir: 输出目录。
            on_plot: 保存后的回调（W&B/ClearML 集成用）。
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.on_plot = on_plot

    # ==================================================================
    # 主入口
    # ==================================================================
    def plot_batch(
        self,
        stage: Stage,
        source: Source,
        images: torch.Tensor,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        xch: int,
        x_modality_name: str,
        precision_contract: Dict[str, str],
        modality_mode: ModalityMode,
        *,
        masks: Optional[torch.Tensor] = None,
        kpts: Optional[torch.Tensor] = None,
        obb: Optional[torch.Tensor] = None,
        sem_masks: Optional[torch.Tensor] = None,
        confs: Optional[torch.Tensor] = None,
        paths: Optional[List[str]] = None,
        names: Optional[Dict[int, str]] = None,
        batch_ni: Optional[int] = None,
        pred_base: Optional[str] = None,
        threaded: bool = False,
        # 标注显示控制
        show_box_labels: bool = True,
        legend: bool = False,
    ) -> None:
        """统一绘图主入口。

        根据 modality_mode 决定输出：
          - rgb_only: 仅 _rgb.jpg
          - x_only: 仅 _{x_modality}.jpg
          - dual: _rgb.jpg + _{x_modality}.jpg + _multimodal.jpg（并排）

        Args:
            ...（见原 docstring）
            show_box_labels: True=框上显示类名/conf；False=只画框
            legend: True=mosaic 右上角画统一图例（颜色↔类名）

        Args:
            stage: train / val / predict。
            source: labels（GT 标签，confs=None）/ pred（预测结果，confs 非 None）。
            images: 多模态张量 [B, 3+xch, H, W]。单模态场景由调用方零填充缺失模态后传入。
            batch_idx: 每个标注的 batch 索引 [N]。
            cls: 类别 [N]。
            bboxes: 归一化 xywh [N, 4]。
            xch: X 模态通道数。
            x_modality_name: X 模态名（thermal/depth/...）。
            precision_contract: {"rgb": ..., "x": ...}。
            modality_mode: dual / rgb_only / x_only。
            masks: 实例分割 [N, H, W]。
            kpts: 关键点 [N, num_kpts, 3]。
            obb: 旋转框 [N, 5] xywhr。
            sem_masks: 语义分割 [B, H, W]。
            confs: 置信度 [N]，None=GT 标签。
            paths: 图像路径列表。
            names: 类别映射。
            batch_ni: train/val 的 batch 序号。
            pred_base: predict 的基础文件名。
            threaded: 是否异步绘图（默认同步）。

        契约：所有坐标输入必须归一化 [0,1]。
        """
        # 1. 统一 batch_idx 类型
        batch_idx = ensure_batch_idx_long(batch_idx)

        # 2. 拆分模态（即使单模态也拆分，缺失模态为零填充）
        C = images.shape[1]
        if C > 3:
            rgb_images, x_images = split_modalities(images, xch)
        else:
            # 单模态场景（predict 调用方未零填充）：按 modality_mode 判定
            if modality_mode == "rgb_only":
                rgb_images = images
                x_images = torch.zeros(
                    images.shape[0], xch, images.shape[2], images.shape[3],
                    device=images.device, dtype=images.dtype,
                )
            elif modality_mode == "x_only":
                rgb_images = torch.zeros(
                    images.shape[0], 3, images.shape[2], images.shape[3],
                    device=images.device, dtype=images.dtype,
                )
                x_images = images
            else:
                raise ValueError(
                    f"plot_batch: 单模态输入(通道={C})但 modality_mode={modality_mode}，"
                    f"双模态场景需传 [B, 3+xch, H, W] 张量。"
                )

        # 3. 生成 display-ready 可视化张量
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract.get("rgb", "legacy"))
        x_visual = visualize_x_to_3ch(
            x_images,
            colorize=False,
            x_modality=x_modality_name,
            precision=precision_contract.get("x", "legacy"),
        )

        # 4. 统一标注张量（obb 优先于 xywh bbox）
        # 若调用方同时给了 bboxes(xywh) 和 obb(xywhr)，优先用 obb
        bboxes_eff = bboxes
        is_obb = False
        if obb is not None and (isinstance(obb, torch.Tensor) and obb.numel() > 0):
            bboxes_eff = obb
            is_obb = True

        # 5. 根据模态模式输出
        if modality_mode == "rgb_only":
            self._plot_single(
                rgb_visual, batch_idx, cls, bboxes_eff, is_obb,
                masks, kpts, sem_masks, confs, paths, names,
                modal_tag="RGB", modality="rgb", source=source,
                stage=stage, batch_ni=batch_ni, pred_base=pred_base,
                x_modality_name=x_modality_name, threaded=threaded,
                show_box_labels=show_box_labels, legend=legend,
            )
        elif modality_mode == "x_only":
            self._plot_single(
                x_visual, batch_idx, cls, bboxes_eff, is_obb,
                masks, kpts, sem_masks, confs, paths, names,
                modal_tag=x_modality_name, modality="x", source=source,
                stage=stage, batch_ni=batch_ni, pred_base=pred_base,
                x_modality_name=x_modality_name, threaded=threaded,
                show_box_labels=show_box_labels, legend=legend,
            )
        else:  # dual
            try:
                # RGB 视图
                self._plot_single(
                    rgb_visual, batch_idx, cls, bboxes_eff, is_obb,
                    masks, kpts, sem_masks, confs, paths, names,
                    modal_tag="RGB", modality="rgb", source=source,
                    stage=stage, batch_ni=batch_ni, pred_base=pred_base,
                    x_modality_name=x_modality_name, threaded=threaded,
                    show_box_labels=show_box_labels, legend=legend,
                )
                # X 视图
                self._plot_single(
                    x_visual, batch_idx, cls, bboxes_eff, is_obb,
                    masks, kpts, sem_masks, confs, paths, names,
                    modal_tag=x_modality_name, modality="x", source=source,
                    stage=stage, batch_ni=batch_ni, pred_base=pred_base,
                    x_modality_name=x_modality_name, threaded=threaded,
                    show_box_labels=show_box_labels, legend=legend,
                )
                # 并排视图
                self._plot_side_by_side(
                    rgb_visual, x_visual, batch_idx, cls, bboxes_eff, is_obb,
                    masks, kpts, sem_masks, confs, paths, names,
                    source=source, stage=stage, batch_ni=batch_ni, pred_base=pred_base,
                    x_modality_name=x_modality_name, threaded=threaded,
                    show_box_labels=show_box_labels, legend=legend,
                )
            except Exception as e:
                LOGGER.warning(f"MultiModalPlotter: 绘制 dual 视图失败: {e}")

    # ==================================================================
    # 单模态视图
    # ==================================================================
    def _plot_single(
        self,
        visual: torch.Tensor,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        is_obb: bool,
        masks: Optional[torch.Tensor],
        kpts: Optional[torch.Tensor],
        sem_masks: Optional[torch.Tensor],
        confs: Optional[torch.Tensor],
        paths: Optional[List[str]],
        names: Optional[Dict[int, str]],
        modal_tag: str,
        modality: str,
        source: Source,
        stage: Stage,
        batch_ni: Optional[int],
        pred_base: Optional[str],
        x_modality_name: str,
        threaded: bool,
        show_box_labels: bool = True,
        legend: bool = False,
    ) -> None:
        """单模态视图：直接调 plot_images_mm。"""
        fname = self.save_dir / self.build_filename(
            stage, source, modality, x_modality_name, batch_ni, pred_base
        )
        # paths 按模态重命名后缀
        paths_mm = self._suffix_paths(paths, modality, x_modality_name) if paths else paths

        plot_images_mm(
            images=visual,
            batch_idx=batch_idx,
            cls=cls,
            bboxes=bboxes,
            coords_normalized=True,
            masks=masks,
            kpts=kpts,
            sem_masks=sem_masks,
            confs=confs,
            paths=paths_mm,
            fname=str(fname),
            names=names,
            on_plot=self.on_plot,
            modal_tag=modal_tag,
            threaded=threaded,
            show_box_labels=show_box_labels,
            legend=legend,
        )

    # ==================================================================
    # 并排视图
    # ==================================================================
    def _plot_side_by_side(
        self,
        rgb_visual: torch.Tensor,
        x_visual: torch.Tensor,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        is_obb: bool,
        masks: Optional[torch.Tensor],
        kpts: Optional[torch.Tensor],
        sem_masks: Optional[torch.Tensor],
        confs: Optional[torch.Tensor],
        paths: Optional[List[str]],
        names: Optional[Dict[int, str]],
        source: Source,
        stage: Stage,
        batch_ni: Optional[int],
        pred_base: Optional[str],
        x_modality_name: str,
        threaded: bool,
        show_box_labels: bool = True,
        legend: bool = False,
    ) -> None:
        """并排视图：拼接图像 + 坐标变换 + 调 plot_images_mm。

        注意：并排视图不传 instance masks（避免坐标系偏移 + 2x mask 变换开销）。
        与 aa093d9 旧代码行为一致：旧版 plot_images 在并排图也不画 masks。
        """
        # 1. 拼接图像 [B,3,H,W] + [B,3,H,W] → [B,3,H,2W]
        side_visual = concat_side_by_side(rgb_visual, x_visual)

        # 2. 坐标变换：单图域标注 → 并排图标注（N → 2N）
        #    masks 传 None：并排视图不画 instance masks（旧代码行为 + 性能优化）
        t = self._transform_for_side_by_side(batch_idx, cls, bboxes, None, kpts, confs)

        # 3. sem_masks: [B,H,W] → [B,H,2W]（左右各一份）
        sem_side = None
        if sem_masks is not None:
            sem_side = torch.cat([sem_masks, sem_masks], dim=2)

        fname = self.save_dir / self.build_filename(
            stage, source, "multimodal", x_modality_name, batch_ni, pred_base
        )
        paths_mm = self._suffix_paths(paths, "multimodal", x_modality_name) if paths else paths

        plot_images_mm(
            images=side_visual,
            batch_idx=t["batch_idx"],
            cls=t["cls"],
            bboxes=t["bboxes"],
            coords_normalized=True,
            masks=None,  # 并排视图不画 instance masks
            kpts=t["kpts"],
            sem_masks=sem_side,
            confs=t["confs"],
            paths=paths_mm,
            fname=str(fname),
            names=names,
            on_plot=self.on_plot,
            modal_tag="Dual",
            threaded=threaded,
            show_box_labels=show_box_labels,
            legend=legend,
        )

    # ==================================================================
    # 并排图坐标变换（核心）
    # ==================================================================
    def _transform_for_side_by_side(
        self,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        masks: Optional[torch.Tensor],
        kpts: Optional[torch.Tensor],
        confs: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """单图域归一化坐标 → 并排图归一化坐标（N → 2N）。

        并排图坐标系（总宽度归一化到 [0,1]）：
          左半 RGB: x_center∈[0,0.5], w∈[0,0.5]
          右半 X:   x_center∈[0.5,1.0], w∈[0,0.5]

        各标注变换：
          - bbox [N,4]→[2N,4]: 前 N 左半(x*=0.5,w*=0.5), 后 N 右半(x=x*0.5+0.5,w*=0.5)
          - mask [N,H,W]→[2N,H,2W]: 前 N mask 在左半, 后 N mask 在右半（避免 plot_images_mm resize 拉伸）
          - kpts [N,K,3]→[2N,K,3]: x 坐标 *=0.5（左）/ x*0.5+0.5（右）
          - obb [N,5]→[2N,5]: 同 bbox，角度 r 不变
          - batch_idx/cls/confs: 复制为 2N

        Args:
            bboxes: 归一化 xywh [N,4] 或 xywhr [N,5]。

        Returns:
            dict 含 batch_idx/cls/bboxes/masks/kpts/confs，均为 2N 长度。
        """
        if bboxes.numel() == 0:
            # 空标注，直接复制 batch_idx/cls
            return {
                "batch_idx": torch.cat([batch_idx, batch_idx], dim=0),
                "cls": torch.cat([cls, cls], dim=0),
                "bboxes": bboxes,
                "masks": masks,
                "kpts": kpts,
                "confs": torch.cat([confs, confs], dim=0) if confs is not None else None,
            }

        # ===== bbox 变换 =====
        bboxes_left = bboxes.clone()
        bboxes_left[:, [0, 2]] *= 0.5  # x, w
        bboxes_right = bboxes.clone()
        bboxes_right[:, 0] = bboxes_right[:, 0] * 0.5 + 0.5
        bboxes_right[:, 2] *= 0.5
        bboxes_side = torch.cat([bboxes_left, bboxes_right], dim=0)

        # ===== mask 变换 [N,H,W] → [2N,H,2W] =====
        masks_side = masks
        if masks is not None and masks.numel() > 0:
            zero = torch.zeros_like(masks)
            masks_left_half = torch.cat([masks, zero], dim=2)   # [N,H,2W] 前 N mask 在左半
            masks_right_half = torch.cat([zero, masks], dim=2)  # [N,H,2W] 后 N mask 在右半
            masks_side = torch.cat([masks_left_half, masks_right_half], dim=0)  # [2N,H,2W]

        # ===== kpts 变换 [N,K,3] → [2N,K,3] =====
        kpts_side = kpts
        if kpts is not None and kpts.numel() > 0:
            kpts_left = kpts.clone()
            kpts_left[..., 0] *= 0.5
            kpts_right = kpts.clone()
            kpts_right[..., 0] = kpts_right[..., 0] * 0.5 + 0.5
            kpts_side = torch.cat([kpts_left, kpts_right], dim=0)

        # ===== batch_idx / cls / confs 复制 =====
        return {
            "batch_idx": torch.cat([batch_idx, batch_idx], dim=0),
            "cls": torch.cat([cls, cls], dim=0),
            "bboxes": bboxes_side,
            "masks": masks_side,
            "kpts": kpts_side,
            "confs": torch.cat([confs, confs], dim=0) if confs is not None else None,
        }

    # ==================================================================
    # 命名与模态判定（静态方法）
    # ==================================================================
    @staticmethod
    def resolve_x_modality_name(
        modality_param: Optional[str],
        data: Optional[dict],
        x_modality_name_attr: Optional[str] = None,
    ) -> str:
        """统一解析 X 模态名（解决 train/val/predict 命名不一致）。

        优先级：
          1. x_modality_name_attr（train/val 的 self.x_modality_name）
          2. data.yaml 的 modality_used / x_modality
          3. modality_param 本身（若非 RGB/dual/x 泛化词）
          4. 兜底 "x"

        Args:
            modality_param: 入口传入的 modality 字符串（如 "thermal"/"RGB"/"X"）。
            data: 数据集配置 dict。
            x_modality_name_attr: 已解析的 x_modality_name 属性。

        Returns:
            真实模态名（如 "thermal"/"depth"），而非泛化的 "x"。
        """
        # 处理 list 类型（x_modality_name_attr 可能来自配置，如 ['rgb', 'sar']）
        attr = x_modality_name_attr
        if isinstance(attr, list):
            non_rgb = [v for v in attr if str(v).lower() != "rgb"]
            attr = non_rgb[0] if non_rgb else None
        if attr and str(attr).lower() not in ("unknown", ""):
            return str(attr).lower()

        if data:
            for key in ("modality_used", "x_modality", "modality"):
                val = data.get(key)
                if val is None:
                    continue
                # 处理 list 类型：modality_used 如 ["rgb", "sar"]，取非 RGB 的第一个作为 X 模态
                if isinstance(val, list):
                    non_rgb = [v for v in val if str(v).lower() != "rgb"]
                    if non_rgb:
                        val = non_rgb[0]
                    else:
                        continue
                if str(val).lower() not in ("rgb", "dual", "x", ""):
                    return str(val).lower()

        if modality_param:
            mp = str(modality_param).lower()
            if mp not in ("rgb", "dual", "x", ""):
                return mp

        return "x"

    @staticmethod
    def build_filename(
        stage: Stage,
        source: Source,
        modality: Literal["rgb", "x", "multimodal"],
        x_modality_name: str,
        batch_ni: Optional[int] = None,
        pred_base: Optional[str] = None,
    ) -> str:
        """构建输出文件名（保持现有命名约定，W&B glob 兼容）。

        规则：
          - predict: pred_{base}_labels_{mod}.jpg
          - train/val: {stage}_batch{ni}_{source}_{mod}.jpg

        其中 modality="x" 时替换为 x_modality_name（如 thermal）。

        Args:
            stage: train / val / predict。
            source: labels / pred。
            modality: rgb / x / multimodal。
            x_modality_name: X 模态名。
            batch_ni: train/val batch 序号。
            pred_base: predict 基础名。

        Returns:
            文件名字符串。
        """
        mod_str = x_modality_name if modality == "x" else modality
        if stage == "predict":
            if pred_base is None:
                raise ValueError("predict 阶段需提供 pred_base")
            return f"pred_{pred_base}_labels_{mod_str}.jpg"
        if batch_ni is None:
            raise ValueError(f"{stage} 阶段需提供 batch_ni")
        return f"{stage}_batch{batch_ni}_{source}_{mod_str}.jpg"

    @staticmethod
    def determine_modality_mode(modality: Optional[str], is_single_modal: bool = False) -> ModalityMode:
        """统一模态模式判定（解决各入口 if-else 不一致）。

        Args:
            modality: 入口的 self.modality 字段。可能值：
                - None / 空 / "dual": 双模态
                - "RGB" / "rgb": RGB 单模态
                - "X" / "thermal" / "depth" / ...: X 单模态
            is_single_modal: predict 阶段的单模态标志（可选，作为辅助判定）。

        Returns:
            "dual" / "rgb_only" / "x_only"
        """
        if modality is None or str(modality).strip() == "" or str(modality).lower() == "dual":
            return "dual"
        m = str(modality).lower()
        if m == "rgb":
            return "rgb_only"
        # "x" 或具体模态名（thermal/depth/ir...）均视为 X 单模态
        return "x_only"

    # ==================================================================
    # 辅助
    # ==================================================================
    @staticmethod
    def _suffix_paths(
        paths: List[str], modality: str, x_modality_name: str
    ) -> List[str]:
        """给 paths 加模态后缀（仅用于网格图文件名标注，不影响实际读取）。"""
        mod_str = x_modality_name if modality == "x" else modality
        return [p.replace(".jpg", f"_{mod_str}.jpg") for p in paths]
