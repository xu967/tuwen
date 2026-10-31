# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
plot_images_mm: 多模态专属的图像网格绘图函数。

贴近原生 plot_images（ultralytics/utils/plotting.py），但专为多模态场景定制。

核心差异（相对原生 plot_images）：
-----------------------------
1. **3 通道断言**：images 必须是 3 通道 display-ready，否则报错（不再静默裁剪多光谱）。
2. **明确坐标契约**：coords_normalized 参数取代 4 处启发式判断（boxes/kpts 的 max()<=1.1 启发式）。
3. **mask 尺寸校验**：要求 mask 尺寸与单张图像一致；不一致时警告并 resize，但不破坏语义。
4. **同步/异步可控**：threaded 参数，默认同步（plot_images_mm 一次多模态调 3 次绘图，异步不可控）。
5. **sem_masks 支持**：原生 plot_images 不支持语义分割叠加，本函数内置支持。
6. **模态标识**：modal_tag 参数，在每张子图角落标注 RGB/thermal/depth 等，便于对比。

设计哲学：明确契约替代启发式判断。
所有输入标注必须已归一化到 [0,1]（coords_normalized=True）或像素坐标（coords_normalized=False），
函数内部不再做任何"看起来像归一化"的启发式判断。

复用组件：
---------
- Annotator: ultralytics.utils.plotting.Annotator（原语层，不修改）
- colors: ultralytics.utils.plotting.colors（调色板，不修改）
- ops.xywh2xyxy / ops.xywhr2xyxyxyxy: ultralytics.utils.ops（bbox 坐标变换，不修改）
"""

import math
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings
from ultralytics.utils.plotting import Annotator, colors


def _auto_grid(bs: int, max_aspect: float = 2.5) -> tuple:
    """紧致 grid：找 rows×cols ≥ bs，最接近方形且浪费最少。

    替代原生 `ceil(sqrt(bs))` 的强制方阵布局，消除非完全平方数 batch 的空白槽位。

    约束：
    - rows ≤ cols（横向优先，符合业界惯例）
    - cols / rows ≤ max_aspect（避免单列布局过于瘦长）
    - 优先 waste 最小，相同 waste 时选 |rows - cols| 最小（更方正）

    Examples:
        _auto_grid(6)   → (2, 3)   # 6 槽位，零浪费（原 3×3=9 浪费 3 槽）
        _auto_grid(8)   → (2, 4)   # 8 槽位，零浪费（原 3×3=9 浪费 1 槽）
        _auto_grid(10)  → (2, 5)   # 10 槽位，零浪费（原 4×4=16 浪费 6 槽）
        _auto_grid(11)  → (3, 4)   # 12 槽位，浪费 1（原 4×4=16 浪费 5）
        _auto_grid(13)  → (3, 5)   # 15 槽位，浪费 2（原 4×4=16 浪费 3）
        _auto_grid(16)  → (4, 4)   # 16 槽位，与原行为一致
    """
    if bs <= 1:
        return (1, 1)
    best = None
    for r in range(1, bs + 1):
        c = -(-bs // r)  # ceil(bs / r)
        if r > c:
            continue
        if c / r > max_aspect:
            continue
        waste = r * c - bs
        if (
            best is None
            or waste < best[2]
            or (waste == best[2] and abs(r - c) < abs(best[0] - best[1]))
        ):
            best = (r, c, waste)
    return (best[0], best[1]) if best else (1, bs)


def plot_images_mm(
    images: Union[torch.Tensor, np.ndarray],
    batch_idx: Union[torch.Tensor, np.ndarray],
    cls: Union[torch.Tensor, np.ndarray],
    bboxes: Union[torch.Tensor, np.ndarray] = None,
    *,
    # 多模态定制参数
    coords_normalized: bool = True,
    sem_masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    modal_tag: Optional[str] = None,
    threaded: bool = False,
    # 任务特定参数
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    kpts: Optional[Union[torch.Tensor, np.ndarray]] = None,
    # 通用参数（与原生 plot_images 对齐）
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Union[Dict[int, str], List[str]]] = None,
    on_plot: Optional[Callable] = None,
    max_size: int = 1920,
    max_subplots: int = 30,
    save: bool = True,
    conf_thres: float = 0.25,
    # 标注显示控制
    show_box_labels: bool = True,   # True=框上显示类名/conf；False=只画框
    legend: bool = False,           # True=mosaic 右上角画统一图例（颜色↔类名）
) -> Optional[np.ndarray]:
    """
    多模态专属图像网格绘图函数。

    绘制一批图像及其标注（bbox / mask / kpts / sem_mask），支持并排图坐标系统和语义分割。
    与原生 plot_images 的区别：
      - images 必须是 3 通道（不再静默裁剪多光谱）。
      - 坐标按 coords_normalized 参数明确处理（不启发式判断）。
      - 支持 sem_masks 语义分割叠加。
      - 支持 modal_tag 在子图角落标注模态。
      - 支持 threaded 参数控制同步/异步（默认同步）。

    Args:
        images: 一批图像 [B, 3, H, W]，float [0,1] 或 uint8 [0,255]。
        batch_idx: 每个标注对应的 batch 索引 [N]。
        cls: 每个标注的类别 [N]。
        bboxes: 每个标注的边界框 [N, 4] (xywh 归一化) 或 [N, 5] (xywhr 归一化)。
            若 coords_normalized=True，值域 [0,1]；否则为像素坐标。
        coords_normalized: True=归一化 xywh/xywhr，False=像素坐标。
        sem_masks: 语义分割掩码 [B, H, W]，类别索引 int。
        modal_tag: 模态标识（如 "RGB" / "thermal"），画在每张子图右上角。
        threaded: True=后台线程异步执行，False=同步执行（默认）。
        confs: 置信度 [N]。None=GT 标签模式（不检查 conf_thres）。
        masks: 实例分割掩码 [N, H_img, W_img]，二进制。
        kpts: 关键点 [N, num_kpts, 3]（COCO 17 点：x, y, conf）。
            若 coords_normalized=True，x,y 值域 [0,1]；否则为像素坐标。
            注意：原生 plot_images 签名默认值写 `(0, 51)` 但实际使用是三维格式。
        paths: 每张图像的文件路径（用于文件名标注）。
        fname: 输出文件名。
        names: 类别索引到名称的映射（字典或列表）。
        on_plot: 保存后的回调（如 W&B/ClearML 集成）。
        max_size: 输出图像最大尺寸。
        max_subplots: 子图最大数量。
        save: True=保存为文件，False=返回 numpy 数组。
        conf_thres: 显示阈值（仅 pred 模式，confs 不为 None 时生效）。

    Returns:
        若 save=False 返回绘制结果 numpy 数组；否则返回 None。
    """
    # 默认空 bboxes
    if bboxes is None:
        bboxes = np.zeros((0, 4), dtype=np.float32)

    def _draw() -> Optional[np.ndarray]:
        # ===== 1. tensor → numpy =====
        def _to_numpy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x) if not isinstance(x, np.ndarray) else x

        im = _to_numpy(images)
        b_idx = _to_numpy(batch_idx)
        c = _to_numpy(cls)
        b = _to_numpy(bboxes)
        cf = _to_numpy(confs)
        m = _to_numpy(masks)
        k = _to_numpy(kpts)
        sm = _to_numpy(sem_masks)

        # ===== 2. 维度规整 =====
        if c.ndim == 2:
            c = c.squeeze(-1)
        if b_idx.ndim == 2:
            b_idx = b_idx.squeeze(-1)

        # ===== 3. 3 通道断言（核心差异 #1）=====
        if im.shape[1] != 3:
            raise ValueError(
                f"plot_images_mm 要求 3 通道 display-ready 图像，得到 {im.shape[1]} 通道。"
                f"请先用 split_modalities + visualize_x_to_3ch 转换为 3 通道。"
            )

        bs, _, h, w = im.shape
        bs = min(bs, max_subplots)

        # ===== 4. NaN 清理 + [0,1]→[0,255] =====
        im = np.nan_to_num(im, nan=0.0, posinf=0.0, neginf=0.0)
        if im.max() <= 1.0:
            im = im * 255.0
        im = im.clip(0, 255).astype(np.uint8)

        # ===== 4.5 sem_masks 预叠加（修复 offset 问题：在 mosaic 拼接前按子图叠加）=====
        # 注意：annotator.semantic_mask 不感知子图偏移，会画在 mosaic (0,0) 被覆盖。
        # 此处在每张图各自区域 overlay，mosaic 拼接后位置自然正确。
        if sm is not None:
            for i in range(bs):
                if i >= sm.shape[0]:
                    break
                sem_i = sm[i]
                if sem_i.shape != (h, w):
                    sem_i = cv2.resize(
                        sem_i.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(np.int64)
                unique_vals = np.unique(sem_i)
                for val in unique_vals:
                    iv = int(val)
                    if iv == 255:  # ignore_index
                        continue
                    color_rgb = colors(iv)  # RGB tuple
                    mask_bool = sem_i == val
                    if mask_bool.any():
                        # im[i] 是 [C,H,W]，im[i][:, mask_bool] 是 [C,N]，color 需 reshape 为 [C,1]
                        c_arr = np.array(color_rgb, dtype=np.float32).reshape(3, 1)
                        im[i][:, mask_bool] = (
                            im[i][:, mask_bool].astype(np.float32) * 0.5
                            + c_arr * 0.5
                        ).astype(np.uint8)

        # ===== 5. mosaic 拼接（紧致 grid：rows×cols ≥ bs 且最接近方形）=====
        rows, cols = _auto_grid(bs)
        mosaic = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
        for i in range(bs):
            x_off, y_off = int(w * (i % cols)), int(h * (i // cols))
            mosaic[y_off : y_off + h, x_off : x_off + w, :] = im[i].transpose(1, 2, 0)

        # ===== 6. resize（同原生，但用 max(rows,cols) 替代 ns）=====
        grid_dim = max(rows, cols)
        scale = max_size / grid_dim / max(h, w)
        orig_h, orig_w = h, w
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, (int(cols * w), int(rows * h)))

        # ===== 7. 标注 =====
        fs = max(int((h + w) * grid_dim * 0.01), 18)
        annotator = Annotator(
            mosaic,
            line_width=round(fs / 10),
            font_size=fs,
            pil=True,
            example=str(names),
        )

        for i in range(bs):
            x_off, y_off = int(w * (i % cols)), int(h * (i // cols))
            annotator.rectangle([x_off, y_off, x_off + w, y_off + h], None, (255, 255, 255), width=2)
            if paths is not None and i < len(paths):
                annotator.text(
                    [x_off + 5, y_off + 5],
                    Path(paths[i]).name[:40],
                    txt_color=(220, 220, 220),
                )

            # 模态标识（核心差异 #6）
            if modal_tag:
                annotator.text(
                    [x_off + w - 80, y_off + 5],
                    modal_tag,
                    txt_color=(255, 255, 0),
                    box_color=(0, 0, 0, 128),
                )

            # 筛选当前子图的标注
            if len(c) == 0:
                continue
            idx = b_idx == i
            classes = c[idx].astype("int")
            is_label = cf is None  # True=GT 标签，False=预测结果

            # ===== 7.1 bbox =====
            if len(b):
                boxes = b[idx].copy()
                conf = cf[idx] if cf is not None else None
                if len(boxes):
                    # 核心差异 #2：明确坐标契约，不启发式判断
                    if coords_normalized:
                        boxes[..., [0, 2]] *= w
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale

                    # 加子图偏移
                    boxes[..., 0] += x_off
                    boxes[..., 1] += y_off

                    is_obb = boxes.shape[-1] == 5
                    boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)

                    for j, box in enumerate(boxes.astype(np.int64).tolist()):
                        ci = classes[j]
                        color = colors(ci)
                        c_name = names.get(ci, ci) if isinstance(names, dict) else ci
                        if is_label or (conf is not None and conf[j] > conf_thres):
                            # show_box_labels=False 时只画框，不画框上文字
                            if show_box_labels:
                                label = f"{c_name}" if is_label else f"{c_name} {conf[j]:.1f}"
                            else:
                                label = ""
                            annotator.box_label(box, label, color=color)

            elif len(classes):
                # 纯分类模式（classify 任务，无 bbox）
                for ci in classes:
                    color = colors(ci)
                    c_name = names.get(ci, ci) if isinstance(names, dict) else ci
                    annotator.text(
                        [x_off, y_off],
                        f"{c_name}",
                        txt_color=color,
                        box_color=(64, 64, 64, 128),
                    )

            # ===== 7.2 keypoints =====
            # 核心差异 #3：明确坐标契约，不启发式判断
            if k is not None and len(k):
                kpts_ = k[idx].copy()
                if len(kpts_):
                    if coords_normalized:
                        kpts_[..., 0] *= w
                        kpts_[..., 1] *= h
                    elif scale < 1:
                        kpts_[...] *= scale
                    kpts_[..., 0] += x_off
                    kpts_[..., 1] += y_off
                    for j in range(len(kpts_)):
                        if is_label or (cf is not None and cf[idx][j] > conf_thres):
                            annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # ===== 7.3 instance masks =====
            # 核心差异 #4：要求 mask 尺寸与子图一致，不一致时警告并 resize
            if m is not None and len(m):
                if idx.shape[0] == m.shape[0]:  # overlap_masks=False
                    image_masks = m[idx]
                else:  # overlap_masks=True
                    image_masks = m[[i]]
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im_arr = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if is_label or (cf is not None and cf[idx][j] > conf_thres):
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            # mask 尺寸与子图不一致 → 警告并 resize
                            LOGGER.warning(
                                f"plot_images_mm: mask shape ({mh}, {mw}) != subplot ({h}, {w}), "
                                f"resizing. 请确保 mask 尺寸正确以避免拉伸。"
                            )
                            mask_resized = cv2.resize(image_masks[j].astype(np.uint8), (w, h))
                            mask_resized = mask_resized.astype(bool)
                        else:
                            mask_resized = image_masks[j].astype(bool)
                        try:
                            im_arr[y_off : y_off + h, x_off : x_off + w, :][mask_resized] = (
                                im_arr[y_off : y_off + h, x_off : x_off + w, :][mask_resized] * 0.4
                                + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im_arr)

            # ===== 7.4 semantic masks 已在 4.5 预叠加，此处无需重复 =====

        # ===== 8. 图例（mosaic 右上角统一图例，颜色↔类名）=====
        if legend and len(c) > 0 and names:
            unique_cls = sorted(set(int(x) for x in c.tolist()))
            if unique_cls:
                im_arr = np.asarray(annotator.im).copy()
                mos_h, mos_w = im_arr.shape[:2]
                line_h = max(int(fs * 1.3), 22)
                block = max(int(fs * 0.7), 12)
                gap = 3
                txt_gap = block + 6
                # 估算图例宽度，定位到右上角（避开 modal_tag 区域）
                legend_w = txt_gap + max(
                    (len(str(names.get(ci, ci) if isinstance(names, dict) else ci)) for ci in unique_cls),
                    default=4,
                ) * int(fs * 0.55) + 10
                start_x = mos_w - legend_w - 8
                start_y = 8
                # 半透明背景板
                bg_box = im_arr[start_y - 4 : start_y + len(unique_cls) * (line_h + gap) + 4,
                                start_x - 8 : start_x + legend_w]
                if bg_box.size:
                    bg_box[...] = (bg_box.astype(np.float32) * 0.4 + 110).clip(0, 255).astype(np.uint8)
                for idx, ci in enumerate(unique_cls):
                    color_rgb = colors(ci)
                    c_name = names.get(ci, ci) if isinstance(names, dict) else ci
                    y_row = start_y + idx * (line_h + gap)
                    # 颜色块
                    im_arr[y_row : y_row + block, start_x : start_x + block] = color_rgb
                    # 类名文字（PIL 已在 annotator.draw，这里用 cv2 兜底描边）
                    cv2.putText(
                        im_arr, str(c_name), (start_x + txt_gap, y_row + block - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, max(fs / 28, 0.5),
                        (0, 0, 0), max(int(fs / 14), 1), cv2.LINE_AA,
                    )
                annotator.fromarray(im_arr)

        # ===== 9. 保存或返回 =====
        if not save:
            return np.asarray(annotator.im)
        annotator.im.save(fname)
        if on_plot:
            on_plot(fname)
        return None

    # ===== 9. 同步/异步控制（核心差异 #5）=====
    if threaded:
        Thread(target=_draw, daemon=True).start()
        return None
    return _draw()
