# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO parameter split utilities.

YOLOVO 模型同时包含视觉侧（backbone + neck + head）和文本侧（MobileCLIP
text tower + VCTE 融合层）。本模块提供按 YAML 边界拆分参数量的工具，
用于训练初始化 / 验证 / 推理时的复杂度日志输出。

边界定义（用户拍板，按 YAML 节点划分）：
  - 文本侧:
      * ``self.text_encoder``（MobileCLIPTextEncoder，含 text_model + proj）
      * ``self.model`` 中对应 YAML ``text:`` 节点的 VCTE 层
        （VCTETextInput / VCTETextAdapter / VCTEVisualAdapter / VCTETextEnhancer）
  - 图像侧: YAML ``backbone:`` + ``head:`` 对应的所有层
            （含 VCTENeckModulator / YOLOVODetect）

仅作用于 YOLOVO，不影响 YOLOMM / RTDETRMM / YOLOWorld / YOLOE 的复杂度输出。
"""

from __future__ import annotations


def _resolve_text_layer_indices(inner) -> set:
    """Resolve YAML ``text:`` layer indices in ``inner.model`` Sequential.

    优先使用 ``inner._text_layer_indices``（在 ``YOLOVOModel.__init__`` /
    ``__setstate__`` 中预先计算）。若缺失则 fallback 到 isinstance 过滤，
    结果与 YAML 划分完全对齐（VCTEVisualAdapter 也归文本侧）。
    """
    indices = getattr(inner, "_text_layer_indices", None)
    if indices is not None:
        return set(indices)

    # Fallback: 按 isinstance 过滤（覆盖旧 checkpoint 未预先记录 indices 的情况）
    try:
        from ultralytics.nn.modules.vcte import (
            VCTETextInput,
            VCTETextAdapter,
            VCTETextEnhancer,
            VCTEVisualAdapter,
        )
    except ImportError:
        return set()

    return {
        i
        for i, m in enumerate(inner.model)
        if isinstance(m, (VCTETextInput, VCTETextAdapter, VCTEVisualAdapter, VCTETextEnhancer))
    }


def count_yolovo_param_split(model) -> dict:
    """Return YOLOVO parameter split into total / vision / text.

    Args:
        model: YOLOVO 模型实例（允许被 DDP 包装）。

    Returns:
        dict with keys ``total``, ``vision``, ``text`` (int counts).
        严格保证 ``total == vision + text``。
        对非 YOLOVO 模型也安全返回（text=0，vision=total）。
    """
    from ultralytics.utils.torch_utils import de_parallel

    inner = de_parallel(model)

    # 1. 收集文本侧参数的 id 集合
    text_param_ids = set()

    # (a) text_encoder 前缀 —— 与 YOLOVOTrainer.build_optimizer / _param_groups_for_debug 对齐
    for name, p in inner.named_parameters():
        if name.startswith("text_encoder."):
            text_param_ids.add(id(p))

    # (b) self.model 中 YAML text: 节点对应的 VCTE 层
    text_indices = _resolve_text_layer_indices(inner)
    for i, m in enumerate(inner.model):
        if i in text_indices:
            for p in m.parameters():
                text_param_ids.add(id(p))

    # 2. 单次遍历分类所有参数，使用 id 集合去重
    total = 0
    text = 0
    for p in inner.parameters():
        n = p.numel()
        total += n
        if id(p) in text_param_ids:
            text += n

    return {"total": total, "vision": total - text, "text": text}


def format_yolovo_complexity_lines(model, report) -> list:
    """Return user-facing YOLOVO complexity log lines with param split.

    输出两行：
      1) 参数量拆分：total + vision + text
      2) GFLOPs 阶段分布（与 ``log_default_complexity`` 的 Stages 行对齐）
    """
    from ultralytics.utils.torch_utils import format_param_count

    split = count_yolovo_param_split(model)
    stage_raw = report.stage_flops()
    gflops_total = report.total_flops / 1e9
    stage_gflops = {k: v / 1e9 for k, v in stage_raw.items()}

    lines = [
        (
            f"Params: {format_param_count(split['total'])} ({split['total']:,})  "
            f"[vision: {format_param_count(split['vision'])} ({split['vision']:,})  |  "
            f"text: {format_param_count(split['text'])} ({split['text']:,})]  "
            f"| GFLOPs(total[default]): {gflops_total:.2f}"
        ),
    ]
    if "backbone" in stage_gflops:
        parts = [
            f"backbone={stage_gflops.get('backbone', 0.0):.2f}",
            f"neck={stage_gflops.get('neck', 0.0):.2f}",
        ]
        if "text" in stage_gflops:
            parts.append(f"text={stage_gflops.get('text', 0.0):.2f}")
        parts.append(f"head={stage_gflops.get('head', 0.0):.2f}")
        lines.append("Stages:  " + "  ".join(parts))
    return lines


def log_yolovo_complexity_with_param_split(model, report, logger) -> None:
    """Log YOLOVO complexity with param split using the provided logger."""
    for line in format_yolovo_complexity_lines(model, report):
        logger.info(line)
