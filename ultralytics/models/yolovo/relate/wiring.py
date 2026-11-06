"""Relate wiring: from 解析、两步构建追加、relate_info 读写与校验、谓词编码。

设计文档: docs/plans/2026-10-05-yolovo-relate-design.md §4/§5
- relate 段永不进 parse_model（B2 方案）：resolve_from 自有解析器，attach_relate_layer
  程序化 add_module 追加并补 i/f/type/save 簿记。
- relate_info: task/primary/from_resolved/rnames/text_encoder/templates/text_dim。
- 谓词编码：与 checkpoint 同一 MobileCLIP + 模板系集成编码（归一化均值）；
  共享串重编码 tripwire cos>0.999 防跨编码器/跨模板静默失效。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from ultralytics.nn.modules.head import OBB, Detect, Pose, Segment
from ultralytics.utils import LOGGER

# 与一级头各特征输入 stride 对应的 P 编号（由一级头决定，不随 neck 变化）
STRIDE_TO_PNAME = {8: "P3", 16: "P4", 32: "P5"}

# 谓词模板系（单一事实源；集成编码取归一化均值）。
# 第一人称短语与 MobileCLIP 物体名词训练分布更接近。
TRAIN_TEMPLATES = ("a photo of {}.", "{}", "a relation of {}.")

# tripwire 共享串：每次编码都附带，用于跨编码器/跨模板一致性校验
_TRIPWIRE_STRING = "relation"

PRIMARY_CLASSES = (Detect, Segment, Pose, OBB)


def find_primary_head(model):
    """倒序查找一级检测头；找不到返回 None。"""
    for m in reversed(model.model):
        if isinstance(m, PRIMARY_CLASSES):
            return m
    return None


def _fmt_allowed(allowed) -> str:
    return ", ".join(f"{a['name']}(L{a['layer']}, s{a['stride']})" for a in allowed)


def resolve_from(model, inputs) -> List[Dict[str, Any]]:
    """解析 inputs（yaml 路径或列表）为 from_resolved 列表。

    每项 {name, layer, stride}。P 编号按一级头特征输入的 stride 解析；
    整数按层号；两种写法可混用；输出按一级头原输入顺序排列。
    非法值抛 ValueError 并打印允许的 P 名与层号。
    """
    head = find_primary_head(model)
    if head is None:
        raise ValueError("模型中没有一级检测头（YOLOVODetect/Segment/OBB/Pose），不能接入关系层。")

    # 一级头合法特征输入：f 中除 -1 外的层号 → stride
    f = head.f if isinstance(head.f, (list, tuple)) else [head.f]
    head_layers = [int(x) for x in f if int(x) != -1]
    # stride 推断：对一级头做一次小输入前向代价高；直接用 head.stride（[nl] 张量）
    strides = head.stride.detach().cpu().tolist() if torch.is_tensor(head.stride) else list(head.stride)
    layer_to_stride = dict(zip(head_layers, [int(s) for s in strides]))

    # 文本层禁止：text 层索引在 YOLOVOModel._text_layer_indices
    text_layers = set(getattr(model, "_text_layer_indices", frozenset()))

    allowed = []
    for layer, stride in layer_to_stride.items():
        if layer in text_layers:
            continue
        allowed.append({"name": STRIDE_TO_PNAME.get(stride, f"L{layer}"), "layer": layer, "stride": stride})

    # 解析 inputs
    if isinstance(inputs, (str,)):
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml

        inputs = YAML.load(check_yaml(str(inputs))).get("from", None)
        if inputs is None:
            raise ValueError("接线 YAML 缺少 `from` 键。")
    if inputs is None:
        inputs = [a["name"] for a in allowed]  # 缺省全接入
    if not isinstance(inputs, (list, tuple)):
        raise ValueError(f"inputs 必须是 yaml 路径或列表，收到 {type(inputs)}。")

    picked = {}
    for item in inputs:
        if isinstance(item, int):
            if item not in layer_to_stride:
                raise ValueError(
                    f"层号 {item} 不是该一级头的特征输入（文本层同样禁止）。允许: "
                    f"{_fmt_allowed(allowed)}"
                )
            stride = layer_to_stride[item]
            if item in text_layers:
                raise ValueError(f"层号 {item} 是类别文本层，不能作为关系特征输入。")
            key, name = item, STRIDE_TO_PNAME.get(stride, f"L{item}")
        elif isinstance(item, str):
            item_u = item.upper()
            by_name = [a for a in allowed if a["name"] == item_u]
            if not by_name:
                raise ValueError(
                    f"P 编号 {item!r} 不在该一级头特征输入内。允许: "
                    f"{_fmt_allowed(allowed)}"
                )
            key, name = by_name[0]["layer"], item_u
        else:
            raise ValueError(f"inputs 条目必须是 int 层号或 str P 编号，收到 {item!r}。")
        if key in picked:
            raise ValueError(f"inputs 中 {item!r} 重复（解析到层 {key}）。")
        picked[key] = {"name": name, "layer": key, "stride": layer_to_stride[key]}

    # 按一级头原输入顺序排列
    resolved = [picked[l] for l in sorted(picked)]
    return resolved


def attach_relate_layer(model, inputs, rnames: Dict[int, str], field_cfg=None, obj_names=None):
    """两步构建的第二步：程序化追加 YOLOVORelate（RA 结构）到一级模型。

    追加后模型 save 列表补一级头输出层号；relate_info 写入 model 属性。
    """
    from ultralytics.models.yolovo.relate.head import YOLOVORelate

    if getattr(model, "relate_info", None):
        return model  # 已带关系层，幂等

    head = find_primary_head(model)
    if head is None:
        raise ValueError("模型中没有一级检测头，不能接入关系层。")

    from_resolved = resolve_from(model, inputs)

    # 谓词编码（与 checkpoint 同一 text tower）
    rnames_norm = {int(k): str(v) for k, v in dict(rnames).items()}
    n_pred = len(rnames_norm)
    if n_pred == 0:
        raise ValueError("rnames 谓词词表为空。")
    W, text_dim, trip_cos = encode_predicate_bank(model, rnames_norm)

    rel = YOLOVORelate(
        primary_head=head,
        feat_layers=[r["layer"] for r in from_resolved],
        feat_strides=[r["stride"] for r in from_resolved],
        n_predicates=n_pred,
        text_dim=text_dim,
        feat_channels=_primary_feat_channels(head, from_resolved),
        field_cfg=field_cfg,
    )
    # 词表 + 类别名嵌入：谓词 W；类别 W_obj 用同一 MobileCLIP 塔。
    # obj_names 来自 data yaml 的 names（RA-4M 497 类 ≠ 一级模型的 20/80 类）。
    W_obj = _encode_class_names(model, obj_names)
    rel.install_vocabulary([rnames_norm[i] for i in sorted(rnames_norm)], W, W_obj)
    # 程序化追加簿记（对照 parse_model 行为）
    head_idx = len(model.model) - 1  # 一级头当前是最后一层（两步构建前提）
    rel.i = len(model.model)
    rel.f = [head_idx] + [r["layer"] for r in from_resolved]
    rel.type = rel.__class__.__name__
    model.model.add_module(str(rel.i), rel)
    model.save.append(head_idx)  # 一级头输出必须入 save

    model.relate_info = {
        "task": getattr(model, "task", None) or "detect",
        "primary": head.__class__.__name__,
        "from_resolved": from_resolved,
        "rnames": rnames_norm,
        "text_encoder": getattr(getattr(model, "text_encoder_cfg", {}), "get", lambda *_: None)("name")
        or "mobileclip_s0",
        "templates": list(TRAIN_TEMPLATES),
        "text_dim": text_dim,
        "tripwire_cos": round(float(trip_cos), 6),
        "field_cfg": rel.field_cfg.to_dict(),
    }
    LOGGER.info(
        f"Relate layer attached: from={[(r['name'], r['layer']) for r in from_resolved]}, "
        f"V={n_pred}, text_dim={text_dim}, tripwire cos={model.relate_info['tripwire_cos']:.6f}"
    )
    return model


@torch.no_grad()
def _primary_feat_channels(head, from_resolved) -> Optional[List[int]]:
    """Infer the stride-aligned feature channels from the primary head's cv2 branches.

    ``_materialize_pool`` used to guess 128 for P4, which silently breaks
    yolo11s/m (P4=256/384).  Return ``None`` if the head layout is unknown;
    the lazy ``_fmap`` path then still works at eval time.
    """
    import torch.nn as nn

    strides = head.stride.detach().cpu().tolist() if torch.is_tensor(head.stride) else list(head.stride)
    chs = []
    for branch in getattr(head, "cv2", []):
        for m in branch.modules():
            if isinstance(m, nn.Conv2d):
                chs.append(int(m.in_channels))
                break
    if len(chs) != len(strides):
        return None
    stride_to_ch = {int(s): c for s, c in zip(strides, chs)}
    picked = [stride_to_ch.get(int(r["stride"])) for r in from_resolved]
    return None if any(c is None for c in picked) else picked


@torch.no_grad()
def _encode_class_names(model, names=None) -> Optional[torch.Tensor]:
    """类别名经同一 MobileCLIP 塔编码 → W_obj [n_cls, text_dim]（Q2 对齐损失用）。"""
    names = names or getattr(model, "names", None)
    if not names:
        return None
    cls_list = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if not cls_list:
        return None
    bank = {i: n for i, n in enumerate(cls_list)}
    W, _, _ = encode_predicate_bank(model, bank)
    return W


@torch.no_grad()
def encode_predicate_bank(model, rnames: Dict[int, str]):
    """MobileCLIP 模板系集成编码谓词词表 → (W [V, text_dim], text_dim, tripwire_cos)。

    每个谓词对全部模板编码后取归一化均值再归一化；tripwire 共享串同样编码，
    供加载侧比对（cos>0.999 表示同一编码器+模板空间）。
    """
    import os

    was_training = model.training
    model.eval()
    # attach 发生在模型搬到 GPU 之前：10,370 谓词 × 3 模板 ≈ 31k 条文本在 CPU 上要跑 6~7 分钟
    # （每次启动都重复）。有卡时把文本塔临时搬到 GPU 编码，结束后搬回原设备；
    # 编码结果只差浮点舍入（~1e-6），tripwire 容差 1e-3 不受影响。YOLOVO_TEXT_ENCODE_GPU=0 关闭。
    te = getattr(model, "text_encoder", None)
    te_dev = None
    if te is not None and torch.cuda.is_available() and os.environ.get("YOLOVO_TEXT_ENCODE_GPU", "1") != "0":
        try:
            te_dev = next(te.parameters()).device
        except StopIteration:
            te_dev = None
        if te_dev is not None and te_dev.type == "cuda":
            te_dev = None  # 已在 GPU，无需搬动
        elif te_dev is not None:
            te.to("cuda")
    try:
        ids = sorted(rnames)
        texts, owner = [], []  # owner: (predicate_id, template_idx)
        for pid in ids:
            for t in TRAIN_TEMPLATES:
                texts.append(t.format(rnames[pid]))
                owner.append((pid, len(TRAIN_TEMPLATES)))
        texts.append(_TRIPWIRE_STRING)
        # 分批编码：RA-4M 的 10,370 谓词 × 3 模板一次进塔会 OOM。
        emb = torch.cat([model.encode_text(texts[i:i + 512]).float().cpu()
                         for i in range(0, len(texts), 512)])
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb)
        trip = emb[-1]
        emb = emb[:-1]
        text_dim = emb.shape[-1]
        V = len(ids)
        emb = emb.reshape(V, len(TRAIN_TEMPLATES), text_dim)
        W = torch.nn.functional.normalize(emb.mean(dim=1), dim=-1)
        trip_cos = float(torch.dot(W[0], trip).clamp(-1, 1)) if V else 0.0
        return W.cpu(), int(text_dim), trip_cos
    finally:
        if te_dev is not None:
            te.to(te_dev)
        if was_training:
            model.train()


def validate_relate_info(info: Any, W_shape: Optional[tuple] = None, expect_tripwire_cos: Optional[float] = None):
    """relate_info 结构校验 + W 形状校验 + tripwire 比对。fail-fast。"""
    if not isinstance(info, dict) or not info:
        raise ValueError("relate_info 缺失或为空：rel.val/rel.predict 只接受二级关系权重。")
    required = ("task", "primary", "from_resolved", "rnames", "text_dim")
    missing = [k for k in required if k not in info]
    if missing:
        raise ValueError(f"relate_info 缺少字段: {missing}。")
    if not isinstance(info["rnames"], dict) or not info["rnames"]:
        raise ValueError("relate_info.rnames 必须是非空 dict。")
    if W_shape is not None:
        V = len(info["rnames"])
        if tuple(W_shape) != (V, info["text_dim"]):
            raise ValueError(f"predicate_bank 形状 {tuple(W_shape)} 与词表不匹配 (V={V}, text_dim={info['text_dim']})。")
    if expect_tripwire_cos is not None:
        ref = float(info.get("tripwire_cos", 0.0))
        if abs(expect_tripwire_cos - ref) > 1e-3:
            raise ValueError(
                f"谓词库 tripwire 不一致: 当前编码 cos={expect_tripwire_cos:.6f} vs "
                f"checkpoint 记录 {ref:.6f}——编码器或模板与训练时不符，拒绝加载（garbage not error）。"
            )


def resolve_rnames(rnames_override: Any = None, data_yaml: Union[str, dict, None] = None) -> Dict[int, str]:
    """谓词词表解析：显式覆盖 > data yaml rnames。"""
    from ultralytics.models.yolovo.relate.dataset import parse_rnames

    if rnames_override is not None:
        return parse_rnames(rnames_override)
    if data_yaml is None:
        raise ValueError("未提供 rnames：请在 data yaml 写 rnames 键或显式传 rnames 参数。")
    if isinstance(data_yaml, (str,)):
        from ultralytics.utils import YAML

        data = YAML.load(data_yaml)
    else:
        data = data_yaml
    if "rnames" not in (data or {}):
        raise ValueError(f"{data_yaml} 缺少 rnames 键（谓词词表）。")
    return parse_rnames(data["rnames"])


def resolve_objnames(data_yaml: Union[str, dict, None] = None) -> Optional[List[str]]:
    """data yaml 的 ``names`` → W_obj 类别词表；缺省则沿用一级模型 names。"""
    if data_yaml is None:
        return None
    if isinstance(data_yaml, str):
        from ultralytics.utils import YAML

        data = YAML.load(data_yaml)
    else:
        data = data_yaml
    names = (data or {}).get("names")
    if not names:
        return None
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]


def load_relate_info(ckpt: Any) -> Optional[Dict]:
    """从 checkpoint dict 读 relate_info；无则 None。"""
    if isinstance(ckpt, dict):
        return ckpt.get("relate_info")
    return None
