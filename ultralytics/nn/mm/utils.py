# Ultralytics Multimodal Utilities
# Helper functions for multimodal system status and validation
# Version: v1.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ultralytics.utils import LOGGER

MM_METADATA_KEY = "metadata"
MM_X_MODALITY_KEY = "x_modality_name"
SINGLE_MODAL_BUILD_KEY = "single_modal_build"

# ---------------------------------------------------------------------------
# Input-source alias table (Decision 2: backward-compat + new A/B/Fuse)
# ---------------------------------------------------------------------------
INPUT_SOURCE_ALIASES: dict[str, str] = {
    # legacy semantics → internal role
    "RGB":  "PRIMARY",
    "X":    "SECONDARY",
    "Dual": "FUSED",
    # new semantics → internal role
    "A":    "PRIMARY",
    "B":    "SECONDARY",
    "Fuse": "FUSED",
    # YOLOVO text modality
    "Text": "TEXT",
    "TXT":  "TEXT",
}

# Case-insensitive lookup map built from INPUT_SOURCE_ALIASES
_CASE_INSENSITIVE_ALIAS_MAP: dict[str, str] = {
    k.casefold(): v for k, v in INPUT_SOURCE_ALIASES.items()
}


@dataclass(frozen=True)
class MultiModalChannelContract:
    """多模态入口通道契约 — 构建期契约，不表达当前消融保留哪一路。

    Attributes:
        primary_ch: 主模态通道数（来自 Pch，默认 3）
        secondary_ch: 次模态通道数（来自 Xch，默认 3）
        fused_ch: 融合输入通道数 = primary_ch + secondary_ch
        primary_modality: 主模态名称（如 ``"rgb"``）
        secondary_modality: 次模态名称（如 ``"depth"``）
        resolved_runtime_modality: 消融时保留的模态 token，未消融时为 None
        is_multimodal_model: 是否为多模态模型
    """

    primary_ch: int
    secondary_ch: int
    fused_ch: int
    primary_modality: str
    secondary_modality: str
    resolved_runtime_modality: Optional[str]
    is_multimodal_model: bool


@dataclass(frozen=True)
class SingleModalBuildSpec:
    """Build-time routing contract for ``single_modal`` YAML injection."""

    raw: str
    role_token: str
    role_name: str
    matched_modality: Optional[str] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "role_token": self.role_token,
            "role_name": self.role_name,
            "matched_modality": self.matched_modality,
        }


def resolve_input_source_role(token: object) -> Optional[str]:
    """Resolve an input-source token to its internal role (case-insensitive).

    Any capitalization of ``'RGB'``, ``'X'``, ``'Dual'``, ``'A'``, ``'B'``,
    ``'Fuse'`` is accepted.

    Returns:
        ``'PRIMARY'``, ``'SECONDARY'``, ``'FUSED'``, or ``None``.
    """
    if not isinstance(token, str):
        return None
    return _CASE_INSENSITIVE_ALIAS_MAP.get(token.strip().casefold())


def _role_name_to_single_modal_token(role_name: str) -> str:
    """Convert an internal input-source role to the stable YAML route token."""
    role = str(role_name).strip().upper()
    if role == "PRIMARY":
        return "A"
    if role == "SECONDARY":
        return "B"
    if role == "FUSED":
        return "Fuse"
    raise ValueError(f"single_modal does not support input role {role_name!r}")


def is_rgb_modality(modality_name: str) -> bool:
    """Return True when *modality_name* refers to RGB (case-insensitive).

    Used to gate colour-space operations (BGR->RGB, HSV augment, etc.).
    Only the exact token ``"rgb"`` (case-insensitive) qualifies.
    """
    return isinstance(modality_name, str) and modality_name.strip().lower() == "rgb"


def _clean_modality_value(value: Any) -> Optional[str]:
    """Return a stripped modality token when the input is a non-empty string."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_modality_lookup_token(value: Any) -> Optional[str]:
    """Normalize a modality token for case-insensitive lookup only."""
    cleaned = _clean_modality_value(value)
    return cleaned.casefold() if cleaned else None


def is_rgb_modality_token(value: Any) -> bool:
    """Return True when *value* semantically refers to RGB regardless of case."""
    return _normalize_modality_lookup_token(value) == 'rgb'


def _coerce_configured_modalities(values: Any) -> list[str]:
    """Normalize configured modality containers while preserving configured token spelling."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    cleaned: list[str] = []
    for value in values:
        token = _clean_modality_value(value)
        if token:
            cleaned.append(token)
    return cleaned


def get_configured_modality_tokens(data: Optional[dict]) -> list[str]:
    """Return configured modality tokens from data.yaml while preserving configured spelling."""
    data = data or {}

    for key in ("modality_used", "models"):
        tokens = _coerce_configured_modalities(data.get(key))
        if tokens:
            return tokens

    mapping = data.get("modality") or data.get("modalities")
    if isinstance(mapping, dict):
        tokens = [_clean_modality_value(key) for key in mapping.keys()]
        return [token for token in tokens if token]

    return []


def find_configured_modality_token(token: Any, configured_tokens: Sequence[str]) -> Optional[str]:
    """Find the configured token that matches *token* case-insensitively."""
    target = _normalize_modality_lookup_token(token)
    if not target:
        return None
    for configured in configured_tokens:
        if _normalize_modality_lookup_token(configured) == target:
            return str(configured).strip()
    return None


def resolve_modality_pair(
    data_or_tokens: Optional[Any],
    *,
    default_secondary: Optional[str] = None,
    strict: bool = False,
) -> tuple[str, str]:
    """Resolve the primary and secondary modality tokens from configuration.

    precision contract 恢复的是值域/加载/增强/预处理语义，
    不是回退模态选择机制。
    resolve_modality_pair()/resolve_requested_modality_token() 继续作为唯一模态解析入口。

    Unlike ``resolve_rgb_x_pair``, this function does NOT require RGB to be
    present.  It simply treats ``modality_used[0]`` as primary and
    ``modality_used[1]`` as secondary.

    Returns:
        (primary_token, secondary_token)
    """
    data = data_or_tokens if isinstance(data_or_tokens, dict) else None
    configured_tokens = (
        list(data_or_tokens)
        if isinstance(data_or_tokens, Sequence) and not isinstance(data_or_tokens, (str, bytes, dict))
        else get_configured_modality_tokens(data)
    )
    configured_tokens = _coerce_configured_modalities(configured_tokens)

    if strict:
        if len(configured_tokens) != 2:
            raise ValueError(
                f"modality configuration must contain exactly 2 modalities, got: {configured_tokens}"
            )
        return configured_tokens[0], configured_tokens[1]

    if len(configured_tokens) >= 2:
        return configured_tokens[0], configured_tokens[1]

    # Single token: primary from config, secondary from fallback
    if configured_tokens:
        primary = configured_tokens[0]
    elif isinstance(data, dict):
        primary = _clean_modality_value(data.get("primary_modality")) or "rgb"
    else:
        primary = "rgb"

    # Fallback: try to extract secondary from dict keys
    if isinstance(data, dict):
        explicit_secondary = _clean_modality_value(data.get("x_modality"))
        if explicit_secondary and configured_tokens:
            explicit_secondary = find_configured_modality_token(explicit_secondary, configured_tokens) or explicit_secondary
        fallback_secondary = (
            explicit_secondary
            or (_clean_modality_value(default_secondary) if default_secondary is not None else None)
            or "unknown"
        )
        return primary, fallback_secondary

    # No dict: use default_secondary or "unknown"
    fallback = (_clean_modality_value(default_secondary) if default_secondary is not None else None) or "unknown"
    return primary, fallback


def resolve_rgb_x_pair(
    data_or_tokens: Optional[Any],
    *,
    default_x: Optional[str] = None,
    strict: bool = False,
) -> tuple[str, str]:
    """Backward-compatible wrapper around ``resolve_modality_pair``.

    Preserves the original function signature so that all existing callers
    continue to work without changes.
    """
    return resolve_modality_pair(
        data_or_tokens,
        default_secondary=default_x,
        strict=strict,
    )


def resolve_requested_modality_token(
    requested: Optional[str],
    data_or_tokens: Optional[Any],
    *,
    default_x: Optional[str] = None,
    strict: bool = False,
) -> Optional[str]:
    """Resolve user-facing modality tokens (RGB/X/concrete) to configured modality tokens."""
    if requested is None:
        return None

    configured_tokens = (
        list(data_or_tokens)
        if isinstance(data_or_tokens, Sequence) and not isinstance(data_or_tokens, (str, bytes, dict))
        else get_configured_modality_tokens(data_or_tokens if isinstance(data_or_tokens, dict) else None)
    )
    configured_tokens = _coerce_configured_modalities(configured_tokens)
    normalized = normalize_modality_token(requested)

    primary_token, secondary_token = resolve_modality_pair(
        configured_tokens if configured_tokens else data_or_tokens,
        default_secondary=default_x,
        strict=strict and bool(configured_tokens),
    )

    if normalized == "RGB":
        return primary_token
    if normalized == "X":
        return secondary_token

    resolved = find_configured_modality_token(normalized, configured_tokens) if configured_tokens else None
    if resolved:
        return resolved

    if strict:
        raise ValueError(f"requested modality {requested!r} not found in configured modalities: {configured_tokens}")

    cleaned = _clean_modality_value(normalized)
    return cleaned


def resolve_single_modal_build_spec(single_modal: str, data_dict: dict) -> SingleModalBuildSpec:
    """Resolve ``single_modal`` into a build-time YAML injection spec."""
    if not isinstance(single_modal, str) or not single_modal.strip():
        raise ValueError("single_modal must be a non-empty string when provided")

    token = single_modal.strip()
    role_name = resolve_input_source_role(token)
    if role_name is not None:
        return SingleModalBuildSpec(
            raw=token,
            role_token=_role_name_to_single_modal_token(role_name),
            role_name=role_name,
            matched_modality=None,
        )

    configured_tokens = get_configured_modality_tokens(data_dict)
    if not configured_tokens:
        raise ValueError(
            f"single_modal='{token}': dataset has no modality_used configured, "
            f"cannot resolve input role"
        )

    for idx, configured_token in enumerate(configured_tokens):
        if _normalize_modality_lookup_token(configured_token) == _normalize_modality_lookup_token(token):
            role_name = "PRIMARY" if idx == 0 else "SECONDARY"
            return SingleModalBuildSpec(
                raw=token,
                role_token=_role_name_to_single_modal_token(role_name),
                role_name=role_name,
                matched_modality=configured_token,
            )

    raise ValueError(f"single_modal='{token}' not found in dataset modality_used: {configured_tokens}")


def resolve_single_modal_input_role(single_modal: str, data_dict: dict) -> str:
    """Compatibility wrapper returning the A/B/Fuse route token for ``single_modal``."""
    return resolve_single_modal_build_spec(single_modal, data_dict).role_token


def has_multimodal_routing_layers(cfg_dict: dict) -> bool:
    """Check whether any layer in the model YAML config has a 5th-field routing annotation."""
    all_layers = cfg_dict.get("backbone", []) + cfg_dict.get("head", [])
    for layer_config in all_layers:
        if len(layer_config) >= 5 and resolve_input_source_role(layer_config[4]) is not None:
            return True
    return False


SINGLE_MODAL_CONFLICT_MSG = (
    "当前模型配置yaml 文件已经存在 input 多模态输入字段，"
    "请勿配置 single_modal 超参数。"
    "多模态模型 yaml 与 single_modal 超参数不兼容，请注释。"
)


def has_multimodal_config(cfg_dict: dict) -> bool:
    """Check if model YAML config has multimodal routing configured.

    Checks both top-level ``input`` field and backbone/head layer 5th-field
    routing annotations.
    """
    if cfg_dict.get("input") is not None:
        if resolve_input_source_role(cfg_dict["input"]) is not None:
            return True
    all_layers = cfg_dict.get("backbone", []) + cfg_dict.get("head", [])
    for layer_config in all_layers:
        if len(layer_config) >= 5 and resolve_input_source_role(layer_config[4]) is not None:
            return True
    return False


def resolve_multimodal_channel_contract(
    *,
    data: Optional[dict] = None,
    requested_modality: Optional[str] = None,
    model_cfg: Optional[dict] = None,
) -> MultiModalChannelContract:
    """从 data.yaml 与模型 YAML 共同解析真实通道契约。

    规则：
    1. ``primary_ch`` 来自 ``Pch``，默认 3
    2. ``secondary_ch`` 来自 ``Xch``，默认 3
    3. 只要模型是多模态模型，``fused_ch = primary_ch + secondary_ch``
    4. ``requested_modality`` 只参与 runtime token 解析，不参与 build-time ``fused_ch`` 决策

    Args:
        data: 已解析的 data.yaml 字典
        requested_modality: 消融时指定的模态 token（如 ``"RGB"``, ``"depth"``）
        model_cfg: 模型 YAML 配置字典（用于检测多模态路由层）

    Returns:
        MultiModalChannelContract: 不可变的通道契约对象
    """
    dataset = data or {}
    primary_ch = int(dataset.get("Pch", 3))
    secondary_ch = int(dataset.get("Xch", 3))
    primary_modality, secondary_modality = resolve_modality_pair(
        dataset,
        default_secondary=extract_x_modality_name_from_data(dataset),
    )
    resolved_runtime_modality = None
    if requested_modality:
        resolved_runtime_modality = resolve_requested_modality_token(
            requested_modality,
            [primary_modality, secondary_modality],
            default_x=secondary_modality,
            strict=True,
        )

    has_mm = False
    if isinstance(model_cfg, dict):
        has_mm = has_multimodal_routing_layers(model_cfg)

    is_mm = has_mm or bool(dataset.get("modality_used") or dataset.get("models"))
    fused_ch = primary_ch + secondary_ch if is_mm else primary_ch
    return MultiModalChannelContract(
        primary_ch=primary_ch,
        secondary_ch=secondary_ch,
        fused_ch=fused_ch,
        primary_modality=primary_modality,
        secondary_modality=secondary_modality,
        resolved_runtime_modality=resolved_runtime_modality,
        is_multimodal_model=is_mm,
    )


def get_modality_path_from_data(data: Optional[dict], modality_name: str) -> str:
    """Return the configured modality directory using case-insensitive key matching.

    Resolution order:
    1. ``data["modality"]`` / ``data["modalities"]`` dictionary (exact match)
    2. Infer from ``modality_used`` + ``train`` path (primary) or ``images_xxx`` convention
    3. Hardcoded fallback: ``rgb -> "images"``, others -> ``"images_{name}"``
    """
    data = data or {}

    # Priority 1: explicit mapping in config
    mapping = data.get("modality") or data.get("modalities")
    if isinstance(mapping, dict):
        resolved_key = find_configured_modality_token(modality_name, mapping.keys())
        if resolved_key is not None and resolved_key in mapping:
            return mapping[resolved_key]

    # Priority 2: infer from modality_used + train path
    modality_used = data.get("modality_used") or data.get("models")
    if isinstance(modality_used, (list, tuple)) and modality_used:
        normalized = _normalize_modality_lookup_token(modality_name)
        first_normalized = _normalize_modality_lookup_token(modality_used[0])
        if normalized == first_normalized:
            # Primary modality: extract directory name from train path
            train_path = data.get("train", "").replace("\\", "/").strip("./")
            for segment in train_path.split("/"):
                if segment.startswith("images"):
                    return segment
            # Fallback: convention images_<modality> or "images" for rgb
            if normalized == "rgb":
                return "images"
            return f"images_{modality_name}"
        else:
            # Secondary modality: images_xxx convention
            return f"images_{modality_name}"

    # Priority 3: final fallback (original behaviour)
    if _normalize_modality_lookup_token(modality_name) == "rgb":
        return "images"
    return f"images_{modality_name}"


def normalize_modality_token(modality: Optional[str]) -> Optional[str]:
    """Normalize modality tokens for internal routing.

    Legacy: rgb -> RGB, x -> X
    New:    a -> PRIMARY, b -> SECONDARY, fuse -> FUSED
    Other concrete modality names (thermal/depth/ir/...) are kept as-is.
    """
    if modality is None:
        return None
    if not isinstance(modality, str):
        return modality
    m = modality.strip()
    low = m.lower()
    # Legacy aliases
    if low == "rgb":
        return "RGB"
    if low == "x":
        return "X"
    # New aliases -> resolve to internal roles
    if low == "a":
        return "PRIMARY"
    if low == "b":
        return "SECONDARY"
    if low == "fuse":
        return "FUSED"
    return m


def extract_x_modality_name_from_data(data: Optional[dict]) -> str:
    """训练期专用：从已解析的数据配置中提取 X 模态名字。"""
    data = data or {}

    explicit_x = _clean_modality_value(data.get("x_modality"))
    if explicit_x:
        configured_tokens = get_configured_modality_tokens(data)
        if configured_tokens:
            resolved = resolve_requested_modality_token(explicit_x, configured_tokens, default_x=explicit_x)
            if resolved and _normalize_modality_lookup_token(resolved) != "rgb":
                return resolved
        return explicit_x

    _rgb_token, x_token = resolve_rgb_x_pair(data, default_x="unknown")
    return x_token or "unknown"


def set_x_modality_name_in_ckpt_metadata(
    ckpt: dict,
    x_modality_name: Optional[str],
    primary_modality_name: Optional[str] = None,
) -> dict:
    """Write modality names to ckpt metadata."""
    metadata = ckpt.get(MM_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[MM_X_MODALITY_KEY] = str(x_modality_name or "unknown").strip() or "unknown"
    if primary_modality_name is not None:
        metadata["primary_modality_name"] = str(primary_modality_name).strip() or "rgb"
    ckpt[MM_METADATA_KEY] = metadata
    return ckpt


def validate_single_modal_and_modality_exclusive(args: Any) -> None:
    """Fail fast when build-time ``single_modal`` and runtime ``modality`` are mixed."""
    if args is None:
        return

    if isinstance(args, dict):
        single_modal = args.get("single_modal")
        modality = args.get("modality")
    else:
        single_modal = getattr(args, "single_modal", None)
        modality = getattr(args, "modality", None)

    if _clean_modality_value(single_modal) and _clean_modality_value(modality):
        raise ValueError(
            "single_modal and modality cannot be used together. "
            "single_modal is build-time YAML injection for plain single-modal models; "
            "modality is runtime ablation/filling for already-multimodal models."
        )


def get_single_modal_build_metadata_from_model(model: Any) -> Optional[dict[str, Any]]:
    """Return verified ``single_modal_build`` metadata from ``model.yaml`` if present."""
    for candidate in (model, getattr(model, "model", None)):
        yaml_dict = getattr(candidate, "yaml", None) if candidate is not None else None
        if isinstance(yaml_dict, dict):
            meta = yaml_dict.get(SINGLE_MODAL_BUILD_KEY)
            if isinstance(meta, dict) and meta and has_multimodal_config(yaml_dict):
                return dict(meta)
    return None


def set_single_modal_build_metadata_in_ckpt(ckpt: dict, model: Any = None, args: Any = None) -> dict:
    """Persist verified ``single_modal`` build metadata into checkpoint fields."""
    meta = get_single_modal_build_metadata_from_model(model)
    raw_arg = getattr(args, "single_modal", None) if args is not None else None
    if isinstance(args, dict):
        raw_arg = args.get("single_modal")

    if not meta:
        if _clean_modality_value(raw_arg):
            raise ValueError(
                f"single_modal={raw_arg!r} was requested, but model.yaml has no verified "
                f"{SINGLE_MODAL_BUILD_KEY!r} metadata. Refusing to mark checkpoint as multimodal."
            )
        return ckpt

    ckpt[SINGLE_MODAL_BUILD_KEY] = dict(meta)
    ckpt["is_multimodal"] = True
    metadata = ckpt.get(MM_METADATA_KEY)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[SINGLE_MODAL_BUILD_KEY] = dict(meta)
    ckpt[MM_METADATA_KEY] = metadata
    return ckpt


def sync_x_modality_name_to_model(
    model: Any,
    x_modality_name: Optional[str],
    primary_modality_name: Optional[str] = None,
) -> str:
    """Sync modality names to model metadata, router, and module attributes."""
    secondary = str(x_modality_name or "unknown").strip() or "unknown"
    primary = str(primary_modality_name or "rgb").strip() or "rgb"

    for candidate in (model, getattr(model, "model", None)):
        if candidate is None:
            continue
        model_metadata = getattr(candidate, "metadata", None)
        if not isinstance(model_metadata, dict):
            model_metadata = {}
        model_metadata[MM_X_MODALITY_KEY] = secondary
        model_metadata["primary_modality_name"] = primary
        setattr(candidate, "metadata", model_metadata)

    # Sync to router
    router_chains = (
        ("mm_router",),
        ("multimodal_router",),
        ("model", "mm_router"),
        ("model", "multimodal_router"),
    )
    for chain in router_chains:
        obj = model
        for attr in chain:
            if not hasattr(obj, attr):
                obj = None
                break
            obj = getattr(obj, attr)
        if obj is not None:
            setattr(obj, "x_modality_type", secondary)
            if hasattr(obj, "primary_modality_type"):
                setattr(obj, "primary_modality_type", primary)

    # Sync to modules
    modules_fn = getattr(model, "modules", None)
    if callable(modules_fn):
        for module in modules_fn():
            if hasattr(module, "_mm_x_modality"):
                module._mm_x_modality = secondary
            if hasattr(module, "_mm_primary_modality"):
                module._mm_primary_modality = primary

    return secondary


def get_runtime_x_modality_name(model: Any) -> str:
    """Read X modality name from checkpoint metadata; fallback to 'unknown'."""
    for candidate in (model, getattr(model, "model", None)):
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict):
            name = metadata.get(MM_X_MODALITY_KEY)
            if isinstance(name, str) and name.strip():
                return name.strip()
    return "unknown"


def get_runtime_primary_modality_name(model: Any) -> str:
    """Read primary modality name from checkpoint metadata; fallback to 'rgb'."""
    for candidate in (model, getattr(model, "model", None)):
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict):
            name = metadata.get("primary_modality_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return "rgb"


def validate_mm_config_format(config):
    """Validate multimodal configuration format correctness"""
    routing_layers = {"PRIMARY": [], "SECONDARY": [], "FUSED": []}

    for section in ['backbone', 'head']:
        for i, layer_config in enumerate(config.get(section, [])):
            if len(layer_config) == 5:
                input_source = layer_config[4]
                role = resolve_input_source_role(input_source)
                if role and role in routing_layers:
                    routing_layers[role].append(i)

    total = sum(len(v) for v in routing_layers.values())
    LOGGER.info(f"MultiModal: config validation complete, {total} routing layers")
    for role, layers in routing_layers.items():
        if layers:
            LOGGER.info(f"MultiModal: {role} routed={len(layers)}")

    return {
        'primary_layers': routing_layers["PRIMARY"],
        'secondary_layers': routing_layers["SECONDARY"],
        'fused_layers': routing_layers["FUSED"],
        'total_routing_layers': total,
    }


def mm_system_status():
    """Display multimodal system status"""
    LOGGER.info("MultiModal: dynamic modality routing system status check...")
    LOGGER.info("MultiModal: supports arbitrary dual-modality (Phase 1)")
    LOGGER.info("MultiModal: input aliases: RGB/X/Dual (legacy) + A/B/Fuse (new)")
    LOGGER.info("MultiModal: config format: [from, repeats, module, args, 'RGB'/'X'/'Dual'/'A'/'B'/'Fuse']")
    LOGGER.info("MultiModal: system version: v2.0 - universal modality routing")
    return True


def check_mm_model_attributes(model):
    """Check multimodal attributes in the model"""
    mm_layers = []
    
    for i, m in enumerate(model.model if hasattr(model, 'model') else []):
        if hasattr(m, '_mm_input_source'):
            layer_info = {
                'layer_index': getattr(m, '_mm_layer_index', i),
                'input_source': getattr(m, '_mm_input_source', None),
                'x_modality': getattr(m, '_mm_x_modality', 'unknown'),
                'version': getattr(m, '_mm_version', 'unknown'),
                'new_input_start': getattr(m, '_mm_new_input_start', False)
            }
            mm_layers.append(layer_info)
    
    if mm_layers:
        LOGGER.info(f"MultiModal: 发现 {len(mm_layers)} 个多模态路由层")
        for layer_info in mm_layers:
            source = layer_info['input_source']
            idx = layer_info['layer_index']
            if layer_info['new_input_start']:
                LOGGER.info(f"MultiModal: Layer {idx} - {source}模态新输入起点")
            else:
                LOGGER.info(f"MultiModal: Layer {idx} - {source}模态路由层")
    else:
        LOGGER.info("MultiModal: 未发现多模态路由层，使用标准模式")
    
    return mm_layers


def get_mm_system_info():
    """Get multimodal system information"""
    return {
        'version': 'v2.0',
        'supported_input_aliases': list(INPUT_SOURCE_ALIASES.keys()),
        'internal_roles': ['PRIMARY', 'SECONDARY', 'FUSED'],
        'supported_architectures': ['YOLO', 'RTDETR'],
    }


def confirm_modality_ablation(modality: Optional[str]) -> None:
    """首次使用 modality 做消融时给出确认提示，并把结果持久化到独立状态文件。"""
    if not modality:
        return

    from ultralytics.nn.mm.ablation_notice import ModalityAblationNoticeStore

    store = ModalityAblationNoticeStore()
    if store.is_confirmed():
        return

    red = "\033[31m"
    reset = "\033[0m"
    msg = (
        f"{red}"
        f"⚠️  MODALITY ABLATION NOTICE\n"
        f"modality 参数仅用于模态消融实验（Modality Ablation Study），并非单模态训练/推理。\n"
        f"\n"
        f"模态消融实验：在双模态模型训练完成后，通过 modality 参数选择性遮蔽某一模态的输入通道，\n"
        f"以评估各模态对模型性能的贡献。此操作不会改变模型结构，仅控制数据流。\n"
        f"{reset}\n"
        f"确认继续？[y/N]: "
    )
    answer = input(msg).strip().lower()
    if answer in ("y", "yes"):
        store.mark_confirmed()
        return
    raise RuntimeError("Modality ablation cancelled by user.")


def check_tensor_dtype_for_half(tensor, context=""):
    """在 half() 转换前检查 float64 张量。"""
    import torch
    if tensor.dtype == torch.float64:
        raise ValueError(
            f"{context} Cannot convert float64 tensor to float16 — "
            f"this would lose twice the precision of a float32 cast. "
            f"Convert to float32 first."
        )


def prepare_multimodal_cfg_dict(cfg, data):
    """Load model YAML as dict and inject ``dataset_config`` field.

    Shared utility called by all multimodal trainer ``get_model`` methods.
    Handles YAML string loading, dict deepcopy and ``dataset_config`` injection.
    ``single_modal`` YAML injection happens in ``prepare_multimodal_build_cfg``.

    Args:
        cfg: Model config (str path or dict).
        data: Dataset config dict (must contain Xch, x_modality, etc.).
    Returns:
        dict: cfg_dict ready for DetectionModel / SegmentationModel / etc.
        Falls back to original *cfg* when YAML loading fails.
    """
    from copy import deepcopy

    if isinstance(cfg, str):
        try:
            from ultralytics.nn.tasks import yaml_model_load
            cfg_dict = yaml_model_load(cfg)
        except Exception:
            return cfg
    elif isinstance(cfg, dict):
        cfg_dict = deepcopy(cfg)
    else:
        return cfg

    cfg_dict['dataset_config'] = dict(data)
    return cfg_dict


def inject_single_modal_yaml(cfg_dict: dict, single_modal: str, data_dict: dict) -> dict:
    """Convert a plain single-modal YAML dict to the internal multimodal build shape."""
    spec = resolve_single_modal_build_spec(single_modal, data_dict)

    cfg_dict["input"] = spec.role_token
    cfg_dict[SINGLE_MODAL_BUILD_KEY] = spec.to_metadata()

    backbone = cfg_dict.get("backbone", [])
    if not backbone:
        raise ValueError("single_modal YAML injection requires a non-empty backbone")

    first = list(backbone[0])
    if len(first) == 4:
        first.append(spec.role_token)
    elif len(first) >= 5:
        first[4] = spec.role_token
    else:
        raise ValueError(f"Invalid first backbone layer for single_modal injection: {first}")
    backbone[0] = first
    return cfg_dict


def prepare_multimodal_build_cfg(cfg, data, args: Any = None, logger: Any = None):
    """Prepare trainer model cfg with build-time ``single_modal`` YAML injection."""
    from copy import deepcopy

    log = logger or LOGGER
    validate_single_modal_and_modality_exclusive(args)
    if isinstance(args, dict):
        arg_model = args.get("model")
        single_modal = args.get("single_modal")
        data_path = args.get("data")
    else:
        arg_model = getattr(args, "model", None) if args is not None else None
        single_modal = getattr(args, "single_modal", None) if args is not None else None
        data_path = getattr(args, "data", None) if args is not None else None
    cfg_in = cfg or arg_model

    if single_modal is None:
        return prepare_multimodal_cfg_dict(cfg_in, data)
    if not isinstance(single_modal, str) or not single_modal.strip():
        raise ValueError("single_modal must be a non-empty string when provided")

    if isinstance(cfg_in, str):
        from ultralytics.nn.tasks import yaml_model_load

        cfg_dict_raw = yaml_model_load(cfg_in)
    elif isinstance(cfg_in, dict):
        cfg_dict_raw = deepcopy(cfg_in)
    else:
        raise ValueError(f"single_modal={single_modal!r} requires a YAML/dict model config, got {type(cfg_in).__name__}")

    if has_multimodal_config(cfg_dict_raw):
        raise ValueError(
            f"single_modal='{single_modal}' can only be used with a plain single-modal YAML. "
            "The provided model YAML already has multimodal routing."
        )

    data_dict = dict(data) if isinstance(data, dict) else {}
    if data_path:
        from ultralytics.utils import YAML, checks

        data_dict = YAML.load(checks.check_yaml(data_path))
    if not data_dict:
        raise ValueError(f"single_modal='{single_modal}' requires dataset config to resolve A/B/Fuse role")

    inject_single_modal_yaml(cfg_dict_raw, single_modal, data_dict)
    if log:
        log.info(
            f"single_modal='{single_modal}' injected as input={cfg_dict_raw.get('input')} "
            "for build-time multimodal routing."
        )

    return prepare_multimodal_cfg_dict(cfg_dict_raw, data)
