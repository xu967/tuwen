#!/usr/bin/env python3
"""YOLOVO distillation YAML TUI.

This rich-based terminal tool guides users through generating or editing the
``distill_yolovo.yaml`` file consumed by ``YOLOVO.train(distill=[yaml, mode])``.
It is intentionally YOLOVO-only: v1 supports RGB teachers and explicit feature
layer mappings.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

console = Console()
_MAX_RETRIES = 10
_CFG_MODELS_ROOT = _PROJECT_ROOT / "ultralytics" / "cfg" / "models"
_DEFAULT_YOLOVO_YAML = _CFG_MODELS_ROOT / "yolovo" / "yolovo11.yaml"


@dataclass
class LayerInfo:
    """Single model layer metadata."""

    index: int
    name: str
    out_channels: int
    input_source: Optional[str] = None
    f: object = None


@dataclass
class TeacherEntry:
    """Teacher information collected by the TUI."""

    name: str
    role: str
    weights: str
    yaml_path: Optional[str] = None
    layers: List[LayerInfo] = field(default_factory=list)


@dataclass
class FeatureMapping:
    """Feature layer mapping between teacher and student."""

    teacher_name: str
    teacher_layer: int | tuple
    student_layer: int | tuple


def _get_out_channels_from_state_dict(state_dict: dict, layer_idx: int) -> int:
    """Infer a layer's output channels from a state_dict."""
    prefix = f"model.{layer_idx}."
    for suffix in ("conv.weight", "weight"):
        key = prefix + suffix
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key, value in state_dict.items():
        if key.startswith(prefix) and hasattr(value, "dim") and value.dim() == 4:
            return int(value.shape[0])
    return -1


def _get_out_channels_from_module(module, state_dict: dict, layer_idx: int) -> int:
    """Try several common module attributes before falling back to state_dict."""
    if hasattr(module, "out_channels"):
        return int(module.out_channels)
    if hasattr(module, "conv") and hasattr(module.conv, "out_channels"):
        return int(module.conv.out_channels)
    if hasattr(module, "cv2") and hasattr(module.cv2, "conv"):
        return int(module.cv2.conv.out_channels)
    if hasattr(module, "cv3") and hasattr(module.cv3, "conv"):
        return int(module.cv3.conv.out_channels)
    return _get_out_channels_from_state_dict(state_dict, layer_idx)


def _select_index_from_list(options: list[str], prompt_text: str) -> int:
    """Display numbered options and return the selected index."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("No.", justify="right", style="cyan", width=4)
    table.add_column("Option", style="white")
    for index, option in enumerate(options):
        table.add_row(str(index + 1), option)
    console.print(table)

    while True:
        raw = Prompt.ask(f"{prompt_text} (序号或名称)").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            console.print(f"[red]序号超出范围 [1, {len(options)}][/red]")
            continue
        except ValueError:
            pass
        for index, option in enumerate(options):
            if raw == option or raw in option:
                return index
        console.print(f"[red]未匹配到选项: '{raw}'[/red]")


def _search_yaml_in_cfg(name: str) -> Optional[Path]:
    """Search ``ultralytics/cfg/models`` for a YAML file by name."""
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    if not _CFG_MODELS_ROOT.is_dir():
        return None
    matches = list(_CFG_MODELS_ROOT.rglob(name))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    console.print(f"[yellow]找到 {len(matches)} 个匹配:[/yellow]")
    labels = [str(match.relative_to(_PROJECT_ROOT)) for match in matches]
    return matches[_select_index_from_list(labels, "选择配置文件")]


def _resolve_model_path(raw_path: str) -> Optional[str]:
    """Resolve absolute paths, relative paths, or YAML basenames."""
    path = Path(raw_path).expanduser()
    if path.is_file():
        return str(path)
    candidate = (_PROJECT_ROOT / path).resolve()
    if candidate.is_file():
        return str(candidate)
    if not path.parent.parts or str(path.parent) == ".":
        found = _search_yaml_in_cfg(raw_path)
        if found:
            console.print(f"[green]自动定位: {found}[/green]")
            return str(found)
    return None


def _detect_and_select_scale(yaml_path: str) -> Optional[str]:
    """Ask for a YOLO scale when the YAML exposes a ``scales`` mapping."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except Exception:
        return None
    scales = cfg.get("scales") if isinstance(cfg, dict) else None
    if not scales or not isinstance(scales, dict):
        return None

    scale_keys = list(scales.keys())
    console.print("\n[cyan]检测到模型 scale 选项:[/cyan]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("No.", justify="right", style="cyan", width=4)
    table.add_column("Scale", style="white", width=8)
    table.add_column("Depth", justify="right", style="magenta", width=8)
    table.add_column("Width", justify="right", style="magenta", width=8)
    table.add_column("Max Ch", justify="right", style="magenta", width=8)
    for index, key in enumerate(scale_keys):
        value = scales[key]
        table.add_row(str(index + 1), key, str(value[0]), str(value[1]), str(value[2]))
    console.print(table)

    while True:
        raw = Prompt.ask("选择 scale (序号或名称)", default=scale_keys[0]).strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(scale_keys):
                return scale_keys[idx]
        except ValueError:
            pass
        if raw in scale_keys:
            return raw
        console.print(f"[red]无效选择: '{raw}'，可选: {', '.join(scale_keys)}[/red]")


def _is_yolovo_model(model) -> bool:
    """Return True for YOLOVOModel-like objects."""
    return type(model).__name__ == "YOLOVOModel" or (
        hasattr(model, "set_input_text") and hasattr(model, "distill_predict")
    )


def _extract_layers(model, state_dict: dict) -> list[LayerInfo]:
    """Extract layer metadata from a model object with a ``model`` Sequential."""
    model_seq = getattr(model, "model", None)
    if model_seq is None:
        console.print("[red]模型缺少 model 属性，无法遍历层[/red]")
        return []
    return _extract_layers_from_sequential(model_seq, state_dict)


def _extract_layers_from_sequential(model_seq, state_dict: dict | None = None) -> list[LayerInfo]:
    """Extract layer metadata from an nn.Sequential body."""
    if state_dict is None:
        state_dict = model_seq.state_dict() if hasattr(model_seq, "state_dict") else {}
    layers = []
    for index, module in enumerate(model_seq):
        layers.append(
            LayerInfo(
                index=index,
                name=type(module).__name__,
                out_channels=_get_out_channels_from_module(module, state_dict, index),
                input_source=getattr(module, "_mm_input_source", None),
                f=getattr(module, "f", None),
            )
        )
    return layers


def _parse_from_yaml(yaml_path: Path, scale: str | None = None) -> tuple[list[LayerInfo], str]:
    """Build a YOLOVOModel from YAML and extract its layer structure."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except Exception as exc:
        console.print(f"[red]读取 YAML 失败: {exc}[/red]")
        return [], "unknown"

    if not isinstance(cfg, dict):
        console.print("[red]YAML 根节点必须是 mapping[/red]")
        return [], "unknown"

    if "text" not in cfg and "text_encoder" not in cfg:
        console.print("[yellow]该 YAML 未检测到 YOLOVO text/text_encoder 字段，仍尝试按 YOLOVO 构建[/yellow]")
    if scale and cfg.get("scales"):
        cfg = dict(cfg)
        cfg["scale"] = scale
        console.print(f"[dim]使用 scale: {scale}[/dim]")

    try:
        from ultralytics.nn.tasks import YOLOVOModel

        model = YOLOVOModel(cfg, ch=int(cfg.get("ch", cfg.get("channels", 3))), nc=cfg.get("nc", 80), verbose=False)
    except Exception as exc:
        console.print(f"[red]无法从 YOLOVO YAML 构建模型: {exc}[/red]")
        return [], "unknown"

    return _extract_layers(model, model.state_dict()), "yolovo"


def _parse_from_pt(pt_path: Path, yaml_hint: str | None = None) -> tuple[list[LayerInfo], str]:
    """Load a checkpoint and extract layer structure.

    Standard YOLO11 checkpoints are displayed through the provided YOLOVO YAML
    hint because the YOLOVO runtime remaps them into a YOLOVOModel teacher.
    """
    console.print(f"[dim]加载权重: {pt_path.name} (pickle 反序列化，请确保文件来源可信)[/dim]")
    try:
        from ultralytics.models.yolovo.model import _ensure_mobileclip_vendor_importable

        _ensure_mobileclip_vendor_importable()
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        console.print(f"[red]加载权重失败: {exc}[/red]")
        return [], "unknown"

    model = ckpt.get("model", ckpt.get("ema", None)) if isinstance(ckpt, dict) else ckpt
    if model is None:
        console.print("[red]权重文件中未找到模型对象[/red]")
        return [], "unknown"
    if hasattr(model, "float"):
        model = model.float()

    if _is_yolovo_model(model):
        return _extract_layers(model, model.state_dict() if hasattr(model, "state_dict") else {}), "yolovo"

    console.print("[yellow]检测到非 YOLOVO checkpoint；YOLOVO distill runtime 仅支持 YOLOVO 或可 remap 的 YOLO11 teacher。[/yellow]")
    if yaml_hint:
        console.print("[dim]将使用辅助 YOLOVO YAML 展示 remap 后的教师层结构[/dim]")
        return _parse_from_yaml(Path(yaml_hint))
    return _extract_layers(model, model.state_dict() if hasattr(model, "state_dict") else {}), "yolo11"


def parse_model_layers(path: str, scale: str | None = None, yaml_hint: str | None = None) -> tuple[list[LayerInfo], str]:
    """Parse model layers from a .pt checkpoint or YOLOVO .yaml."""
    model_path = Path(path)
    if not model_path.is_file():
        console.print(f"[red]文件不存在: {path}[/red]")
        return [], "unknown"
    if model_path.suffix == ".pt":
        return _parse_from_pt(model_path, yaml_hint=yaml_hint)
    if model_path.suffix in (".yaml", ".yml"):
        return _parse_from_yaml(model_path, scale=scale)
    console.print(f"[red]不支持的文件格式: {model_path.suffix}，需要 .pt 或 .yaml/.yml[/red]")
    return [], "unknown"


_SOURCE_STYLE = {
    "RGB": "green",
    "X": "blue",
    "Dual": "yellow",
}


def display_layer_table(layers: list[LayerInfo], title: str) -> None:
    """Display model layers as a rich table."""
    table = Table(title=title, show_lines=False)
    table.add_column("Index", justify="right", style="cyan", width=6)
    table.add_column("Module", style="white", min_width=20)
    table.add_column("Out Ch", justify="right", style="magenta", width=8)
    table.add_column("Input Source", justify="center", width=14)
    table.add_column("From", justify="center", width=12)

    for layer in layers:
        if layer.input_source:
            style = _SOURCE_STYLE.get(layer.input_source, "white")
            source = f"[{style}]{layer.input_source}[/{style}]"
        else:
            source = "-"
        channels = str(layer.out_channels) if layer.out_channels > 0 else "?"
        from_value = f"[dim]{layer.f}[/dim]" if isinstance(layer.f, list) else str(layer.f if layer.f is not None else "-")
        table.add_row(str(layer.index), layer.name, channels, source, from_value)
    console.print(table)
    console.print()


def _resolve_layer_indices(spec) -> list[int]:
    """Expand an int or closed range tuple into layer indices."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, tuple) and len(spec) == 2:
        return list(range(spec[0], spec[1] + 1))
    raise TypeError(f"不支持的层号类型: {type(spec).__name__}({spec!r})，预期 int 或 (start, end)")


def validate_channel_alignment(
    teacher_spec,
    student_spec,
    teacher_layers: list[LayerInfo],
    student_layers: list[LayerInfo],
) -> list[str]:
    """Validate feature mapping channel compatibility."""
    teacher_indices = _resolve_layer_indices(teacher_spec)
    student_indices = _resolve_layer_indices(student_spec)
    errors = []

    if len(teacher_indices) != len(student_indices):
        return [f"范围长度不匹配: 教师 {len(teacher_indices)} 层 vs 学生 {len(student_indices)} 层"]

    teacher_map = {layer.index: layer for layer in teacher_layers}
    student_map = {layer.index: layer for layer in student_layers}
    for teacher_idx, student_idx in zip(teacher_indices, student_indices):
        teacher_layer = teacher_map.get(teacher_idx)
        student_layer = student_map.get(student_idx)
        if teacher_layer is None:
            errors.append(f"教师层 {teacher_idx} 不存在")
            continue
        if student_layer is None:
            errors.append(f"学生层 {student_idx} 不存在")
            continue
        if teacher_layer.out_channels <= 0 or student_layer.out_channels <= 0:
            errors.append(
                f"层 T{teacher_idx}({teacher_layer.out_channels}ch) -> "
                f"S{student_idx}({student_layer.out_channels}ch): 无法确定通道数，请手动确认"
            )
        elif teacher_layer.out_channels != student_layer.out_channels:
            errors.append(
                f"通道不匹配: 教师层 {teacher_idx} ({teacher_layer.out_channels}ch) -> "
                f"学生层 {student_idx} ({student_layer.out_channels}ch)"
            )
    return errors


def _layer_spec_to_yaml(spec) -> int | list:
    """Convert a layer spec into YAML-serializable data."""
    if isinstance(spec, int):
        return spec
    if isinstance(spec, (tuple, list)):
        return [int(item) for item in spec]
    return int(spec)


def build_yaml_dict(
    teachers: list[TeacherEntry],
    mode: str,
    feature_mappings: list[FeatureMapping],
    output_teachers: list[str],
) -> dict:
    """Build the distillation YAML dictionary."""
    data = {"version": 1, "teachers": []}
    for teacher in teachers:
        entry = {"name": teacher.name, "role": "rgb", "weights": teacher.weights}
        if teacher.yaml_path:
            entry["yaml"] = teacher.yaml_path
        data["teachers"].append(entry)

    mappings = {}
    if mode in ("feature", "both") and feature_mappings:
        mappings["feature"] = [
            {
                "teacher": mapping.teacher_name,
                "teacher_layer": _layer_spec_to_yaml(mapping.teacher_layer),
                "student_layer": _layer_spec_to_yaml(mapping.student_layer),
            }
            for mapping in feature_mappings
        ]
    if mode in ("output", "both") and output_teachers:
        mappings["output"] = [{"teacher": name} for name in output_teachers]
    if mappings:
        data["mappings"] = mappings
    return data


def merge_yaml(existing: dict, new: dict) -> dict:
    """Merge existing distill YAML data with newly generated data."""
    merged = dict(existing)
    merged["version"] = new.get("version", existing.get("version", 1))

    teachers_by_name = {teacher["name"]: teacher for teacher in existing.get("teachers", [])}
    for teacher in new.get("teachers", []):
        if teacher["name"] in teachers_by_name:
            console.print(f"[yellow]教师 '{teacher['name']}' 已存在，跳过[/yellow]")
        else:
            teachers_by_name[teacher["name"]] = teacher
    merged["teachers"] = list(teachers_by_name.values())

    existing_mappings = existing.get("mappings", {})
    new_mappings = new.get("mappings", {})
    merged_mappings = {}

    feature = list(existing_mappings.get("feature", []))
    feature_keys = {
        (item.get("teacher"), str(item.get("teacher_layer")), str(item.get("student_layer")))
        for item in feature
    }
    for item in new_mappings.get("feature", []):
        key = (item.get("teacher"), str(item.get("teacher_layer")), str(item.get("student_layer")))
        if key in feature_keys:
            console.print(f"[yellow]特征映射 {key} 已存在，跳过[/yellow]")
            continue
        feature.append(item)
        feature_keys.add(key)
    if feature:
        merged_mappings["feature"] = feature

    output = list(existing_mappings.get("output", []))
    output_names = {item["teacher"] for item in output}
    for item in new_mappings.get("output", []):
        if item["teacher"] not in output_names:
            output.append(item)
            output_names.add(item["teacher"])
    if output:
        merged_mappings["output"] = output

    if merged_mappings:
        merged["mappings"] = merged_mappings
    return merged


def write_yaml(data: dict, output_path: str) -> None:
    """Write YAML and validate it against the project schema."""
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)
    console.print(f"[green]已写入: {path}[/green]")

    try:
        from ultralytics.nn.mm.distill.schema import load_distill_config

        load_distill_config(str(path))
        console.print("[green]校验通过: YAML 符合 DistillConfig schema[/green]")
    except Exception as exc:
        console.print(f"[red]校验失败: {exc}[/red]")
        console.print("[yellow]请检查生成的 YAML 是否符合 schema 要求[/yellow]")


def _parse_layer_input(raw: str) -> int | tuple:
    """Parse ``6`` or ``3-6`` into a layer spec."""
    raw = raw.strip()
    try:
        value = int(raw)
    except ValueError:
        pass
    else:
        if value < 0:
            raise ValueError(f"层号不支持负数索引: {value}，请使用 0 或正整数")
        return value

    if "-" in raw:
        parts = raw.split("-", 1)
        try:
            start, end = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            raise ValueError(f"无效的层范围格式: '{raw}'，应为 '起始-结束' 如 '3-6'")
        if start < 0 or end < 0:
            raise ValueError(f"层号不支持负数索引: [{start}, {end}]")
        if start > end:
            raise ValueError(f"范围起始 {start} 大于结束 {end}")
        return (start, end)
    raise ValueError(f"无效的层号: '{raw}'，应为整数或范围如 '3-6'")


def _validate_layer_exists(spec, layers: list[LayerInfo], label: str) -> None:
    """Validate that a layer spec exists in the provided layer list."""
    indices = _resolve_layer_indices(spec)
    max_index = max((layer.index for layer in layers), default=-1)
    for index in indices:
        if index < 0 or index > max_index:
            raise ValueError(f"{label}层 {index} 超出范围 [0, {max_index}]")


def _step_load_existing() -> Optional[dict]:
    """Optionally load an existing distillation YAML."""
    console.print(Panel("YOLOVO 蒸馏配置工具", style="bold cyan"))
    console.print()

    if not Confirm.ask("是否加载现有蒸馏 YAML 进行编辑？", default=False):
        return None

    path = Prompt.ask("现有蒸馏 YAML 路径", default="distill_yolovo.yaml")
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    if not resolved.is_file():
        console.print(f"[red]文件不存在: {resolved}[/red]")
        return None

    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        console.print(f"[green]已加载现有配置: {resolved}[/green]")
        from ultralytics.nn.mm.distill.schema import load_distill_config

        load_distill_config(str(resolved))
        console.print("[green]现有配置 schema 校验通过[/green]")
        teachers = data.get("teachers", [])
        mappings = data.get("mappings", {})
        console.print(f"  教师数量: {len(teachers)}")
        console.print(f"  特征映射: {len(mappings.get('feature', []))} 条")
        console.print(f"  输出蒸馏: {len(mappings.get('output', []))} 条")
        console.print()
        return data
    except Exception as exc:
        console.print(f"[red]加载失败: {exc}[/red]")
        return None


def _step_student_model() -> tuple[list[LayerInfo], str]:
    """Ask for the YOLOVO student model and parse its layers."""
    console.print(Panel("[bold]Step 1: 学生模型[/bold]", style="cyan"))
    console.print("[dim]支持完整路径、相对路径或 YOLOVO YAML 文件名。[/dim]")
    for attempt in range(_MAX_RETRIES):
        raw_path = Prompt.ask("学生模型路径 (.pt/.yaml，或 YAML 名称，输入 q 退出)")
        if raw_path.strip().lower() == "q":
            raise KeyboardInterrupt
        if not raw_path.strip():
            console.print(f"[yellow]请输入模型路径或 q 退出 (剩余 {_MAX_RETRIES - attempt - 1} 次)[/yellow]")
            continue
        resolved = _resolve_model_path(raw_path)
        if resolved is None:
            console.print(f"[red]未找到: {raw_path}[/red]")
            console.print(f"[yellow]请重新输入 (剩余 {_MAX_RETRIES - attempt - 1} 次)[/yellow]")
            continue
        scale = _detect_and_select_scale(resolved) if resolved.endswith((".yaml", ".yml")) else None
        layers, family = parse_model_layers(resolved, scale=scale)
        if layers:
            scale_label = f", scale={scale}" if scale else ""
            display_layer_table(layers, f"学生模型层结构 ({family}{scale_label}) - {Path(resolved).name}")
            if family != "yolovo":
                console.print("[yellow]学生模型不是 YOLOVO；YOLOVO distill 训练可能无法使用该配置。[/yellow]")
            return layers, family
        console.print(f"[yellow]请重新输入有效路径 (剩余 {_MAX_RETRIES - attempt - 1} 次)[/yellow]")
    console.print("[red]超过最大重试次数，退出[/red]")
    raise SystemExit(1)


def _step_teachers(student_family: str) -> list[TeacherEntry]:
    """Collect one or more RGB teachers."""
    console.print(Panel("[bold]Step 2: 教师模型[/bold]", style="cyan"))
    console.print(
        "[dim]请选择用于蒸馏的教师模型。当前只支持 RGB 教师；"
        "如果使用标准 yolo11*.pt，后续需要提供 YOLOVO YAML 以便转换为 YOLOVO 教师结构。[/dim]"
    )
    teachers: list[TeacherEntry] = []

    while True:
        weights = Prompt.ask("教师模型 .pt 文件路径")
        if not weights.endswith(".pt"):
            console.print("[red]请输入 .pt 格式的教师模型文件路径。[/red]")
            continue
        weights_path = Path(weights).expanduser()
        if not weights_path.is_absolute():
            weights_path = (_PROJECT_ROOT / weights_path).resolve()
        if not weights_path.is_file():
            console.print(f"[red]文件不存在: {weights_path}[/red]")
            continue

        default_name = "teacher_rgb" if not teachers else f"teacher_rgb_{len(teachers) + 1}"
        name = Prompt.ask("教师名称", default=default_name)
        if any(teacher.name == name for teacher in teachers):
            console.print(f"[red]教师名称 '{name}' 已存在，请使用其他名称[/red]")
            continue

        yaml_path = None
        if Confirm.ask("是否提供辅助 YAML？", default=True):
            yaml_raw = Prompt.ask("辅助 YAML 路径或名称", default=str(_DEFAULT_YOLOVO_YAML))
            resolved_yaml = _resolve_model_path(yaml_raw)
            if resolved_yaml:
                yaml_path = resolved_yaml
            else:
                console.print(f"[yellow]未找到: {yaml_raw}，已忽略[/yellow]")

        layers, family = parse_model_layers(str(weights_path), yaml_hint=yaml_path)
        if layers:
            display_layer_table(layers, f"教师 '{name}' (rgb, {family}) 层结构 - {weights_path.name}")
        if family not in ("yolovo", "yolo11", "unknown"):
            console.print(f"[yellow]教师家族 ({family}) 与 YOLOVO 不一致，蒸馏训练时可能报错[/yellow]")
        if student_family != "yolovo":
            console.print(f"[yellow]学生家族 ({student_family}) 不是 YOLOVO，请确认。[/yellow]")

        teachers.append(
            TeacherEntry(
                name=name,
                role="rgb",
                weights=str(weights_path),
                yaml_path=yaml_path,
                layers=layers,
            )
        )
        if not Confirm.ask("添加更多教师？", default=False):
            break

    return teachers


def _step_distill_mode() -> str:
    """Ask which distillation mode to configure."""
    console.print(Panel("[bold]Step 3: 蒸馏模式[/bold]", style="cyan"))
    mode_options = [
        ("output", "输出蒸馏 (检测头输出级别)"),
        ("feature", "特征蒸馏 (中间层特征对齐)"),
        ("both", "同时使用两种蒸馏"),
    ]
    for index, (key, desc) in enumerate(mode_options):
        console.print(f"  [cyan]{index + 1}[/cyan] [bold]{key}[/bold] - {desc}")
    console.print()

    while True:
        raw = Prompt.ask("选择蒸馏模式 (序号或名称)").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(mode_options):
                return mode_options[idx][0]
        except ValueError:
            pass
        for key, _ in mode_options:
            if raw == key:
                return key
        console.print(f"[red]无效选择: '{raw}'[/red]")


def _step_feature_mappings(
    teachers: list[TeacherEntry],
    student_layers: list[LayerInfo],
) -> list[FeatureMapping]:
    """Collect feature distillation mappings."""
    console.print(Panel("[bold]Step 4: 特征层映射[/bold]", style="cyan"))
    mappings: list[FeatureMapping] = []

    for teacher in teachers:
        console.print(f"\n[bold]配置教师 '{teacher.name}' 的层映射:[/bold]")
        if not teacher.layers:
            console.print(f"[red]教师 '{teacher.name}' 无层信息，跳过[/red]")
            continue

        while True:
            display_layer_table(teacher.layers, f"教师 '{teacher.name}' 层结构")
            while True:
                raw = Prompt.ask("教师层号 (单层如 '27' 或范围如 '27-29')")
                try:
                    teacher_spec = _parse_layer_input(raw)
                    _validate_layer_exists(teacher_spec, teacher.layers, "教师")
                    break
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")

            display_layer_table(student_layers, "学生模型层结构")
            while True:
                raw = Prompt.ask("学生层号 (单层如 '27' 或范围如 '27-29')")
                try:
                    student_spec = _parse_layer_input(raw)
                    _validate_layer_exists(student_spec, student_layers, "学生")
                    break
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")

            errors = validate_channel_alignment(teacher_spec, student_spec, teacher.layers, student_layers)
            if errors:
                console.print("[red]通道对齐校验失败:[/red]")
                for error in errors:
                    console.print(f"  [red]- {error}[/red]")
                if not Confirm.ask("仍然添加此映射？", default=False):
                    continue
            else:
                teacher_indices = _resolve_layer_indices(teacher_spec)
                student_indices = _resolve_layer_indices(student_spec)
                teacher_map = {layer.index: layer for layer in teacher.layers}
                student_map = {layer.index: layer for layer in student_layers}
                for teacher_idx, student_idx in zip(teacher_indices, student_indices):
                    teacher_layer = teacher_map.get(teacher_idx)
                    student_layer = student_map.get(student_idx)
                    console.print(
                        f"  [green]T{teacher_idx}({teacher_layer.name if teacher_layer else '?'}, "
                        f"{teacher_layer.out_channels if teacher_layer else '?'}ch) -> "
                        f"S{student_idx}({student_layer.name if student_layer else '?'}, "
                        f"{student_layer.out_channels if student_layer else '?'}ch)[/green]"
                    )

            mappings.append(
                FeatureMapping(
                    teacher_name=teacher.name,
                    teacher_layer=teacher_spec,
                    student_layer=student_spec,
                )
            )
            console.print(f"[green]已添加映射 (当前共 {len(mappings)} 条)[/green]")
            if not Confirm.ask(f"为教师 '{teacher.name}' 添加更多映射？", default=False):
                break

    return mappings


def _step_output_teachers(teachers: list[TeacherEntry]) -> list[str]:
    """Select teachers used for output distillation."""
    console.print(Panel("[bold]Step 5: 输出蒸馏教师[/bold]", style="cyan"))
    selected = []
    for teacher in teachers:
        if Confirm.ask(f"  教师 '{teacher.name}' 参与输出蒸馏？", default=True):
            selected.append(teacher.name)
    return selected


def _step_preview_and_write(data: dict, existing_data: Optional[dict]) -> None:
    """Preview generated YAML and write it to disk."""
    console.print(Panel("[bold]Step 6: 预览与输出[/bold]", style="cyan"))
    if existing_data:
        data = merge_yaml(existing_data, data)
        console.print("[dim]已与现有配置合并[/dim]")

    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    console.print(Panel(yaml_str, title="YOLOVO 蒸馏配置预览", border_style="green"))

    output_path = Prompt.ask("输出文件路径", default="distill_yolovo.yaml")
    resolved = Path(output_path).expanduser()
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    if resolved.is_file() and not Confirm.ask(f"文件 '{resolved}' 已存在，覆盖？", default=False):
        output_path = Prompt.ask("请输入新的文件路径")

    write_yaml(data, output_path)


def main() -> None:
    """Run the interactive YOLOVO distillation config workflow."""
    console.print()
    console.print("[bold cyan]YOLOVO 蒸馏配置工具[/bold cyan]")
    console.print()

    try:
        existing_data = _step_load_existing()
        student_layers, student_family = _step_student_model()
        teachers = _step_teachers(student_family)
        if not teachers:
            console.print("[red]未添加任何教师，退出[/red]")
            return

        mode = _step_distill_mode()
        feature_mappings: list[FeatureMapping] = []
        if mode in ("feature", "both"):
            feature_mappings = _step_feature_mappings(teachers, student_layers)
            if not feature_mappings:
                console.print("[yellow]未配置任何特征映射[/yellow]")
                if mode == "feature":
                    console.print("[red]feature 模式至少需要一条特征映射，退出[/red]")
                    return

        output_teachers: list[str] = []
        if mode in ("output", "both"):
            output_teachers = _step_output_teachers(teachers)
            if not output_teachers:
                console.print("[yellow]未选择任何输出蒸馏教师[/yellow]")
                if mode == "output":
                    console.print("[red]output 模式至少需要一个教师，退出[/red]")
                    return

        data = build_yaml_dict(teachers, mode, feature_mappings, output_teachers)
        _step_preview_and_write(data, existing_data)
        console.print()
        console.print("[bold green]配置完成[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[dim]已取消[/dim]")
    except Exception as exc:
        console.print(f"\n[red]发生错误: {exc}[/red]")
        raise


if __name__ == "__main__":
    main()
