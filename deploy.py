#!/usr/bin/env python3
"""
deploy.py - YOLOVO deployment pipeline.

固定输出目录:
    /root/mutilmanger/YOLOVO

目录约定:
    YOLOVO/
      back/          # 历史备份目录
      <active>/      # 唯一活跃版本目录

用法:
    /root/miniconda3/envs/yolon/bin/python deploy.py
    /root/miniconda3/envs/yolon/bin/python deploy.py --dry-run
    /root/miniconda3/envs/yolon/bin/python deploy.py -y
    /root/miniconda3/envs/yolon/bin/python deploy.py --version 26.5.29
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent
DST_ROOT = Path("/root/mutilmanger/YOLOVO")
VERSION_FILE = SRC_ROOT / "ultralytics" / "cfg" / "default.yaml"
SCRIPT_REGISTRY_FILE = SRC_ROOT / "deploy_scripts.json"
SCRIPT_REGISTRY_VERSION = 1

SYNC_LIST = [
    "ultralytics/",
    "train.py",
    "val.py",
    "predict.py",
    "train_obb.py",
    "val_obb.py",
    "predict_obb.py",
    "train_pose.py",
    "val_pose.py",
    "predict_pose.py",
    "pruneVO.py",
    "finetrainVO.py",
    "tools/",
    "scripts/",
    "deploy.py",
    "deploy_scripts.json",
    ".gitignore",
    "pyproject.toml",
    "yolo11n.pt",
    "新手指南.txt",
    "本项目福利.txt",
    "distill_yolovo.yaml",
]

RSYNC_EXCLUDES = ["__pycache__/", "*.pyc", ".git/", "*.egg-info/", ".pytest_cache/"]
SCRIPT_DISCOVERY_IGNORES = {"__init__.py", "deploy.py"}

_USE_COLOR = sys.stdout.isatty()


def _c(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def log_step(n: int, total: int, msg: str) -> None:
    print(_c(36, f"\n{'=' * 56}"))
    print(_c(36, f"  [{n}/{total}] {msg}"))
    print(_c(36, f"{'=' * 56}"))


def log_success(msg: str) -> None:
    print(_c(32, f"  [OK] {msg}"))


def log_warn(msg: str) -> None:
    print(_c(33, f"  [WARN] {msg}"))


def log_error(msg: str) -> None:
    print(_c(31, f"  [FAIL] {msg}"))


def _dir_size(path: Path) -> str:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(Path(dirpath) / filename)
            except OSError:
                pass
    if total < 1024 * 1024:
        return f"{total / 1024:.0f}K"
    return f"{total / (1024 * 1024):.0f}M"


def generate_version_tag() -> str:
    now = datetime.now()
    return f"{now.year % 100}.{now.month}.{now.day}"


def get_current_mm_version() -> str:
    if not VERSION_FILE.is_file():
        return "unknown"
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'^mm_project_version:\s*["\']?([^"\'\s]+)', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def bump_version(new_version: str, dry_run: bool) -> str:
    if not VERSION_FILE.is_file():
        log_error(f"版本定义文件不存在: {VERSION_FILE}")
        sys.exit(1)

    text = VERSION_FILE.read_text(encoding="utf-8")
    old_version = get_current_mm_version()
    if old_version == new_version:
        log_warn(f"版本号未变化 ({new_version})，跳过更新")
        return old_version

    pattern = r'^(mm_project_version:\s*["\']?)[^"\'\s]+(["\']?\s*.*)$'
    new_text, count = re.subn(pattern, rf"\g<1>{new_version}\2", text, flags=re.MULTILINE)
    if count == 0:
        log_error(f"未在 {VERSION_FILE} 中找到 mm_project_version 字段")
        sys.exit(1)

    if dry_run:
        log_success(f"[模拟] 版本号: {old_version} -> {new_version}")
    else:
        VERSION_FILE.write_text(new_text, encoding="utf-8")
        log_success(f"版本号: {old_version} -> {new_version}")
    return old_version


def _dedupe_sorted(items: list[str]) -> list[str]:
    return sorted({str(item).strip() for item in items if str(item).strip()})


def _normalize_root_script(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("脚本路径不能为空")

    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(SRC_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"脚本必须位于项目根目录下: {raw}") from exc

    if len(path.parts) != 1:
        raise ValueError(f"只支持项目根目录下的 .py 脚本: {raw}")
    if path.suffix != ".py":
        raise ValueError(f"只支持 .py 脚本: {raw}")
    if path.name in SCRIPT_DISCOVERY_IGNORES or path.name.startswith("_"):
        raise ValueError(f"该脚本属于部署工具/内部调试文件，不参与自动发现: {raw}")
    return path.name


def load_script_registry() -> dict[str, list[str] | int]:
    default: dict[str, list[str] | int] = {
        "version": SCRIPT_REGISTRY_VERSION,
        "include": [],
        "exclude": [],
    }
    if not SCRIPT_REGISTRY_FILE.exists():
        return default

    try:
        data = json.loads(SCRIPT_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log_error(f"读取脚本打包清单失败: {SCRIPT_REGISTRY_FILE} ({exc})")
        sys.exit(1)

    if not isinstance(data, dict):
        log_error(f"脚本打包清单格式错误: {SCRIPT_REGISTRY_FILE} 不是 JSON object")
        sys.exit(1)

    include = data.get("include", [])
    exclude = data.get("exclude", [])
    if not isinstance(include, list) or not isinstance(exclude, list):
        log_error("脚本打包清单格式错误: include/exclude 必须是列表")
        sys.exit(1)

    try:
        normalized_include = _dedupe_sorted([_normalize_root_script(str(v)) for v in include])
        normalized_exclude = _dedupe_sorted([_normalize_root_script(str(v)) for v in exclude])
    except ValueError as exc:
        log_error(f"脚本打包清单包含非法条目: {exc}")
        sys.exit(1)

    overlap = sorted(set(normalized_include) & set(normalized_exclude))
    if overlap:
        log_error(f"脚本打包清单冲突，同一文件同时 include/exclude: {overlap}")
        sys.exit(1)

    return {
        "version": SCRIPT_REGISTRY_VERSION,
        "include": normalized_include,
        "exclude": normalized_exclude,
    }


def save_script_registry(registry: dict[str, list[str] | int]) -> None:
    payload = {
        "version": SCRIPT_REGISTRY_VERSION,
        "include": _dedupe_sorted(list(registry.get("include", []))),
        "exclude": _dedupe_sorted(list(registry.get("exclude", []))),
    }
    SCRIPT_REGISTRY_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def discover_unlisted_root_scripts() -> list[str]:
    built_in_scripts = {item for item in SYNC_LIST if item.endswith(".py")}
    scripts: list[str] = []
    for path in SRC_ROOT.glob("*.py"):
        name = path.name
        if name in built_in_scripts or name in SCRIPT_DISCOVERY_IGNORES or name.startswith("_"):
            continue
        scripts.append(name)
    return sorted(scripts)


def apply_script_registry_overrides(
    registry: dict[str, list[str] | int],
    *,
    include_scripts: list[str],
    exclude_scripts: list[str],
) -> bool:
    include = set(registry.get("include", []))
    exclude = set(registry.get("exclude", []))
    changed = False

    for raw in include_scripts:
        name = _normalize_root_script(raw)
        if name not in include or name in exclude:
            include.add(name)
            exclude.discard(name)
            changed = True

    for raw in exclude_scripts:
        name = _normalize_root_script(raw)
        if name not in exclude or name in include:
            exclude.add(name)
            include.discard(name)
            changed = True

    registry["include"] = _dedupe_sorted(list(include))
    registry["exclude"] = _dedupe_sorted(list(exclude))
    return changed


def resolve_extra_script_sync_items(args: argparse.Namespace) -> list[str]:
    registry = load_script_registry()
    try:
        changed = apply_script_registry_overrides(
            registry,
            include_scripts=args.include_script,
            exclude_scripts=args.exclude_script,
        )
    except ValueError as exc:
        log_error(str(exc))
        sys.exit(1)

    if changed and not args.dry_run:
        save_script_registry(registry)
        log_success(f"已更新脚本打包清单: {SCRIPT_REGISTRY_FILE}")

    discovered = discover_unlisted_root_scripts()
    include = set(registry.get("include", []))
    exclude = set(registry.get("exclude", []))
    unclassified = sorted(set(discovered) - include - exclude)

    if unclassified:
        log_warn("发现尚未分类的项目根目录 Python 脚本:")
        for name in unclassified:
            print(f"    - {name}")

        if args.dry_run:
            log_warn("模拟模式不写入清单；正式部署前需要选择 include 或 exclude")
        elif args.yes or not sys.stdin.isatty():
            log_error(
                "非交互/--yes 模式不能询问新脚本是否打包。请先运行交互式 deploy.py，"
                "或使用 --include-script/--exclude-script 记录选择后再部署。"
            )
            sys.exit(2)
        else:
            for name in unclassified:
                answer = input(f"  是否以后自动打包 {name}？[y/N] ").strip().lower()
                if answer in {"y", "yes"}:
                    include.add(name)
                    exclude.discard(name)
                else:
                    exclude.add(name)
                    include.discard(name)
            registry["include"] = _dedupe_sorted(list(include))
            registry["exclude"] = _dedupe_sorted(list(exclude))
            save_script_registry(registry)
            log_success(f"已记录脚本打包选择: {SCRIPT_REGISTRY_FILE}")

    include = set(registry.get("include", []))
    missing = sorted(name for name in include if not (SRC_ROOT / name).is_file())
    if missing:
        log_warn(f"脚本打包清单中包含不存在的文件，将跳过: {missing}")

    return sorted(name for name in include if (SRC_ROOT / name).is_file())


def detect_current_version() -> str:
    if not DST_ROOT.is_dir():
        log_error(f"部署目标目录不存在: {DST_ROOT}")
        log_error("请先创建 YOLOVO 交付根目录，并保留 back/ 与唯一活跃版本目录。")
        sys.exit(1)

    entries = [
        entry.name
        for entry in DST_ROOT.iterdir()
        if entry.is_dir() and entry.name != "back" and not entry.name.startswith(".")
    ]

    if len(entries) == 0:
        log_error(f"{DST_ROOT} 下没有找到活跃版本目录")
        sys.exit(1)
    if len(entries) > 1:
        log_error(f"{DST_ROOT} 下存在多个活跃版本目录: {entries}")
        log_error("无法确定当前活跃版本，请手动清理后重试")
        sys.exit(1)

    return entries[0]


def get_date_tag() -> str:
    now = datetime.now()
    return f"{now.year % 100}_{now.month}_{now.day}"


def do_backup(current: str, dry_run: bool) -> Path:
    src = DST_ROOT / current
    dst = DST_ROOT / "back" / current

    if dst.exists():
        log_error(f"备份目标已存在: {dst}")
        log_error("请手动删除旧备份后重试")
        sys.exit(1)

    if dry_run:
        log_success(f"[模拟] 备份 {src} -> {dst}")
        return dst

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        log_success(f"备份完成: {dst} ({_dir_size(dst)})")
    except OSError as exc:
        log_error(f"备份失败: {exc}")
        sys.exit(1)

    return dst


def do_rename(current: str, new_name: str, dry_run: bool) -> Path:
    src = DST_ROOT / current
    dst = DST_ROOT / new_name

    if current == new_name:
        log_warn(f"新日期与当前版本同名 ({new_name})，跳过重命名")
        return dst

    if dst.exists():
        log_error(f"重命名目标已存在: {dst}")
        log_error("请手动处理冲突后重试")
        sys.exit(1)

    if dry_run:
        log_success(f"[模拟] 重命名 {src} -> {dst}")
        return dst

    try:
        src.rename(dst)
        log_success(f"重命名完成: {dst}")
    except OSError as exc:
        log_error(f"重命名失败: {exc}")
        sys.exit(1)

    return dst


def do_sync(target_dir: Path, dry_run: bool, extra_items: list[str] | None = None) -> tuple[int, int, list[str]]:
    rsync_exclude_args = [arg for exc in RSYNC_EXCLUDES for arg in ("--exclude", exc)]
    dry_run_arg = ["--dry-run"] if dry_run else []
    sync_items = [*SYNC_LIST, *_dedupe_sorted(extra_items or [])]

    ok = 0
    fail = 0
    failed_items: list[str] = []

    for item in sync_items:
        src_path = SRC_ROOT / item.rstrip("/")
        is_dir = item.endswith("/")

        if not src_path.exists():
            log_warn(f"跳过（源不存在）: {item}")
            fail += 1
            failed_items.append(f"{item} (源文件不存在)")
            continue

        if is_dir:
            dst_path = target_dir / item
            cmd = [
                "rsync",
                "-av",
                *dry_run_arg,
                *rsync_exclude_args,
                str(src_path) + "/",
                str(dst_path),
            ]
        else:
            cmd = [
                "rsync",
                "-av",
                *dry_run_arg,
                *rsync_exclude_args,
                str(src_path),
                str(target_dir) + "/",
            ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
            if result.returncode == 0:
                log_success(f"同步: {item}")
                ok += 1
            else:
                log_error(f"同步失败: {item}")
                if result.stderr:
                    print(f"    {result.stderr.strip()}")
                fail += 1
                failed_items.append(item)
        except subprocess.TimeoutExpired:
            log_error(f"同步超时: {item}")
            fail += 1
            failed_items.append(f"{item} (超时)")
        except Exception as exc:
            log_error(f"同步异常: {item} ({exc})")
            fail += 1
            failed_items.append(f"{item} ({exc})")

    return ok, fail, failed_items


def print_report(
    backup_path: Path,
    active_path: Path,
    ok: int,
    fail: int,
    failed_items: list[str],
    dry_run: bool,
    old_version: str = "",
    new_version: str = "",
) -> None:
    mode = " [模拟]" if dry_run else ""
    print()
    print(_c(36, "=" * 56))
    print(_c(36, f"  YOLOVO 部署完成{mode}"))
    print(_c(36, "=" * 56))

    size_str = "（模拟）" if dry_run else (_dir_size(backup_path) if backup_path.exists() else "N/A")
    rel_backup = backup_path.relative_to(DST_ROOT)
    rel_active = active_path.relative_to(DST_ROOT)
    print(f"  输出根目录: {DST_ROOT}")
    print(f"  版本: {old_version} -> {new_version}")
    print(f"  备份: YOLOVO/{rel_backup}/ ({size_str})")
    print(f"  活跃: YOLOVO/{rel_active}/")
    print(f"  同步: {ok} 成功, {fail} 失败")

    for item in failed_items:
        print(_c(31, f"    [FAIL] {item}"))

    print(_c(36, "=" * 56))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOVO 部署流水线")
    parser.add_argument("--dry-run", action="store_true", help="仅模拟，不执行实际操作")
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认提示")
    parser.add_argument("--version", type=str, default=None, help="手动指定版本号（默认自动生成 年.月.日 格式）")
    parser.add_argument(
        "--include-script",
        action="append",
        default=[],
        metavar="SCRIPT.py",
        help="将项目根目录下的新 .py 脚本记录为以后自动打包，可重复传入",
    )
    parser.add_argument(
        "--exclude-script",
        action="append",
        default=[],
        metavar="SCRIPT.py",
        help="将项目根目录下的新 .py 脚本记录为以后不自动打包，可重复传入",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_steps = 5

    log_step(0, total_steps, "检查脚本打包清单")
    extra_scripts = resolve_extra_script_sync_items(args)
    if extra_scripts:
        log_success(f"额外脚本自动打包: {', '.join(extra_scripts)}")

    log_step(1, total_steps, "检测 YOLOVO 目标目录")
    current = detect_current_version()
    date_tag = get_date_tag()
    log_success(f"输出根目录: {DST_ROOT}")
    log_success(f"当前活跃版本: {current}")
    log_success(f"今日日期标签: {date_tag}")

    log_step(2, total_steps, "更新项目版本号")
    new_version = args.version or generate_version_tag()
    old_version = bump_version(new_version, args.dry_run)

    same_day = current == date_tag
    if same_day:
        log_warn("同一天重复部署，跳过重命名步骤")

    action = "模拟部署" if args.dry_run else "部署"
    rename_hint = "" if same_day else f"\n  重命名: {current} -> {date_tag}"
    print(f"\n  即将执行: {action}")
    print(f"  输出根目录: {DST_ROOT}")
    print(f"  版本号: {old_version} -> {new_version}")
    print(f"  备份: YOLOVO/{current}/ -> YOLOVO/back/{current}/")
    if rename_hint:
        print(f"  {rename_hint.strip()}")
    print(f"  同步: {len(SYNC_LIST) + len(extra_scripts)} 项")

    if not args.yes and not args.dry_run:
        try:
            answer = input("\n  确认执行？[y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("  已取消。")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n  已取消。")
            return 0

    log_step(3, total_steps, "备份当前版本")
    backup_path = do_backup(current, args.dry_run)

    new_name = date_tag if not same_day else current
    active_path = do_rename(current, new_name, args.dry_run)

    log_step(4, total_steps, "增量同步")
    ok, fail, failed_items = do_sync(active_path, args.dry_run, extra_items=extra_scripts)

    print_report(
        backup_path,
        active_path,
        ok,
        fail,
        failed_items,
        args.dry_run,
        old_version=old_version,
        new_version=new_version,
    )

    if fail > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
