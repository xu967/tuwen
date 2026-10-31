"""Shared helpers for BARV: config, letterbox, label grids, reproducibility."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str((PROJECT_ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_uid_split(uid: str, splits: dict) -> str:
    """Stable image-level split from a hashed uid (no leakage across images)."""
    h = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 1000
    edges = [
        ("train_core", splits["train_core"]),
        ("train_teacher", splits["train_teacher"]),
        ("dev", splits["dev"]),
        ("calib", splits["calib"]),
        ("test", splits["test"]),
    ]
    acc = 0.0
    for name, frac in edges:
        acc += float(frac) * 1000
        if h < acc:
            return name
    return "test"


def letterbox_params(h: int, w: int, size: int) -> dict:
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    nw, nh = max(1, min(size, nw)), max(1, min(size, nh))
    return {
        "scale": scale,
        "nw": nw,
        "nh": nh,
        "px": (size - nw) // 2,
        "py": (size - nh) // 2,
        "size": size,
    }


def letterbox_image(rgb: np.ndarray, p: dict, fill: int = 114) -> np.ndarray:
    out = np.full((p["size"], p["size"], 3), fill, dtype=np.uint8)
    out[p["py"] : p["py"] + p["nh"], p["px"] : p["px"] + p["nw"]] = cv2.resize(
        rgb, (p["nw"], p["nh"]), interpolation=cv2.INTER_LINEAR
    )
    return out


def letterbox_label(label: np.ndarray, p: dict, fill: int = -1) -> np.ndarray:
    out = np.full((p["size"], p["size"]), fill, dtype=np.int32)
    out[p["py"] : p["py"] + p["nh"], p["px"] : p["px"] + p["nw"]] = cv2.resize(
        label.astype(np.int32), (p["nw"], p["nh"]), interpolation=cv2.INTER_NEAREST
    )
    return out


def grid_label(label_lb: np.ndarray, step: int) -> np.ndarray:
    """Downsample a letterboxed label map to a feature grid (nearest)."""
    return label_lb[::step, ::step]


def box_fill_mask(box_orig, p: dict, grid: int, step: int) -> np.ndarray:
    """Filled rectangle mask on a feature grid; box_orig is xyxy in original pixels."""
    x1, y1, x2, y2 = box_orig
    x1 = int(np.floor(x1 * p["scale"] + p["px"])) // step
    y1 = int(np.floor(y1 * p["scale"] + p["py"])) // step
    x2 = int(np.ceil(x2 * p["scale"] + p["px"])) // step
    y2 = int(np.ceil(y2 * p["scale"] + p["py"])) // step
    x1, x2 = max(0, min(grid - 1, x1)), max(0, min(grid - 1, x2))
    y1, y2 = max(0, min(grid - 1, y1)), max(0, min(grid - 1, y2))
    if x2 <= x1:
        x2 = min(grid, x1 + 1)
    if y2 <= y1:
        y2 = min(grid, y1 + 1)
    m = np.zeros((grid, grid), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def to_tensor_rgb(img_rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1).float() / 255.0
    return x


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
