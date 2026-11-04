# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""YOLOVO relationship secondary-task dataset.

Consumes the SAME images/labels as the primary task, with single-file mixed
annotations: standard YOLO box rows plus ``R <sub> <obj> <rel_id> [score]
[spatial]`` relation rows.  Instance addressing is the 0-based box-row index
within the label file.  Illegal indices / rel ids raise at scan time (never
silently dropped), so the standard ``verify_image_label`` (which flags R rows
as corrupt) is NOT used here -- see docs/plans/2026-10-05-yolovo-relate-design.md §3.1.

Augmentation policy (§6 training-pollution checklist): only HSV + LetterBox.
Every geometric/joining augmentation that can reorder or drop box rows
(mosaic / mixup / cutmix / copy-paste / fliplr / flipud / affine) is disabled
because it would silently invalidate the row-index addressing of R lines.
GT-box jitter (NMS-box distribution bridging, §11) is applied at collate time
via :func:`make_jitter_collate` -- it perturbs box geometry only and keeps
topology, instance identity, pair indices and labels intact.
"""

from __future__ import annotations

import hashlib
import os
from copy import copy
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from ultralytics.data.augment import Compose, Format, LetterBox, RandomHSV
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import (
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from ultralytics.utils import LOGGER, TQDM

from itertools import repeat

# Separate cache file so the primary task's labels.cache is never overwritten
# (and never mis-read: the relate cache carries extra keys).
RELATE_CACHE_NAME = "labels.relate.cache"

# data.yaml `rcoords` conventions for box rows.
RCOORDS_MODES = ("xywhn", "xyxy")


def parse_rnames(rnames: Any, source: str = "data.yaml rnames") -> Dict[int, str]:
    """Validate and normalize a predicate vocabulary ``rnames`` mapping.

    Same shape convention as ``names``: ``{id: predicate}``.  Missing or
    duplicate ids raise SyntaxError per design §3.1.
    """
    if rnames is None:
        raise SyntaxError(
            f"关系数据集缺少谓词词表：{source} 必须提供 `rnames`（与 names 同构，如 {{0: riding, 1: on}}）。"
        )
    if isinstance(rnames, dict):
        items = []
        for k, v in rnames.items():
            try:
                items.append((int(k), str(v)))
            except (TypeError, ValueError) as e:
                raise SyntaxError(f"{source} 中 rnames 含非法条目 {k!r}: {v!r}（id 必须可转 int）。") from e
    else:  # list/tuple -> positional ids
        items = [(i, str(v)) for i, v in enumerate(rnames)]
    ids = [i for i, _ in items]
    if not items:
        raise SyntaxError(f"{source} 中 rnames 为空词表。")
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise SyntaxError(f"{source} 中 rnames id 重复：{dupes}。")
    if sorted(ids) != list(range(len(ids))):
        # 静默重编号会让谓词整体错位（1-based 的 rnames 尤其危险）
        raise SyntaxError(
            f"{source} 中 rnames id 必须是 0..{len(ids) - 1} 连续，得到 {sorted(ids)[:5]}...")
    names = [v for _, v in sorted(items)]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SyntaxError(f"{source} 中 rnames 谓词名重复：{dupes}。")
    return {i: n for i, n in enumerate(names)}


def parse_relate_rversion(data: Optional[dict]) -> int:
    """Read the annotation-format version switch ``rversion`` (default 1)."""
    version = (data or {}).get("rversion", 1)
    try:
        version = int(version)
    except (TypeError, ValueError) as e:
        raise SyntaxError(f"data.yaml rversion 必须是 int，得到 {version!r}。") from e
    if version != 1:
        raise SyntaxError(f"不支持的 rversion={version}；当前解析器只支持 rversion: 1。")
    return version


def parse_relate_rcoords(data: Optional[dict]) -> str:
    """Read the box-row coordinate convention ``rcoords`` (default ``xywhn``)."""
    coords = str((data or {}).get("rcoords") or "xywhn").lower()
    if coords not in RCOORDS_MODES:
        raise SyntaxError(f"不支持的 rcoords={coords!r}；可选：{list(RCOORDS_MODES)}。")
    return coords


def parse_relate_rswap(data: Optional[dict]) -> bool:
    """Read the swap-pair switch ``rswap`` (default False; hinge auto-generates)."""
    return bool((data or {}).get("rswap", False))


def parse_relate_lines(
    lb_file: str,
    prefix: str,
    im_file: str,
    shape: Tuple[int, int],
    num_cls: int,
    num_rel: int,
    rcoords: str = "xywhn",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse one mixed label file into (boxes, relations, rel_weights, spatial_flags).

    Box rows keep the standard YOLO 5-column semantics (``cls x y w h``);
    ``R`` rows follow ``R sub obj rel_id [score] [spatial]`` with 0-based box
    row addressing.  Out-of-range indices / rel ids / malformed rows raise
    ValueError carrying file name + offending line content.
    """
    h, w = shape
    with open(lb_file, encoding="utf-8") as f:
        rows = [x.split() for x in f.read().strip().splitlines() if len(x)]

    box_rows, rel_rows = [], []
    for lineno, parts in enumerate(rows):
        if parts[0] == "R":
            rel_rows.append((lineno, parts))
        else:
            box_rows.append((lineno, parts))

    n = len(box_rows)
    lb = np.zeros((n, 5), dtype=np.float32)
    for k, (lineno, parts) in enumerate(box_rows):
        if len(parts) < 5:
            raise ValueError(
                f"{prefix}{im_file}: {os.path.basename(lb_file)} 第 {lineno + 1} 行框行需 ≥5 列 "
                f"(cls x y w h)，实际 {len(parts)} 列: '{' '.join(parts)}'"
            )
        try:
            cls, x, y, bw, bh = (float(v) for v in parts[:5])
        except ValueError as e:
            raise ValueError(
                f"{prefix}{im_file}: {os.path.basename(lb_file)} 第 {lineno + 1} 行含非数值列: "
                f"'{' '.join(parts)}'"
            ) from e
        if not (0 <= cls < num_cls):
            raise ValueError(
                f"{prefix}{im_file}: {os.path.basename(lb_file)} 第 {lineno + 1} 行类别 id {int(cls)} "
                f"超出 names 范围 0..{num_cls - 1}: '{' '.join(parts)}'"
            )
        lb[k] = (cls, x, y, bw, bh)

    if rcoords == "xyxy":  # pixel corners -> normalized xywh (class column untouched)
        boxes = lb[:, 1:]
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / w
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        lb[:, 1:] = np.stack([cx, cy, bw, bh], axis=1)

    coords = lb[:, 1:]
    if n and (coords.min() < -0.01 or coords.max() > 1.01):
        bad = np.argwhere((coords < -0.01) | (coords > 1.01))
        r, c = bad[0]
        raise ValueError(
            f"{prefix}{im_file}: {os.path.basename(lb_file)} 第 {box_rows[int(r)][0] + 1} 行坐标越界 "
            f"(列 {c + 1}={coords[int(r), int(c)]:.4f}，rcoords={rcoords}): "
            f"'{' '.join(box_rows[int(r)][1])}'"
        )

    nr = len(rel_rows)
    relations = np.zeros((nr, 3), dtype=np.float32)
    rel_weights = np.ones((nr,), dtype=np.float32)
    spatial_flags = np.zeros((nr,), dtype=np.float32)
    for k, (lineno, parts) in enumerate(rel_rows):
        where = f"{os.path.basename(lb_file)} 第 {lineno + 1} 行"
        raw = " ".join(parts)
        if len(parts) < 4:
            raise ValueError(
                f"{prefix}{im_file}: {where} R 行至少 4 列 (R sub obj rel_id)，实际 {len(parts)} 列: '{raw}'"
            )
        try:
            sub, obj, rel_id = (int(float(v)) for v in parts[1:4])
        except ValueError as e:
            raise ValueError(f"{prefix}{im_file}: {where} R 行索引列非整数: '{raw}'") from e
        if not (0 <= sub < n):
            raise ValueError(
                f"{prefix}{im_file}: {where} 主语框行号 {sub} 越界（该文件共 {n} 个框行，合法 0..{n - 1}）: '{raw}'"
            )
        if not (0 <= obj < n):
            raise ValueError(
                f"{prefix}{im_file}: {where} 宾语框行号 {obj} 越界（该文件共 {n} 个框行，合法 0..{n - 1}）: '{raw}'"
            )
        if not (0 <= rel_id < num_rel):
            raise ValueError(
                f"{prefix}{im_file}: {where} rel_id={rel_id} 不在 rnames 词表内（合法 0..{num_rel - 1}）: '{raw}'"
            )
        if len(parts) >= 5:
            try:
                rel_weights[k] = float(parts[4])
            except ValueError as e:
                raise ValueError(f"{prefix}{im_file}: {where} R 行分数列非数值: '{raw}'") from e
        if len(parts) >= 6:
            try:
                spatial_flags[k] = float(parts[5])
            except ValueError as e:
                raise ValueError(f"{prefix}{im_file}: {where} R 行空间列非数值: '{raw}'") from e
        relations[k] = (sub, obj, rel_id)
    return lb, relations, rel_weights, spatial_flags


def scan_relate_label(args: Tuple) -> Tuple:
    """Verify one image/label pair for the relate task (R-line aware).

    Same return contract style as ``verify_image_label`` but with relation
    arrays appended.  Structural annotation errors raise (fail-fast per
    design); only image-level problems degrade to corrupt-message rows.
    """
    im_file, lb_file, prefix, num_cls, num_rel, rcoords = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""
    empty = (np.zeros((0, 5), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
             np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = (im.size[1], im.size[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        if os.path.isfile(lb_file):
            nf = 1
            lb, relations, rel_weights, spatial_flags = parse_relate_lines(
                lb_file, prefix, im_file, shape, num_cls, num_rel, rcoords
            )
        else:
            nm = 1
            lb, relations, rel_weights, spatial_flags = empty
        return im_file, lb, shape, relations, rel_weights, spatial_flags, nm, nf, ne, nc, msg
    except (ValueError, SyntaxError):
        # 结构性错误（框索引越界、谓词 id 非法）必须中止；当成 corrupt 图丢掉的
        # 话，坏标注只会退化成"训练集悄悄少了一批图"。
        raise
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return (None,) + empty + (None, nm, nf, ne, nc, msg)


class YOLOVORelateDataset(YOLODataset):
    """YOLO dataset with R-line relation parsing and relation-safe augmentation.

    Per-sample label dicts carry the standard keys plus ``relations`` [R,3]
    (sub idx, obj idx, rel id), ``rel_weights`` [R] and ``spatial_flags`` [R].
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", rnames: Any = None, **kwargs):
        self.rnames = parse_rnames(rnames or (data or {}).get("rnames"))
        parse_relate_rversion(data)
        self.rcoords = parse_relate_rcoords(data)
        self.rswap = parse_relate_rswap(data)
        super().__init__(*args, data=data, task=task, **kwargs)

    def _cache_hash(self) -> str:
        """标签缓存指纹：路径+字节数之外再混入谓词词表与 rcoords。

        只看路径和文件大小时，改了 data.yaml 的 rnames / rcoords（关系行内容不变）
        会静默命中旧 cache，整份数据集按旧谓词 id 或旧坐标约定训练。
        注意 get_hash 只累加总字节数：同字数的标签改动仍会漏检，改标注后删缓存最稳。
        """
        fp = hashlib.sha256(
            (repr(sorted(self.rnames.items())) + "|" + str(self.rcoords)).encode()).hexdigest()
        return get_hash(self.label_files + self.im_files) + fp[:16]

    # ------------------------------------------------------------------
    # Label scanning (R-line aware, own cache file)
    # ------------------------------------------------------------------

    def cache_labels(self, path: Path = None) -> Dict:
        """Scan labels splitting R rows; structural errors abort training."""
        path = Path(path) if path else Path(self.label_files[0]).parent / RELATE_CACHE_NAME
        x = {"labels": []}
        nm, nf, ne, nc = 0, 0, 0, 0
        msgs = []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        num_cls = len(self.data["names"])
        num_rel = len(self.rnames)
        with ThreadPool(min(8, os.cpu_count() or 1)) as pool:
            results = pool.imap(
                func=scan_relate_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(num_cls),
                    repeat(num_rel),
                    repeat(self.rcoords),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, relations, rel_weights, spatial_flags, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm, nf, ne, nc = nm + nm_f, nf + nf_f, ne + ne_f, nc + nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": [],
                            "keypoints": None,
                            "normalized": True,
                            "bbox_format": "xywh",
                            "relations": relations,
                            "rel_weights": rel_weights,
                            "spatial_flags": spatial_flags,
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}.")
        x["hash"] = self._cache_hash()
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """Load labels from the relate-specific cache or rescan."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent / RELATE_CACHE_NAME
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            # 必须用 _cache_hash()：它比 get_hash() 多混了词表指纹，拿裸
            # get_hash() 比对永远不等，缓存从未被复用（每次构建都全量重扫）。
            assert cache["hash"] == self._cache_hash()
        except (FileNotFoundError, AssertionError, AttributeError, KeyError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists:
            from ultralytics.utils import LOCAL_RANK

            if LOCAL_RANK in {-1, 0}:
                d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                TQDM(None, desc=self.prefix + d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))

        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid images found in {cache_path}.")
        self.im_files = [lb["im_file"] for lb in labels]
        return labels

    # ------------------------------------------------------------------
    # Relation-safe transforms / label plumbing
    # ------------------------------------------------------------------

    def update_labels(self, include_class: Optional[List[int]]) -> None:
        """Refuse class filtering: dropping box rows breaks R-row addressing."""
        if include_class is not None:
            raise ValueError(
                "YOLOVORelateDataset 不支持 classes 过滤——删框行会静默破坏 R 行的行号寻址；"
                "请在生成数据集时预先过滤。"
            )
        if self.single_cls:
            for i in range(len(self.labels)):
                self.labels[i]["cls"][:, 0] = 0

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """HSV + LetterBox only (relation-index-safe augmentation policy)."""
        if self.augment:
            hyp = copy(hyp)  # never mutate the shared args namespace
            for key in (
                "mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud",
                "degrees", "translate", "scale", "shear", "perspective", "bgr",
            ):
                setattr(hyp, key, 0.0)
            transforms = Compose(
                [
                    LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                ]
            )
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp: Dict) -> None:
        """Mosaic already disabled; just keep the transform pipeline stable."""
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: Dict) -> Dict:
        """Keep relation arrays attached to the per-sample label dict."""
        relations = label.get("relations")
        rel_weights = label.get("rel_weights")
        spatial_flags = label.get("spatial_flags")
        label = super().update_labels_info(label)
        label["relations"] = relations
        label["rel_weights"] = rel_weights
        label["spatial_flags"] = spatial_flags
        return label

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Standard YOLO collate; relation keys stay as lists of per-image tensors."""
        return relate_collate(batch, jitter=0.0)


# ----------------------------------------------------------------------
# Collate / box jitter (§11)
# ----------------------------------------------------------------------

def _as_relation_tensors(sample: Dict) -> Dict:
    out = dict(sample)
    for key, width in (("relations", 3),):
        v = out.get(key)
        if isinstance(v, np.ndarray):
            out[key] = torch.from_numpy(v)
        elif isinstance(v, (list, tuple)):
            out[key] = torch.tensor(v, dtype=torch.float32).reshape(-1, width) if len(v) else torch.zeros((0, width))
    for key in ("rel_weights", "spatial_flags"):
        v = out.get(key)
        if isinstance(v, np.ndarray):
            out[key] = torch.from_numpy(v)
        elif isinstance(v, (list, tuple)):
            out[key] = torch.tensor(v, dtype=torch.float32) if len(v) else torch.zeros((0,))
    return out


def jitter_boxes_xywhn(boxes: torch.Tensor, mag: float = 0.10, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Scale-relative GT-box jitter in normalized xywh space.

    Center shift ∝ box size (U(-mag, mag) × w/h); width/height multiplied by
    independent log-uniform factors ``exp(U(-mag, mag))``.  Output is clamped
    into the image; degenerate boxes get a floor of 1e-4.  Topology, instance
    order and relation indices are untouched (train/eval pair distribution
    consistency is preserved by construction).
    """
    if boxes.numel() == 0 or mag <= 0:
        return boxes
    xywh = boxes.float()
    g = generator
    cx = xywh[:, 0] + (torch.rand_like(xywh[:, 0]) * 2 - 1) * mag * xywh[:, 2]
    cy = xywh[:, 1] + (torch.rand_like(xywh[:, 1]) * 2 - 1) * mag * xywh[:, 3]
    fw = torch.exp((torch.rand_like(xywh[:, 2]) * 2 - 1) * mag)
    fh = torch.exp((torch.rand_like(xywh[:, 3]) * 2 - 1) * mag)
    w = (xywh[:, 2] * fw).clamp(min=1e-4)
    h = (xywh[:, 3] * fh).clamp(min=1e-4)
    # keep the box inside the unit image
    cx = cx.clamp(w / 2, 1 - w / 2)
    cy = cy.clamp(h / 2, 1 - h / 2)
    return torch.stack([cx, cy, w, h], dim=1).to(boxes.dtype)


def relate_collate(batch: List[Dict], jitter: float = 0.0, generator: Optional[torch.Generator] = None) -> Dict:
    """Collate relation samples: standard YOLO collate + optional GT-box jitter.

    Relation tensors are re-chunked per image and exposed as lists aligned
    with ``batch_idx`` offsets so ``batch["relations"][i]`` addresses the i-th
    image with local (already offset-free) box indices.
    """
    batch = [_as_relation_tensors(b) for b in batch]
    rel_keys = ("relations", "rel_weights", "spatial_flags")
    saved = {k: [b.pop(k) for b in batch] for k in rel_keys}
    new_batch = YOLODataset.collate_fn(batch)
    if jitter and jitter > 0:
        new_batch["bboxes"] = jitter_boxes_xywhn(new_batch["bboxes"], mag=jitter, generator=generator)
    for k in rel_keys:
        new_batch[k] = saved[k]
    return new_batch


def make_jitter_collate(mag: float = 0.10):
    """Return a DataLoader collate applying GT-box jitter of magnitude ``mag``.

    ``mag=0`` disables jitter (validation loaders); two preset magnitudes
    0.10 / 0.15 are supported for the calibration sweep (§11).
    """

    def _collate(batch: List[Dict]) -> Dict:
        return relate_collate(batch, jitter=float(mag))

    return _collate


def build_relate_dataset(
    cfg,
    img_path: str,
    batch: Optional[int],
    data: Dict[str, Any],
    mode: str = "train",
    stride: int = 32,
    rnames: Any = None,
) -> YOLOVORelateDataset:
    """Build a YOLOVORelateDataset from trainer/validator-style args."""
    return YOLOVORelateDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        # rect 必须与训练保持一致：训练是方形 640 letterbox，验证原本被强制成矩形，
        # 实测 1024×768 下图归一化坐标差 0.0586，59% 的 GT 框中心偏移 >0.01、
        # 箱高比均值 1.259——是系统性的 train→val 协变量偏移，不是噪声。
        rect=bool(getattr(cfg, "rect", False)),
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task=cfg.task if getattr(cfg, "task", None) in ("segment", "obb", "pose") else "detect",
        classes=None,  # class filtering refused: breaks row addressing
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        rnames=rnames,
    )
