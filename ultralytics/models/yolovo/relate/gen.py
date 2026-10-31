"""RelGen — YOLOVO 关系数据集离线生成器（prior 后端，无 LLM）.

对标 DEMGen 范式: ``RelGen(...).run(data.yaml) -> stats``。

管线（纯确定性，CPU 可跑）:
  1. 读标准 YOLO 检测数据集（images/ + labels/ 框标注, data.yaml）
  2. 框预处理: 过滤过小框（min_box）, 每图框数上限（max_boxes）
  3. 语义层: 类别对先验表（prior_vg.json, VG 统计）按频率采样谓词
  4. 空间层: 纯几何谓词（RA verify_spatial_v2 规则的先验版）,
     方向由框相对位置确定（left/right/above/below）,
     contact 需 IoU>0 或边界距 < eps, containment 需 ⊂0.5
  5. 门控: 每图关系数预算（rel_budget, 默认按框数 n*(n-1)*ratio 截断）
  6. 输出: {save_dir}/labels/<split>/<stem>.txt（框行 + R 行混合格式）
     + images 软链 + data.yaml（rnames/rversion/_gen_report.json）

用法:
    gen = RelGen(tau=0.56, save_dir=None)
    stats = gen.run("path/to/data.yaml")
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ultralytics.utils import LOGGER

# 空间谓词: 几何规则直接可判定的闭集（RA spatial frame 子集 + 结构谓词）
GEO_PREDS = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
    "beside": "beside",
    "inside": "inside",
    "on": "on",
}

# 先验黑名单：VG 噪声/低信息谓词（"part of" 多为部位归属标注，
# 会被同类同框对灾难性放大；结构语义由几何层 inside/on 覆盖）
_PRIOR_BLACKLIST = {"part of"}

_HERE = Path(__file__).parent
_DEFAULT_PRIOR = _HERE / "prior_vg.json"


class RelGen:
    """关系数据集生成器。

    Args:
        tau: 词表频率截断阈值（累计质量占比低于 tau 的谓词丢弃）。
        prior: 先验表 JSON 路径；None 用内置 prior_vg.json。
        min_box: 框面积（归一化 wh 乘积）低于此值的框不参与关系。
        max_boxes: 每图参与关系的最大框数（与训练侧 max_objects=60 对齐）。
        rel_budget: 每图关系预算系数（乘以 n*(n-1)）。
        geo_always: True 时几何关系始终生成（先验仅做语义增补）。
        seed: 随机种子（谓词采样可复现）。
        save_dir: 输出目录；None 时为 {path}_related。
        rversion: 词表版本；None 时自动（已存在报告则 +1）。
        device: 兼容 DEMGen 签名，prior 后端忽略（纯 CPU）。
    """

    def __init__(
        self,
        tau: float = 0.56,
        prior: Optional[str] = None,
        min_box: float = 0.0004,
        max_boxes: int = 60,
        rel_budget: float = 0.3,
        geo_always: bool = True,
        seed: int = 0,
        save_dir: Optional[str] = None,
        rversion: Optional[int] = None,
        device: str = "cpu",
    ):
        self.tau = tau
        self.prior_path = Path(prior) if prior else _DEFAULT_PRIOR
        self.min_box = min_box
        self.max_boxes = max_boxes
        self.rel_budget = rel_budget
        self.geo_always = geo_always
        self.rng = random.Random(seed)
        self.save_dir = save_dir
        self.rversion = rversion
        self.device = device  # prior 后端无模型，占位
        self.prior: Dict = {}
        self._cls_alias: Dict[str, str] = {}

    # ---- 先验表 --------------------------------------------------------

    def load_prior(self) -> None:
        if not self.prior_path.exists():
            raise FileNotFoundError(
                f"先验表不存在: {self.prior_path}。先运行 tools/relgen/build_prior_vg.py 生成。"
            )
        data = json.loads(self.prior_path.read_text())
        self.prior = data.get("table", {})
        stats = data.get("stats", {})
        LOGGER.info(f"[RelGen] prior loaded: entries={stats.get('entries')}, "
                    f"sub_classes={stats.get('sub_classes')}")

    def lookup(self, sub_cls_name: str, obj_cls_name: str) -> Dict[str, float]:
        """类别对 -> 谓词频率分布。

        保守命中规则（M3FD 实测教训）:
        - 不做对称回退（(s,o) 语义常不对称: riding vs ridden）
        - 条目内谓词数 >= 2 且总支持度 >= 15 才视为可信（单谓词条目
          多为 VG 噪声, 如 car x car: {'part of': 11}, 会被同框对放大）
        """
        s, o = self._norm_cls(sub_cls_name), self._norm_cls(obj_cls_name)
        entry = self.prior.get(s, {}).get(o, {})
        if len(entry) >= 2 and sum(entry.values()) >= 15:
            return dict(entry)
        return {}

    def _norm_cls(self, name: str) -> str:
        n = (name or "").strip().lower()
        n = " ".join(w for w in n.replace("/", " ").split() if w not in ("a", "an", "the"))
        if not self._cls_alias:
            # 与 build_prior_vg.CLS_ALIAS 同源
            for k in ("men", "man", "woman", "women", "people", "guy", "girl", "boy", "lady"):
                self._cls_alias[k] = "person"
            self._cls_alias.update({"vehicle": "car", "streetlight": "lamp",
                                    "traffic light": "lamp", "light": "lamp"})
        return self._cls_alias.get(n, n)

    # ---- 几何层 --------------------------------------------------------

    @staticmethod
    def geo_predicate(a: np.ndarray, b: np.ndarray) -> Optional[str]:
        """框 a(主) b(宾) -> 空间谓词（RA verify 规则先验版）。

        a/b: [cx, cy, w, h] 归一化。
        """
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        iou_w = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
        iou_h = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
        inter = iou_w * iou_h
        union = aw * ah + bw * bh - inter
        iou = inter / union if union > 0 else 0.0
        area_ratio = (aw * ah) / max(bw * bh, 1e-9)
        # containment: a ⊂ b 面积占比 > 0.5
        if iou > 0 and area_ratio > 0.5 and inter / max(aw * ah, 1e-9) > 0.8:
            return "inside"
        # contact: 边界接触或重叠
        if iou > 0.05 and abs(ay - by) < (ah + bh) / 4:
            return "on"
        dx, dy = ax - bx, ay - by
        if abs(dx) > (aw + bw) / 2 and abs(dx) > abs(dy):
            return "to the right of" if dx > 0 else "to the left of"
        if abs(dy) > (ah + bh) / 2:
            return "below" if dy > 0 else "above"
        return "beside"

    # ---- 主流程 --------------------------------------------------------

    def run(self, data_yaml: str, split: Optional[str] = None) -> "RelGenStats":
        from ultralytics.data.utils import check_det_dataset

        self.load_prior()
        src = check_det_dataset(data_yaml)
        src_path = Path(src.get("path", ".")).resolve()
        out_root = Path(self.save_dir).resolve() if self.save_dir else src_path.parent / f"{src_path.name}_related"
        splits = [split] if split else [s for s in ("train", "val", "test")
                                        if src.get(s) and Path(src_path, src[s]).exists()]
        names: Dict[int, str] = {int(k): str(v) for k, v in dict(src.get("names", {})).items()}

        # rversion 自动递增
        report_path = out_root / "_gen_report.json"
        rver = self.rversion
        if rver is None:
            rver = 1
            if report_path.exists():
                try:
                    rver = int(json.loads(report_path.read_text()).get("rversion", 0)) + 1
                except (ValueError, json.JSONDecodeError):
                    rver = 1

        counter: Counter = Counter()
        per_split = {}
        # 第一遍：逐 split 生成关系（挂内存），词表确定后再落盘
        staged = {}  # split -> [(stem, boxes, rels)]
        for sp in splits:
            imgs = sorted(Path(src_path, src[sp]).glob("*.*"))
            staged[sp] = []
            n_rels = n_boxes = n_no_box = n_no_gen = 0
            for im in imgs:
                lb_src = src_path / "labels" / sp / f"{im.stem}.txt"
                boxes = self._read_boxes(lb_src)
                if len(boxes) < 2:
                    staged[sp].append((im.stem, boxes, []))
                    n_no_box += 1
                    continue
                rels = self._gen_rels(boxes, names, counter)
                n_rels += len(rels)
                n_boxes += len(boxes)
                if not rels:
                    n_no_gen += 1
                staged[sp].append((im.stem, boxes, rels))
            per_split[sp] = {"imgs": len(imgs), "boxes": n_boxes, "rels": n_rels,
                             "no_rel_few_boxes": n_no_box, "no_rel_gen_empty": n_no_gen}

        # tau 截断词表 -> rel_id 映射；词表外谓词丢弃
        rnames = self._vocab(counter)
        pid = {p: i for i, p in rnames.items()}
        dropped = {p: c for p, c in counter.items() if p not in pid}
        if dropped:
            LOGGER.info(f"[RelGen] tau 截断丢弃谓词: {sorted(dropped)}")
        for sp in splits:
            out_lb = out_root / "labels" / sp
            out_lb.mkdir(parents=True, exist_ok=True)
            n_tau_drop = 0
            for stem, boxes, rels in staged[sp]:
                kept = [(s, o, pid[p]) for s, o, p in rels if p in pid]
                if rels and not kept:
                    n_tau_drop += 1
                self._write(out_lb / f"{stem}.txt", boxes, kept)
            if n_tau_drop:
                LOGGER.warning(f"[RelGen] {sp}: {n_tau_drop} 图关系全被 tau 截断丢弃")
            per_split[sp]["no_rel_tau_dropped"] = n_tau_drop
            # images 软链
            img_dir = Path(src_path, src[sp])
            for im in sorted(img_dir.glob("*.*")):
                dst_im = out_root / "images" / sp / im.name
                dst_im.parent.mkdir(parents=True, exist_ok=True)
                if not dst_im.exists():
                    os.symlink(im.resolve(), dst_im)

        report = {"rnames": rnames, "counts": dict(counter.most_common()),
                  "stats": per_split, "tau": self.tau, "rversion": rver,
                  "dropped": dropped,
                  "config": {"min_box": self.min_box, "max_boxes": self.max_boxes,
                             "rel_budget": self.rel_budget, "geo_always": self.geo_always}}
        report_path.write_text(json.dumps(report, indent=1))
        self._write_data_yaml(out_root, src, splits, rnames, rver)
        LOGGER.info(f"[RelGen] done -> {out_root} (rversion={rver}, V={len(rnames)})")
        return RelGenStats(per_split, rnames, report_path)

    # ---- 内部 ----------------------------------------------------------

    @staticmethod
    def _read_boxes(lb_file: Path) -> np.ndarray:
        if not lb_file.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = [l.split() for l in lb_file.read_text().strip().splitlines() if l.strip()]
        arr = np.array([[float(v) for v in r[:5]] for r in rows if len(r) >= 5],
                       dtype=np.float32).reshape(-1, 5)
        return arr

    def _gen_rels(self, boxes: np.ndarray, names: Dict[int, str], counter: Counter) -> List[tuple]:
        mask = boxes[:, 3] * boxes[:, 4] >= self.min_box
        idx = np.flatnonzero(mask)[: self.max_boxes]
        n = len(idx)
        if n < 2:
            return []
        budget = max(2, int(n * (n - 1) * self.rel_budget))
        cands = []
        for ii in range(n):
            for jj in range(n):
                if ii == jj:
                    continue
                si, oi = int(idx[ii]), int(idx[jj])
                sname = names.get(int(boxes[si, 0]), "")
                oname = names.get(int(boxes[oi, 0]), "")
                prior = {p: c for p, c in self.lookup(sname, oname).items()
                         if p not in _PRIOR_BLACKLIST}
                if prior:
                    # 频率加权采样（而非 top-1）：保留谓词多样性
                    ps, ws = zip(*prior.items())
                    pred = self.rng.choices(ps, weights=ws, k=1)[0]
                    kind = "semantic"
                elif self.geo_always:
                    pred = self.geo_predicate(boxes[si, 1:5], boxes[oi, 1:5])
                    kind = "spatial"
                    if pred is None:
                        continue
                else:
                    continue
                cands.append((si, oi, pred, kind))
        # 语义优先填充预算
        sem = [c for c in cands if c[3] == "semantic"]
        geo = [c for c in cands if c[3] == "spatial"]
        self.rng.shuffle(geo)
        chosen = sem[:budget] + geo[: max(0, budget - len(sem))][:budget]
        for _, _, p, _ in chosen:
            counter[p] += 1
        # rel_id 在词表确定后重写（_write 时由外部 vocab 映射）
        return [(s, o, p) for s, o, p, _ in chosen]

    @staticmethod
    def _write(lb_out: Path, boxes: np.ndarray, rels) -> None:
        """rels: [(sub, obj, rel_id)]，rel_id 已是词表内 id。"""
        lines = [" ".join(f"{v:.6f}" for v in row) for row in boxes]
        lines += [f"R {s} {o} {r}" for s, o, r in rels]
        lb_out.parent.mkdir(parents=True, exist_ok=True)
        lb_out.write_text("\n".join(lines) + ("\n" if lines else ""))

    def _vocab(self, counter: Counter) -> Dict[int, str]:
        items = counter.most_common()
        total = sum(c for _, c in items) or 1
        acc = 0.0
        kept = []
        for p, c in items:
            acc += c / total
            kept.append(p)
            if acc >= self.tau:
                break
        return {i: p for i, p in enumerate(kept)}

    def _write_data_yaml(self, out_root: Path, src, splits, rnames, rver) -> None:
        lines = [f"# auto-generated by RelGen (prior backend, tau={self.tau}, rversion {rver})",
                 f"path: {out_root}"]
        for sp in splits:
            lines.append(f"{sp}: images/{sp}")
        lines.append("names:")
        for k, v in dict(src.get("names", {})).items():
            lines.append(f"  {k}: {v}")
        lines.append("rnames:")
        for k, v in rnames.items():
            lines.append(f"  {k}: '{v}'")
        lines.append(f"rversion: {rver}")
        (out_root / "data.yaml").write_text("\n".join(lines) + "\n")


class RelGenStats:
    """success = 落盘后仍含 ≥1 条关系的图（扣除少框/生成空/tau 截断三类）。"""

    def __init__(self, per_split, rnames, report_path):
        self.per_split = per_split
        self.rnames = rnames
        self.report_path = report_path
        self.total = sum(s["imgs"] for s in per_split.values())
        self.success = sum(
            s["imgs"] - s.get("no_rel_few_boxes", 0) - s.get("no_rel_gen_empty", 0)
            - s.get("no_rel_tau_dropped", 0)
            for s in per_split.values()
        )
        self.failed = 0  # prior 后端无 IO 失败；口径问题见 per_split 分项
        self.failures = []

    def __repr__(self):
        return f"RelGenStats(success={self.success}/{self.total}, V={len(self.rnames)})"
