"""Evaluation metrics for GLP: public relation metrics + field-specific ones.

Public:
  top1 / top5 predicate accuracy, macro-F1, relation existence P/R/F1.

Custom (field-specific, directly tied to the two GLP claims):
  FMA  field-mass agreement -- GT predicate field mass inside the GT object box
       versus outside it.  Higher is better.
  WBG  background-mass gap -- w_bg on GT predicates should be lower than on
       non-GT candidates; the gap measures abstention separation.
  AURC risk-coverage area of using w_bg as an abstention score.
  SFC  subject-frame consistency -- local kernel means must not change when the
       subject box is translated/scaled (V2's equivariance claim, lower better).
  EE   to_image_frame round-trip error (analytic sanity, lower better).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _phi(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def gaussian_box_mass(mu: torch.Tensor, sigma: torch.Tensor,
                      box: torch.Tensor) -> torch.Tensor:
    """Mass of diagonal Gaussians inside axis-aligned boxes.

    ``mu``/``sigma`` are ``[..., K, 2]``; ``box`` is ``[..., 4]`` cxcywh.
    ``box`` leading dims are right-aligned to ``mu`` (e.g. ``[K,4]`` against
    ``[K,M,AG,2]`` becomes ``[K,1,4]``).
    """
    while box.dim() < mu.dim() - 1:
        box = box.unsqueeze(-2)
    x1 = box[..., 0] - box[..., 2] * 0.5
    x2 = box[..., 0] + box[..., 2] * 0.5
    y1 = box[..., 1] - box[..., 3] * 0.5
    y2 = box[..., 1] + box[..., 3] * 0.5
    a = _phi((x2.unsqueeze(-1) - mu[..., 0]) / sigma[..., 0].clamp_min(1e-4)) - \
        _phi((x1.unsqueeze(-1) - mu[..., 0]) / sigma[..., 0].clamp_min(1e-4))
    b = _phi((y2.unsqueeze(-1) - mu[..., 1]) / sigma[..., 1].clamp_min(1e-4)) - \
        _phi((y1.unsqueeze(-1) - mu[..., 1]) / sigma[..., 1].clamp_min(1e-4))
    return (a * b).clamp(0.0, 1.0)


def field_mass_in_box(field: Dict[str, torch.Tensor], box: torch.Tensor) -> torch.Tensor:
    """Mixture mass inside ``box`` for every pair/candidate: ``[K, M]``."""
    mu, sigma = field["mu"], field["sigma"]
    w = field["logw"].exp()
    return (w * gaussian_box_mass(mu, sigma, box)).sum(-1)


def _topk_hit(logits: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """逐 GT 行：top-k 是否命中该行自己的谓词。"""
    if labels.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=logits.device)
    top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
    return (top == labels.unsqueeze(-1)).any(-1)


def _slot_hit(hit: torch.Tensor, si: torch.Tensor, n_slots: int) -> torch.Tensor:
    """GT 行级命中 -> 槽位级命中数：一个有序对命中任一 GT 谓词即算对。"""
    if hit.numel() == 0:
        return torch.zeros((), dtype=torch.long, device=hit.device)
    acc = torch.zeros(n_slots, dtype=torch.long, device=hit.device)
    acc.scatter_add_(0, si, hit.long())
    return (acc > 0).sum()


def _macro_f1(pred: torch.Tensor, labels: torch.Tensor, n_cls: int) -> float:
    """bincount 版 macro-F1：原实现按类循环 10k 次、每次一次 GPU 同步，单轮要几十秒。"""
    if labels.numel() == 0:
        return 0.0
    tp = torch.bincount(labels[pred == labels], minlength=n_cls).float()
    # 分母只按 GT 类：2809 类里被 argmax 偶然输出一次的 predicted-only 类会被
    # 恒记为 F1=0，把宏平均压成常数噪声（原实现是 pred+label 的并集）。
    lab_count = torch.bincount(labels, minlength=n_cls).float()
    denom = torch.bincount(pred, minlength=n_cls).float() + lab_count
    keep = lab_count > 0
    if not keep.any():
        return 0.0
    return float((2 * tp[keep] / denom[keep]).mean())


def aurc(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if confidence.numel() == 0:
        return 0.0
    order = confidence.argsort(descending=True)
    correct = correct[order].float()
    n = correct.numel()
    risks = 1.0 - correct.cumsum(0) / torch.arange(1, n + 1, device=correct.device)
    # trapezoidal integration over coverage
    cov = torch.arange(1, n + 1, device=correct.device) / n
    return float((risks * (cov - torch.cat([torch.zeros(1, device=cov.device), cov[:-1]]))).sum())


class FieldMetricAccumulator:
    """Accumulate per-image relation/field metrics for logging."""

    def __init__(self, n_cls: int):
        self.n_cls = n_cls
        self.n_pred = 0
        # 累计量保持张量形态，只在 compute()（epoch 末）取一次值：
        # 逐图 .item()/.tolist() 每个 batch 要掐断 16 次 GPU 流水。
        self.top1 = self.top5 = self.tp = self.fp = self.fn = self.cand_hit = 0
        self.fma: List[torch.Tensor] = []
        self.wbg_gap: List[torch.Tensor] = []
        self.aurc_conf: List[torch.Tensor] = []
        self.aurc_ok: List[torch.Tensor] = []
        self._pred_all: List[torch.Tensor] = []
        self._lab_all: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, out: Dict, gt_list: Sequence[Sequence[Tuple[int, int, int]]] = ()):
        # 训练时 GT 被强制塞进候选，散列后的 top1 恒≈1（构造出来的假象）；
        # 指标必须用未散列的检索分数，才和推理口径（raw top-M）对齐。
        logits = out.get("logits_raw", out["logits"]).float()
        valid = out["valid_mask"].bool()
        # GT 用 (slot, pred) 稀疏行表示：同一有序对可以有多个谓词，last-wins 的单标签
        # 会把第 2..k 个谓词记成「预测错误」。
        gt = out.get("slot_gt")
        if gt is None:
            pair_gt: Dict[Tuple[int, int], List[int]] = {}
            for (s, o, p) in (gt_list[0] if gt_list else []):
                pair_gt.setdefault((int(s), int(o)), []).append(int(p))
            sk, pk = [], []
            for k, (s, o) in enumerate(zip(out["sub_idx"].tolist(), out["obj_idx"].tolist())):
                if valid[k]:
                    for p in pair_gt.get((s, o), ()):
                        sk.append(k)
                        pk.append(p)
            gt = (torch.tensor(sk, dtype=torch.long, device=logits.device),
                  torch.tensor(pk, dtype=torch.long, device=logits.device))
        si, lb = gt
        K = logits.shape[0]
        slot_has = torch.zeros(K, dtype=torch.bool, device=logits.device)
        if si.numel():
            slot_has.scatter_(0, si, True)
            self.n_pred = self.n_pred + slot_has.sum()
            # 候选预算截断会让部分 GT 有序对根本不进候选（>64 对的图），它们既不在
            # 分子也不在分母，会把 top-k 指标系统性抬高。用 gt_list 对账补进分母。
            if gt_list:
                self.n_pred = self.n_pred + max(
                    0, len({(int(s), int(o)) for (s, o, _p) in gt_list[0]}) - int(slot_has.sum()))
            lg = logits[si]
            # 检索口径的 GT 是否落进 field 候选：cand_idx 是「强制塞入 GT」之后的候选表，
            # 拿它比对恒为 1，必须用未散列的检索分数重新取 top-M（推理时就是这么取的）。
            if "cand_idx" in out:
                k = int(out["cand_idx"].shape[-1])
                self.cand_hit = self.cand_hit + _slot_hit(
                    (lg.topk(k, dim=-1).indices == lb.unsqueeze(-1)).any(-1), si, K)
            self.top1 = self.top1 + _slot_hit(_topk_hit(lg, lb, 1), si, K)
            self.top5 = self.top5 + _slot_hit(_topk_hit(lg, lb, 5), si, K)
            self._pred_all.append(lg.argmax(-1))
            self._lab_all.append(lb)

        # existence P/R/F1 over all valid pairs (threshold 0.5)；无 GT 的图也要计 fp
        exist = torch.sigmoid(out["rel_logits"].float()[valid]) >= 0.5
        pos = slot_has[valid]
        self.tp = self.tp + (exist & pos).sum()
        self.fp = self.fp + (exist & ~pos).sum()
        self.fn = self.fn + (~exist & pos).sum()

        if si.numel() and "field" in out and "cand_idx" in out:
            field = out["field"]
            cand = out["cand_idx"]                       # [K,M]
            obj_box = out["boxes"][0][out["obj_idx"]]    # [K,4]
            mass = field_mass_in_box(field, obj_box)     # [K,M]
            gy = cand[si] == lb.unsqueeze(-1)            # [G,M]
            self.fma.append(((2 * mass[si] - 1).clamp(-1, 1))[gy])
            if "w_bg" in field:
                wbg = field["w_bg"][si]                   # [G,M]
                pos_w, neg_w = wbg[gy], wbg[~gy]
                if pos_w.numel() and neg_w.numel():
                    # positive = GT predicates carry less background mass
                    self.wbg_gap.append(neg_w.mean() - pos_w.mean())
                self.aurc_conf.append(1.0 - wbg.max(-1).values)
                self.aurc_ok.append(lg.argmax(-1) == lb)

    def compute(self) -> Dict[str, float]:
        out = {}
        if self.n_pred:
            n_pred = float(self.n_pred)
            out["rel_top1"] = float(self.top1) / n_pred
            out["rel_top5"] = float(self.top5) / n_pred
            pred = torch.cat(self._pred_all) if self._pred_all else torch.zeros(0)
            lab = torch.cat(self._lab_all) if self._lab_all else torch.zeros(0)
            out["rel_macro_f1"] = _macro_f1(pred, lab, self.n_cls)
            out["rel_gt_in_cand"] = float(self.cand_hit) / n_pred
        if float(self.tp + self.fp + self.fn):
            tp, fp, fn = float(self.tp), float(self.fp), float(self.fn)
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            out["rel_exist_precision"] = prec
            out["rel_exist_recall"] = rec
            out["rel_exist_f1"] = 2 * prec * rec / max(prec + rec, 1e-9)
        if self.fma:
            fma = torch.cat(self.fma)
            out["field_fma"] = float(fma.mean()) if fma.numel() else 0.0
        if self.wbg_gap:
            out["field_wbg_gap"] = float(torch.stack(self.wbg_gap).mean())
        if self.aurc_conf:
            out["field_aurc"] = aurc(torch.cat(self.aurc_conf), torch.cat(self.aurc_ok))
        return out


@torch.no_grad()
def subject_frame_consistency(field_read, queries: torch.Tensor, scene: torch.Tensor,
                              anchors: torch.Tensor, geo: torch.Tensor,
                              e_cand: torch.Tensor, shift=(0.1, -0.05),
                              scale: float = 1.2) -> Dict[str, float]:
    """SFC/EE: local field kernels must not move when the subject box moves.

    Only V2 (``local_frame``) can pass: its generator input excludes absolute
    cues, so ``mu_local`` is invariant.  V1 is the negative control.
    """
    _, f1 = field_read(queries, scene, anchors, geo, e_cand, return_field=True)
    a2 = anchors.clone()
    # subject box: translate + scale about its centre
    a2[..., 0, 0] = anchors[..., 0, 0] + shift[0]
    a2[..., 0, 1] = anchors[..., 0, 1] + shift[1]
    a2[..., 0, 2] = anchors[..., 0, 2] * scale
    a2[..., 0, 3] = anchors[..., 0, 3] * scale
    _, f2 = field_read(queries, scene, a2, geo, e_cand, return_field=True)
    if "mu_local" not in f1:
        return {"field_sfc_mu": float("nan"), "field_sfc_sigma": float("nan"),
                "field_ee": float("nan")}
    mu_err = (f1["mu_local"] - f2["mu_local"]).norm() / f1["mu_local"].norm().clamp_min(1e-9)
    sig_err = (f1["sigma_local"] - f2["sigma_local"]).norm() / \
        f1["sigma_local"].norm().clamp_min(1e-9)
    # analytic round-trip: image-frame kernels mapped back must equal local ones
    c = f1["subject_box"][..., None, None, :2]
    s = f1["subject_box"][..., None, None, 2:].clamp_min(1e-4)
    mu_back = (f1["mu"] - c.expand_as(f1["mu"])) / s.expand_as(f1["mu"])
    ee = (mu_back - f1["mu_local"]).norm() / f1["mu_local"].norm().clamp_min(1e-9)
    return {"field_sfc_mu": float(mu_err), "field_sfc_sigma": float(sig_err),
            "field_ee": float(ee)}
