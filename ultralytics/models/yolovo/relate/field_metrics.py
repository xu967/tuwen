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


def _topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    if labels.numel() == 0:
        return 0
    top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
    return int((top == labels.unsqueeze(-1)).any(-1).sum())


def _macro_f1(pred: torch.Tensor, labels: torch.Tensor, n_cls: int) -> float:
    if labels.numel() == 0:
        return 0.0
    f1s = []
    for c in range(n_cls):
        tp = int(((pred == c) & (labels == c)).sum())
        fp = int(((pred == c) & (labels != c)).sum())
        fn = int(((pred != c) & (labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


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
        self.top1 = self.top5 = self.n_pred = 0
        self.fma: List[float] = []
        self.wbg_gap: List[float] = []
        self.aurc_conf: List[torch.Tensor] = []
        self.aurc_ok: List[torch.Tensor] = []
        self.tp = self.fp = self.fn = 0
        self._pred_all: List[torch.Tensor] = []
        self._lab_all: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, out: Dict, gt_list: Sequence[Sequence[Tuple[int, int, int]]]):
        gt = gt_list[0] if gt_list else []
        pair_gt = {}
        for (s, o, p) in gt:
            pair_gt[(int(s), int(o))] = int(p)
        if not pair_gt:
            return
        sub = out["sub_idx"].tolist()
        obj = out["obj_idx"].tolist()
        valid = out["valid_mask"].bool()
        logits = out["logits"].float()
        labels = torch.full((logits.shape[0],), -1, dtype=torch.long, device=logits.device)
        for k, (s, o) in enumerate(zip(sub, obj)):
            if valid[k] and (s, o) in pair_gt:
                labels[k] = pair_gt[(s, o)]

        has = labels >= 0
        if has.any():
            lg, lb = logits[has], labels[has]
            self.n_pred += int(lb.numel())
            self.top1 += _topk_correct(lg, lb, 1)
            self.top5 += _topk_correct(lg, lb, 5)
            pred = lg.argmax(-1)
            self._pred_all.append(pred.cpu())
            self._lab_all.append(lb.cpu())

            # existence P/R/F1 over all valid pairs (threshold 0.5)
            v = valid
            exist = torch.sigmoid(out["rel_logits"].float()[v]) >= 0.5
            pos = has[v]
            self.tp += int((exist & pos).sum())
            self.fp += int((exist & ~pos).sum())
            self.fn += int((~exist & pos).sum())

            if "field" in out and "cand_idx" in out:
                field = out["field"]
                cand = out["cand_idx"]                       # [K,M]
                boxes = out["boxes"][0]
                obj_idx = out["obj_idx"]
                obj_box = boxes[obj_idx]                     # [K,4]
                mass = field_mass_in_box(field, obj_box)     # [K,M]
                gtm = cand[has] == lb.unsqueeze(-1)           # [n,M]
                if gtm.any():
                    mass_gt = mass[has][gtm]
                    self.fma.extend((2 * mass_gt - 1).clamp(-1, 1).tolist())
                if "w_bg" in field:
                    wbg = field["w_bg"][has]                  # [n,M]
                    pos = wbg[gtm]
                    neg = wbg[~gtm]
                    if pos.numel() and neg.numel():
                        # positive = GT predicates carry less background mass
                        self.wbg_gap.append(float(neg.mean() - pos.mean()))
                    conf = 1.0 - wbg.max(-1).values
                    ok = lg.argmax(-1) == lb
                    self.aurc_conf.append(conf.cpu())
                    self.aurc_ok.append(ok.cpu())

    def compute(self) -> Dict[str, float]:
        out = {}
        if self.n_pred:
            out["rel_top1"] = self.top1 / self.n_pred
            out["rel_top5"] = self.top5 / self.n_pred
            pred = torch.cat(self._pred_all) if self._pred_all else torch.zeros(0)
            lab = torch.cat(self._lab_all) if self._lab_all else torch.zeros(0)
            out["rel_macro_f1"] = _macro_f1(pred, lab, self.n_cls)
            prec = self.tp / max(self.tp + self.fp, 1)
            rec = self.tp / max(self.tp + self.fn, 1)
            out["rel_exist_precision"] = prec
            out["rel_exist_recall"] = rec
            out["rel_exist_f1"] = 2 * prec * rec / max(prec + rec, 1e-9)
        if self.fma:
            out["field_fma"] = float(sum(self.fma) / len(self.fma))
        if self.wbg_gap:
            out["field_wbg_gap"] = float(sum(self.wbg_gap) / len(self.wbg_gap))
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
