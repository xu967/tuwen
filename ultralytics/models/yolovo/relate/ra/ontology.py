"""Predicate ontology built from vocabulary text cosines (Q3A).

RA's PredicateOntology provides the three [V, V] tables its InfoNCE and swap
hinge consume: pos_mask (synonym sharing), inverse_mask (antonym pairs),
neg_lw (per-column negative down-weighting from semantic closeness). RA
builds them from a corpus; we build the same structure from MobileCLIP
cosines of the vocabulary itself:

- sym / pos_mask: cos(w_i, w_j) >= sym_thr pairs (and the diagonal)
- inverse_mask: pairs from a small hand antonym list intersected with vocab
- neg_lw[g, c]: down-weight of column c when contrasting against target g —
  close columns (likely synonyms) are discounted, distant ones stay 1.0
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

# hand antonym list intersected with the vocabulary (empty for M3FD's 15)
ANTONYM_PAIRS = (
    ("above", "below"), ("left of", "right of"), ("left", "right"),
    ("in front of", "behind"), ("over", "under"), ("on", "under"),
    ("inside", "outside"), ("up", "down"),
)

SYM_COS_THRESHOLD = 0.90      # cos >= this -> synonym pair
CLOSE_COS_THRESHOLD = 0.80    # closeness discount starts here
MIN_NEG_LW = 0.25             # floor of the down-weight

# 显式对称谓词表（规范化后的名字）。不能用文本余弦推对称：MobileCLIP 上
# cos≥0.90 的近邻中位 240 个，「有同义词」≠「方向对称」——实测 96.97% 的谓词
# 会被误判为对称，swap 方向损失几乎全跳过。按需增删。
SYMMETRIC_PREDICATES = frozenset({
    "near", "next to", "beside", "by", "around", "touching", "facing", "hugging",
    "with", "close to", "adjacent to", "parallel to", "across from", "against", "along",
})


def build_ontology(W: torch.Tensor, names=None, sym_thr: float = SYM_COS_THRESHOLD,
                   close_thr: float = CLOSE_COS_THRESHOLD,
                   min_lw: float = MIN_NEG_LW) -> Dict[str, torch.Tensor]:
    """Build pos/inverse/neg_lw tables from a L2-normalised vocab matrix.

    Args:
        W: [V, D] normalised predicate embeddings (MobileCLIP template-encoded)
        names: optional vocab strings for the antonym intersection
    Returns:
        dict with pos_mask [V,V] bool, inverse_mask [V,V] bool, neg_lw [V,V],
        sym [V] bool (symmetric predicate = swap-exempt)
    """
    V = W.shape[0]
    Wn = torch.nn.functional.normalize(W.float(), dim=-1)
    cos = Wn @ Wn.T                                   # [V, V]

    pos_mask = cos >= sym_thr
    pos_mask.fill_diagonal_(True)

    inverse_mask = torch.zeros(V, V, dtype=torch.bool)
    if names is not None:
        idx = {n: i for i, n in enumerate(names)}
        for a, b in ANTONYM_PAIRS:
            if a in idx and b in idx:
                inverse_mask[idx[a], idx[b]] = True
                inverse_mask[idx[b], idx[a]] = True

    # neg_lw[g, c]: log-space closeness discount, floor at min_lw; diagonal 0
    closeness = ((cos - close_thr) / (1.0 - close_thr)).clamp(0.0, 1.0)
    neg_lw = 1.0 - closeness * (1.0 - min_lw)
    # 近义列保留 closeness 折扣（文件头承诺的「likely synonyms are discounted」）；
    # 只有反义列豁免。原实现把 pos_mask 也设回 1.0，等于最难的近义负例从不降权。
    neg_lw = neg_lw.masked_fill(inverse_mask, 1.0)
    neg_lw.fill_diagonal_(0.0)

    # 对称谓词只认显式表，不用余弦推断（见 SYMMETRIC_PREDICATES 的注释）。
    if names is not None:
        sym = torch.tensor([n in SYMMETRIC_PREDICATES for n in names], dtype=torch.bool)
    else:
        sym = torch.zeros(V, dtype=torch.bool)
    return {"pos_mask": pos_mask, "inverse_mask": inverse_mask,
            "neg_lw": neg_lw, "sym": sym}
