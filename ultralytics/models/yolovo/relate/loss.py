"""YOLOVORelateLoss: v3 四项关系损失（分开平均，再按权重相加）。

L = λ_geo·L_geo + λ_e·L_exist + λ_p·L_pred + λ_s·L_swap  （1/1/1/0.5）

v3 归约契约（batch 内）：
- rel_geo   所有图全部有效有序对放在一起平均（正样本=标注对，分母=有效对数）
- rel_exist 第一级留下的 400 对放在一起平均（focal γ=2；标注 w=1，未标注 w=0.3）
- rel_pred  只对「有标注 ∧ 留在最终 128 对」的配对平均；空时返回连参 0
            logits = cos/0.07（不乘 logit_scale）
- rel_swap  margin=0.05；对称词豁免；同图同谓词双向都标的那条跳过（交换对采样时已强制保留）

对称词豁免表：beside / standing beside / talking to（词表内无反义词对，
反义词负列逻辑保留但本词表不触发）。有关系标注的图不加权。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

LAMBDA_GEO = 1.0
LAMBDA_EXIST = 1.0
LAMBDA_PRED = 1.0
LAMBDA_SWAP = 0.5
FOCAL_GAMMA = 2.0
NEG_WEIGHT = 0.3
INFONCE_TEMP = 0.07
SWAP_MARGIN = 0.05

# v3 swap 豁免：语义对称谓词（无方向性，交换监督无意义）
SYMMETRIC_SKIP = {"beside", "standing beside", "talking to"}

# 反义词对（词表内出现时作为负列加入；当前 15 谓词配不出 → 不触发，逻辑保留）
DEFAULT_INVERSE_PAIRS = (
    ("above", "below"), ("left of", "right of"), ("left", "right"),
    ("in front of", "behind"), ("over", "under"), ("on", "under"),
    ("inside", "outside"), ("up", "down"),
)


class YOLOVORelateLoss:
    """v3 关系损失：逐图累积四项的原始量，__call__ 末尾 batch 分开平均再加权。"""

    loss_names = ("rel_geo", "rel_exist", "rel_pred", "rel_swap")

    def __init__(self, relate_module, rnames: Dict[int, str], inverse_pairs: Optional[Tuple] = DEFAULT_INVERSE_PAIRS):
        self.rel = relate_module
        self.rnames = {int(k): str(v) for k, v in dict(rnames).items()}
        rows = [
            (i, j)
            for i, wi in self.rnames.items()
            for j, wj in self.rnames.items()
            if i < j and (wi, wj) in set(inverse_pairs or ())
        ]
        self.inverse_ids = torch.tensor(sum(([a, b] for a, b in rows), []), dtype=torch.long) if rows else None
        # 对称豁免索引
        self.symmetric_ids = {pid for pid, w in self.rnames.items() if w in SYMMETRIC_SKIP}

    # ----------------------------------------------------------------- utils

    def _zero_connected(self, device) -> torch.Tensor:
        """零损失但连参（空集合 backward 不崩）。"""
        return 0.0 * next(self.rel.parameters()).sum()

    @staticmethod
    def _focal_bce(score_logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float = FOCAL_GAMMA):
        """带权 focal BCE（输入视为 logit）。按权重归一平均。"""
        bce = F.binary_cross_entropy_with_logits(score_logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = weights * (1 - p_t).pow(gamma) * bce
        return loss.sum() / weights.sum().clamp(min=1e-6)

    # ---------------------------------------------------------- per-image 累积

    def accumulate(self, rel_out: Dict, targets: Dict, acc: Dict) -> None:
        """单图四项的原始量追加进 acc（sum / count 结构）。

        rel_out 来自 head.replay_image：pair_idx_all/geo_logit_all（全对）、
        pair_idx1/exist_score_400（第一级 400 对）、pair_idx2/pair_emb（最终 128 对）。
        """
        device = rel_out["geo_logit_all"].device
        boxes = rel_out["boxes"]
        n = rel_out["n_boxes"]
        relations = targets.get("relations")
        gt = [(int(r[0]), int(r[1]), int(r[2])) for r in relations] if relations is not None and len(relations) else []
        gt_keyset = {s * n + o for (s, o, _) in gt}

        # ---- rel_geo：全对 BCE（正=标注对；分母=有效对数，无 clamp）----
        p_all = rel_out["pair_idx_all"]
        keys_all = p_all[:, 0] * n + p_all[:, 1]
        y_geo = torch.tensor([1.0 if int(k) in gt_keyset else 0.0 for k in keys_all], device=device)
        acc["geo_sum"] = acc["geo_sum"] + F.binary_cross_entropy_with_logits(
            rel_out["geo_logit_all"].float(), y_geo, reduction="sum"
        )
        acc["geo_cnt"] = acc["geo_cnt"] + p_all.shape[0]

        # ---- rel_exist：400 对 focal（w 只乘在每一项上；分母=对数，无 clamp）----
        if "exist_score_400" in rel_out and rel_out["exist_score_400"].numel():
            p1 = rel_out["pair_idx1"]
            keys1 = p1[:, 0] * n + p1[:, 1]
            hit = torch.tensor([int(k) in gt_keyset for k in keys1], device=device)
            s400 = rel_out["exist_score_400"].float()
            w = torch.where(hit, torch.ones_like(s400), torch.full_like(s400, NEG_WEIGHT))
            bce = F.binary_cross_entropy_with_logits(s400, hit.float(), reduction="none")
            p_t = torch.exp(-bce)
            term = w * (1 - p_t).pow(FOCAL_GAMMA) * bce
            acc["exist_sum"] = acc["exist_sum"] + term.sum()
            acc["exist_cnt"] = acc["exist_cnt"] + s400.shape[0]

        # ---- rel_pred：标注 ∧ 留在 128；logits=cos/0.07；负列含反义词（本词表不触发）----
        if gt:
            p2 = rel_out["pair_idx2"]
            keys2 = rel_out["keys2"]  # head 已算好（s*n+o）
            emb = rel_out["pair_emb"].float()
            W = self.rel.predicate_bank.to(emb.device).float()
            cos = F.normalize(emb, dim=-1) @ F.normalize(W, dim=-1).T  # [P, V]

            # 行映射：k → row（对齐，杜绝串行）
            row_of = {int(k): i for i, k in enumerate(keys2.tolist())}
            pos = [(row_of[s * n + o], int(pid)) for (s, o, pid) in gt if (s * n + o) in row_of]
            if pos:
                rows_t = torch.tensor([r for r, _ in pos], device=device)
                pids_t = torch.tensor([p_ for _, p_ in pos], device=device)
                logits = cos[rows_t] / INFONCE_TEMP
                # 反义词负列：本词表配不出 → 空扩展，逻辑保留
                if self.inverse_ids is not None and self.inverse_ids.numel():
                    logits = logits  # 反义列已在全词表内（cross_entropy 覆盖），无需额外列
                acc["pred_sum"] = acc["pred_sum"] + F.cross_entropy(logits, pids_t, reduction="sum")
                acc["pred_cnt"] = acc["pred_cnt"] + len(pos)

                # ---- rel_swap：同谓词双向才跳过；行对齐取余弦 ----
                gt_spo = {(s, o, pid) for (s, o, pid) in gt}
                sw_terms = []
                for row_i, (s, o, pid) in zip(rows_t.tolist(), [(s, o, p_) for (s, o, p_) in gt if (s * n + o) in row_of]):
                    if pid in self.symmetric_ids:
                        continue
                    if (o, s, pid) in gt_spo:
                        continue  # 同谓词双向都标 → 跳过
                    k_sw = o * n + s
                    if k_sw not in row_of:
                        continue  # 交换对未存活（采样已强制，兜底）
                    c_f = cos[row_i, int(pid)]
                    c_sw = cos[row_of[k_sw], int(pid)]
                    sw_terms.append(F.relu(SWAP_MARGIN + c_sw - c_f))
                if sw_terms:
                    acc["swap_sum"] = acc["swap_sum"] + torch.stack(sw_terms).sum()
                    acc["swap_cnt"] = acc["swap_cnt"] + len(sw_terms)

    # ----------------------------------------------------------------- call

    def finalize(self, acc: Dict, device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._zero_connected(device)  # 空集合项的连参零
        l_geo = acc["geo_sum"] / max(acc["geo_cnt"], 1) if acc["geo_cnt"] > 0 else z
        l_exist = acc["exist_sum"] / max(acc["exist_cnt"], 1) if acc["exist_cnt"] > 0 else z
        l_pred = acc["pred_sum"] / max(acc["pred_cnt"], 1) if acc["pred_cnt"] > 0 else z
        l_swap = acc["swap_sum"] / max(acc["swap_cnt"], 1) if acc["swap_cnt"] > 0 else z

        loss = LAMBDA_GEO * l_geo + LAMBDA_EXIST * l_exist + LAMBDA_PRED * l_pred + LAMBDA_SWAP * l_swap
        items = torch.stack([l_geo.detach(), l_exist.detach(), l_pred.detach(), l_swap.detach()])
        return loss, items

    @staticmethod
    def new_acc() -> Dict:
        return {"geo_sum": 0.0, "geo_cnt": 0, "exist_sum": 0.0, "exist_cnt": 0.0, "exist_w": 0.0,
                "pred_sum": 0.0, "pred_cnt": 0, "swap_sum": 0.0, "swap_cnt": 0}

    # v2 兼容入口（trainer 若仍按图调用）：直接累计后 finalize
    def __call__(self, rel_out: Dict, targets: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        acc = self.new_acc()
        self.accumulate(rel_out, targets, acc)
        return self.finalize(acc, rel_out["geo_logit_all"].device)
