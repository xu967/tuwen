"""YOLOVORelate v4: faithful port of the RA (RelateAnything) relation head.

Structure (relsgg.py ported; backbone swapped for the YOLOVO neck):

    neck P4 (40x40xC4) --1x1--> F_map [B,40,40,768]
    SoftSpatialPool          one feature per box (cross-attention, RA pooling.py)
    RelatednessPairSampler   two-stage sampling (geo 400 -> inner/16 128)
    pair_proj(cat[v_sub,v_obj,v_union,v_contact,geo]) -> pair_feat
    RelationTransformer      2 self + 2 cross layers (RA transformer.py)
    DeformableRelRead        deformable scene read, null slots (RA deformable.py)
    RelationInteractionBlock n_dep=2 + n_gnd=1 joint layers (RA backbone_block.py)
    dual query: q = norm(proj(r) + g0*sub_text(v_sub) + g1*obj_text(v_obj))
                q_spa = spa_proj(cat[r, geo_feat])
    VocabHead.score_query_dual -> logits  (alpha-routed cosine mixture)

Losses (RA config defaults):
    nce(0.07,512,0.5 hard) + 0.1*obj_align + 0.5*swap(0.05)
    + 0.25*sigmoid_aux + 0.05*bg(5) + 1.0*geo + 1.0*rel(focal 0.3 PU)
Score contract: sigmoid(a*(pred_logit + w*pair_logit) + b), Platt after.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .field import FieldConfig, PredicateFieldRead, field_grid, field_stats
from .ra.backbone_block import RelationInteractionBlock
from .ra.deformable import DeformableRelRead
from .ra.geometry import BoxPromptEncoder, RelGeomEncoder
from .ra.pooling import SoftSpatialPool, union_box
from .ra.transformer import RelationTransformer

NUM_GEO = 19


class _VocabHead(nn.Module):
    """RA vocab_head.py ported: dual-expert scaled cosine with W-routed alpha."""

    def __init__(self, d_model: int = 512, text_dim: int = 512,
                 logit_scale_init: float = 5.0, gate_hidden: int = 128):
        super().__init__()
        self.text_dim = text_dim
        hidden = max(d_model * 2, text_dim // 2)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, text_dim, bias=False),
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.logit_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("W", torch.empty(0))
        # 可学习参数而不是 buffer：原来只在 set_vocabulary 里算一次，gate_mlp 永远
        # 没有梯度，路由退化成全体谓词固定 ~0.49。
        self.register_parameter("alpha", None)
        self.pred_names: List[str] = []
        self.gate_mlp = nn.Sequential(
            nn.Linear(text_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def set_vocabulary(self, names: List[str], W: torch.Tensor) -> None:
        self.pred_names = list(names)
        self.W = F.normalize(W.float(), dim=-1).clone()
        self._update_alpha()

    @torch.no_grad()
    def _update_alpha(self) -> None:
        if self.W.numel():
            a = torch.sigmoid(self.gate_mlp(self.W).squeeze(-1)).clone()
            # gate_mlp 只提供初值；之后 alpha 由 NCE/场损失自己学。
            self.alpha = nn.Parameter(a)

    def score_query_dual(self, q_sem: torch.Tensor, q_spa: torch.Tensor) -> torch.Tensor:
        """Per-predicate mixture of the two experts' cosines, scaled and shifted."""
        if self.W.numel() == 0:
            raise RuntimeError("no vocabulary installed")
        alpha = self.alpha
        cos_sem = F.normalize(q_sem, dim=-1) @ self.W.T
        cos_spa = F.normalize(q_spa, dim=-1) @ self.W.T
        cos = (1.0 - alpha) * cos_sem + alpha * cos_spa
        scale = self.logit_scale.exp().clamp(max=100.0)
        return cos * scale + self.logit_bias

    def score_candidates(self, q_sem: torch.Tensor, q_spa: torch.Tensor,
                         cand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a top-M candidate subset per pair.

        ``q_sem``/``q_spa`` are ``[B,K,M,D]``, ``cand`` is ``[B,K,M]``.
        Returns ``(cos, logit)`` each ``[B,K,M]``.
        """
        Wc = self.W[cand]                                # [B,K,M,D]
        a = self.alpha[cand].unsqueeze(-1)               # [B,K,M,1]
        cos_sem = (F.normalize(q_sem, dim=-1) * Wc).sum(-1)
        cos_spa = (F.normalize(q_spa, dim=-1) * Wc).sum(-1)
        cos = (1.0 - a.squeeze(-1)) * cos_sem + a.squeeze(-1) * cos_spa
        scale = self.logit_scale.exp().clamp(max=100.0)
        return cos, cos * scale + self.logit_bias


class _Sampler(nn.Module):
    """RA sampler.py ported verbatim in logic (batched [B, N] form).

    geo_scorer: 19 -> 64 -> 1 MLP; f_sub/f_obj: LayerNorm+MLP -> rel_dim;
    second-stage score = <f_s, f_o> / sqrt(rel_dim) (fixed scalar divisor).
    """

    def __init__(self, feat_dim: int, geo_budget: int = 400, final_budget: int = 128,
                 rel_dim: int = 256, neg_weight: float = 0.3, swap_include: bool = True):
        super().__init__()
        self.geo_budget, self.final_budget = geo_budget, final_budget
        self.rel_dim, self.neg_weight, self.swap_include = rel_dim, neg_weight, swap_include
        self.geo_scorer = nn.Sequential(nn.Linear(NUM_GEO, 64), nn.GELU(), nn.Linear(64, 1))
        self.f_sub = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, rel_dim),
                                   nn.GELU(), nn.Linear(rel_dim, rel_dim))
        self.f_obj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, rel_dim),
                                   nn.GELU(), nn.Linear(rel_dim, rel_dim))

    def forward(self, boxes: torch.Tensor, obj_feats: torch.Tensor,
                box_counts: Optional[torch.Tensor] = None,
                is_gt: Optional[torch.Tensor] = None, flat_labels: Optional[torch.Tensor] = None):
        """Eval form (no losses): returns sub_idx/obj_idx [B,K], valid, rel_logits [B,K]."""
        B, N, _ = boxes.shape
        device = boxes.device
        if box_counts is None:
            box_counts = torch.full((B,), N, dtype=torch.long, device=device)
        ar = torch.arange(N, device=device)
        valid_box = ar.unsqueeze(0) < box_counts.unsqueeze(1)
        not_self = ar.unsqueeze(0) != ar.unsqueeze(1)
        pair_valid = (valid_box.unsqueeze(2) & valid_box.unsqueeze(1) & not_self).reshape(B, N * N)

        geo_feats = RelGeomEncoder.features(
            boxes.unsqueeze(2).expand(B, N, N, 4),
            boxes.unsqueeze(1).expand(B, N, N, 4)).reshape(B, N * N, NUM_GEO)
        geo_scores = self.geo_scorer(geo_feats).squeeze(-1)

        NEG = torch.finfo(geo_scores.dtype).min / 2
        sel_scores = geo_scores.masked_fill(~pair_valid, NEG)
        force = is_gt if is_gt is not None else None
        if force is not None:
            force = force & pair_valid
            sel_scores = sel_scores.masked_fill(force, float("inf"))
        K1 = min(self.geo_budget, N * N)
        _, top1 = sel_scores.topk(K1, dim=1)
        alive1 = torch.gather(pair_valid, 1, top1)

        zs, zo = self.f_sub(obj_feats), self.f_obj(obj_feats)
        sub_i1, obj_i1 = top1 // N, top1 % N
        z_s = torch.gather(zs, 1, sub_i1.unsqueeze(-1).expand(-1, -1, self.rel_dim))
        z_o = torch.gather(zo, 1, obj_i1.unsqueeze(-1).expand(-1, -1, self.rel_dim))
        rel_scores1 = (z_s * z_o).sum(-1) / (self.rel_dim ** 0.5)

        sel2 = rel_scores1.masked_fill(~alive1, NEG)
        if force is not None:
            sel2 = sel2.masked_fill(torch.gather(force, 1, top1), float("inf"))
        K2 = min(self.final_budget, K1)
        _, top2 = sel2.topk(K2, dim=1)

        flat_sel = torch.gather(top1, 1, top2)
        sub_idx, obj_idx = flat_sel // N, flat_sel % N
        valid_mask = torch.gather(alive1, 1, top2)
        rel_logits = torch.gather(rel_scores1, 1, top2)
        pred_labels = None
        if flat_labels is not None:
            pred_labels = torch.gather(flat_labels, 1, flat_sel)
            pred_labels = torch.where(valid_mask, pred_labels, torch.full_like(pred_labels, -1))
        return sub_idx, obj_idx, valid_mask, pred_labels, rel_logits


def _gt_grid(gt_list, B: int, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """is_gt [B,N*N] and predicate label per pair (-1 none), from per-image gt lists.

    同一有序对的多个谓词在这里 last-wins，只保留最后一个。这个单标签仅用于
    「把 GT 列强制塞进 top-M 候选」和诊断；训练目标走 _losses 里按 gt_list
    重建的多热 gt_hot，不受此限制。
    """
    is_gt = torch.zeros(B, N * N, dtype=torch.bool, device=device)
    labels = torch.full((B, N * N), -1, dtype=torch.long, device=device)
    for b, gt in enumerate(gt_list):
        if not gt:
            continue
        for (s, o, p) in gt:
            if s < N and o < N:
                is_gt[b, s * N + o] = True
                labels[b, s * N + o] = int(p)
    return is_gt, labels


def _cxwh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], -1)


class YOLOVORelate(nn.Module):
    """v4 RA-faithful relation head over the YOLOVO neck.

    forward(x) x = [primary_out, *feat_maps(P3,P4,P5)] — P4 is the base map.
    """

    def __init__(self, primary_head, feat_layers: List[int], feat_strides: List[int],
                 n_predicates: int, text_dim: int = 512, feat_channels: Optional[List[int]] = None,
                 hidden_dim: int = 512, geo_budget: int = 400, final_budget: int = 128,
                 eval_budget: int = 500, max_objects: int = 60,
                 backbone_dim: int = 768, d_model: int = 512,
                 field_cfg: Optional[FieldConfig] = None):
        super().__init__()
        self.feat_layers, self.feat_strides = list(feat_layers), list(feat_strides)
        self.n_predicates, self.text_dim = n_predicates, text_dim
        self.geo_budget, self.final_budget, self.eval_budget = geo_budget, final_budget, eval_budget
        self.max_objects, self.d_model = max_objects, d_model
        # P4 index inside the head's from-list (stride 16)
        self.base_idx = self.feat_strides.index(16) if 16 in self.feat_strides else 0
        self._feat_channels = feat_channels

        # base-map adapter: P4 channels -> backbone_dim (Q1B)
        self.fmap_proj: Optional[nn.Module] = None

        self.spatial_pool = SoftSpatialPool(d_model=backbone_dim)
        self.sampler = _Sampler(feat_dim=backbone_dim, geo_budget=geo_budget,
                                final_budget=final_budget)
        self.geo_encoder = RelGeomEncoder(d_model=d_model)
        # pair 输入 = v_sub/v_obj/v_union/v_contact(768×4) + geo_d(geo_encoder 输出=d_model)
        self.pair_proj = nn.Linear(backbone_dim * 4 + d_model, d_model)
        self.box_prompt_encoder = BoxPromptEncoder(d_model=d_model)
        self.rel_transformer = RelationTransformer(d_model=d_model, backbone_dim=backbone_dim)
        self.deformable_read = DeformableRelRead(d_model=d_model)
        self.rel_interaction = RelationInteractionBlock(d_model=d_model, scene_dim=backbone_dim)
        self.vocab_head = _VocabHead(d_model=d_model, text_dim=text_dim)

        # GLP predicate field (off by default; V1/V2 selected by FieldConfig)
        self.field_cfg = FieldConfig.from_any(field_cfg) if field_cfg is not None \
            else FieldConfig(mode="off")
        # 场残差范数上限：早期健康值在 8~25，超过 16 后开始发散（见 _forward_pairs）。
        self.field_delta_max = 16.0
        # 候选列与 raw 检索分数的合成方式：field（覆写，原行为）| max（只加分）| raw（不用场）
        self.field_fuse = "field"
        self.field_read: Optional[PredicateFieldRead] = None
        self.q_res_proj: Optional[nn.Module] = None
        if self.field_cfg.mode != "off":
            self.field_read = PredicateFieldRead(d_model=d_model, cfg=self.field_cfg)
            self.q_res_proj = nn.Sequential(
                nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, text_dim))
            for m in self.q_res_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # dual-query composition (gates start small: training begins from the
        # pure pair-context query, RA init 0.1)
        self.sub_text_proj = nn.Linear(backbone_dim, text_dim, bias=False)
        self.obj_text_proj = nn.Linear(backbone_dim, text_dim, bias=False)
        nn.init.xavier_uniform_(self.sub_text_proj.weight)
        nn.init.xavier_uniform_(self.obj_text_proj.weight)
        # PredCls 的定义输入是 GT 框 + GT 类别；冻结的类名嵌入经此投到视觉特征空间。
        self.cls_proj = nn.Linear(text_dim, backbone_dim, bias=False)
        nn.init.xavier_uniform_(self.cls_proj.weight)
        self.compose_norm = nn.LayerNorm(text_dim)
        self.compose_gate = nn.Parameter(torch.tensor([0.1, 0.1]))
        self.register_buffer("W_obj", torch.empty(0))   # class-name embeddings (Q2)
        hidden = max(d_model * 2, text_dim // 2)
        self.spa_proj = nn.Sequential(
            nn.Linear(d_model + NUM_GEO, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, text_dim, bias=False))
        for m in self.spa_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # ontology tables (Q3A) — installed with the vocabulary
        self.onto: Optional[Dict[str, torch.Tensor]] = None

        # v3-compat capture/stash channels
        self.capture = False
        self._captured = None
        self._batch_feats = None
        self._batch_primary = None

        self.i = -1
        self.f: List[int] = []
        self.type = self.__class__.__name__

    # ------------------------------------------------------------ install

    def install_vocabulary(self, names: List[str], W: torch.Tensor, W_obj: Optional[torch.Tensor] = None):
        """W [V, text_dim] predicate bank; W_obj [n_cls, text_dim] class names (Q2)."""
        self.vocab_head.set_vocabulary(names, W)
        from .ra.ontology import build_ontology
        self.onto = build_ontology(self.vocab_head.W, names)
        if W_obj is not None:
            self.W_obj = F.normalize(W_obj.float(), dim=-1).clone()

        # RA relsgg.py install_losses: nce 是 scale-free 的（作用于 raw cos），
        # logit_scale/logit_bias 只由 sigmoid/bg 训练。两项权重都 <= 0 时冻结，
        # 避免验证打分链（cos*scale+bias）的仿射参数无监督漂移。
        # 冻结状态记录在 self._affine_frozen，供 trainer._freeze_primary 在
        # 每 epoch 重冻结时遵循（避免 epoch 级 requires_grad=True 把它撤销）。
        from ..relate.trainer import RelateTrainerMixin  # 延迟导入避免环
        sig_on = getattr(RelateTrainerMixin, "LAMBDA_RA", {}).get("sigmoid", 0.0) > 0.0
        bg_on = getattr(RelateTrainerMixin, "LAMBDA_RA", {}).get("bg", 0.0) > 0.0
        self._affine_frozen = not (sig_on or bg_on)
        for p in (self.vocab_head.logit_scale, self.vocab_head.logit_bias):
            p.requires_grad_(not self._affine_frozen)
        if self._affine_frozen:
            LOGGER.info("[Relate] vocab_head.logit_scale/bias frozen: sigmoid/bg both disabled (RA rule)")

    def _ensure_fmap_proj(self, ch4: int, device) -> nn.Module:
        if self.fmap_proj is None:
            self.fmap_proj = nn.Sequential(nn.Conv2d(ch4, 768, 1), nn.GELU()).to(device)
        return self.fmap_proj

    def _fmap(self, base: torch.Tensor) -> torch.Tensor:
        """P4 [B,C,h,w] -> F_map [B,h,w,768]."""
        z = self._ensure_fmap_proj(base.shape[1], base.device)(base)  # [B,768,h,w]
        return z.permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------ forward

    def forward(self, x, feat_batch_idx: Optional[int] = None):
        """x = [primary_out, *feat_maps]; train stashes the base map, eval captures."""
        primary_out, feats = x[0], list(x[1:])
        base = feats[self.base_idx].detach().float()          # [B, C4, h, w]
        if feat_batch_idx is not None:
            base = base[feat_batch_idx: feat_batch_idx + 1]
        if not self.training:
            if self.capture:
                # 逐图列表：validator 按 batch 顺序逐图调 score（第 b 图用 feats[b]）
                self._captured = {"feats": [base[b:b+1] for b in range(base.shape[0])],
                                  "strides": [16]}
            return primary_out
        self._batch_feats = [base]
        self._batch_primary = primary_out
        return {"det": primary_out, "rel": None}


    # --------------------------------------------------------- full compute

    def _forward_pairs(self, base: torch.Tensor, boxes_b: torch.Tensor,
                       box_counts: Optional[torch.Tensor] = None,
                       gt_list: Optional[List[List[Tuple[int, int, int]]]] = None,
                       eval_budget: Optional[int] = None,
                       field_ablate: Optional[str] = None,
                       box_cls: Optional[torch.Tensor] = None):
        """Batched pair pipeline (no losses). ``box_counts [B]`` marks the valid
        prefix of each padded box row, so batches of variable-size images share
        one forward instead of B per-image calls."""
        F_map = self._fmap(base)                              # [B,h,w,768]
        B, h, w, _ = F_map.shape
        N = boxes_b.shape[1]
        device = boxes_b.device

        v_obj = self.spatial_pool(F_map, boxes_b)             # [B,N,768]
        if box_cls is not None and self.W_obj.numel():
            # 类别是 PredCls 的定义输入；不喂的话模型只能从外观反推，实测低于
            # 只用 (subj_cls,obj_cls) 查表的 32.3%。W_obj 冻结，cls_proj 可学。
            if self.W_obj.device != device:
                self.W_obj = self.W_obj.to(device)
            cid = box_cls.to(device).long().clamp(0, self.W_obj.shape[0] - 1)
            v_obj = v_obj + self.cls_proj(self.W_obj[cid])
        is_gt = labels = None
        training = gt_list is not None
        if training:
            is_gt, labels = _gt_grid(gt_list, B, N, device)
            if self.sampler.swap_include:
                swapped = is_gt.reshape(B, N, N).transpose(1, 2).reshape(B, N * N)
                force = is_gt | swapped
            else:
                force = is_gt
        else:
            force = None
        budget = eval_budget if not training else None
        saved = saved_final = None
        if budget is not None:
            saved, saved_final = self.sampler.geo_budget, self.sampler.final_budget

        try:
            if budget is not None:
                # Both stages must honour eval_budget; otherwise stage 2 caps
                # at final_budget=128 and topk=500 silently returns 128.
                self.sampler.geo_budget = budget
                self.sampler.final_budget = budget
            sub_idx, obj_idx, valid_mask, pred_labels, rel_logits = self.sampler(
                boxes_b, v_obj, box_counts=box_counts, is_gt=force, flat_labels=labels)
        finally:
            if saved is not None:
                self.sampler.geo_budget = saved
                self.sampler.final_budget = saved_final

        K = sub_idx.shape[1]
        # _gt_grid 的 labels 是 last-wins：同一有序对的第 2..k 个谓词会被覆盖。候选强制
        # 和场指标都需要「这个有序对的全部 GT 谓词」，这里建一次 (slot, pred) 稀疏索引，
        # 两处共用，避免再各自反查一遍。
        gt_slots = None
        if training and gt_list is not None:
            gt_slots = []
            empty = torch.empty(0, dtype=torch.long, device=device)
            for b in range(B):
                if not gt_list[b]:
                    gt_slots.append((empty, empty))
                    continue
                g = torch.tensor(gt_list[b], dtype=torch.long, device=device)     # [G,3]
                flat = sub_idx[b] * N + obj_idx[b]                                # [K]
                hit = flat.unsqueeze(1) == (g[:, 0] * N + g[:, 1]).unsqueeze(0)   # [K,G]
                k_idx, g_idx = hit.nonzero(as_tuple=True)
                gt_slots.append((k_idx, g[g_idx, 2]))
        B_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        sub_boxes_k = boxes_b[B_idx, sub_idx]
        obj_boxes_k = boxes_b[B_idx, obj_idx]
        v_sub = v_obj[B_idx, sub_idx]
        v_obj_k = v_obj[B_idx, obj_idx]
        union_boxes_k = union_box(sub_boxes_k, obj_boxes_k)

        # contact zone: intersection if overlap else the gap box (RA relsgg.py)
        sub_xy, obj_xy = _cxwh_to_xyxy(sub_boxes_k), _cxwh_to_xyxy(obj_boxes_k)
        ix1 = torch.maximum(sub_xy[..., 0], obj_xy[..., 0])
        iy1 = torch.maximum(sub_xy[..., 1], obj_xy[..., 1])
        ix2 = torch.minimum(sub_xy[..., 2], obj_xy[..., 2])
        iy2 = torch.minimum(sub_xy[..., 3], obj_xy[..., 3])
        contact_k = torch.stack([(ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5,
                                 (ix2 - ix1).abs(), (iy2 - iy1).abs()], -1)
        pool_boxes = torch.cat([union_boxes_k, contact_k], dim=1)
        v_pool = self.spatial_pool(F_map, pool_boxes)
        v_union, v_contact = v_pool[:, :K], v_pool[:, K:]

        # RelGeomEncoder returns features when called with .features; the module
        # call projects to d_model — we need BOTH: raw 19 for spa_proj, projected
        # for pair input. Use the static features + own MLP-consistent projection:
        geo19 = RelGeomEncoder.features(sub_boxes_k, obj_boxes_k)  # [B,K,19]
        geo_d = self.geo_encoder(sub_boxes_k, obj_boxes_k)         # [B,K,d_model]

        pair_input = torch.cat([v_sub, v_obj_k, v_union, v_contact, geo_d], dim=-1)
        pair_feat = self.pair_proj(pair_input)

        box_tokens = self.box_prompt_encoder.encode_pairs(sub_xy, obj_xy)  # [B,K,4,d]
        padding_mask = ~valid_mask
        r = self.rel_transformer(pair_feat, F_map, box_tokens=box_tokens,
                                 pair_padding_mask=padding_mask)
        scene_d = self.rel_transformer.scene_proj(F_map).permute(0, 3, 1, 2).contiguous()
        anchors = torch.stack([sub_boxes_k, obj_boxes_k, union_boxes_k, contact_k], dim=2)
        r = self.deformable_read(r, scene_d, anchors)
        scene_flat = F_map.reshape(B, h * w, -1)
        r = self.rel_interaction(r, scene_flat, query_padding_mask=padding_mask,
                                 grid_hw=(h, w))

        # 七项损失复用：整批算一次 N×N 几何打分与 f_sub/f_obj，逐图损失不再各自重算。
        geo_scores = self.sampler.geo_scorer(RelGeomEncoder.features(
            boxes_b.unsqueeze(2).expand(B, N, N, 4),
            boxes_b.unsqueeze(1).expand(B, N, N, 4)).reshape(B, N * N, NUM_GEO)).squeeze(-1)
        zs, zo = self.sampler.f_sub(v_obj), self.sampler.f_obj(v_obj)

        g = self.compose_gate
        q = self.compose_norm(self.vocab_head.proj(r)
                              + g[0] * self.sub_text_proj(v_sub)
                              + g[1] * self.obj_text_proj(v_obj_k))
        q_spa = self.spa_proj(torch.cat([r, geo19], dim=-1))
        logits = self.vocab_head.score_query_dual(q, q_spa)     # [B,K,V]

        out = {"sub_idx": sub_idx, "obj_idx": obj_idx, "valid_mask": valid_mask,
               "rel_logits": rel_logits, "logits": logits, "v_obj": v_obj,
               "boxes": boxes_b, "n_boxes": N, "q": q, "q_spa": q_spa,
               "geo19": geo19, "is_gt": is_gt, "labels": labels, "force": force,
               "gt_slots": gt_slots,
               "box_counts": box_counts, "geo_scores": geo_scores, "zs": zs, "zo": zo}
        if self.field_read is not None:
            if self.vocab_head.W.device != logits.device:
                self.vocab_head.W = self.vocab_head.W.to(logits.device)
                self.vocab_head.alpha = self.vocab_head.alpha.to(logits.device)
            V = self.vocab_head.W.shape[0]
            m = min(int(self.field_cfg.n_candidates), V)
            cand = self._select_candidates(logits, pred_labels, valid_mask, m, training,
                                           gt_slots=gt_slots)
            e_cand = self.vocab_head.W[cand]
            if field_ablate == "agnostic":
                e_cand = e_cand.mean(dim=2, keepdim=True).expand_as(e_cand)
            elif field_ablate == "shuffled":
                e_cand = e_cand.roll(1, dims=2)
            r_field, field_out = self.field_read(
                r, scene_d, anchors, geo19, e_cand, return_field=True)
            delta = r_field - r.unsqueeze(2)
            # 残差无界会让 q_f 的归一化梯度失控：实测 delta_norm 从 24 涨到 50 后，
            # 下一轮 loss 直接变 NaN。按范数做投影截断（超出的部分等比缩回），不新增参数。
            dnorm = delta.norm(dim=-1, keepdim=True)
            delta = delta * (self.field_delta_max / dnorm.clamp_min(self.field_delta_max))
            res = self.q_res_proj(delta)
            # delta 截断只防 fp16 溢出，防不住退化：q_res_proj 的输出仍可达 ||q|| 的
            # 228 倍，且候选维度变化为 0，散列后把 raw 检索分数整个盖成常量噪声。
            # 把输出投影到 ||q||/10 的球内，场头只能「轻推」排序，检索头重新主导。
            cap = (q.detach().norm(dim=-1) / 10.0).clamp_min(1.0)[..., None, None]
            res = res * (cap / res.norm(dim=-1, keepdim=True).clamp_min(cap))
            q_f = q.unsqueeze(2) + res
            q_spa_f = q_spa.unsqueeze(2) + res
            cos_m, logit_m = self.vocab_head.score_candidates(q_f, q_spa_f, cand)
            # 保留未散列的检索分数：scatter 会把 GT 列覆盖成 field 分数，检索头
            # （q·W）若只在散列后的分数上训练，正样本列永远拿不到梯度，推理时
            # raw top-M 就召回不到 GT（实测 GT 排名 332~9112，top-4 命中 0/17）。
            out["logits_raw"] = logits
            fused = logits.scatter(-1, cand, logit_m.to(logits.dtype))
            # 场分数整体低于 raw 时，覆写等于把检索头的赢家压下去（实测 acc −2.9pp）。
            # max：只允许场加分；raw：评测不用场（消融）。getattr：旧 ckpt 没这个属性。
            mode = getattr(self, "field_fuse", "field")
            if mode == "max":
                fused = torch.maximum(logits, fused)
            elif mode == "raw":
                fused = logits
            out["logits"] = fused
            out["cand_idx"] = cand
            out["field_cos"] = cos_m
            out["field"] = field_out
            diag = {"delta_norm": dnorm.detach().flatten(1).mean(-1)}
            gamma = getattr(self.field_read, "gamma", None)
            if gamma is not None:
                diag["gamma_l1"] = gamma.detach().abs().sum().expand(B)
            if "w_bg" in field_out:
                diag["w_bg_mean"] = field_out["w_bg"].detach().mean(dim=(1, 2))
            out["field_diag"] = diag
        return out

    def _compute(self, base: torch.Tensor, boxes_b: torch.Tensor,
                 gt_list: Optional[List[List[Tuple[int, int, int]]]] = None,
                 eval_budget: Optional[int] = None,
                 box_cls: Optional[torch.Tensor] = None,
                 field_ablate: Optional[str] = None):
        """Single-image view over :meth:`_forward_pairs` (+ losses when training).

        boxes_b [1, N, 4] normalised cxcywh; gt_list per-image triples.
        Returns dict with indices, logits, and (train) the seven loss terms.
        """
        o = self._forward_pairs(base, boxes_b, None, gt_list, eval_budget, field_ablate, box_cls)
        B = o["sub_idx"].shape[0]
        out = {"sub_idx": o["sub_idx"][0], "obj_idx": o["obj_idx"][0],
               "valid_mask": o["valid_mask"][0], "rel_logits": o["rel_logits"][0],
               "logits": o["logits"][0], "v_obj": o["v_obj"], "boxes": boxes_b,
               "n_boxes": o["n_boxes"], "geo_scores": o["geo_scores"],
               "zs": o["zs"], "zo": o["zo"]}
        # _losses 用 out.get("logits_raw", out["logits"]) 兜底；不往外传会静默 fallback
        # 到场散列后的 logits，NCE 凭空低 0.6 nat，而检索排序毫无改善。
        if "logits_raw" in o:
            out["logits_raw"] = o["logits_raw"][0]
        if "field" in o:
            out["cand_idx"] = o["cand_idx"][0]
            out["field_cos"] = o["field_cos"][0]
            out["field"] = {
                k: (v[0] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v)
                for k, v in o["field"].items()
            }
            out["field_diag"] = {
                k: (v[0] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v)
                for k, v in o["field_diag"].items()
            }
        if gt_list is None:
            return out
        out.update(self._losses(out, gt_list, o["q"], o["is_gt"],
                                o["sub_idx"], o["obj_idx"], o["valid_mask"],
                                o["v_obj"], boxes_b, o["n_boxes"], box_cls,
                                box_counts=o["box_counts"]))
        return out

    # ------------------------------------------------------------- losses

    def _losses(self, out, gt_list, q, is_gt, sub_idx, obj_idx, valid_mask,
                v_obj, boxes_b, N, box_cls=None, box_counts=None):
        """Seven RA loss terms for one image (sum form; trainer averages per set)."""
        from .ra.ontology import build_ontology  # noqa: F401  (tables already on self.onto)
        device = q.device
        n_b = int(box_counts[0]) if box_counts is not None else N
        if self.onto is not None:
            for kt, tt in self.onto.items():
                self.onto[kt] = tt.to(device)
        if self.vocab_head.W.device != device:
            self.vocab_head.W = self.vocab_head.W.to(device)
            self.vocab_head.alpha = self.vocab_head.alpha.to(device)
        if self.W_obj.numel() and self.W_obj.device != device:
            self.W_obj = self.W_obj.to(device)
        V = self.vocab_head.W.shape[0]
        zeros = q.new_zeros

        # gt hot per slot [K, V]：按 gt_list 全量多热。不能走 labels 网格——那是
        # last-wins 的单标签，同一有序对的第 2..k 个谓词会被覆盖（训练集 23.04%
        # 的关系行），sigmoid/nce 因此只能学到其中一个。
        K = sub_idx.shape[1]
        gt_hot = torch.zeros(K, V, dtype=torch.bool, device=device)
        if gt_list[0]:
            g = torch.tensor(gt_list[0], dtype=torch.long, device=device)   # [G,3]
            flat = sub_idx[0] * N + obj_idx[0]                              # [K]
            hit = (flat.unsqueeze(1) == (g[:, 0] * N + g[:, 1]).unsqueeze(0)) \
                & valid_mask[0].unsqueeze(1)
            slot_i, gt_i = hit.nonzero(as_tuple=True)
            gt_hot[slot_i, g[gt_i, 2]] = True
        has_gt = gt_hot.any(-1) & valid_mask[0]
        hot = gt_hot[has_gt].float()

        terms = {}

        # 全词表 cosine 直接由 out["logits"] 还原：dual-score 就是 cos*scale+bias，
        # 已含 field 候选列的 scatter。原实现每图重算 [K,512]@[512,V] 两遍（forward 2 次 + backward 4 次），
        # 单项 4.9 ms/图，是 _losses 里最大开销。
        # 不给 scale/bias 加 detach：两者都带图时 NCE 对仿射参数的梯度精确抵消
        # （NCE 只作用于 cos）；detach 反而留下 Σ f_c·cos_c 的非零梯度，持续把
        # logit_scale 往下压。
        scale = self.vocab_head.logit_scale.exp().clamp(max=100.0)
        # NCE/swap 走未散列的检索分数（检索头），sigmoid/bg 走散列后的分数（场头）。
        cos_all = (out.get("logits_raw", out["logits"]) - self.vocab_head.logit_bias) / scale

        # 1. InfoNCE (batch-local: within-image columns; hard negatives by neg_lw)
        if has_gt.any():
            cos = cos_all[has_gt]
            logits_n = cos / 0.07
            neg_lw = self.onto["neg_lw"] if self.onto else torch.ones(V, V, device=device)
            pos_mask = self.onto["pos_mask"] if self.onto else torch.eye(V, dtype=torch.bool, device=device)
            inv_mask = self.onto["inverse_mask"] if self.onto else torch.zeros(V, V, dtype=torch.bool, device=device)
            hot_b = hot.bool()
            a_idx, g_idx = hot_b.nonzero(as_tuple=True)
            # 稀疏行 gather 代替 hot @ [V,V]：10k 谓词时后者每图要读 430MB 权重表。
            pos_any = torch.zeros_like(hot)
            pos_any.index_add_(0, a_idx, pos_mask[g_idx].to(hot.dtype))
            inv_any = torch.zeros_like(hot)
            inv_any.index_add_(0, a_idx, inv_mask[g_idx].to(hot.dtype))
            # neg_lw 是乘性权重 ∈[0.25,1]（ontology.py）；作为 log 偏置必须取 log
            # （≤0），直接相加等于把远负例抬高 +0.25~+1.0、近义列反而零偏置。
            # 同一槽位多个 GT 时 log 相加 = 权重连乘，与「都要区分」一致。
            lw = torch.zeros_like(hot)
            lw.index_add_(0, a_idx, neg_lw[g_idx].clamp_min(1e-8).log())
            lw = lw.masked_fill((pos_any > 0) | (inv_any > 0), 0.0)
            den = torch.logsumexp(logits_n + lw, dim=-1)
            per = den[a_idx] - logits_n[a_idx, g_idx]
            terms["nce"] = per.mean()

            # 2. swap hinge (RA losses.swap_direction_hinge; sym exempt)
            sym = self.onto["sym"] if self.onto else torch.zeros(V, dtype=torch.bool, device=device)
            cos_q = cos_all
            terms_swap = []
            gt_sym = {(int(x), int(y), int(z)) for (x, y, z) in gt_list[0]}
            sv, ov, vm = sub_idx[0], obj_idx[0], valid_mask[0]
            for (s, o, p) in gt_list[0]:
                if sym[p] or (o, s, p) in gt_sym:
                    continue
                kf_m = vm & (sv == s) & (ov == o)
                kb_m = vm & (sv == o) & (ov == s)
                if not (bool(kf_m.any()) and bool(kb_m.any())):
                    continue
                kf = int(kf_m.nonzero(as_tuple=True)[0][0])
                kb = int(kb_m.nonzero(as_tuple=True)[0][0])
                terms_swap.append(F.relu(0.05 + cos_q[kb, p] - cos_q[kf, p]))
            terms["swap"] = torch.stack(terms_swap).mean() if terms_swap else q.new_zeros(())

            # 3. per-cell sigmoid aux (RA lambda_sigmoid=0.25 weighting)
            # RA relsgg.py: 直接 bce_with_logits(lg, tgt, weight=w)，无 sigmoid/logit 往返。
            # 这一项用散列后的分数（含 field），是场头唯一的正样本上行监督。
            lg = out["logits"][has_gt]
            tgt = hot
            pos_mass = tgt.sum(-1, keepdim=True)
            n_neg = ((tgt <= 0)).sum(-1, keepdim=True).clamp(min=1)
            w = torch.where(tgt > 0, torch.ones_like(tgt), pos_mass / n_neg)
            sig = F.binary_cross_entropy_with_logits(lg, tgt, weight=w, reduction="none")
            terms["sigmoid"] = sig.sum(-1).mean()
        else:
            terms["nce"] = terms["swap"] = terms["sigmoid"] = q.new_zeros(())

        # 4. object-text alignment (Q2): GT boxes' v_obj vs class-name embeddings
        # RA _object_text_loss: 双投影(sub/obj_text_proj)各 CE 取均值，且除以
        # infonce_temp=0.07（裸余弦进 CE 梯度过弱）。
        if self.W_obj.numel() and box_cls is not None and len(box_cls):
            n_al = len(box_cls)
            assert n_al == n_b, (
                f"obj loss alignment: box_cls ({n_al}) != valid boxes ({n_b})")
            feats_al = v_obj[0][:n_al]
            labels_al = box_cls[:n_al].to(device).long()
            obj_loss = q.new_zeros(())
            for proj in (self.sub_text_proj, self.obj_text_proj):
                qo_n = F.normalize(proj(feats_al), dim=-1)
                sim = qo_n @ self.W_obj.T / 0.07
                obj_loss = obj_loss + F.cross_entropy(sim, labels_al)
            terms["obj"] = obj_loss * 0.5
        else:
            terms["obj"] = q.new_zeros(())

        # 5. background suppression (RA lambda_bg=0.05, top-5)
        # RA relsgg.py: (per_slot * w_slot).sum() / w_slot.sum() —— w 是归一化
        # 权重（PU 估计或 neg_weight 兜底），不是额外缩放。无 el_sampler 统计
        # 时走兜底分支，w_slot = neg_weight = 0.3 恒定，等效 per_slot.mean()。
        neg_slots = valid_mask[0] & ~has_gt
        if neg_slots.any():
            lg_neg = out["logits"][neg_slots]
            kk = min(5, lg_neg.shape[-1])
            per_slot = F.softplus(lg_neg.topk(kk, dim=-1).values).mean(-1)
            w_slot = torch.full_like(per_slot, self.sampler.neg_weight)
            terms["bg"] = (per_slot * w_slot).sum() / w_slot.sum().clamp(min=1e-6)
        else:
            terms["bg"] = q.new_zeros(())

        # 6/7. geo + rel(focal) from the sampler stage. 几何特征与 f_sub/f_obj
        # 由 _forward_pairs 整批算好后经 out 传入（原实现每图重算一遍 N×N）。
        gs = out.get("geo_scores")
        if gs is None:
            b = boxes_b.shape[0]
            gs = self.sampler.geo_scorer(RelGeomEncoder.features(
                boxes_b.unsqueeze(2).expand(b, N, N, 4),
                boxes_b.unsqueeze(1).expand(b, N, N, 4)).reshape(b, N * N, NUM_GEO)).squeeze(-1)
        geo_scores = gs[0]
        ar = torch.arange(N, device=device)
        valid_box = ar < n_b
        pair_valid = ((valid_box.unsqueeze(0) & valid_box.unsqueeze(1))
                      & (ar.unsqueeze(0) != ar.unsqueeze(1))).reshape(-1)
        n_valid = pair_valid.float().sum().clamp(min=1.0)
        geo_bce = F.binary_cross_entropy_with_logits(geo_scores, is_gt[0].float(), reduction="none")
        terms["geo"] = (geo_bce * pair_valid.float()).sum() / n_valid

        zs, zo = out.get("zs"), out.get("zo")
        if zs is None:
            zs, zo = self.sampler.f_sub(v_obj), self.sampler.f_obj(v_obj)
        z_s = zs[0][out["sub_idx"]]
        z_o = zo[0][out["obj_idx"]]
        rel_scores1 = (z_s * z_o).sum(-1) / (self.sampler.rel_dim ** 0.5)
        gt_slot = gt_hot.any(-1)
        w_e = torch.where(gt_slot, torch.ones_like(rel_scores1),
                          torch.full_like(rel_scores1, self.sampler.neg_weight))
        p = torch.sigmoid(rel_scores1)
        p_t = gt_slot.float() * p + (1 - gt_slot.float()) * (1 - p)
        rel_bce = F.binary_cross_entropy_with_logits(rel_scores1, gt_slot.float(), reduction="none") \
            * (1 - p_t).pow(2.0)
        m = out["valid_mask"].float()
        terms["rel"] = (rel_bce * w_e * m).sum() / m.sum().clamp(min=1.0)

        return terms

    # ------------------------------------------------------- field helpers

    def _select_candidates(self, logits: torch.Tensor, gtp: Optional[torch.Tensor],
                           valid: torch.Tensor, m: int, training: bool,
                           gt_slots: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                           ) -> torch.Tensor:
        """Top-M predicate columns per pair; every GT predicate forced in during training."""
        B, K, V = logits.shape
        scores = logits.clone()
        if training and gtp is not None:
            ok = (gtp >= 0).unsqueeze(-1)
            idx = gtp.clamp(min=0).unsqueeze(-1)
            forced = scores.scatter(2, idx, torch.full_like(idx, float("inf"), dtype=scores.dtype))
            scores = torch.where(ok, forced, scores)
        # gtp 是 last-wins 的单标签，多谓词对被覆盖掉的那些在这里补回来。
        if training and gt_slots is not None:
            for b, (si, pi) in enumerate(gt_slots):
                if si.numel():
                    scores[b, si, pi] = float("inf")
        scores = scores.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        return scores.topk(min(m, V), dim=-1).indices

    # ------------------------------------------------------------ compat

    def set_train_boxes(self, boxes, gt_pairs=None):  # v3-compat no-op kept for trainer
        pass

    @torch.no_grad()
    def score(self, boxes: torch.Tensor, topk: Optional[int] = None, img_idx: int = 0,
              cls: Optional[torch.Tensor] = None,
              pred_topk: Optional[int] = None):
        """Eval scoring on captured features. boxes [N,4] cxcywh normalised.

        img_idx：批 capture 时本图在批内的序号（validator 逐图调用按序传）。
        cls：与 boxes 对齐的物体类别 id；PredCls/ SGCls 必须传（GT 类别是定义输入）。
        pred_topk：额外返回每个 pair 的 top-k 谓词 id [P,k]（ngR 口径指标用）；
        返回值随之变成 ``(rows, top_ids)``。
        Returns [P,5]: sub, obj, pred_id, fused, exist."""
        cap = self._captured
        if cap is None:
            raise RuntimeError("score() requires a captured forward first.")
        # 验证器可能用 half() 副本：打分链全程 fp32（余弦/排序对精度敏感）
        self.float()
        feats_list = cap["feats"]
        base = feats_list[img_idx if img_idx < len(feats_list) else 0].float()
        boxes = boxes.to(base.device).detach().float()
        if boxes.shape[0] < 2:
            rows = boxes.new_zeros((0, 5))
            return (rows, boxes.new_zeros((0, pred_topk), dtype=torch.long)) \
                if pred_topk is not None else rows
        out = self._compute(base, boxes.unsqueeze(0), gt_list=None,
                            eval_budget=topk or self.eval_budget,
                            box_cls=cls.unsqueeze(0) if cls is not None else None)
        pred_logit = out["logits"]                     # [K, V] scaled cosine
        pair_logit = out["rel_logits"]                 # [K]   inner/16
        fused = torch.sigmoid(pred_logit + pair_logit.unsqueeze(-1))  # a=1,b=0 until Platt
        best = fused.argmax(-1)
        best_f = fused.gather(-1, best.unsqueeze(-1)).squeeze(-1)
        exist = torch.sigmoid(pair_logit)
        keep = out["valid_mask"]
        rows = torch.stack([out["sub_idx"][keep].float(), out["obj_idx"][keep].float(),
                            best[keep].float(), best_f[keep], exist[keep]], dim=1)
        if pred_topk is None:
            return rows
        ids = fused.topk(min(int(pred_topk), fused.shape[-1]), dim=-1).indices[keep]
        return rows, ids

    @torch.no_grad()
    def score_all(self, boxes: torch.Tensor, topk: Optional[int] = None,
                  img_idx: int = 0, field_ablate: Optional[str] = None,
                  return_field: bool = False, cls: Optional[torch.Tensor] = None):
        """Full eval output on captured features.

        Returns ``(sub, obj, valid, logits, rel_logits)``; with
        ``return_field=True`` appends ``(field, cand_idx)`` for RES/field
        diagnostics (MD §3.3).
        """
        cap = self._captured
        if cap is None:
            raise RuntimeError("score_all() requires a captured forward first.")
        self.float()
        base = cap["feats"][img_idx if img_idx < len(cap["feats"]) else 0].float()
        boxes = boxes.to(base.device).detach().float()
        if boxes.shape[0] < 2:
            dev = boxes.device
            V = int(self.vocab_head.W.shape[0]) if self.vocab_head.W.numel() else 0
            ret = (torch.zeros(0, dtype=torch.long, device=dev),
                   torch.zeros(0, dtype=torch.long, device=dev),
                   torch.zeros(0, dtype=torch.bool, device=dev),
                   torch.zeros((0, V), device=dev),
                   torch.zeros(0, device=dev))
            return ret + ((None, None) if return_field else ())
        out = self._compute(base, boxes.unsqueeze(0), gt_list=None,
                            eval_budget=topk or self.eval_budget,
                            box_cls=cls.unsqueeze(0) if cls is not None else None,
                            field_ablate=field_ablate)
        ret = (out["sub_idx"], out["obj_idx"], out["valid_mask"],
               out["logits"], out["rel_logits"])
        if return_field:
            return ret + (out.get("field"), out.get("cand_idx"))
        return ret

    def extra_repr(self) -> str:
        return (f"n_predicates={self.n_predicates}, text_dim={self.text_dim}, "
                f"feat_layers={self.feat_layers}, budgets=({self.geo_budget},{self.final_budget},{self.eval_budget})")
