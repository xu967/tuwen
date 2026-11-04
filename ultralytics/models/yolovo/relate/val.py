"""Relate validator：指标 mixin + 按任务工厂（SGDet/SGCls 双协议）。

设计文档 §7：
- 指标分层：存在性 AP、谓词 top-1 acc、R@20/50/100 + mR@50/100 + F1@50
  （graph-constrained SGDet：双端 IoU≥0.5 + 谓词相同）、PairRecall 门禁、GT_MRR。
- SGCls（GT 框 oracle）模式开关（args.protocol）。
- NMS 复用任务验证器 postprocess；关系打分调 YOLOVORelate.score（capture 特征）。
- 指标不写进检测 mAP。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou


def _xywh2xyxy(b: torch.Tensor) -> torch.Tensor:
    """归一化/像素 xywh -> xyxy（box_iou 需要 xyxy）。"""
    if b is None or b.numel() == 0:
        return b
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 2] / 2
    y2 = b[:, 1] + b[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


class RelateEvaluator:
    """分层指标累计器（纯 torch，图粒度）。"""

    def __init__(self, iou_thr: float = 0.5, match_by_index: bool = False):
        self.iou_thr = iou_thr
        # PredCls 下 pred_boxes 就是 GT 框，按索引直接配对即可。走 IoU 会因重叠框
        # 把预测行配到另一对（实测压低 acc 约 2pp）。
        self.match_by_index = match_by_index
        self.reset()

    def reset(self):
        self.n_imgs = 0
        self.pred_correct = 0
        self.pred_total = 0
        self.pa_correct = 0   # v3 predicate_acc 分子
        self.pa_total = 0     # v3 predicate_acc 分母（两端框匹配的标注对）
        self.pa_pair_correct = 0  # pair-level：同一有序对的多条谓词只算一次
        self.pa_pair_total = 0
        self.det_pair_hits = 0  # v3 DetPairRecall
        self.det_pair_total = 0
        self.recall_hits = {k: 0 for k in (20, 50, 100)}
        self.mrecall_hits = {}  # pid -> {k: hits}
        self.mrecall_total = {}  # pid -> count
        self.gt_total = 0
        self.gt_ranks = []
        self.pair_recall_hits = 0
        self.pair_recall_total = 0
        # 存在性 AP：正例按 GT 有序对去重取最高分，未被任何预测行匹配上的对
        # 记入 exist_missed 留在分母里（原实现只除以"预测出的正例数"，AP 被放大
        # 约 1/PairRecall 倍）。
        self._exist_pos = []
        self._exist_neg = []
        self.exist_missed = 0
        self.pair_topk_hits = {}   # k -> 命中数（该对 top-k 谓词命中任一 GT 谓词）
        self.pair_topk_total = 0
        self.pair_topk_k = 0       # 列数固定，避免某轮没命中时整列消失导致 csv 错位

    def update(self, pred_boxes, pred_rels, gt_boxes, gt_rels, topk=100, pred_topk=None):
        """v3 指标更新。pred_rels [P,5] = (s, o, pred_id, 融合分, 存在性分)。

        ``pred_topk [P,k]``：每个预测行的 top-k 谓词 id，用于 pair-level top-k 补充指标。
        设备统一：入口全部拉到 CPU。
        - predicate_acc：分母=候选中两端框 IoU≥0.5 的标注行；分子=其中最高谓词==标注
        - pair_acc：按唯一有序对去重，命中该对任一 GT 谓词即对
        - pair_existence_AP：正例按有序对去重，未被匹配的 GT 对计入分母
        - DetPairRecall：GT 两端框是否都被检测出（60 框内）；PairRecall：有序对在候选中
        """
        self.n_imgs += 1
        pred_boxes = pred_boxes.cpu() if torch.is_tensor(pred_boxes) else pred_boxes
        pred_rels = pred_rels.cpu() if torch.is_tensor(pred_rels) else pred_rels
        gt_boxes = gt_boxes.cpu() if torch.is_tensor(gt_boxes) else gt_boxes
        gt_rels = gt_rels.cpu() if torch.is_tensor(gt_rels) else gt_rels
        pred_topk = pred_topk.cpu() if torch.is_tensor(pred_topk) else pred_topk
        R = gt_rels.shape[0] if gt_rels is not None and gt_rels.numel() else 0
        self.gt_total += R
        # 本图唯一有序对 -> 该对的全部 GT 谓词
        pair_pids = {}
        for r in range(R):
            pair_pids.setdefault((int(gt_rels[r, 0]), int(gt_rels[r, 1])), set()).add(
                int(gt_rels[r, 2]))
        if pred_rels is None or pred_rels.numel() == 0:
            self.gt_ranks.extend([float("inf")] * R)  # 未召回的 GT 不该给 MRR 贡献分
            # 无预测时也要进分母，否则"一条都没预测出来"的图会被整体剔除
            self.pair_recall_total += R
            self.pa_pair_total += len(pair_pids)
            self.pair_topk_total += len(pair_pids)
            self.exist_missed += len(pair_pids)
            for r in range(R):
                pid = int(gt_rels[r, 2])
                self.mrecall_total[pid] = self.mrecall_total.get(pid, 0) + 1
            # 双召回仍计（候选空时两端召回可能非零）
            if R:
                self._det_pair_recall(gt_boxes, gt_rels, pred_boxes, R, only_det=True)
            return

        order = pred_rels[:, 3].argsort(descending=True)  # 融合分排序（R@K 用）
        pred_sorted = pred_rels[order]

        ps, po = pred_sorted[:, 0].long(), pred_sorted[:, 1].long()
        pp = pred_sorted[:, 2].long()
        if pred_topk is not None:
            pred_topk = pred_topk[order]
            self.pair_topk_k = max(self.pair_topk_k, int(pred_topk.shape[1]))
        exist_scores = pred_sorted[:, 4] if pred_sorted.shape[1] >= 5 else pred_sorted[:, 3]
        # 同一张 IoU 矩阵下面到处复用。PredCls 下 pred 框就是 GT 框，用单位阵等价于
        # 按索引配对，且不会被重叠框配到另一对。只比形状不够：检测框数与 GT 框数
        # 偶然相等时会把两套不同的框按索引错配。
        if R and self.match_by_index and torch.is_tensor(pred_boxes) \
                and pred_boxes.shape == gt_boxes.shape and torch.equal(pred_boxes, gt_boxes):
            iou = torch.eye(gt_boxes.shape[0], dtype=torch.float32)
        else:
            iou = box_iou(_xywh2xyxy(gt_boxes), _xywh2xyxy(pred_boxes)) if R else None

        matched_any = torch.zeros(pred_sorted.shape[0], dtype=torch.bool)
        if R:
            # DetPairRecall：GT 两端框各自是否有任一检测框 IoU≥thr（不看配对）
            endp_ok = ((iou >= self.iou_thr).any(dim=1))  # [R_gt] 每个GT框
            for r in range(R):
                gs, go = int(gt_rels[r, 0]), int(gt_rels[r, 1])
                self.det_pair_total += 1
                self.det_pair_hits += int(bool(endp_ok[gs]) and bool(endp_ok[go]))

            # PairRecall + predicate_acc（v3 分母）：两端框匹配的标注行
            for r in range(R):
                gs, go, gp = int(gt_rels[r, 0]), int(gt_rels[r, 1]), int(gt_rels[r, 2])
                end_match = (iou[gs, ps] >= self.iou_thr) & (iou[go, po] >= self.iou_thr)
                self.pair_recall_total += 1
                paired = bool(end_match.any())
                self.pair_recall_hits += int(paired)
                self.mrecall_total[gp] = self.mrecall_total.get(gp, 0) + 1
                if paired:
                    self.pa_total += 1
                    # 分子=该对最高分谓词==标注：取首个匹配行（融合分排序下的 argmax 记录）
                    self.pa_correct += int(int(pp[end_match.nonzero()[0, 0]]) == gp)

            # 唯一有序对：pair_acc、pair-level top-k、存在性 AP 的正例都按对去重。
            # 行级 predicate_acc 把「同一有序对的第 2+ 谓词」永远算错（val 19.0% 的行）。
            for (gs, go), pids in pair_pids.items():
                m = (iou[gs, ps] >= self.iou_thr) & (iou[go, po] >= self.iou_thr)
                hit = m.nonzero().flatten()
                self.pa_pair_total += 1
                self.pair_topk_total += 1
                if not hit.numel():
                    self.exist_missed += 1
                    continue
                top = int(hit[0])
                self.pa_pair_correct += int(int(pp[top]) in pids)
                self._exist_pos.append(float(exist_scores[top]))
                matched_any |= m
                if pred_topk is not None:
                    ids = pred_topk[top].tolist()
                    for k in range(1, len(ids) + 1):
                        if set(ids[:k]) & pids:
                            self.pair_topk_hits[k] = self.pair_topk_hits.get(k, 0) + 1

        # 三元组 R@K / mR@K（graph-constrained，融合分排序）
        for r in range(R):
            pid = int(gt_rels[r, 2])
            gs, go, gp = int(gt_rels[r, 0]), int(gt_rels[r, 1]), int(gt_rels[r, 2])
            ok = (iou[gs, ps] >= self.iou_thr) & (iou[go, po] >= self.iou_thr) & (pp == gp)
            pos = ok.nonzero(as_tuple=False)
            rank = int(pos[0, 0]) + 1 if pos.numel() else float("inf")
            self.gt_ranks.append(rank)
            for k in self.recall_hits:
                if rank <= k:
                    self.recall_hits[k] += 1
                    self.mrecall_hits.setdefault(pid, {})[k] = self.mrecall_hits.get(pid, {}).get(k, 0) + 1

        # 存在性 AP 的负例：没被任何 GT 有序对匹配上的预测行（正例已按对去重）
        self._exist_neg.extend(exist_scores[~matched_any].tolist())

    def _det_pair_recall(self, gt_boxes, gt_rels, pred_boxes, R, only_det=False):
        """候选空时的 DetPairRecall 兜底统计。"""
        iou = box_iou(_xywh2xyxy(gt_boxes), _xywh2xyxy(pred_boxes))
        endp_ok = (iou >= self.iou_thr).any(dim=1)
        for r in range(R):
            gs, go = int(gt_rels[r, 0]), int(gt_rels[r, 1])
            self.det_pair_total += 1
            self.det_pair_hits += int(bool(endp_ok[gs]) and bool(endp_ok[go]))

    @staticmethod
    def _ap(pos: List[float], neg: List[float], missed: int = 0) -> float:
        """rank-based AP：正样本在合并排序中的平均精度（PR 曲线下面积近似）。

        ``missed`` 是没被任何预测行匹配上的 GT 正例数。它们不进排序，但必须留在
        分母里——原实现除以"预测出的正例数"，漏检越多 AP 越高（约放大 1/PairRecall）。
        """
        if not pos and not missed:
            return 0.0
        import numpy as np

        scores = np.array(pos + neg)
        labels = np.array([1] * len(pos) + [0] * len(neg))
        order = np.argsort(-scores)
        labels = labels[order]
        tp = np.cumsum(labels)
        precision = tp / (np.arange(len(labels)) + 1)
        return float((precision * labels).sum() / max(1, len(pos) + missed))

    def stats(self) -> Dict:
        out = {"n_imgs": self.n_imgs, "n_gt": self.gt_total}
        if self.gt_total:
            for k, v in self.recall_hits.items():
                out[f"R@{k}"] = v / self.gt_total
            # mR@K（含按 support 分桶：1 / 2-4 / 5-19 / 20+，桶内宏平均）
            for k in (50, 100):
                buckets = {"le1": [], "2_4": [], "5_19": [], "ge20": []}
                for pid, total in self.mrecall_total.items():
                    h = self.mrecall_hits.get(pid, {}).get(k, 0)
                    r = h / total
                    b = ("le1" if total <= 1 else "2_4" if total <= 4
                         else "5_19" if total <= 19 else "ge20")
                    buckets[b].append(r)
                per_class = [r for vals in buckets.values() for r in vals]
                out[f"mR@{k}"] = sum(per_class) / len(per_class) if per_class else 0.0
                for b, vals in buckets.items():
                    out[f"mR@{k}_sup_{b}"] = sum(vals) / len(vals) if vals else 0.0
            r50 = out.get("R@50", 0.0)
            mr50 = out.get("mR@50", 0.0)
            out["F1@50"] = 2 * r50 * mr50 / max(1e-9, r50 + mr50)
            out["pair_existence_AP"] = self._ap(self._exist_pos, self._exist_neg, self.exist_missed)
            out["predicate_acc"] = self.pa_correct / max(1, self.pa_total)  # v3 分母
            out["pa_denominator"] = self.pa_total
            out["pair_acc"] = self.pa_pair_correct / max(1, self.pa_pair_total)
            out["pair_acc_denominator"] = self.pa_pair_total
            # 每对 top-k 谓词的命中率：标准 graph-constrained R@K 只取 argmax，
            # 这一组是 ngR@K 口径的补充指标，不能和 R@50 混比。列数固定为 pair_topk_k，
            # 否则某轮没有低 k 命中时整列消失，results.csv 的表头会对不上。
            for k in range(1, self.pair_topk_k + 1):
                out[f"pair_acc@{k}"] = self.pair_topk_hits.get(k, 0) / max(1, self.pair_topk_total)
            out["DetPairRecall"] = self.det_pair_hits / self.det_pair_total if self.det_pair_total else 0.0
            out["PairRecall"] = (
                self.pair_recall_hits / self.pair_recall_total if self.pair_recall_total else 0.0
            )
            ranks = self.gt_ranks
            out["GT_MRR"] = sum(1.0 / r for r in ranks if r != float("inf")) / max(1, len(ranks))
        return out


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.engine.validator import BaseValidator as _TypedValBase
else:
    _TypedValBase = object


class RelateMetricsMixin(_TypedValBase):  # type: ignore[misc,valid-type]
    # score() 顺带返回每对 top-k 谓词，用于 pair_acc@k（ngR 口径）补充指标。
    _PAIR_TOPK = 5
    """覆写 BaseValidator 虚方法：只算关系指标，不算检测 mAP。"""

    relate_info: Optional[dict] = None

    # ---- 生命周期 ------------------------------------------------------

    def init_metrics(self, model=None):
        """读 relate_info、断言、编码校验；准备 evaluator。"""
        super().init_metrics(model)  # 父类初始化不可跳过（end2end/iouv/seen 等，postprocess 依赖）
        from ultralytics.models.yolovo.relate.wiring import validate_relate_info

        # 模型优先级：init_metrics(model) 传入（训练期）> self.relate_model_ref > data.relate_model
        model = model or getattr(self, "relate_model_ref", None)
        if model is None and isinstance(self.data, dict):
            model = self.data.get("relate_model")
        info = self.relate_info or getattr(model, "relate_info", None)
        if not info:
            raise RuntimeError("rel.val 需要 relate_info（validator.relate_info 或模型属性）。")
        # W 形状校验
        from ultralytics.models.yolovo.relate.head import YOLOVORelate

        for m in (model.modules() if model is not None else []):
            if isinstance(m, YOLOVORelate):
                validate_relate_info(info, tuple(m.vocab_head.W.shape))
        self.relate_model_ref = model  # postprocess 复用
        # 验证模型（EMA 副本）的 input_cache 在序列化时被清空；前向需要 text 缓存
        if model is not None and hasattr(model, "set_input_text") and not model.has_input("text"):
            was_training = model.training
            model.eval()
            model.set_input_text()
            if was_training:
                model.train()
        # Preserve a protocol set by RelateNamespace.val before init_metrics;
        # init_metrics used to overwrite it with args (which has no protocol).
        self.protocol = getattr(self, "protocol", None) or \
            getattr(self.args, "protocol", "sgdet") or "sgdet"
        self.evaluator = RelateEvaluator(match_by_index=self.protocol == "sgcls")
        # PredCls：SGDet 之外并行记一份。pred 框恒为 GT 框，按索引配对。
        self.evaluator_pc = RelateEvaluator(match_by_index=True)
        self._rel_head = None
        self.rel_stats = {}
        LOGGER.info(f"[RelateVal] protocol={self.protocol}, V={len(info['rnames'])}")

    def preprocess(self, batch):
        """forward 前开 capture（下一 postprocess 消费后关闭）。"""
        batch = super().preprocess(batch)
        model = getattr(self, "relate_model_ref", None)
        if model is not None:
            from ultralytics.models.yolovo.relate.head import YOLOVORelate

            for m in model.modules():
                if isinstance(m, YOLOVORelate):
                    m.capture = True
                    break
        return batch

    def postprocess(self, preds):
        """NMS + capture 特征打分（特征由紧前的 forward 捕获）。

        签名对齐 DetectionValidator.postprocess(preds)。
        归一化尺寸从 capture 特征反推：P3 网格 × stride = letterbox 输入尺寸。
        """
        from ultralytics.models.yolovo.relate.head import YOLOVORelate

        results = super().postprocess(preds)
        model = getattr(self, "relate_model_ref", None)
        rel = None
        if model is not None:
            for m in model.modules():
                if isinstance(m, YOLOVORelate):
                    rel = m
        if rel is None:
            return results
        rel.capture = False
        # update_metrics 的截断域要和这里一致（max_objects 不是常量 60）
        self._rel_max_objects = int(getattr(rel, "max_objects", 60))
        cap = rel._captured
        ih, iw = 640, 640
        if cap and cap.get("feats"):
            fh, fw = cap["feats"][0].shape[-2:]
            # capture 存的是 base_idx（P4）的特征，不是第 0 个接线层
            stride = rel.feat_strides[rel.base_idx]
            ih, iw = fh * stride, fw * stride
        for bi, r in enumerate(results if isinstance(results, list) else [results]):
            if not isinstance(r, dict) or "bboxes" not in r:
                continue
            b = r["bboxes"]  # xyxy（letterbox 输入坐标系）
            conf = r.get("conf")
            cls = r.get("cls")  # 检测类别；postprocess 不一定提供，缺省则不打类别
            # 按检测置信度留 60 框再打分（max_objects 契约）
            if conf is not None and b.shape[0] > rel.max_objects:
                keep = conf.squeeze(-1).topk(rel.max_objects).indices
                b = b[keep]
                if cls is not None:
                    cls = cls[keep]
            if b.numel() == 0:
                r["rel"] = torch.zeros((0, 5))
                continue
            wh = (b[:, 2:4] - b[:, 0:2]).clamp(min=0)
            cx = (b[:, 0] + b[:, 2]) / 2
            cy = (b[:, 1] + b[:, 3]) / 2
            boxes = torch.stack([cx / iw, cy / ih, wh[:, 0] / iw, wh[:, 1] / ih], dim=1)
            rel_out = (rel.score(boxes.cpu(), img_idx=bi,
                                 cls=cls.cpu().reshape(-1) if cls is not None else None)
                       if boxes.shape[0] >= 2 else torch.zeros((0, 5)))
            r["rel"] = rel_out
        return results

    def update_metrics(self, preds, batch):
        """从 results 取 rel 打分与 GT relations 更新 evaluator。"""
        import torch as _t

        for i, det in enumerate(preds if isinstance(preds, list) else [preds]):
            r = self.results[i] if hasattr(self, "results") and i < len(self.results) else det
            rel_out = getattr(r, "rel", None) if not isinstance(r, dict) else r.get("rel")
            # GT
            idx = batch["batch_idx"].long()
            relations = batch.get("relations") or []
            # 本图 GT
            # 框：从 batch labels 切本图（归一化 xywh）
            labels = batch["bboxes"][idx == i] if "bboxes" in batch else _t.zeros((0, 4))
            # GT 类别与 labels 对齐；训练期 batch["cls"] 被重映射成局部文本 id，
            # preprocess_batch 存的 cls_global 才是物体类 id。
            cls_all = batch.get("cls_global", batch.get("cls"))
            cls_i = None
            if cls_all is not None and cls_all.numel() == idx.numel():
                cls_i = cls_all[idx == i].reshape(-1).long().cpu()
                if cls_i.numel() != labels.shape[0]:
                    cls_i = None
            rels_i = relations[i] if i < len(relations) else _t.zeros((0, 3))
            if rels_i is not None and len(rels_i):
                rels_i = rels_i.to(labels.device).float()
            # 预测框（graph-constrained 输入）：dict 形态（postprocess 输出 xyxy→归一化 xywh）
            pred_rels = rel_out if rel_out is not None else _t.zeros((0, 4))
            if isinstance(det, dict) and det.get("bboxes") is not None and len(det["bboxes"]):
                b = det["bboxes"]
                conf = det.get("conf")
                max_obj = int(getattr(self, "_rel_max_objects", 60))
                if conf is not None and b.shape[0] > max_obj:  # 与 postprocess 截断一致
                    keep = conf.squeeze(-1).topk(max_obj).indices
                    b = b[keep]
                wh = (b[:, 2:4] - b[:, 0:2]).clamp(min=0)
                cx = (b[:, 0] + b[:, 2]) / 2
                cy = (b[:, 1] + b[:, 3]) / 2
                ih, iw = batch["img"].shape[-2:]
                pred_boxes = _t.stack([cx / iw, cy / ih, wh[:, 0] / iw, wh[:, 1] / iw], dim=1).cpu()
            else:
                # 没有检测框 ≠ "预测框就是 GT 框"。用 GT 会让 DetPairRecall 恒 1.0。
                pred_boxes = _t.zeros((0, 4))
            # PredCls（文献口径；本实现只喂框、不预测物体类别，SGCls 与 PredCls 同义）：
            # 用 GT 框重新打分。protocol=sgcls 时它替代 SGDet 主指标；否则在同一次
            # 前向里额外记一份 pc_*——SGDet 被冻结检测器压死时，这是唯一能判断关系头
            # 本身好坏的口径。
            rel = self._rel_head  # 逐图重扫 model.modules() 太慢，init_metrics 里缓存一次
            if rel is None and labels.shape[0] >= 2:
                model = getattr(self, "relate_model_ref", None) or getattr(self, "model", None)
                model = model.module if hasattr(model, "module") else model
                if model is not None:
                    rel = next((m for m in model.modules()
                                if m.__class__.__name__ == "YOLOVORelate"), None)
                self._rel_head = rel
            sgcls = getattr(self, "protocol", "sgdet") == "sgcls"
            res = top_k = None
            if rel is not None and labels.shape[0] >= 2:
                res, top_k = rel.score(labels.cpu(), img_idx=i, cls=cls_i,
                                       pred_topk=self._PAIR_TOPK)
            # PredCls 口径恒记一份（<2 框的图没有 pair 可打分，但 GT 行仍要计入）。
            # sgcls 下主轴也用 GT 框，两份数值相同但列集合因此不随协议变化。
            pc_rels = res if res is not None else _t.zeros((0, 5))
            self.evaluator_pc.update(labels.cpu(), pc_rels, labels, rels_i, pred_topk=top_k)
            if sgcls:
                if res is not None:
                    pred_rels, pred_boxes = res, labels.cpu()
                self.evaluator.update(pred_boxes, pred_rels, labels, rels_i, pred_topk=top_k)
            else:
                self.evaluator.update(pred_boxes, pred_rels, labels, rels_i)

    def get_stats(self):
        self.rel_stats = self.evaluator.stats()
        d = dict(self.rel_stats)
        if getattr(self, "evaluator_pc", None) is not None and self.evaluator_pc.n_imgs:
            for k, v in self.evaluator_pc.stats().items():
                d[f"pc_{k}"] = v
        # checkpoint 选择指标：优先 PredCls 的 R@50。SGDet 的 R@50 在冻结检测器下恒≈0，
        # 用它选等于在噪声里挑冠军；F1@50 里的 mR@50 分母有 62% 是 ≤2 样本的类，同样
        # 由噪声主导。不提供 fitness 时父类回退到最后 batch 负损失——空标注图损失≈0 会胜出。
        d["fitness"] = float(d["pc_R@50"] if "pc_R@50" in d else d.get("R@50", 0.0))
        self.rel_stats = d  # print_results 打的是这份，pc_* 才看得到
        return d

    def finalize_metrics(self):
        pass

    # ---- 关系可视化（docs/plans/2026-10-08-relate-plotting-integration.md）----

    _PANEL_W = 460
    _TITLE_H = 34

    _DEJAVU_PATHS = (
        "/usr/share/fonts/dejavu",
        "/usr/share/fonts/truetype/dejavu",
    )

    @classmethod
    def _pil_font(cls, size: int = 12, bold: bool = False):
        """DejaVu（Bold/regular 双探测路径，RA 同款）→ PIL default 三级降级。"""
        import os

        from PIL import ImageFont

        name = f"DejaVuSans{'-Bold' if bold else ''}.ttf"
        for d in cls._DEJAVU_PATHS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        try:
            return ImageFont.truetype(name, size)  # 系统字体搜索路径兜底
        except Exception:  # noqa: BLE001 - 环境无 DejaVu 时降级
            return ImageFont.load_default()

    @staticmethod
    def _label_bg(draw, xy, text, font, pad=2):
        """白底黑字标签，坐标钳位画布内。"""
        x, y = xy
        w = draw.textlength(text, font=font) + 2 * pad
        h = (font.size if hasattr(font, "size") else 10) + 2 * pad
        x = max(0, min(x - w / 2, draw.im.size[0] - w))
        y = max(0, min(y - h / 2, draw.im.size[1] - h))
        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 230))
        draw.text((x + pad, y + pad), text, fill=(0, 0, 0), font=font)

    # Okabe-Ito 色盲安全调色板（RA e3_panels.py 同款）
    _REL_PALETTE = [
        (230, 159, 0), (86, 180, 233), (0, 158, 115), (213, 94, 0),
        (0, 114, 178), (204, 121, 167), (140, 86, 75), (127, 127, 127),
    ]

    def _plot_rel_panel(self, img_pil, boxes_n, cats, cat_names, edges, rnames, title, colour_of=None, max_edges=7):
        """单 panel：used 框 + 箭头边 + 谓词标签（对齐 RA draw_panel）。

        颜色语义：颜色=【实例号】（used 顺序发色，调色板循环），不是类别色——
        边色 = sub 实例色，可视觉追踪到发起框；框标签底色 = 实例色。
        可读性预算：边数 > max_edges 时按 fused 分数（edges 第4列）截断，
        超预算提示 "showing top-K of N"。RA 原版直接弃画 >7 边的图，这里软截断。

        Args:
            img_pil: PIL 原图。
            boxes_n: [N,4] 归一化 xyxy。
            cats: [N] 类别 id。
            edges: [E,>=3] (sub,obj,pred_id[,fused])。
            rnames: 谓词名 dict。
            title: 标题条文本。
            colour_of: 保留兼容入参，None 时用实例调色板。
            max_edges: 可读性边数上限（RA 规则 7）。
        """
        import math

        from PIL import Image, ImageDraw

        edges = edges.tolist() if torch.is_tensor(edges) else [list(e) for e in (edges or [])]
        n_total = len(edges)
        if n_total > max_edges and all(len(e) >= 4 for e in edges):
            edges = sorted(edges, key=lambda e: -float(e[3]))[:max_edges]  # fused 降序截断
        used = sorted({int(e[0]) for e in edges} | {int(e[1]) for e in edges})

        W = self._PANEL_W
        H = round(img_pil.height * W / img_pil.width)
        canvas = Image.new("RGB", (W, H + self._TITLE_H), (250, 250, 250))
        canvas.paste(img_pil.resize((W, H), Image.LANCZOS), (0, self._TITLE_H))
        draw = ImageDraw.Draw(canvas, "RGBA")
        font = self._pil_font(12)
        tfont = self._pil_font(13, bold=True)
        draw.rectangle([0, 0, W, self._TITLE_H - 1], fill=(32, 32, 32))
        t = f"{title}  ({n_total} edges)" if n_total else title
        draw.text((8, 9), t, fill=(255, 255, 255), font=tfont)
        oy = self._TITLE_H

        # 实例独立发色（RA colour_of(bi)）：used 顺序分配调色板
        inst_colour = {bi: self._REL_PALETTE[k % len(self._REL_PALETTE)] for k, bi in enumerate(used)}

        # 框：只画边引用的框；标签底色 = 实例色，黑字（RA 同款）
        centers = {}
        for bi in used:
            if bi >= len(boxes_n):
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes_n[bi]]
            x1, y1, x2, y2 = x1 * W, y1 * H + oy, x2 * W, y2 * H + oy
            c = inst_colour[bi]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
            name = cat_names.get(int(cats[bi]), "?") if bi < len(cats) else "?"
            tag = f"{bi}:{name}"
            tw = draw.textlength(tag, font=font)
            ty = max(oy, y1 - 15)  # 钳位画布内（RA clamp）
            tx = min(max(0, x1), W - tw - 6)
            draw.rectangle([tx, ty, tx + tw + 6, ty + 15], fill=c)
            draw.text((tx + 3, ty + 1), tag, fill=(0, 0, 0), font=font)
            centers[bi] = ((x1 + x2) / 2, (y1 + y2) / 2)
        # 边：sub 中心 -> obj 中心，线色 = sub 实例色（半透明 235，RA 同款）
        for e in edges:
            s, o = int(e[0]), int(e[1])
            if s not in centers or o not in centers:
                continue
            p0, p1 = centers[s], centers[o]
            c = inst_colour[s] + (235,)
            draw.line([p0, p1], fill=c, width=3)
            ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            L, A = 11.0, 0.5
            ux, uy = math.cos(ang), math.sin(ang)
            tip = p1
            draw.polygon([
                tip,
                (tip[0] - L * ux + 6 * uy, tip[1] - L * uy - 6 * ux),
                (tip[0] - L * ux - 6 * uy, tip[1] - L * uy + 6 * ux),
            ], fill=c)
            pid = int(e[2])
            label = str(rnames.get(pid, pid)) if isinstance(rnames, dict) else str(pid)
            tw = draw.textlength(label, font=font)
            mx = min(max(0, (p0[0] + p1[0]) / 2 - tw / 2 - 3), W - tw - 6)
            my = min(max(oy, (p0[1] + p1[1]) / 2 - 8), canvas.size[1] - 17)
            draw.rectangle([mx, my, mx + tw + 6, my + 16], fill=(255, 255, 255, 235))
            draw.text((mx + 3, my + 1), label, fill=(20, 20, 20), font=font)
        if not n_total:
            draw.text((10, oy + 10), "(no edge)", fill=(120, 120, 120), font=font)
        elif n_total > len(edges):
            draw.text((10, oy + 10), f"(showing top-{len(edges)} of {n_total})", fill=(120, 120, 120), font=font)
        return canvas

    def plot_relations(self, batch, preds, rels, ni):
        """GT|PRED 双联关系图：输出 val_batch{ni}_rel.jpg。"""
        from PIL import Image

        rnames = (self.relate_info or {}).get("rnames", {})
        names = self.names or {}
        rows = []
        bs = len(batch["im_file"])
        for i in range(bs):
            img = Image.open(batch["im_file"][i]).convert("RGB")
            ow, oh = img.width, img.height
            # GT panel
            idx = batch["batch_idx"].long().cpu()
            labels = batch["bboxes"][idx == i].float().cpu()  # 归一化 xywh（原图坐标系）
            gt_rel = batch.get("relations") or []
            gt_rel_i = gt_rel[i] if i < len(gt_rel) else torch.zeros((0, 3))
            gt_rel_i = gt_rel_i.float().cpu() if gt_rel_i is not None else torch.zeros((0, 3))
            gt_boxes = _xywh2xyxy(labels) if len(labels) else torch.zeros((0, 4))
            gt_cats = batch["cls"][idx == i].long().flatten().cpu() if "cls" in batch else torch.zeros(0, dtype=torch.long)
            p_gt = self._plot_rel_panel(img, gt_boxes, gt_cats, names, gt_rel_i[:, :3], rnames, f"GT #{i}")
            # PRED panel
            r = preds[i] if i < len(preds) else {}
            rel_out = rels[i]
            if rel_out is None or not isinstance(r, dict) or "bboxes" not in r or r["bboxes"] is None or not len(r["bboxes"]):
                p_pr = self._plot_rel_panel(img, torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long), names, torch.zeros((0, 3)), rnames, f"PRED #{i}")
            else:
                b = r["bboxes"].float().cpu()
                conf = r.get("conf")
                max_obj = 60
                for m in (self.relate_model_ref.modules() if getattr(self, "relate_model_ref", None) is not None else []):
                    from ultralytics.models.yolovo.relate.head import YOLOVORelate

                    if isinstance(m, YOLOVORelate):
                        max_obj = m.max_objects
                        break
                if conf is not None and b.shape[0] > max_obj:  # 复现 postprocess topk 截断，对齐 rel 索引域
                    keep = conf.squeeze(-1).cpu().topk(max_obj).indices
                    b = b[keep]
                # 逆 letterbox：减 pad / 除 gain -> 原图像素 xyxy -> 归一化
                gain, pad = batch["ratio_pad"][i]
                pad_x, pad_y = float(pad[0]), float(pad[1])
                gain = float(gain) if not hasattr(gain, "__len__") else float(gain[0])
                gain_w = gain if not hasattr(batch["ratio_pad"][i][0], "__len__") else float(batch["ratio_pad"][i][0][1])
                pb = b.clone()
                pb[:, [0, 2]] = (pb[:, [0, 2]] - pad_x) / gain_w / ow
                pb[:, [1, 3]] = (pb[:, [1, 3]] - pad_y) / gain / oh
                pcls = r["cls"].long().flatten().cpu()
                if conf is not None and r["cls"].shape[0] > max_obj:
                    pcls = pcls[conf.squeeze(-1).cpu().topk(max_obj).indices]
                edges = rel_out.float().cpu() if rel_out is not None and torch.is_tensor(rel_out) else torch.zeros((0, 5))
                edges = edges[: len(gt_rel_i)]  # 公平预算：PRED 边数 ≤ GT 边数
                p_pr = self._plot_rel_panel(img, pb, pcls, names, edges, rnames, f"PRED #{i}")  # 保留 fused 列供 max_edges 截断
            rows.append((p_gt, p_pr))
        H = max(max(a.height, b.height) for a, b in rows)
        canvas = Image.new("RGB", (self._PANEL_W * 2, H * len(rows)), (0, 0, 0))
        for ri, (a, b) in enumerate(rows):
            canvas.paste(a, (0, ri * H))
            canvas.paste(b, (self._PANEL_W, ri * H))
        canvas.save(self.save_dir / f"val_batch{ni}_rel.jpg", quality=85)

    def plot_predictions(self, batch, preds, ni, max_det=None):
        """剥离 rel 键（根治父类 torch.cat 设备崩点）后再走父类绘图。"""
        rels = []
        for p in preds:
            rels.append(p.pop("rel", None) if isinstance(p, dict) else None)
            if isinstance(p, dict):
                for k, v in list(p.items()):
                    if isinstance(v, torch.Tensor):
                        p[k] = v.cpu()
        super().plot_predictions(batch, preds, ni, max_det)
        if self.args.plots:
            self.plot_relations(batch, preds, rels, ni)

    def print_results(self):
        s = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in self.rel_stats.items())
        LOGGER.info(f"[RelateVal] {s}")

    def get_desc(self) -> str:
        """验证进度条表头：父类写死 Box(P/R/mAP50/mAP50-95)，关系验证既不算也不打印这些列。"""
        return "%22s" % "RelationVal (R@K / mR@K / F1@50 / ExAP)"

    # ---- 检测侧全部置空 --------------------------------------------------

    def _prepare_batch(self, si, batch):
        return {}

    def _prepare_pred(self, pbatch, preds):
        return None


_TASK_VALIDATORS = {}


def make_relate_validator(task: str):
    """按任务 MRO 拼 Relate validator（先例 make_pruned_trainer_class）。"""
    if task in _TASK_VALIDATORS:
        return _TASK_VALIDATORS[task]
    from ultralytics.models.yolovo.obb.val import YOLOVOOBBValidator
    from ultralytics.models.yolovo.pose.val import YOLOVOPoseValidator
    from ultralytics.models.yolovo.segment.val import YOLOVOSegValidator
    from ultralytics.models.yolovo.val import YOLOVOValidator

    base = {
        "detect": YOLOVOValidator,
        "segment": YOLOVOSegValidator,
        "obb": YOLOVOOBBValidator,
        "pose": YOLOVOPoseValidator,
    }[task]
    cls = type(f"Relate{base.__name__}", (RelateMetricsMixin, base), {})
    _TASK_VALIDATORS[task] = cls
    return cls
