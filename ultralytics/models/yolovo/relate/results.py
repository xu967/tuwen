"""RelateResults: 关系预测结果容器。

Results.__init__ 固定 _keys 五项，直接挂属性会被切片/导出路径丢弃，
必须走子类（设计文档 §7.3）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ultralytics.engine.results import Results


class RelateResults(Results):
    """Results 子类：每图关系三元组 (sub_idx, obj_idx, pred_id, score) + 谓词名表。"""

    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None,
                 relations=None, rnames=None, **kwargs):
        super().__init__(orig_img, path, names, boxes, masks, probs, keypoints, obb, **kwargs)
        self._relations = relations  # [P,4] tensor or None
        self._rnames = rnames or {}

    @property
    def relations(self):
        return self._relations

    @property
    def rnames(self):
        return self._rnames

    def triplets(self, topk: Optional[int] = None, min_score: float = 0.0):
        """可读三元组列表 [(sub_label, pred_name, obj_label, score), ...]。"""
        rel = self._relations
        if rel is None or not len(rel):
            return []
        rel = rel[rel[:, 3] >= min_score]
        if topk:
            rel = rel[rel[:, 3].argsort(descending=True)][:topk]
        cls = self.boxes.cls.tolist() if self.boxes is not None else []
        for row in rel.tolist():
            s, o, pid, sc = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            s_name = self.names.get(int(cls[s]), f"obj{s}") if s < len(cls) else f"obj{s}"
            o_name = self.names.get(int(cls[o]), f"obj{o}") if o < len(cls) else f"obj{o}"
            out.append((s_name, self._rnames.get(pid, f"rel{pid}"), o_name, sc))
        return out

    def plot_rel(self, panel_w: int = 460, max_edges: int = 7):
        """单图关系可视化：实例独立配色 + 箭头边 + 谓词标签，返回 PIL Image（复用 val 绘制）。"""
        from PIL import Image

        from .val import RelateMetricsMixin

        img = Image.fromarray(self.orig_img[..., ::-1])  # BGR -> RGB
        boxes = self.boxes.xyxy if self.boxes is not None else torch.zeros((0, 4))
        h, w = self.orig_shape
        boxes_n = boxes.clone()
        boxes_n[:, [0, 2]] /= w
        boxes_n[:, [1, 3]] /= h
        cats = self.boxes.cls.long() if self.boxes is not None else torch.zeros(0, dtype=torch.long)
        host = RelateMetricsMixin()  # _plot_rel_panel 只用类常量与纯函数，无 validator 状态
        host._PANEL_W = panel_w
        return host._plot_rel_panel(
            img, boxes_n.cpu(), cats.cpu(), self.names,
            self._relations, self._rnames, f"PRED {self.path}", max_edges=max_edges,
        )

    def __repr__(self):
        n = 0 if self._relations is None else len(self._relations)
        return f"RelateResults(boxes={0 if self.boxes is None else len(self.boxes)}, relations={n})"
