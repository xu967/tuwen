"""YOLOVORelatePredictor: 关系预测器（NMS → top-60 → capture 打分）。

设计文档 §7.3：覆写 postprocess 与结果构造；RelateResults 子类携带 relations；
capture 标志在循环内开/关；普通 predict() 不受影响。
"""

from __future__ import annotations

from typing import Optional

import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import LOGGER

MAX_OBJECTS_DEFAULT = 60


class YOLOVORelatePredictor(DetectionPredictor):
    """Detection predictor + relation scoring on NMS-surviving boxes."""

    relate_info: Optional[dict] = None
    rnames_override = None


    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """模型装配完成（设备已定）后再补 YOLOVO 文本缓存，warmup 前生效（同 YOLOVOPredictor 先例）。"""
        if not self.model:
            self.setup_model(model)
        inner = getattr(self.model, "model", self.model)  # AutoBackend -> nn.Module
        if hasattr(inner, "set_input_text") and not inner.has_input("text"):
            names = (getattr(self.relate_info, "get", lambda *a: None)("names") or getattr(inner, "names", None) or {})
            if names:
                inner.set_input_text(list(names.values()))
        yield from super().stream_inference(source, model, *args, **kwargs)

    def preprocess(self, im):
        """forward 前开 capture（同 val 先例），postprocess 消费后关闭。"""
        im = super().preprocess(im)
        inner = getattr(getattr(self, "model", None), "model", None)
        if inner is not None:
            from ultralytics.models.yolovo.relate.head import YOLOVORelate

            for m in inner.modules():
                if isinstance(m, YOLOVORelate):
                    m.capture = True
                    break
        return im

    def postprocess(self, preds, img, orig_imgs):
        """任务 NMS → top-60 截断 → YOLOVORelate.score → 关系结果注入。"""
        from ultralytics.models.yolovo.relate.head import YOLOVORelate

        results = super().postprocess(preds, img, orig_imgs)

        backend = getattr(self, "model", None)
        inner = getattr(backend, "model", backend)  # AutoBackend -> nn.Module
        rel = None
        for m in getattr(inner, "model", []):
            if isinstance(m, YOLOVORelate):
                rel = m
                break
        if rel is None:
            LOGGER.warning("[RelatePredict] 模型中没有 YOLOVORelate 层，跳过关系打分。")
            return results

        rnames = {int(k): str(v) for k, v in (self.relate_info or {}).get("rnames", {}).items()}
        try:
            # capture 特征需要一次 eval 前向——predict 管线已在上面跑过 super().postprocess，
            # 但特征暂存发生在模型前向（已在 inference 里）。此处检查暂存并在需要时重放。
            from ultralytics.models.yolovo.relate.results import RelateResults

            new_results = []
            for bi, r in enumerate(results):
                boxes = r.boxes.xywhn if r.boxes is not None and len(r.boxes) else torch.zeros((0, 4))
                boxes = boxes.cpu() if torch.is_tensor(boxes) else torch.as_tensor(boxes)
                if boxes.shape[0] > MAX_OBJECTS_DEFAULT:
                    conf = r.boxes.conf if r.boxes is not None else None
                    conf = conf.cpu() if torch.is_tensor(conf) else torch.as_tensor(conf)
                    keep = conf.topk(MAX_OBJECTS_DEFAULT).indices
                    boxes = boxes[keep]
                    sub_map = keep
                else:
                    sub_map = None
                if rel._captured is None or boxes.shape[0] < 2:
                    rel_out = torch.zeros((0, 4))
                else:
                    # img_idx 必须传：capture 里是按 batch 顺序存的逐图特征，
                    # 不传就永远用第 0 张图的特征给整批打分。
                    rel_out = rel.score(boxes, img_idx=bi).cpu()
                if sub_map is not None and rel_out.numel():
                    # 恢复原始框索引
                    rel_out[:, 0] = sub_map[rel_out[:, 0].long()].float()
                    rel_out[:, 1] = sub_map[rel_out[:, 1].long()].float()
                rr = RelateResults(
                    orig_img=r.orig_img, path=r.path, names=r.names,
                    boxes=r.boxes.data if r.boxes is not None else None,
                    masks=r.masks.data if r.masks is not None else None,
                    probs=r.probs.data if r.probs is not None else None,
                    keypoints=r.keypoints.data if r.keypoints is not None else None,
                    obb=r.obb.data if r.obb is not None else None,
                    relations=rel_out, rnames=rnames,
                )

                new_results.append(rr)
            return new_results
        finally:
            rel.capture = False

    def write_results(self, i, p, im, s):
        """save=True 时额外写一张关系可视化图（原图+箭头边+谓词标签）。"""
        from pathlib import Path

        s = super().write_results(i, p, im, s)
        if self.args.save:
            r = self.results[i]
            if hasattr(r, "plot_rel"):
                stem = Path(p).stem
                r.plot_rel().save(self.save_dir / f"{stem}_rel.jpg", quality=85)
        return s
