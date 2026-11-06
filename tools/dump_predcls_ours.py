"""Dump our relation head's PredCls predictions (GT boxes) for protocol-matched comparison.

Runs the repo's own validator (so scoring is identical to results.csv) and records, per image,
the graph-constrained rows ``[sub, obj, pred_id, fused_score]`` together with the GT triplets.

usage:
  python tools/dump_predcls_ours.py <ckpt> <data_yaml> <out.pt> [--no-cls]

--no-cls : do not feed GT object classes to the head (paper setting: boxes only).
"""
import argparse
import os
import sys

_CUDNN_LIB = "/root/rivermind-data/cudnn9/nvidia/cudnn/lib"
_LD = os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isdir(_CUDNN_LIB) and _CUDNN_LIB not in _LD.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN_LIB + (f":{_LD}" if _LD else "")
    os.execv(sys.executable, [sys.executable, *sys.argv])
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

from ultralytics import YOLOVO  # noqa: E402
from ultralytics.models.yolovo.relate import val as relval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--no-cls", action="store_true")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

records = []
names_queue = []

_orig_update_metrics = None
_orig_eval_update = relval.RelateEvaluator.update


def _patched_update_metrics(self, preds, batch):
    names_queue.extend(list(batch.get("im_file", [])))
    return _orig_update_metrics(self, preds, batch)


def _patched_eval_update(self, pred_boxes, pred_rels, gt_boxes, gt_rels, topk=100, pred_topk=None):
    # only the PredCls evaluator (match_by_index=True) feeds the dump; in protocol=predcls the main
    # evaluator is also match_by_index, so tag the pc instance explicitly via the validator.
    if getattr(self, "_dump_target", False):
        gt = gt_rels[:, :3].long().cpu() if gt_rels is not None and len(gt_rels) else torch.zeros((0, 3), dtype=torch.long)
        pr = pred_rels[:, :4].float().cpu() if pred_rels is not None and len(pred_rels) else torch.zeros((0, 4))
        records.append({"image": names_queue[len(records)] if len(records) < len(names_queue) else None,
                        "n_boxes": int(gt_boxes.shape[0]), "gt": gt, "pred": pr})
    return _orig_eval_update(self, pred_boxes, pred_rels, gt_boxes, gt_rels, topk=topk, pred_topk=pred_topk)


_orig_init_metrics = None


def _patched_init_metrics(self, model=None):
    r = _orig_init_metrics(self, model)
    self.evaluator_pc._dump_target = True
    return r


model = YOLOVO(args.ckpt)
VCls = relval.make_relate_validator("detect")
_orig_update_metrics = VCls.update_metrics
_orig_init_metrics = VCls.init_metrics
VCls.update_metrics = _patched_update_metrics
VCls.init_metrics = _patched_init_metrics
relval.RelateEvaluator.update = _patched_eval_update

if args.no_cls:
    rel = next(m for m in model.model.modules() if m.__class__.__name__ == "YOLOVORelate")
    _orig_score = rel.score

    def _score_nocls(boxes, topk=None, img_idx=0, cls=None, pred_topk=None):
        return _orig_score(boxes, topk=topk, img_idx=img_idx, cls=None, pred_topk=pred_topk)

    rel.score = _score_nocls

stats = model.rel.val(data=args.data, protocol="predcls", batch=args.batch, workers=4,
                      project="/root/rivermind-data/scratch_audit/runs", name="dump_tmp", exist_ok=True)
stats = {k: float(v) for k, v in dict(stats).items() if hasattr(v, "__float__")}
torch.save({"records": records, "validator_stats": stats, "ckpt": args.ckpt, "data": args.data,
            "no_cls": args.no_cls}, args.out)
print(f"dumped {len(records)} images -> {args.out}; validator pc_R@50={stats.get('pc_R@50')}")
