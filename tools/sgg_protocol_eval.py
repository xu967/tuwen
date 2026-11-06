"""Protocol-matched SGG evaluation (paper 2609.12552 §5.1 definitions) over dumped predictions.

Input: a ``.pt`` dump with a list of per-image records::

    {"image": str, "n_boxes": int,
     "gt": LongTensor[G,3]  (sub, obj, pred_id),
     "pred": FloatTensor[P,4] (sub, obj, pred_id, score)}   # graph-constrained: one row per ordered pair

Metrics (ground-truth boxes, graph-constrained):
  R@K      per-image recall of GT triplets among the K highest-scoring predicted triplets, averaged over images
  mR@K     mean over predicates of per-predicate recall (recall of each predicate pooled over all its GT triplets)
  F1@K     harmonic mean of R@K and mR@K
  R@K_micro  total hits / total GT (the results.csv convention of this repo, kept for continuity)
  buckets  rare (<50) / common (50-500) / frequent (>500) per-predicate recall, by GT count in the evaluated split
  pair_acc top-1 predicate of each GT ordered pair hits any of its GT predicates (per pair, pooled)

Usage: python tools/sgg_protocol_eval.py dump1.pt [dump2.pt ...] [--ks 20 50 100] [--names name1 name2 ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch


def evaluate(records, ks=(20, 50, 100)):
    n_img = 0
    r_img = {k: [] for k in ks}
    hit_pred = {k: defaultdict(int) for k in ks}
    gt_pred = defaultdict(int)
    hit_micro = {k: 0 for k in ks}
    n_gt_total = 0
    pair_hit = pair_total = 0
    for rec in records:
        gt = rec["gt"].long()
        if gt.numel() == 0:
            continue
        n_img += 1
        pred = rec["pred"].float()
        G = gt.shape[0]
        n_gt_total += G
        for p in gt[:, 2].tolist():
            gt_pred[p] += 1
        # rank predictions by score (graph-constrained dump: one row per ordered pair)
        if pred.numel():
            order = pred[:, 3].argsort(descending=True)
            pred = pred[order]
            key_pred = (pred[:, 0].long() * 100000 + pred[:, 1].long()) * 100000 + pred[:, 2].long()
        else:
            key_pred = torch.zeros(0, dtype=torch.long)
        key_gt = (gt[:, 0] * 100000 + gt[:, 1]) * 100000 + gt[:, 2]
        # rank of first matching prediction for each GT triplet (inf if none)
        if key_pred.numel():
            match = key_gt.unsqueeze(1) == key_pred.unsqueeze(0)                  # [G,P]
            has = match.any(1)
            first = torch.where(has, match.float().argmax(1), torch.full((G,), 10**9))
        else:
            first = torch.full((G,), 10**9)
        for k in ks:
            hit = first < k
            r_img[k].append(hit.float().mean().item())
            hit_micro[k] += int(hit.sum())
            for p, h in zip(gt[:, 2].tolist(), hit.tolist()):
                hit_pred[k][p] += int(h)
        # pair accuracy: top-1 predicate per GT ordered pair
        pair_gt = defaultdict(set)
        for s, o, p in gt.tolist():
            pair_gt[(s, o)].add(p)
        top1 = {}
        for s, o, p, sc in pred.tolist():  # pred sorted desc
            key = (int(s), int(o))
            if key not in top1:
                top1[key] = int(p)
        for key, ps in pair_gt.items():
            pair_total += 1
            pair_hit += int(top1.get(key, -1) in ps)
    out = {"n_images": n_img, "n_gt": n_gt_total, "n_predicates": len(gt_pred)}
    for k in ks:
        R = sum(r_img[k]) / max(len(r_img[k]), 1)
        per_pred = {p: hit_pred[k][p] / gt_pred[p] for p in gt_pred}
        mR = sum(per_pred.values()) / max(len(per_pred), 1)
        out[f"R@{k}"] = R
        out[f"mR@{k}"] = mR
        out[f"F1@{k}"] = 2 * R * mR / max(R + mR, 1e-9)
        out[f"R@{k}_micro"] = hit_micro[k] / max(n_gt_total, 1)
        for name, lo, hi in (("rare", 0, 50), ("common", 50, 500), ("frequent", 500, 10**9)):
            sel = [per_pred[p] for p in gt_pred if lo <= gt_pred[p] < hi]
            out[f"mR@{k}_{name}"] = sum(sel) / len(sel) if sel else float("nan")
            out[f"n_pred_{name}"] = len(sel)
    out["pair_acc"] = pair_hit / max(pair_total, 1)
    out["n_pairs"] = pair_total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--json", default=None, help="write all results to this json")
    args = ap.parse_args()
    names = args.names or args.dumps
    results = {}
    for name, path in zip(names, args.dumps):
        recs = torch.load(path, map_location="cpu", weights_only=False)
        results[name] = evaluate(recs["records"] if isinstance(recs, dict) else recs, tuple(args.ks))
    cols = ["R@50", "mR@50", "F1@50", "R@50_micro", "mR@50_rare", "mR@50_common", "mR@50_frequent", "R@20", "R@100", "mR@100", "pair_acc"]
    print("model".ljust(40) + "".join(c.rjust(15) for c in cols))
    for name, r in results.items():
        print(name[:40].ljust(40) + "".join(f"{100 * r[c]:15.2f}" if isinstance(r[c], float) else str(r[c]).rjust(15) for c in cols))
    first = next(iter(results.values()))
    print(f"(images={first['n_images']}, GT triplets={first['n_gt']}, predicates in split={first['n_predicates']}, "
          f"rare/common/frequent predicates={first['n_pred_rare']}/{first['n_pred_common']}/{first['n_pred_frequent']})")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
