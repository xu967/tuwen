"""与官方 RelateAnything 的差距拆解：按谓词类型 / 频次的 R@50，top-1 混淆，以及谓词文本空间几何。

读取 reports/baseline_cmp 的 dump（tools/dump_predcls_ours.py / run_ra_official_predcls.py 产出，同 val 同词表），
不跑模型。文本空间部分读两份 W：我们的（best.pt 里 vocab_head.W，MobileCLIP-S0）与官方的（model.pth 的 W，蒸馏 student）。

usage:
  PYTHONPATH=/tmp/yolovo-audit-numpy2 python tools/gap_analysis.py \
      --dumps RA_official=reports/baseline_cmp/ra_official_vits16plus.pt ours_ep2_nocls=... ours_ep4_cls=... \
      --ours-ckpt runs/test/glp_full_e4_resume/weights/best.pt --out reports/baseline_cmp/gap_analysis.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_CUDNN_LIB = "/root/rivermind-data/cudnn9/nvidia/cudnn/lib"
_LD = os.environ.get("LD_LIBRARY_PATH", "")
if os.path.isdir(_CUDNN_LIB) and _CUDNN_LIB not in _LD.split(":"):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN_LIB + (f":{_LD}" if _LD else "")
    os.execv(sys.executable, [sys.executable, *sys.argv])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

# ---------------- 配置区 ----------------
DIRECTIONAL_KEYS = ("left", "right", "above", "below", "behind", "in front", "under", "over", "on top", "beneath",
                    "underneath", "top of", "bottom", "front of", "higher", "lower", "ahead", "atop")
PROXIMITY_KEYS = ("near", "next to", "beside", "close to", "adjacent", "by ", "alongside", "nearby", "with", "between",
                  "among", "around", "across", "along", "far", "away")
CONTACT_KEYS = ("on", "in", "attached", "touching", "against", "inside", "holding", "wearing", "carrying", "hanging",
                "mounted", "resting", "sitting", "standing", "lying", "leaning", "parked", "covering", "supporting")
ANTONYM_PAIRS = [("above", "below"), ("left of", "right of"), ("to the left of", "to the right of"),
                 ("in front of", "behind"), ("over", "under"), ("on top of", "underneath"), ("inside", "outside"),
                 ("pushing", "pulling"), ("entering", "exiting"), ("taller than", "shorter than"),
                 ("larger than", "smaller than"), ("in front of", "in back of")]
SYNONYM_PAIRS = [("holding", "grasping"), ("holding", "gripping"), ("riding", "riding on"), ("next to", "beside"),
                 ("on top of", "atop"), ("sitting on", "seated on"), ("near", "close to"), ("wearing", "dressed in"),
                 ("looking at", "watching"), ("standing on", "stands on"), ("behind", "in back of"),
                 ("carrying", "transporting"), ("attached to", "fixed to"), ("in", "inside")]
# ----------------------------------------


def kind_of(name: str) -> str:
    n = f" {name.lower()} "
    if any(f" {k}" in n or n.strip().startswith(k) for k in DIRECTIONAL_KEYS):
        return "directional"
    if any(k in n for k in PROXIMITY_KEYS):
        return "proximity"
    if any(n.strip() == k or n.strip().startswith(k + " ") or n.strip().endswith(" " + k) for k in CONTACT_KEYS):
        return "contact/support"
    return "action/other"


def per_predicate(records, k=50):
    """per-predicate hit@k, GT count, and top-1 confusion at annotated pairs (graph-constrained dump)."""
    hit = defaultdict(int)
    cnt = defaultdict(int)
    conf = defaultdict(lambda: defaultdict(int))   # gt_pred -> predicted top1 -> count
    for rec in records:
        gt = rec["gt"].long()
        if gt.numel() == 0:
            continue
        pred = rec["pred"].float()
        if pred.numel():
            pred = pred[pred[:, 3].argsort(descending=True)]
            key_pred = (pred[:, 0].long() * 100000 + pred[:, 1].long()) * 100000 + pred[:, 2].long()
        else:
            key_pred = torch.zeros(0, dtype=torch.long)
        key_gt = (gt[:, 0] * 100000 + gt[:, 1]) * 100000 + gt[:, 2]
        if key_pred.numel():
            match = key_gt.unsqueeze(1) == key_pred.unsqueeze(0)
            has = match.any(1)
            first = torch.where(has, match.float().argmax(1), torch.full((gt.shape[0],), 10**9))
        else:
            first = torch.full((gt.shape[0],), 10**9)
        top1 = {}
        for s, o, p, _ in pred.tolist():
            top1.setdefault((int(s), int(o)), int(p))
        for (s, o, p), f in zip(gt.tolist(), first.tolist()):
            cnt[p] += 1
            hit[p] += int(f < k)
            conf[p][top1.get((s, o), -1)] += 1
    return hit, cnt, conf


def text_geometry(W: torch.Tensor, names, tag):
    Wn = torch.nn.functional.normalize(W.float(), dim=-1)
    idx = {n: i for i, n in enumerate(names)}
    cos = Wn @ Wn.T
    V = cos.shape[0]
    off = cos[~torch.eye(V, dtype=torch.bool)]
    nn90 = ((cos >= 0.90).sum(1) - 1).float()
    # participation ratio (effective dimension) of the centred embedding
    Wc = Wn - Wn.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(Wc.T @ Wc / V).clamp_min(0)
    eff_dim = float(ev.sum() ** 2 / (ev ** 2).sum())
    def pair_cos(pairs):
        out = {}
        for a, b in pairs:
            if a in idx and b in idx:
                out[f"{a} | {b}"] = round(float(cos[idx[a], idx[b]]), 3)
        return out
    ant = pair_cos(ANTONYM_PAIRS)
    syn = pair_cos(SYNONYM_PAIRS)
    res = {"tag": tag, "V": V, "mean_offdiag_cos": round(float(off.mean()), 3),
           "median_neighbours_cos_ge_0.90": float(nn90.median()), "eff_dim": round(eff_dim, 1),
           "antonym_cos": ant, "antonym_mean": round(sum(ant.values()) / max(len(ant), 1), 3),
           "synonym_cos": syn, "synonym_mean": round(sum(syn.values()) / max(len(syn), 1), 3)}
    print(f"[{tag}] V={V} mean offdiag cos={res['mean_offdiag_cos']} median #nn(cos>=.9)={res['median_neighbours_cos_ge_0.90']:.0f} "
          f"eff_dim={res['eff_dim']} antonym mean cos={res['antonym_mean']} synonym mean cos={res['synonym_mean']}")
    for kk, v in ant.items():
        print(f"    antonym  {kk}: {v}")
    for kk, v in syn.items():
        print(f"    synonym  {kk}: {v}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="name=path")
    ap.add_argument("--predicates", default="RA-4M/predicates.json")
    ap.add_argument("--ours-ckpt", default=None)
    ap.add_argument("--official-ckpt", default="assets/relsgg-vits16plus/model.pth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    preds = json.load(open(args.predicates))
    names = [p["name"] for p in sorted(preds, key=lambda p: p["id"])]
    # dump pred ids are 0-based positions in the yaml rnames order (= id-1); verify by GT count sanity below
    kinds = [kind_of(n) for n in names]
    report = {"per_model": {}, "kind_counts": {k: kinds.count(k) for k in set(kinds)}}

    base_cnt = None
    for spec in args.dumps:
        name, path = spec.split("=", 1)
        recs = torch.load(path, map_location="cpu", weights_only=False)
        recs = recs["records"] if isinstance(recs, dict) else recs
        hit, cnt, conf = per_predicate(recs)
        base_cnt = base_cnt or cnt
        by_kind = defaultdict(lambda: [0, 0, 0])   # hits, gts, n_pred
        pp = {}
        for p, c in cnt.items():
            r = hit[p] / c
            pp[p] = r
            kd = kinds[p] if p < len(kinds) else "?"
            by_kind[kd][0] += hit[p]; by_kind[kd][1] += c; by_kind[kd][2] += 1
        kind_tab = {kd: {"R@50_micro": round(v[0] / v[1], 4), "mR@50": round(sum(pp[p] for p in cnt if kinds[p] == kd) / v[2], 4),
                         "n_gt": v[1], "n_pred": v[2]} for kd, v in by_kind.items()}
        # frequency buckets by GT count in this split
        freq_tab = {}
        for bname, lo, hi in (("rare<50", 0, 50), ("common50-500", 50, 500), ("frequent>500", 500, 10**9)):
            sel = [p for p in cnt if lo <= cnt[p] < hi]
            freq_tab[bname] = {"R@50_micro": round(sum(hit[p] for p in sel) / max(sum(cnt[p] for p in sel), 1), 4),
                               "mR@50": round(sum(pp[p] for p in sel) / max(len(sel), 1), 4),
                               "n_gt": sum(cnt[p] for p in sel), "n_pred": len(sel)}
        # top-20 predicates by GT count with per-predicate recall and top confusion
        top = sorted(cnt, key=lambda p: -cnt[p])[:20]
        top_tab = []
        for p in top:
            c = conf[p]
            best = sorted(c.items(), key=lambda kv: -kv[1])[:3]
            top_tab.append({"pred": names[p], "n_gt": cnt[p], "R@50": round(pp[p], 3),
                            "top1_at_gt_pairs": [(names[q] if 0 <= q < len(names) else "none", round(v / cnt[p], 2)) for q, v in best]})
        report["per_model"][name] = {"by_kind": kind_tab, "by_freq": freq_tab, "top20": top_tab,
                                     "n_predicates_with_hit": sum(1 for p in cnt if hit[p] > 0)}
        print(f"\n=== {name} ===")
        print("  by kind:", json.dumps(kind_tab))
        print("  by freq:", json.dumps(freq_tab))
        print(f"  predicates with >=1 hit: {report['per_model'][name]['n_predicates_with_hit']} / {len(cnt)}")
        for row in top_tab[:12]:
            print(f"  {row['pred']:>18s} n={row['n_gt']:5d} R@50={row['R@50']:.3f}  top1@GT pairs: {row['top1_at_gt_pairs']}")

    if args.ours_ckpt:
        from ultralytics import YOLOVO
        m = YOLOVO(args.ours_ckpt)
        rel = next(mm for mm in m.model.modules() if mm.__class__.__name__ == "YOLOVORelate")
        W_ours = rel.vocab_head.W.detach().cpu()
        assert W_ours.shape[0] == len(names), (W_ours.shape, len(names))
        report["text_space_ours"] = text_geometry(W_ours, names, "ours(MobileCLIP-S0)")
    if args.official_ckpt and os.path.exists(args.official_ckpt):
        ck = torch.load(args.official_ckpt, map_location="cpu", weights_only=False)
        known = {n: i for i, n in enumerate(ck["pred_names"])}
        sd = ck["model"]
        keep = [i for i, n in enumerate(names) if n in known]
        W_off = torch.stack([sd["vocab_head.W"][known[names[i]]] for i in keep])
        report["text_space_official"] = text_geometry(W_off, [names[i] for i in keep], "official(student)")
        report["text_space_official"]["n_names_in_common"] = len(keep)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
