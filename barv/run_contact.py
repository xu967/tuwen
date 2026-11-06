"""BARV 难负例对照：几何 LR 基线 vs 几何 MLP（no_visual） vs 视觉 scorer，按负例来源分别报 AUROC。

回答 pilot 留下的问题：负例改成「几何接触、类别合法、未标注」之后，视觉特征还能不能超过几何捷径。
三组模型都在同一 train_core（按谓词配平）上训练，dev 选 epoch，test 只报一次。

usage:
  python -m barv.run_contact --config configs/barv_psg_contact.yaml [--cache data/barv_cache_contact] \
      [--out runs/barv_contact/seed42] [--seed 42]
输出：<out>/geometry_only.pkl(+_dev.json)、no_visual.pt、relation.pt(+_history.json)、report.json、report.md
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .common import PROJECT_ROOT, load_config, save_json, seed_everything
from .metrics import binary_metrics, per_predicate_metrics
from .prepare import CODE_SOURCE
from .train import QueryCache, balance_positives, predict_logits, train_prior_baseline, train_relation

# ---------------- 配置区（命令行可覆盖） ----------------
DEFAULT_CONFIG = "configs/barv_psg_contact.yaml"
DEFAULT_OUT = "runs/barv_contact/seed42"
NEG_SOURCES = (2, 3)        # 2 = contact_other（该对标了别的谓词）、3 = contact_unlabeled（完全没标，含标注噪声）
# -------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _by_source(y, p, src, pred, names, sources=NEG_SOURCES):
    """正例 + 某一来源的负例 → 指标；回答「视觉能不能把这类负例和正例分开」。"""
    out = {}
    for s in sources:
        m = (y == 1) | (src == s)
        if (src == s).sum() == 0:
            continue
        r = binary_metrics(y[m], p[m])
        r["n_neg"] = int((src == s).sum())
        r["per_predicate_auroc"] = {k: v["auroc"] for k, v in per_predicate_metrics(y[m], p[m], pred[m], names).items()}
        out[CODE_SOURCE.get(s, str(s))] = r
    return out


def evaluate(name, y, p, src, pred, names, masks):
    rep = {}
    for split, m in masks.items():
        rep[split] = {"overall": binary_metrics(y[m], p[m]),
                      "by_source": _by_source(y[m], p[m], src[m], pred[m], names),
                      "per_predicate_auroc": {k: v["auroc"] for k, v in per_predicate_metrics(y[m], p[m], pred[m], names).items()}}
    print(f"[{name}] " + "  ".join(f"{s}: auroc={r['overall']['auroc']:.4f} n={r['overall']['n']}" for s, r in rep.items()), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--cache", default=None, help="覆盖 cfg.paths.cache（目录，内含 queries.npz）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    seed_everything(seed)
    cache_dir = args.cache or cfg["paths"]["cache"]
    cache_path = os.path.join(cache_dir, "queries.npz")
    out_dir = str((PROJECT_ROOT / args.out).resolve())
    os.makedirs(out_dir, exist_ok=True)
    cache = QueryCache(cache_path)
    names = list(cfg["data"]["predicates"])
    y = cache.label.numpy().astype(int)
    pred = cache.pred.numpy().astype(int)
    src = cache.z["source"].astype(int)
    print(f"cache {cache_path}: {len(cache)} queries, pos={int((y == 1).sum())}, "
          f"neg by source={{{', '.join(f'{CODE_SOURCE.get(int(s), s)}={int((src == s).sum())}' for s in np.unique(src[y == 0]))}}}", flush=True)

    train_core = balance_positives(cache, cache.split_mask("train_core"),
                                   ratio=float(cfg["data"].get("max_neg_pos_ratio", 4)), seed=seed)
    masks = {"dev": cache.split_mask("dev"), "test": cache.split_mask("test")}
    print(f"train_core={int(train_core.sum())} dev={int(masks['dev'].sum())} test={int(masks['test'].sum())}", flush=True)
    device = torch.device(cfg["hardware"].get("train_device", "cuda:0"))
    report = {"cache": cache_path, "seed": seed, "n_train_core": int(train_core.sum()),
              "n_dev": int(masks["dev"].sum()), "n_test": int(masks["test"].sum()),
              "neg_source_counts": {CODE_SOURCE.get(int(s), str(s)): int((src == s).sum()) for s in np.unique(src[y == 0])},
              "models": {}}

    # 1) 几何 LR（pilot 的 geometry_only 口径：geom 24 维 + 谓词 one-hot，sklearn LR）
    prior = train_prior_baseline(cache, train_core, masks["dev"], os.path.join(out_dir, "geometry_only.pkl"), seed=seed)
    clf = prior["clf"]
    g, one = cache.geom.numpy(), np.eye(int(cache.pred.max()) + 1, dtype=np.float32)[pred]
    p_lr = clf.predict_proba(np.concatenate([g, one], 1))[:, 1]
    report["models"]["geometry_lr"] = evaluate("geometry_lr", y, p_lr, src, pred, names, masks)

    # 2) 同一网络、把像素特征置零（no_visual）：排除「网络容量」解释
    nv = train_relation(cache, cfg, train_core, masks["dev"], os.path.join(out_dir, "no_visual.pt"), pool_mode="no_visual", seed=seed)
    p_nv = _sigmoid(predict_logits(nv["model"], cache, np.arange(len(cache)), device, pool_mode="no_visual"))
    report["models"]["no_visual_mlp"] = evaluate("no_visual_mlp", y, p_nv, src, pred, names, masks)

    # 3) 视觉 scorer（coarse 320 px 池化特征 + 几何 + 谓词）
    vis = train_relation(cache, cfg, train_core, masks["dev"], os.path.join(out_dir, "relation.pt"), pool_mode="full", seed=seed)
    p_vis = _sigmoid(predict_logits(vis["model"], cache, np.arange(len(cache)), device, pool_mode="full"))
    report["models"]["visual"] = evaluate("visual", y, p_vis, src, pred, names, masks)

    save_json(report, os.path.join(out_dir, "report.json"))
    lines = ["# BARV 难负例对照", "",
             f"cache `{cache_path}`，seed {seed}，train_core {report['n_train_core']} / dev {report['n_dev']} / test {report['n_test']}；"
             f"负例来源 {report['neg_source_counts']}", "",
             "| 模型 | dev AUROC | test AUROC | test vs contact_other | test vs contact_unlabeled | test bacc |",
             "|---|---:|---:|---:|---:|---:|"]
    for mname, rep in report["models"].items():
        t = rep["test"]
        bs = t["by_source"]
        lines.append(f"| {mname} | {rep['dev']['overall']['auroc']:.4f} | {t['overall']['auroc']:.4f} | "
                     f"{bs.get('contact_other', {}).get('auroc', float('nan')):.4f} | "
                     f"{bs.get('contact_unlabeled', {}).get('auroc', float('nan')):.4f} | {t['overall']['balanced_acc']:.4f} |")
    lines += ["", "test 按谓词 AUROC（visual / geometry_lr）：", ""]
    pp_v, pp_g = report["models"]["visual"]["test"]["per_predicate_auroc"], report["models"]["geometry_lr"]["test"]["per_predicate_auroc"]
    for k in names:
        if k in pp_v:
            lines.append(f"- {k}: {pp_v[k]:.4f} / {pp_g.get(k, float('nan')):.4f}")
    lines += ["", "pilot（远离负例，`runs/barv/pilot_seed42`）：visual test AUROC 0.9960，geometry_lr 0.9942。"]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
