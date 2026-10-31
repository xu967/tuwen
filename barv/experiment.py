"""Orchestrate full-data vs 40%-subset BARV experiments and write metrics/reports."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .common import save_json, seed_everything
from .engine import (choose_margin, choose_threshold, fit_temperature, load_images, load_relation,
                     load_value, run_method)
from .metrics import (aurc, binary_metrics, cluster_bootstrap, paired_bootstrap,
                      per_predicate_metrics, risk_coverage)
from .train import (QueryCache, balance_positives, compute_teacher_gains, precompute_teacher_fine,
                    select_lambda, split_teacher_images, subset_train_mask,
                    train_evidence_update, train_relation, train_value)

METHODS = ["coarse", "full", "fixed_contact", "fixed_subject", "fixed_object", "fixed_union",
           "random", "entropy", "mve"]
SPLITS = ["train_core", "train_teacher", "dev", "calib", "test", "psgval"]


def _indices(cache, names):
    m = np.zeros(len(cache), bool)
    for n in names:
        m |= cache.split_mask(n)
    return np.nonzero(m)[0]


def _summarize(pred: dict, cache: QueryCache, pred_names: list[str]) -> dict:
    y, p, cost, uid = pred["y"], pred["p"], pred["cost"], pred["image_uid"]
    thr = float(pred.get("threshold", 0.5))
    q2i = {q: i for i, q in enumerate(cache.qid)}
    rows = np.array([q2i[q] for q in pred["qid"]])
    split = cache.split[torch.as_tensor(rows)].numpy()
    out = {"overall": binary_metrics(y, p, thr), "per_split": {}, "per_predicate": {},
           "cost": {"mean_units": float(cost.mean()), "fine_calls": pred["fine_calls"],
                    "cache_hits": pred["cache_hits"],
                    "ms_per_query": float(pred["ms_per_query"]),
                    "fine_calls_per_query": float(pred["fine_calls"] / max(1, len(y))),
                    "cache_hit_rate": float(pred["cache_hits"] / max(1, pred["fine_calls"] + pred["cache_hits"])),
                    "dropped_images": int(pred.get("dropped_images", 0)),
                    "dropped_queries": int(pred.get("dropped_queries", 0))},
           "aurc": aurc(y, p, thr)}
    for code, name in enumerate(SPLITS):
        m = split == code
        if m.sum() and len(np.unique(y[m])) > 1:
            out["per_split"][name] = binary_metrics(y[m], p[m], thr)
    predid = cache.pred[torch.as_tensor(rows)].numpy()
    out["per_predicate"] = per_predicate_metrics(y, p, predid, pred_names, thr)
    return out


def _attach_qids(pred, cache):
    # run_method already returns qids in cache order per image; split/pred lookup by qid
    return pred


def run_one(cache: QueryCache, cfg: dict, out_dir: str, frac: float, seed: int,
            test_images: dict, calib_images: dict, fine_path: str, pred_names: list[str]) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)
    train_mask = (subset_train_mask(cache, frac, seed) if frac < 0.999
                  else cache.image_mask_for_splits(("train_core", "train_teacher")))
    train_core = balance_positives(cache, train_mask & cache.split_mask("train_core"),
                                   ratio=float(cfg["data"].get("max_neg_pos_ratio", 4)), seed=seed)
    teacher_mask = train_mask & cache.split_mask("train_teacher")
    dev = cache.split_mask("dev")
    print(f"[run] frac={frac} seed={seed} core_q={train_core.sum()} teacher_q={teacher_mask.sum()}", flush=True)

    rel_pt = os.path.join(out_dir, "relation.pt")
    train_relation(cache, cfg, train_core, dev, rel_pt, pool_mode="full", seed=seed)
    # Fine update is fit on 70% of train_teacher images; teacher gains/value use
    # the disjoint 30%, so value supervision is out-of-sample for the updater.
    upd_mask, gain_mask = split_teacher_images(cache, teacher_mask, seed, 0.7)
    save_json({"frac": frac, "seed": seed, "n_train_core_queries": int(train_core.sum()),
               "n_teacher_queries": int(teacher_mask.sum()),
               "n_train_core_images": int(len(np.unique(cache.image_uid[train_core]))),
               "n_teacher_images": int(len(np.unique(cache.image_uid[teacher_mask]))),
               "n_update_queries": int(upd_mask.sum()), "n_gain_queries": int(gain_mask.sum())},
              os.path.join(out_dir, "run_config.json"))
    adapted_pt = os.path.join(out_dir, "relation_adapted.pt")
    train_evidence_update(cache, fine_path, rel_pt, cfg, upd_mask, adapted_pt, seed=seed)
    gains_pt = os.path.join(out_dir, "teacher_gains.npz")
    compute_teacher_gains(cache, fine_path, adapted_pt, cfg, gain_mask, gains_pt)
    val_pt = os.path.join(out_dir, "value.pt")
    train_value(gains_pt, cfg, val_pt, seed=seed, n_pred=int(cache.pred.max()) + 1)
    lam_info = select_lambda(gains_pt, val_pt, cfg)
    lam = float(lam_info["lambda"])
    cfg["runtime"]["stop_margin"] = float(lam_info["stop_threshold"])

    relation = load_relation(adapted_pt)
    value = load_value(val_pt)
    test_idx = _indices(cache, ("test", "psgval"))
    calib_idx = _indices(cache, ("calib",))
    preds = {}
    for m in METHODS:
        print(f"[eval] {out_dir} {m}", flush=True)
        r = run_method(cache, cfg, relation, value, m, test_idx, test_images, lam)
        preds[m] = r
        np.savez(os.path.join(out_dir, f"pred_{m}.npz"), **{k: v for k, v in r.items() if k != "method"})

    # Post-policy calibration on calib (never test).
    rc = run_method(cache, cfg, relation, value, "mve", calib_idx, calib_images, lam)
    temp = fit_temperature(rc["logit"], rc["y"])
    p_cal = 1.0 / (1.0 + np.exp(-rc["logit"] / temp))
    thr = choose_threshold(p_cal, rc["y"])
    margin = choose_margin(p_cal, float(cfg["runtime"]["abstain_coverage"]), thr)
    p_test_cal = 1.0 / (1.0 + np.exp(-preds["mve"]["logit"] / temp))
    abstain = np.abs(p_test_cal - thr) < margin
    preds["mve_calibrated"] = {**preds["mve"], "p": p_test_cal, "abstain": abstain, "threshold": thr}
    preds["mve_abstain"] = {**preds["mve"], "p": p_test_cal, "abstain": abstain, "threshold": thr}
    calib_info = {"temperature": temp, "abstain_margin": margin, "threshold": thr,
                  "target_coverage": float(cfg["runtime"]["abstain_coverage"]),
                  "calib_risk_coverage": risk_coverage(rc["y"], p_cal, np.abs(p_cal - thr) < margin, thr)}

    metrics = {"run": {"frac": frac, "seed": seed, "lambda": lam,
                        "stop_threshold": float(cfg["runtime"]["stop_margin"]),
                        "calibration": calib_info},
               "methods": {}}
    for m, r in preds.items():
        s = _summarize(r, cache, pred_names)
        if m in ("mve_abstain",):
            s["selective"] = risk_coverage(r["y"], r["p"], r["abstain"], thr)
        s["bootstrap"] = cluster_bootstrap(r["y"], r["p"], r["image_uid"],
                                           n_boot=int(cfg["train"]["n_bootstrap"]), seed=seed,
                                           threshold=float(r.get("threshold", 0.5)))
        metrics["methods"][m] = s
        print(f"[metrics] {m} bacc={s['overall']['balanced_acc']:.4f} auroc={s['overall']['auroc']:.4f} "
              f"cost={s['cost']['mean_units']:.3f}", flush=True)
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    return {"dir": out_dir, "lambda": lam, "calibration": calib_info, "metrics": metrics}


def compare_runs(cfg: dict, run_dirs: dict[str, str], out_dir: str) -> dict:
    runs = {name: json.load(open(os.path.join(d, "metrics.json"))) for name, d in run_dirs.items()}
    out = {"runs": {}, "comparison": {}}
    for name, m in runs.items():
        out["runs"][name] = {meth: {"balanced_acc": v["overall"]["balanced_acc"],
                                    "macro_f1": v["overall"]["macro_f1"],
                                    "auroc": v["overall"]["auroc"],
                                    "auprc": v["overall"]["auprc"],
                                    "brier": v["overall"]["brier"], "ece": v["overall"]["ece"],
                                    "cost_units": v["cost"]["mean_units"],
                                    "fine_calls_per_query": v["cost"]["fine_calls_per_query"],
                                    "ms_per_query": v["cost"]["ms_per_query"]}
                            for meth, v in m["methods"].items()}
    full = out["runs"].get("full")
    subsets = [out["runs"][k] for k in out["runs"] if k.startswith("sub40")]
    if full and subsets:
        for meth in full:
            keys = ["balanced_acc", "macro_f1", "auroc", "auprc", "cost_units", "ms_per_query"]
            vals = {k: [s[meth][k] for s in subsets if meth in s] for k in keys}
            out["comparison"][meth] = {
                "full": {k: full[meth][k] for k in keys},
                "sub40_mean": {k: float(np.mean(v)) for k, v in vals.items() if v},
                "sub40_std": {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items() if v},
                "delta_sub_minus_full": {k: float(np.mean(v) - full[meth][k]) for k, v in vals.items() if v},
            }
    # paired bootstrap for MVE full vs each subset
    if "full" in run_dirs:
        full_pred = np.load(os.path.join(run_dirs["full"], "pred_mve.npz"), allow_pickle=True)
        per_sub = {}
        for name, d in run_dirs.items():
            if not name.startswith("sub40"):
                continue
            sp = np.load(os.path.join(d, "pred_mve.npz"), allow_pickle=True)
            # align by qid (same evaluation set)
            common = np.intersect1d(full_pred["qid"], sp["qid"])
            fa = {q: i for i, q in enumerate(full_pred["qid"])}
            sb = {q: i for i, q in enumerate(sp["qid"])}
            ia = np.array([fa[q] for q in common])
            ib = np.array([sb[q] for q in common])
            per_sub[name] = paired_bootstrap(full_pred["y"][ia], full_pred["p"][ia], sp["p"][ib],
                                             full_pred["image_uid"][ia],
                                             n_boot=int(cfg["train"]["n_bootstrap"]), seed=0)
        out["comparison"]["mve_paired_bootstrap_vs_full"] = per_sub
    save_json(out, os.path.join(out_dir, "comparison.json"))
    return out
