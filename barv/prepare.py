"""Build PSG queries and cache frozen-backbone query features."""
from __future__ import annotations

import os
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import (PROJECT_ROOT, grid_label, image_uid_split, letterbox_image, letterbox_label,
                     letterbox_params, load_config, save_json, seed_everything)
from .model import build_backbone, pool_regions
from .psg import audit_positive_gaps, build_image_queries, load_psg, panoptic_ids, resolve_paths

SPLITS = ["train_core", "train_teacher", "dev", "calib", "test", "psgval"]
SPLIT_CODE = {s: i for i, s in enumerate(SPLITS)}
CODE_SPLIT = {i: s for s, i in SPLIT_CODE.items()}


def select_records(cfg: dict) -> list[dict]:
    d = load_psg(cfg["paths"]["psg_json"])
    cfg["_predicate_classes"] = d["predicate_classes"]
    seen = {}
    for rec in d["data"]:
        uid = str(rec.get("coco_image_id", rec["image_id"]))
        if uid in seen:
            # Duplicate COCO entries share segments but can carry different
            # official triples; union them so no GT positive is lost.
            merged = set(map(tuple, seen[uid]["relations"])) | set(map(tuple, rec["relations"]))
            seen[uid]["relations"] = [list(t) for t in sorted(merged)]
            continue
        rec = dict(rec)
        rec["relations"] = [list(t) for t in rec["relations"]]
        seen[uid] = rec
    records = []
    for uid, rec in seen.items():
        split = "psgval" if rec["file_name"].startswith("val2017/") else image_uid_split(uid, cfg["split"])
        rec = dict(rec)
        rec["split"] = split
        rec["uid"] = uid
        records.append(rec)
    if cfg.get("limit_records"):
        records = records[: int(cfg["limit_records"])]
    return records


def calibrate_thresholds(cfg: dict, records: list[dict]) -> dict:
    train = [r for r in records if r["split"] in ("train_core", "train_teacher")]
    audit = audit_positive_gaps(train, cfg, cfg["data"]["predicates"], cfg["_predicate_classes"],
                                sample_n=int(cfg["data"].get("audit_images", 2000)),
                                seed=int(cfg["seed"]))
    floor = float(cfg["data"].get("neg_gap_floor_px", 10))
    margin = float(cfg["data"].get("neg_gap_p99_margin_px", 6))
    thresholds, notes = {}, {}
    for p in cfg["data"]["predicates"]:
        a = audit.get(p, {})
        mx = a.get("max")
        if mx is None or a.get("n", 0) == 0:
            thr = float("inf")           # no safe negative exists for this predicate
        else:
            # threshold must exceed EVERY audited positive, otherwise a pair with
            # the same geometry as an official positive could be labeled negative
            thr = max(floor, float(mx) * 1.25 + margin)
        thresholds[p] = float(thr)
        notes[p] = {"max_positive_gap_px": mx, "threshold_px": thresholds[p],
                    "n_positive_audited": a.get("n", 0),
                    "p99_positive_gap_px": a.get("p99")}
    audit["_thresholds"] = thresholds
    audit["_notes"] = notes
    return audit


class QueryPrepDataset(Dataset):
    def __init__(self, records: list[dict], cfg: dict, thresholds: dict):
        self.records = records
        self.cfg = cfg
        self.thresholds = thresholds

    def __len__(self):
        return len(self.records)

    def _masks(self, label_lb, p, grid, step, queries):
        def down(m):
            return cv2.resize(m.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_AREA) > 0
        out = np.zeros((len(queries), 4, grid, grid), dtype=np.uint8)
        for i, q in enumerate(queries):
            sub = down(label_lb == q["sub_seg"])
            obj = down(label_lb == q["obj_seg"])
            if not sub.any():
                sub = box_fill_mask_cfg(q["boxes"][0], p, grid, step)
            if not obj.any():
                obj = box_fill_mask_cfg(q["boxes"][1], p, grid, step)
            out[i, 0] = sub
            out[i, 1] = obj
            out[i, 2] = np.logical_or(sub, obj)
            out[i, 3] = box_fill_mask_cfg(q["boxes"][3], p, grid, step)
        return out

    def __getitem__(self, i):
        rec = self.records[i]
        img_path, pan_path = resolve_paths(rec, self.cfg)
        if not (os.path.exists(img_path) and os.path.exists(pan_path)):
            return None
        bgr = cv2.imread(img_path)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        label = panoptic_ids(pan_path)
        if label.shape[:2] != (h, w):
            label = cv2.resize(label.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        queries, stat = build_image_queries(
            rec, label, self.cfg["data"]["predicates"], self.thresholds,
            self.cfg["_predicate_classes"],
            max_negs_per_pred_image=int(self.cfg["data"].get("max_negs_per_pred_image", 3)),
            max_negs_per_image=int(self.cfg["data"].get("max_negs_per_image", 6)),
            min_instance_px=int(self.cfg["data"].get("min_instance_px", 64)),
            seed=int(self.cfg["seed"]),
        )
        if not queries:
            return None
        size = int(self.cfg["model"]["coarse_size"])
        p = letterbox_params(h, w, size)
        img320 = letterbox_image(rgb, p)
        lab320 = letterbox_label(label, p)
        m20 = self._masks(lab320, p, size // 16, 16, queries)
        m10 = self._masks(lab320, p, size // 32, 32, queries)
        item = {"img": img320, "m20": m20, "m10": m10, "queries": queries, "stat": stat,
                "split": rec["split"], "img_full": None, "m40": None, "m20f": None}
        if rec["split"] not in ("train_core", "train_teacher"):
            fsize = int(self.cfg["model"]["full_size"])
            pf = letterbox_params(h, w, fsize)
            item["img_full"] = letterbox_image(rgb, pf)
            lab_full = letterbox_label(label, pf)
            item["m40"] = self._masks(lab_full, pf, fsize // 16, 16, queries)
            item["m20f"] = self._masks(lab_full, pf, fsize // 32, 32, queries)
        return item


def box_fill_mask_cfg(box_orig, p, grid, step):
    x1, y1, x2, y2 = box_orig
    x1 = int(np.floor(x1 * p["scale"] + p["px"])) // step
    y1 = int(np.floor(y1 * p["scale"] + p["py"])) // step
    x2 = int(np.ceil(x2 * p["scale"] + p["px"])) // step
    y2 = int(np.ceil(y2 * p["scale"] + p["py"])) // step
    x1, x2 = max(0, min(grid - 1, x1)), max(0, min(grid - 1, x2))
    y1, y2 = max(0, min(grid - 1, y1)), max(0, min(grid - 1, y2))
    if x2 <= x1:
        x2 = min(grid, x1 + 1)
    if y2 <= y1:
        y2 = min(grid, y1 + 1)
    m = np.zeros((grid, grid), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _collate(items):
    items = [x for x in items if x is not None]
    if not items:
        return None
    out = {"img": torch.stack([torch.from_numpy(x["img"]).permute(2, 0, 1) for x in items]),
           "m20": [x["m20"] for x in items], "m10": [x["m10"] for x in items],
           "queries": [x["queries"] for x in items], "splits": [x["split"] for x in items],
           "stats": [x["stat"] for x in items], "img_full": None, "m40": None, "m20f": None}
    full = [i for i, x in enumerate(items) if x["img_full"] is not None]
    if full:
        out["img_full"] = {i: torch.from_numpy(items[i]["img_full"]).permute(2, 0, 1) for i in full}
        out["m40"] = {i: items[i]["m40"] for i in full}
        out["m20f"] = {i: items[i]["m20f"] for i in full}
    return out


META_FIELDS = ["qid", "image_uid", "coco_id", "file_name", "pan_seg_file_name",
               "pred", "label", "source", "split", "gap", "overlap",
               "sub_cat", "obj_cat", "sub_seg", "obj_seg", "sub_idx", "obj_idx",
               "frames"]


def cache_features(cfg: dict, records: list[dict], thresholds: dict, out_path: str,
                   num_workers: int | None = None) -> dict:
    seed_everything(int(cfg["seed"]))
    ds = QueryPrepDataset(records, cfg, thresholds)
    nw = int(num_workers if num_workers is not None else cfg.get("num_workers", 16))
    dl = DataLoader(ds, batch_size=int(cfg["hardware"].get("prep_batch", 24)), shuffle=False,
                    num_workers=nw, collate_fn=_collate, persistent_workers=nw > 0,
                    prefetch_factor=4 if nw > 0 else None)
    backbone = build_backbone(cfg).cuda()
    amp = bool(cfg["hardware"].get("amp", True))
    dtype = torch.float16 if amp else torch.float32

    pools, full_pools, metas = [], [], {k: [] for k in META_FIELDS}
    n_unknown, n_images, n_missing = 0, 0, 0
    t0 = time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(dl):
            if batch is None:
                continue
            imgs = batch["img"].cuda(non_blocking=True).float()
            imgs = backbone.preprocess(imgs)
            with torch.autocast("cuda", dtype=dtype):
                f3, f4 = backbone.forward_coarse(imgs)
            f3, f4 = f3.float(), f4.float()
            for b, qs in enumerate(batch["queries"]):
                m3 = torch.from_numpy(batch["m20"][b]).cuda().float()
                m4 = torch.from_numpy(batch["m10"][b]).cuda().float()
                p = pool_regions(f3[b], f4[b], m3, m4)  # [R,4,768]
                pools.append(p.half().cpu().numpy())
                n_unknown += batch["stats"][b]["n_unknown_skipped"]
                n_images += 1
                for q in qs:
                    for k in META_FIELDS:
                        if k == "frames":
                            continue
                        if k == "pred":
                            metas[k].append(q["pid"])
                        elif k == "label":
                            metas[k].append(q["label"])
                        elif k == "source":
                            metas[k].append(0 if q["source"] == "psg_positive" else 1)
                        elif k == "split":
                            metas[k].append(SPLIT_CODE[batch["splits"][b]])
                        elif k == "gap":
                            metas[k].append(q["gap"])
                        elif k == "overlap":
                            metas[k].append(q["overlap"])
                        else:
                            metas[k].append(q[k])
                    metas["frames"].append(np.concatenate(
                        [np.asarray(q["geom"], np.float32), np.asarray(q["actions"], np.float32).reshape(-1),
                         np.asarray(q["boxes"], np.float32).reshape(-1)], axis=0))
            if batch["img_full"]:
                for b, img_full in batch["img_full"].items():
                    x = backbone.preprocess(img_full.cuda().float().unsqueeze(0))
                    with torch.autocast("cuda", dtype=dtype):
                        ff3, ff4 = backbone.forward_coarse(x)
                    mm40 = torch.from_numpy(batch["m40"][b]).cuda().float()
                    mm20 = torch.from_numpy(batch["m20f"][b]).cuda().float()
                    pf = pool_regions(ff3[0].float(), ff4[0].float(), mm40, mm20)
                    full_pools.append(pf.half().cpu().numpy())
            if (bi + 1) % 20 == 0:
                print(f"[prep] batch {bi+1}, images={n_images}, elapsed={time.time()-t0:.0f}s", flush=True)

    if not pools:
        raise RuntimeError("no queries built; check dataset paths and config")
    pool = np.concatenate(pools, axis=0).astype(np.float16)
    frames = np.stack(metas["frames"]).astype(np.float32)
    geom_dim = 24
    split_codes = np.asarray(metas["split"], np.int8)
    pred_arr = np.asarray(metas["pred"], np.int8)
    label_arr = np.asarray(metas["label"], np.int8)
    full_mask_orig = (split_codes != SPLIT_CODE["train_core"]) & (split_codes != SPLIT_CODE["train_teacher"])
    full_pool = np.concatenate(full_pools, axis=0).astype(np.float16) if full_pools else np.zeros((0, 4, 768), np.float16)
    assert full_pool.shape[0] == int(full_mask_orig.sum()), (full_pool.shape[0], int(full_mask_orig.sum()))

    # Post-filter: predicates with no provable negatives are not a binary task;
    # cap negative/positive ratio per predicate so one easy predicate cannot dominate.
    keep = np.ones(len(pool), bool)
    ratio = float(cfg["data"].get("max_neg_pos_ratio", 4))
    min_neg = int(cfg["data"].get("min_neg_per_pred", 50))
    dropped = {}
    rng = np.random.default_rng(int(cfg["seed"]))
    for i, name in enumerate(cfg["data"]["predicates"]):
        pos = np.nonzero((pred_arr == i) & (label_arr == 1))[0]
        neg = np.nonzero((pred_arr == i) & (label_arr == 0))[0]
        if len(neg) < min_neg:
            keep[(pred_arr == i)] = False
            dropped[name] = {"reason": "insufficient verified negatives", "n_pos": int(len(pos)),
                             "n_neg": int(len(neg))}
            continue
        cap = max(min_neg, int(ratio * max(1, len(pos))))
        if len(neg) > cap:
            drop = rng.choice(neg, size=len(neg) - cap, replace=False)
            keep[drop] = False
    if not keep.any():
        raise RuntimeError("all predicates filtered out; relax negative thresholds")
    full_keep = keep[full_mask_orig]
    full_pool = full_pool[full_keep]
    pool = pool[keep]
    frames = frames[keep]
    split_codes = split_codes[keep]
    pred_arr = pred_arr[keep]
    label_arr = label_arr[keep]
    for k in list(metas):
        if k != "frames":
            metas[k] = np.asarray(metas[k])[keep]
    geom = frames[:, :geom_dim]
    actions = frames[:, geom_dim:geom_dim + 24].reshape(-1, 6, 4)
    boxes = frames[:, geom_dim + 24:geom_dim + 24 + 16].reshape(-1, 4, 4)
    full_mask = (split_codes != SPLIT_CODE["train_core"]) & (split_codes != SPLIT_CODE["train_teacher"])
    assert full_pool.shape[0] == int(full_mask.sum()), (full_pool.shape[0], int(full_mask.sum()))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path, pool=pool, geom=geom, actions=actions, boxes=boxes,
        qid=np.asarray(metas["qid"]), image_uid=np.asarray(metas["image_uid"]),
        coco_id=np.asarray(metas["coco_id"]), file_name=np.asarray(metas["file_name"]),
        pan_seg_file_name=np.asarray(metas["pan_seg_file_name"]),
        pred=np.asarray(metas["pred"], np.int8), label=np.asarray(metas["label"], np.int8),
        source=np.asarray(metas["source"], np.int8), split=split_codes,
        gap=np.asarray(metas["gap"], np.float32), overlap=np.asarray(metas["overlap"], np.float32),
        sub_cat=np.asarray(metas["sub_cat"], np.int16), obj_cat=np.asarray(metas["obj_cat"], np.int16),
        sub_seg=np.asarray(metas["sub_seg"], np.int32), obj_seg=np.asarray(metas["obj_seg"], np.int32),
        sub_idx=np.asarray(metas["sub_idx"], np.int16), obj_idx=np.asarray(metas["obj_idx"], np.int16),
        full_index=np.nonzero(full_mask)[0].astype(np.int64),
    )
    np.save(out_path.replace(".npz", "_fullpool.npy"), full_pool)
    stats = {
        "n_queries": int(pool.shape[0]), "n_images": n_images, "n_missing_images": n_missing,
        "n_unknown_skipped": int(n_unknown),
        "n_pos": int((label_arr == 1).sum()), "n_neg": int((label_arr == 0).sum()),
        "pred_counts": dict(Counter(pred_arr.tolist())),
        "dropped_predicates": dropped,
        "split_counts": {CODE_SPLIT[c]: int((split_codes == c).sum()) for c in np.unique(split_codes)},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    save_json(stats, out_path.replace(".npz", "_stats.json"))
    return stats


def main(cfg_path: str, limit: int | None = None, cache: str | None = None):
    cfg = load_config(cfg_path)
    if limit:
        cfg["limit_records"] = limit
    if cache:
        cfg["paths"]["cache"] = str((PROJECT_ROOT / cache).resolve())
    seed_everything(int(cfg["seed"]))
    records = select_records(cfg)
    print("records", len(records), Counter(r["split"] for r in records))
    audit = calibrate_thresholds(cfg, records)
    cache_dir = cfg["paths"]["cache"]
    os.makedirs(cache_dir, exist_ok=True)
    save_json(audit, os.path.join(cache_dir, "positive_gap_audit.json"))
    out = os.path.join(cache_dir, "queries.npz")
    stats = cache_features(cfg, records, audit["_thresholds"], out,
                           num_workers=int(cfg.get("num_workers", 16)))
    save_json({"stats": stats, "thresholds": audit["_thresholds"], "audit": audit},
              os.path.join(cache_dir, "prepare_report.json"))
    print(stats)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/barv_psg.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.limit, args.cache)
