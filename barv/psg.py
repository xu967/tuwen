"""PSG schema parsing, mask geometry, and verified query construction.

Labels follow the v2 protocol:
  positive: official PSG relation triple (verified positive)
  negative: relation-specific geometric rule proves the predicate cannot hold
            (e.g. contact predicates require subject/object mask contact; a pair
             separated by more than the calibrated gap is a verified negative)
  unknown : everything else, never used as a negative in BCE.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .common import letterbox_label, letterbox_params

# ---------------------------------------------------------------------------
# Predicate applicability rules (PSG 0-based category ids)
# ---------------------------------------------------------------------------
ANIMALS = set(range(14, 24))
PERSON = {0}
HOLDABLE = {
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
}
# umbrella (25) is annotated as "wearing" when merely held overhead; it violates
# the mask-contact rule, so it is excluded from wearing negatives.
WEARABLE = {24, 26, 27, 35}
SITTABLE = {13, 56, 57, 59, 60, 61}
STANDABLE = {
    8, 30, 31, 36, 37, 87, 96, 97, 98, 100, 102, 105, 106,
    122, 123, 125, 126, 130, 132,
}
RIDEABLE = {1, 3, 8, 17, 30, 31, 36, 37}
LEANABLE = {13, 56, 57, 84, 109, 110, 111, 112, 121, 131}
LIEABLE = {13, 57, 59, 102, 106, 122, 125, 132}
TOUCH_OBJECTS = set(range(1, 80))


@dataclass(frozen=True)
class Rule:
    name: str
    subjects: frozenset
    objects: frozenset
    max_obj_area_ratio: float | None = None


RULES: dict[str, Rule] = {
    "holding": Rule("holding", frozenset(PERSON), frozenset(HOLDABLE), 0.60),
    "carrying": Rule("carrying", frozenset(PERSON), frozenset(HOLDABLE | {28}), 1.20),
    "wearing": Rule("wearing", frozenset(PERSON), frozenset(WEARABLE), 2.00),
    "sitting on": Rule("sitting on", frozenset(PERSON), frozenset(SITTABLE), 4.00),
    "standing on": Rule("standing on", frozenset(PERSON), frozenset(STANDABLE), None),
    "riding": Rule("riding", frozenset(PERSON), frozenset(RIDEABLE), None),
    "touching": Rule("touching", frozenset(PERSON | ANIMALS), frozenset(TOUCH_OBJECTS), None),
    "leaning on": Rule("leaning on", frozenset(PERSON), frozenset(LEANABLE), None),
    "lying on": Rule("lying on", frozenset(PERSON), frozenset(LIEABLE), None),
}
DEFAULT_PREDICATES = list(RULES)


def load_psg(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_paths(rec: dict, cfg: dict) -> tuple[str, str]:
    import os

    img = os.path.join(cfg["paths"]["images"], os.path.basename(rec["file_name"]))
    pan = os.path.join(cfg["paths"]["coco"], rec["pan_seg_file_name"])
    return img, pan


def audit_positive_gaps(records: list[dict], cfg: dict, predicates: list[str],
                        predicate_classes: list[str], sample_n: int = 2000, seed: int = 0,
                        geom_size: int = 320) -> dict:
    """Measure contact-gap quantiles of official positives to calibrate negative rules.

    A predicate whose positives are routinely far apart is not safely decidable by
    mask contact; its threshold is raised (and the audit report says so).
    """
    rng = np.random.default_rng(seed)
    pick = records if len(records) <= sample_n else [records[i] for i in rng.choice(len(records), sample_n, replace=False)]
    official = {name: predicate_classes.index(name) for name in predicates}
    gaps = {p: [] for p in predicates}
    skipped = 0
    for rec in pick:
        img_path, pan_path = resolve_paths(rec, cfg)
        try:
            label = panoptic_ids(pan_path)
        except FileNotFoundError:
            skipped += 1
            continue
        h, w = label.shape
        inst = instance_table(rec, label)
        by_idx = {r["idx"]: r for r in inst}
        pgeom = letterbox_params(h, w, geom_size)
        lab_geom = letterbox_label(label, pgeom)
        masks = {r["idx"]: (lab_geom == r["seg_id"]) for r in inst}
        for s, o, p in rec["relations"]:
            if p not in official.values() or s not in by_idx or o not in by_idx:
                continue
            if masks[s].sum() == 0 or masks[o].sum() == 0:
                continue
            edt = distance_transform_edt(~masks[o], return_indices=True)
            st = pair_stats(masks[s], masks[o], pgeom["scale"], edt)
            if st is not None:
                gaps[predicate_classes[p]].append(st[0])
    out = {}
    for p, vals in gaps.items():
        v = np.asarray(vals, np.float32)
        out[p] = {
            "n": int(v.size),
            "p50": float(np.percentile(v, 50)) if v.size else None,
            "p95": float(np.percentile(v, 95)) if v.size else None,
            "p99": float(np.percentile(v, 99)) if v.size else None,
            "max": float(v.max()) if v.size else None,
            "frac_contact_2px": float((v <= 2).mean()) if v.size else None,
        }
    out["_audit"] = {"images_sampled": len(pick), "images_skipped_missing": skipped,
                     "gaps_note": "gap measured on 320px letterboxed panoptic masks (adjacent-mask d=1 -> 0), converted to original px"}
    return out


def panoptic_ids(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    return arr[..., 0] + 256 * arr[..., 1] + 65536 * arr[..., 2]


def xywh_to_xyxy(b) -> np.ndarray:
    x, y, w, h = [float(v) for v in b]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def instance_table(rec: dict, label: np.ndarray) -> list[dict]:
    """One row per local segment (relation indices point at this list)."""
    segs, anns = rec["segments_info"], rec["annotations"]
    if len(segs) != len(anns):
        raise ValueError(f"segments/annotations mismatch for image {rec['image_id']}")
    h, w = label.shape
    out = []
    for i, (s, a) in enumerate(zip(segs, anns)):
        # PSG JSON stores absolute xyxy despite bbox_mode=0 (verified against
        # image bounds and mask areas); do not run the COCO xywh conversion.
        box = np.asarray(a["bbox"], np.float32).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
        out.append(
            {
                "idx": i,
                "seg_id": int(s["id"]),
                "cat": int(s["category_id"]),
                "area": float(s.get("area", (box[2] - box[0]) * (box[3] - box[1]))),
                "box": box,
                "isthing": int(s.get("isthing", 1)),
            }
        )
    return out


def pair_stats(sub: np.ndarray, obj: np.ndarray, scale: float, obj_edt=None):
    """Mask gap (original px), overlap fraction, and closest-point midpoint (geom px)."""
    ns, no = int(sub.sum()), int(obj.sum())
    if ns == 0 or no == 0:
        return None
    inter = np.logical_and(sub, obj)
    ni = int(inter.sum())
    if ni > 0:
        ys, xs = np.nonzero(inter)
        return 0.0, ni / max(1, min(ns, no)), (float(xs.mean()), float(ys.mean()))
    dt, inds = obj_edt if obj_edt is not None else distance_transform_edt(~obj, return_indices=True)
    d = dt[sub]
    k = int(np.argmin(d))
    ys, xs = np.nonzero(sub)
    sy, sx = int(ys[k]), int(xs[k])
    oy, ox = int(inds[0][sy, sx]), int(inds[1][sy, sx])
    gap_orig = max(0.0, float(d[k]) - 1.0) / scale
    return gap_orig, 0.0, ((sx + ox) * 0.5, (sy + oy) * 0.5)


def square_xyxy(cx: float, cy: float, side: float, w: float, h: float) -> np.ndarray:
    side = max(16.0, float(side))
    x1, y1 = cx - side / 2, cy - side / 2
    x2, y2 = cx + side / 2, cy + side / 2
    x1, x2 = max(0.0, x1), min(w, x2)
    y1, y2 = max(0.0, y1), min(h, y2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def build_action_boxes(sub_box, obj_box, contact_xy, gap, w, h) -> np.ndarray:
    """Six fixed, reproducible evidence actions (xyxy in original pixels)."""
    sx1, sy1, sx2, sy2 = sub_box
    ox1, oy1, ox2, oy2 = obj_box
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    sside, oside = max(sx2 - sx1, sy2 - sy1), max(ox2 - ox1, oy2 - oy1)
    ux1, uy1 = min(sx1, ox1), min(sy1, oy1)
    ux2, uy2 = max(sx2, ox2), max(sy2, oy2)
    ucx, ucy = (ux1 + ux2) / 2, (uy1 + uy2) / 2
    uside = max(ux2 - ux1, uy2 - uy1)
    cx, cy = contact_xy
    actions = [
        square_xyxy(cx, cy, 2.0 * max(gap, 2.0) + 0.45 * min(sside, oside), w, h),
        square_xyxy(scx, scy, 1.25 * sside, w, h),
        square_xyxy(ocx, ocy, 1.25 * oside, w, h),
        square_xyxy(ucx, ucy, 1.10 * uside, w, h),
        square_xyxy(ucx, ucy, 2.00 * uside, w, h),
        square_xyxy(cx, cy, max(24.0, 0.55 * min(sside, oside)), w, h),
    ]
    return np.stack(actions)


def geometry_features(sub_box, obj_box, union_box, gap, overlap, w, h) -> np.ndarray:
    sb, ob, ub = sub_box.astype(np.float32), obj_box.astype(np.float32), union_box.astype(np.float32)
    sbn = np.array([sb[0] / w, sb[1] / h, sb[2] / w, sb[3] / h], np.float32)
    obn = np.array([ob[0] / w, ob[1] / h, ob[2] / w, ob[3] / h], np.float32)
    ubn = np.array([ub[0] / w, ub[1] / h, ub[2] / w, ub[3] / h], np.float32)
    sa = max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1])) / (w * h)
    oa = max(1.0, (ob[2] - ob[0]) * (ob[3] - ob[1])) / (w * h)
    ua = max(1.0, (ub[2] - ub[0]) * (ub[3] - ub[1])) / (w * h)
    ix1, iy1 = max(sb[0], ob[0]), max(sb[1], ob[1])
    ix2, iy2 = min(sb[2], ob[2]), min(sb[3], ob[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    iou = inter / max(1e-6, (sb[2] - sb[0]) * (sb[3] - sb[1]) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter)
    dx = ((sb[0] + sb[2]) - (ob[0] + ob[2])) / 2 / w
    dy = ((sb[1] + sb[3]) - (ob[1] + ob[3])) / 2 / h
    diag = float(np.hypot(w, h))
    feats = np.concatenate(
        [
            sbn, obn, ubn,
            np.array(
                [
                    sa, oa, ua, iou, dx, dy,
                    np.log(max(1e-4, oa / max(sa, 1e-6))),
                    (sb[2] - sb[0]) / max(1.0, sb[3] - sb[1]),
                    (ob[2] - ob[0]) / max(1.0, ob[3] - ob[1]),
                    sa / max(oa, 1e-6),
                    (ub[2] - ub[0]) / w,
                    (ub[3] - ub[1]) / h,
                ],
                np.float32,
            ),
        ]
    )
    return feats.astype(np.float32)


def _select_negatives(cands: list[dict], pos_pairs: set, cap_pred: int, rng) -> list[dict]:
    for c in cands:
        c["same_cat"] = 0 if (c["sub_cat"], c["obj_cat"]) in pos_pairs else 1
        c["jitter"] = rng.random()
    cands.sort(key=lambda c: (c["same_cat"], c["jitter"]))
    return cands[:cap_pred]


def build_image_queries(
    rec: dict,
    label: np.ndarray,
    predicates: list[str],
    neg_gap_px: dict,
    predicate_classes: list[str],
    geom_size: int = 320,
    *,
    max_negs_per_pred_image: int = 3,
    max_negs_per_image: int = 6,
    min_instance_px: int = 64,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Construct positive / verified-negative queries for one image."""
    h, w = label.shape
    inst = instance_table(rec, label)
    by_idx = {r["idx"]: r for r in inst}
    pgeom = letterbox_params(h, w, geom_size)
    lab_geom = letterbox_label(label, pgeom)
    masks = {r["idx"]: (lab_geom == r["seg_id"]) for r in inst}
    areas = {i: int(m.sum()) for i, m in masks.items()}
    rng = np.random.default_rng(seed + int(hashlib.sha1(str(rec["image_id"]).encode()).hexdigest()[:6], 16))

    gt = {(int(s), int(o), int(p)) for s, o, p in rec["relations"]}
    pred_id = {name: i for i, name in enumerate(predicates)}  # model embedding index
    pred_off = {name: predicate_classes.index(name) for name in predicates}

    stats_cache: dict[tuple[int, int], tuple] = {}
    edt_cache: dict[int, tuple] = {}

    def stats(s: int, o: int):
        key = (s, o)
        if key not in stats_cache:
            if areas.get(s, 0) == 0 or areas.get(o, 0) == 0:
                stats_cache[key] = None
            else:
                if o not in edt_cache:
                    edt_cache[o] = distance_transform_edt(~masks[o], return_indices=True)
                stats_cache[key] = pair_stats(masks[s], masks[o], pgeom["scale"], edt_cache[o])
        return stats_cache[key]

    queries: list[dict] = []
    unknown = 0
    for pname in predicates:
        pid = pred_id[pname]
        off = pred_off[pname]
        rule = RULES[pname]
        total_area = float(h * w)

        pos_pairs = {(s, o) for s, o, p in gt if p == off and s in by_idx and o in by_idx}
        selected: list[dict] = []

        # Verified positives: official PSG triples (kept even if outside the
        # heuristic category set, so our rule never deletes a positive).
        for s, o in sorted(pos_pairs):
            st = stats(s, o)
            if st is None:
                continue
            gap, overlap, cpt = st
            selected.append(
                dict(pred=pname, pid=pid, psg_pid=off, s=s, o=o, label=1, source="psg_positive", gap=gap,
                     overlap=overlap, contact=((cpt[0] - pgeom["px"]) / pgeom["scale"],
                                               (cpt[1] - pgeom["py"]) / pgeom["scale"]))
            )

        # Candidate verified negatives from geometric rules.
        subs = [r["idx"] for r in inst if r["cat"] in rule.subjects and r["area"] >= min_instance_px]
        objs = [r["idx"] for r in inst if r["cat"] in rule.objects and r["area"] >= min_instance_px]
        cands = []
        for s in subs:
            sa = by_idx[s]["area"]
            for o in objs:
                if s == o or (s, o) in pos_pairs:
                    continue
                if rule.max_obj_area_ratio is not None and by_idx[o]["area"] > rule.max_obj_area_ratio * sa:
                    continue
                st = stats(s, o)
                if st is None:
                    continue
                gap, overlap, cpt = st
                if gap > float(neg_gap_px[pname]):
                    cands.append(
                        dict(pred=pname, pid=pid, psg_pid=off, s=s, o=o, label=0, source="geom_far",
                             gap=gap, overlap=overlap,
                             contact=((cpt[0] - pgeom["px"]) / pgeom["scale"],
                                      (cpt[1] - pgeom["py"]) / pgeom["scale"]),
                             sub_cat=by_idx[s]["cat"], obj_cat=by_idx[o]["cat"])
                    )
                else:
                    unknown += 1
        selected.extend(_select_negatives(cands, {(by_idx[s]["cat"], by_idx[o]["cat"]) for s, o in pos_pairs},
                                          max_negs_per_pred_image, rng))

        for q in selected:
            s, o = q["s"], q["o"]
            if s not in by_idx or o not in by_idx:
                continue
            sb, ob = by_idx[s]["box"], by_idx[o]["box"]
            ub = np.array([min(sb[0], ob[0]), min(sb[1], ob[1]), max(sb[2], ob[2]), max(sb[3], ob[3])],
                          dtype=np.float32)
            actions = build_action_boxes(sb, ob, q["contact"], q["gap"], w, h)
            q.update(
                sub_idx=s, obj_idx=o, sub_seg=by_idx[s]["seg_id"], obj_seg=by_idx[o]["seg_id"],
                sub_cat=by_idx[s]["cat"], obj_cat=by_idx[o]["cat"],
                boxes=np.stack([sb, ob, ub, actions[0]]),
                actions=actions,
                geom=geometry_features(sb, ob, ub, q["gap"], q["overlap"], w, h),
                image_hw=(h, w),
                image_uid=str(rec["image_id"]),
                coco_id=str(rec.get("coco_image_id", rec["image_id"])),
                file_name=rec["file_name"],
                pan_seg_file_name=rec["pan_seg_file_name"],
            )
            q["qid"] = hashlib.sha1(
                f"{rec['image_id']}|{by_idx[s]['seg_id']}|{by_idx[o]['seg_id']}|{q['pid']}".encode()
            ).hexdigest()[:16]
            queries.append(q)

    # Keep at most max_negs_per_image negatives, positives always retained.
    pos = [q for q in queries if q["label"] == 1]
    neg = [q for q in queries if q["label"] == 0]
    if len(neg) > max_negs_per_image:
        neg.sort(key=lambda q: (q.get("same_cat", 1), q.get("jitter", 0.0)))
        neg = neg[:max_negs_per_image]
    return pos + neg, {"n_unknown_skipped": unknown, "n_pos": len(pos), "n_neg": len(neg)}
