"""Rebuild OpenPSG-style records from the HF mirror of PSG train (JosephZ/psg_train_sg)
plus the local COCO-2017 panoptic annotations.

The HF rows carry ``image_id``, the JPEG bytes, ``objects`` (``"<category>.<k>"`` ids with
xyxy pixel boxes) and ``relationships`` (subject / predicate / object by object id).  PSG
objects are exactly the COCO panoptic segments of the same image, so each object is matched
to a panoptic ``segments_info`` entry by category name + box (xywh -> xyxy) and gets its
``seg_id`` for the mask.  Output records have the fields ``barv/psg.py`` consumes:
``image_id, file_name, pan_seg_file_name, segments_info[{id, category_id, area, isthing}],
annotations[{bbox xyxy}], relations[[s, o, p]]`` with PSG 0-based category ids
(80 things then 53 stuff, COCO panoptic order) and predicate ids into ``predicate_classes``.

usage:
  python -m barv.hf_psg --parquet-dir /root/rivermind-data/psg_hf \
      --panoptic-json /root/rivermind-data/psg_hf/panoptic/annotations/panoptic_train2017.json \
      --out /root/rivermind-data/psg_hf/psg_train_from_hf.json [--images-out DIR] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

# OpenPSG predicate list (56 classes, official order).
PSG_PREDICATES = [
    "over", "in front of", "beside", "on", "in", "attached to", "hanging from", "on back of",
    "falling off", "going down", "painted on", "walking on", "running on", "crossing", "standing on",
    "lying on", "sitting on", "flying over", "jumping over", "jumping from", "wearing", "holding",
    "carrying", "looking at", "guiding", "kissing", "eating", "drinking", "feeding", "biting",
    "catching", "picking", "playing with", "chasing", "climbing", "cleaning", "playing", "touching",
    "pushing", "pulling", "opening", "cooking", "talking to", "throwing", "slicing", "driving",
    "riding", "parked on", "driving on", "about to hit", "kicking", "swinging", "entering", "exiting",
    "enclosing", "leaning on",
]


def psg_categories(panoptic_json: dict):
    cats = panoptic_json["categories"]
    things = [c for c in cats if c["isthing"]]
    stuff = [c for c in cats if not c["isthing"]]
    ordered = things + stuff                      # PSG: 80 things then 53 stuff
    name_to_pid = {c["name"]: i for i, c in enumerate(ordered)}
    coco_to_pid = {c["id"]: i for i, c in enumerate(ordered)}
    return name_to_pid, coco_to_pid, [c["name"] for c in ordered]


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], np.float32)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def layout_key(items):
    """Image fingerprint: sorted (category name, x1, y1, x2, y2) of all its segments/objects.
    The HF ``image_id`` is PSG's internal id, not the COCO id, so images are matched to the
    COCO panoptic record by this layout (PSG objects are exactly the panoptic segments;
    100/100 exact matches on shard 0)."""
    return tuple(sorted(items))


def build_layout_index(panoptic_jsons, cat_by_id: dict):
    index = {}
    for split, d in panoptic_jsons:
        for a in d["annotations"]:
            items = [(cat_by_id[s["category_id"]], s["bbox"][0], s["bbox"][1],
                      s["bbox"][0] + s["bbox"][2], s["bbox"][1] + s["bbox"][3]) for s in a["segments_info"]]
            index[layout_key(items)] = (split, a)
    return index


def convert_row(row: dict, pan_ann: dict, name_to_pid: dict, coco_to_pid: dict, pred_to_id: dict,
                stats: Counter):
    """One HF row + its panoptic annotation -> OpenPSG-style record (or None)."""
    objects = json.loads(row["objects"]) if isinstance(row["objects"], str) else row["objects"]
    rels = json.loads(row["relationships"]) if isinstance(row["relationships"], str) else row["relationships"]
    segs = pan_ann["segments_info"]
    seg_boxes = [_xywh_to_xyxy(s["bbox"]) for s in segs]
    seg_pid = [coco_to_pid[s["category_id"]] for s in segs]
    used = set()
    obj_index = {}
    segments_info, annotations = [], []
    for k, ob in enumerate(objects):
        name = ob["id"].rsplit(".", 1)[0]
        pid = name_to_pid.get(name)
        if pid is None:
            stats["obj_unknown_category"] += 1
            continue
        box = np.asarray(ob["bbox"], np.float32)
        best, best_iou = None, 0.0
        for j, (sb, sp) in enumerate(zip(seg_boxes, seg_pid)):
            if j in used or sp != pid:
                continue
            v = _iou(box, sb)
            if v > best_iou:
                best, best_iou = j, v
        if best is None or best_iou < 0.7:
            stats["obj_unmatched"] += 1
            continue
        used.add(best)
        s = segs[best]
        obj_index[ob["id"]] = len(segments_info)
        segments_info.append({"id": int(s["id"]), "category_id": int(pid), "area": int(s["area"]),
                              "isthing": int(pid < 80), "iscrowd": int(s.get("iscrowd", 0))})
        annotations.append({"bbox": [float(x) for x in seg_boxes[best]], "category_id": int(pid)})
        stats["obj_matched"] += 1
        stats["obj_iou_lt_0.99"] += int(best_iou < 0.99)
    relations = []
    for r in rels:
        p = pred_to_id.get(r["predicate"])
        s, o = obj_index.get(r["subject"]), obj_index.get(r["object"])
        if p is None:
            stats["rel_unknown_predicate"] += 1
            continue
        if s is None or o is None:
            stats["rel_dropped_endpoint"] += 1
            continue
        relations.append([s, o, p])
        stats["rel_kept"] += 1
    if len(segments_info) < 2:
        stats["img_lt2_segments"] += 1
        return None
    split = pan_ann["_split"]
    return {"image_id": int(pan_ann["image_id"]), "psg_image_id": str(row["image_id"]),
            "coco_image_id": int(pan_ann["image_id"]),
            "file_name": f"{split}2017/" + pan_ann["file_name"].replace(".png", ".jpg"),
            "pan_seg_file_name": f"panoptic_{split}2017/" + pan_ann["file_name"],
            "segments_info": segments_info, "annotations": annotations, "relations": relations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--panoptic-json", required=True, help="panoptic_train2017.json")
    ap.add_argument("--panoptic-val-json", default=None, help="panoptic_val2017.json (PSG also uses val2017 images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images-out", default=None, help="if set, write JPEGs here (basename of file_name)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    pan = json.load(open(args.panoptic_json))
    name_to_pid, coco_to_pid, classes = psg_categories(pan)
    cat_by_id = {c["id"]: c["name"] for c in pan["categories"]}
    jsons = [("train", pan)]
    if args.panoptic_val_json:
        jsons.append(("val", json.load(open(args.panoptic_val_json))))
    index = build_layout_index(jsons, cat_by_id)
    print(f"layout index: {len(index)} panoptic images", flush=True)
    pred_to_id = {p: i for i, p in enumerate(PSG_PREDICATES)}
    stats = Counter()
    records = []
    files = sorted(glob.glob(os.path.join(args.parquet_dir, "train-*.parquet")))
    if args.images_out:
        os.makedirs(args.images_out, exist_ok=True)
    for f in files:
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:  # incomplete download in progress
            print(f"skip {os.path.basename(f)}: {e}", flush=True)
            stats["shards_skipped"] += 1
            continue
        for rg in range(pf.num_row_groups):
            cols = ["image_id", "objects", "relationships"] + (["image"] if args.images_out else [])
            for row in pf.read_row_group(rg, columns=cols).to_pylist():
                objects = json.loads(row["objects"])
                items = [(o["id"].rsplit(".", 1)[0], *[int(round(x)) for x in o["bbox"]]) for o in objects]
                hit = index.get(layout_key(items))
                if hit is None:
                    stats["img_no_panoptic_match"] += 1
                    continue
                pa = dict(hit[1])
                pa["_split"] = hit[0]
                rec = convert_row(row, pa, name_to_pid, coco_to_pid, pred_to_id, stats)
                if rec is None:
                    continue
                if args.images_out:
                    dst = os.path.join(args.images_out, os.path.basename(rec["file_name"]))
                    if not os.path.exists(dst):
                        with open(dst, "wb") as fh:
                            fh.write(row["image"]["bytes"])
                records.append(rec)
                stats["img_kept"] += 1
                if args.limit and len(records) >= args.limit:
                    break
            if args.limit and len(records) >= args.limit:
                break
        print(f"{os.path.basename(f)}: kept {stats['img_kept']} images so far; {dict(stats)}", flush=True)
        if args.limit and len(records) >= args.limit:
            break
    out = {"thing_classes": classes[:80], "stuff_classes": classes[80:], "predicate_classes": PSG_PREDICATES,
           "data": records, "source": "JosephZ/psg_train_sg + COCO panoptic_train2017", "stats": dict(stats)}
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out, "images:", len(records), "stats:", dict(stats))


if __name__ == "__main__":
    main()
