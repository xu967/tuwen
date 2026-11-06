"""Run the released RelateAnything model (paper 2609.12552, maelic/relsgg-vits16plus) on a
relate data yaml with ground-truth boxes and dump graph-constrained predictions in the
format of tools/sgg_protocol_eval.py.

Protocol (= official benchmark/eval_zeroshot.py defaults): GT boxes, no object labels,
image resized to 448x448, vocabulary = the yaml's ``rnames`` encoded by the shipped text
student (template ensemble), sampler eval budget 500 (-> min(500, geo_budget 400) = 400 pairs),
score = sigmoid(a * (pred_logit + pair_logit) + b) with the shipped calibration (monotone).

Environment: needs the official repo (git@github.com:Maelic/RelateAnything.git) on PYTHONPATH and
transformers >= 4.56 (DINOv3). The release was exported with transformers 5.x, whose DINOv3
state_dict nests one more ``model.`` level; keys are remapped here, loading is then exact.

usage:
  USE_TF=0 PYTHONPATH=/root/rivermind-data/ra_overlay:/tmp/yolovo-audit-numpy2:/root/rivermind-data/RelateAnything \
  python tools/run_ra_official_predcls.py --release assets/relsgg-vits16plus --data RA-4M/data_full_monitor.yaml \
      --out reports/baseline_cmp/ra_official.pt [--budget 500] [--device cuda]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--release", default="assets/relsgg-vits16plus")
ap.add_argument("--data", default="RA-4M/data_full_monitor.yaml")
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=500, help="eval pair budget (official default 500)")
ap.add_argument("--max-boxes", type=int, default=100, help="official eval default max_objects=100")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

from relsgg.checkpoint import load_checkpoint, build_model_from_ckpt  # noqa: E402
from relsgg.scoring import ScoreContract  # noqa: E402
from relsgg.text import student as _student  # noqa: E402
from relsgg.text.student import encode_texts_student  # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402


def _ensure_fast_tokenizer(self):
    """The release ships ``tokenizer.json`` only (fast format); transformers 4.x's slow
    CLIPTokenizer needs vocab.json+merges.txt, so use the fast class. Patched on the class
    because the official backbone loader re-imports ``transformers`` (module-level patches are lost)."""
    if self._tokenizer is None:
        from transformers import CLIPTokenizerFast
        self._tokenizer = CLIPTokenizerFast.from_pretrained(self.tokenizer_src or _student.CLIP_TOKENIZER_ID)
    return self._tokenizer


_student.PredicateTextStudent._ensure_tokenizer = _ensure_fast_tokenizer

t0 = time.time()
ckpt_path = os.path.join(args.release, "model.pth")
ck = load_checkpoint(ckpt_path)
model = build_model_from_ckpt(ck, weights="ema", strict=False).eval()
sd, tgt = ck["model"], model.state_dict()
remap = {}
for k, v in sd.items():
    k2 = "backbone.model." + k[len("backbone.model.model."):] if (k not in tgt and k.startswith("backbone.model.model.")) else k
    remap[k2] = v
assert set(remap) == set(tgt), (sorted(set(tgt) - set(remap))[:5], sorted(set(remap) - set(tgt))[:5])
assert all(tuple(remap[k].shape) == tuple(tgt[k].shape) for k in tgt)
model.load_state_dict(remap, strict=True)
model.to(args.device)
# official evaluate(): raw.sampler.final_budget = min(eval_budget, geo_budget)
model.sampler.final_budget = min(args.budget, model.sampler.geo_budget)
print(f"model loaded ({time.time() - t0:.0f}s); budgets geo={model.sampler.geo_budget} final={model.sampler.final_budget}")

# vocabulary: yaml rnames -> text student embeddings (known strings equal ckpt W exactly, cos=1.0)
data = yaml.safe_load(open(args.data))
rn = {int(k): str(v) for k, v in data["rnames"].items()}
ids = sorted(rn)
names = [rn[i] for i in ids]
known = {n: i for i, n in enumerate(ck["pred_names"])}
W = torch.zeros(len(names), sd["vocab_head.W"].shape[1])
todo = [i for i, n in enumerate(names) if n not in known]
for i, n in enumerate(names):
    if n in known:
        W[i] = sd["vocab_head.W"][known[n]]
if todo:
    W_new = encode_texts_student([names[i] for i in todo], os.path.join(args.release, "text_student.pt"),
                                 templates=TRAIN_TEMPLATES, device="cpu")
    W[torch.tensor(todo)] = W_new.float().cpu()
print(f"vocabulary: {len(names)} strings, {len(names) - len(todo)} from checkpoint bank, {len(todo)} encoded by text student")
model.vocab_head.set_vocabulary_matrix(names, W.to(args.device))
model.reparameterize()
contract = ScoreContract.for_checkpoint(ckpt_path)
print("score contract:", contract)

root = data["path"]
img_dir = os.path.join(root, data["val"])
lbl_dir = os.path.join(root, data["val"].replace("images", "labels"))
files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
if args.limit:
    files = files[: args.limit]
records = []
n_trunc = 0
with torch.no_grad():
    for n_done, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        boxes, rels = [], []
        for line in open(os.path.join(lbl_dir, stem + ".txt")):
            p = line.split()
            if not p:
                continue
            if p[0] == "R":
                rels.append((int(p[1]), int(p[2]), int(p[3])))
            else:
                boxes.append([float(x) for x in p[1:5]])  # normalised cxcywh (yaml rcoords default)
        gt = torch.tensor(rels, dtype=torch.long).reshape(-1, 3)
        N = len(boxes)
        rec = {"image": f, "n_boxes": N, "gt": gt, "pred": torch.zeros((0, 4))}
        if N >= 2:
            if N > args.max_boxes:
                n_trunc += 1
            pil = Image.open(f).convert("RGB").resize((448, 448), Image.BILINEAR)
            img = torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(args.device)
            b = torch.tensor(boxes[: args.max_boxes], dtype=torch.float32).unsqueeze(0).to(args.device)
            out = model(img, b, box_counts=torch.tensor([b.shape[1]], device=args.device), targets=None)
            logits = out["logits"][0].float()
            pair = out["pair_logits"][0].float()
            scores = contract.scores(logits, pair)               # sigmoid(a*(pred+pair)+b), [K,V]
            best_s, best_p = scores.max(dim=-1)
            si, oi = out["sub_idx"][0], out["obj_idx"][0]
            keep = out["valid_mask"][0].bool() & (si != oi)
            rec["pred"] = torch.stack([si[keep].float(), oi[keep].float(), best_p[keep].float(), best_s[keep]], 1).cpu()
        records.append(rec)
        if (n_done + 1) % 100 == 0:
            print(f"  {n_done + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
out = {"records": records, "release": args.release, "data": args.data, "budget": model.sampler.final_budget,
       "max_boxes": args.max_boxes, "n_images_truncated": n_trunc, "vocab_size": len(names),
       "vocab_from_bank": len(names) - len(todo), "contract": repr(contract)}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(out, args.out)
print(f"dumped {len(records)} images -> {args.out} ({time.time() - t0:.0f}s); images with >{args.max_boxes} boxes truncated: {n_trunc}")
