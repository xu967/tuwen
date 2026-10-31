"""Train the coarse relation scorer, offline fine-action teacher, and MVE value head.

Protocol notes
- Relation scorer: trained on train_core only; dev selects the epoch.
- Teacher: frozen scorer applied out-of-sample to train_teacher images; fine
  embeddings are model-independent and cached offline (training compute only).
- Value head: trained on teacher gains; 90/10 image split inside train_teacher
  for early stopping. Test/psgval labels are never touched.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .metrics import binary_metrics
from .model import ACTION_FEAT_DIM, ACTION_TYPES, RelationScorer, ValueHead, action_features, build_backbone

SPLITS = ["train_core", "train_teacher", "dev", "calib", "test", "psgval"]


class QueryCache:
    def __init__(self, path: str):
        z = np.load(path)
        self.z = z
        self.pool = torch.from_numpy(z["pool"]).float()
        self.geom = torch.from_numpy(z["geom"]).float()
        self.pred = torch.from_numpy(z["pred"]).long()
        self.label = torch.from_numpy(z["label"]).long()
        self.split = torch.from_numpy(z["split"]).long()
        self.actions = torch.from_numpy(z["actions"]).float()
        self.boxes = torch.from_numpy(z["boxes"]).float()
        self.image_uid = z["image_uid"]
        self.qid = z["qid"]
        self.coco_id = z["coco_id"]
        self.file_name = z["file_name"]
        self.pan_seg_file_name = z["pan_seg_file_name"]
        self.full_index = torch.from_numpy(z["full_index"]).long()
        full_path = path.replace(".npz", "_fullpool.npy")
        self.pool_full = torch.from_numpy(np.load(full_path)).float() if os.path.exists(full_path) else None

    def __len__(self):
        return self.pool.shape[0]

    def split_mask(self, name: str) -> np.ndarray:
        return (self.z["split"] == SPLITS.index(name))

    def image_mask_for_splits(self, names) -> np.ndarray:
        m = np.zeros(len(self), bool)
        for n in names:
            m |= self.split_mask(n)
        return m


def subset_train_mask(cache: QueryCache, frac: float, seed: int) -> np.ndarray:
    """Sample a fraction of training IMAGES; all queries of an image stay together."""
    train_img = np.unique(cache.image_uid[cache.image_mask_for_splits(("train_core", "train_teacher"))])
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * len(train_img))))
    pick = set(rng.choice(train_img, size=k, replace=False).tolist())
    keep_img = np.array([u in pick for u in cache.image_uid])
    return keep_img & cache.image_mask_for_splits(("train_core", "train_teacher"))


def balance_positives(cache: QueryCache, mask: np.ndarray, ratio: float = 4.0,
                      seed: int = 0) -> np.ndarray:
    """Down-sample positives per predicate to at most ratio x its negatives.

    The geometric rule is deliberately conservative for heavy-tailed predicates
    (e.g. holding has 13x more positives than provable negatives), so training on
    the raw pool would over-predict those predicates. Dropping positive examples
    is label-safe; no negative is invented.
    """
    out = mask.copy()
    rng = np.random.default_rng(seed)
    pred = cache.pred.numpy()
    label = cache.label.numpy()
    for i in np.unique(pred[mask]):
        pos = np.nonzero(mask & (pred == i) & (label == 1))[0]
        neg = np.nonzero(mask & (pred == i) & (label == 0))[0]
        cap = int(ratio * max(1, len(neg)))
        if len(pos) > cap:
            drop = rng.choice(pos, size=len(pos) - cap, replace=False)
            out[drop] = False
    return out


def _batch_to_device(cache: QueryCache, idx, device):
    i = torch.as_tensor(idx, dtype=torch.long)
    return (cache.pool[i].to(device, non_blocking=True), cache.geom[i].to(device, non_blocking=True),
            cache.pred[i].to(device, non_blocking=True), cache.label[i].to(device, non_blocking=True))


@torch.no_grad()
def predict_logits(model, cache, idx, device, batch=8192, pool_mode="full"):
    model.eval()
    out = np.zeros(len(idx), np.float32)
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        pool, geom, pred, _ = _batch_to_device(cache, ii, device)
        if pool_mode == "no_visual":
            pool = torch.zeros_like(pool)
        out[s:s + batch] = model(pool, geom, pred)[1].float().cpu().numpy()
    return out


def train_relation(cache: QueryCache, cfg: dict, train_mask: np.ndarray, dev_mask: np.ndarray,
                   out_path: str, pool_mode: str = "full", seed: int | None = None) -> dict:
    seed_everything(int(seed if seed is not None else cfg["seed"]))
    device = torch.device(cfg["hardware"].get("train_device", "cuda:0"))
    tr, dv = np.nonzero(train_mask)[0], np.nonzero(dev_mask)[0]
    y = cache.label.numpy()
    pos = float((y[tr] == 1).sum())
    neg = float((y[tr] == 0).sum())
    pos_weight = torch.tensor(min(20.0, max(0.2, neg / max(pos, 1.0))), device=device)
    n_pred = int(cache.pred.max()) + 1
    model = RelationScorer(pool_dim=4 * 768, geom_dim=24, n_pred=n_pred,
                           pred_dim=int(cfg["model"]["pred_dim"]), hidden=int(cfg["model"]["hidden"]),
                           state_dim=int(cfg["model"]["state_dim"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["relation_lr"]),
                            weight_decay=float(cfg["train"]["relation_wd"]))
    bs = int(cfg["train"]["relation_batch"])
    best = {"auroc": -1.0, "epoch": -1, "state": None}
    hist = []
    rng = np.random.default_rng(int(seed if seed is not None else cfg["seed"]))
    for ep in range(int(cfg["train"]["relation_epochs"])):
        model.train()
        perm = rng.permutation(tr)
        losses = []
        for s in range(0, len(perm), bs):
            ii = perm[s:s + bs]
            pool, geom, pred, yy = _batch_to_device(cache, ii, device)
            if pool_mode == "no_visual":
                pool = torch.zeros_like(pool)
            logit = model(pool, geom, pred)[1]
            loss = F.binary_cross_entropy_with_logits(logit, yy.float(), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        logits = predict_logits(model, cache, dv, device, pool_mode=pool_mode)
        m = binary_metrics(y[dv], 1 / (1 + np.exp(-logits)))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev": m})
        print(f"[relation] ep={ep} loss={np.mean(losses):.4f} dev_auroc={m['auroc']:.4f} "
              f"dev_bacc={m['balanced_acc']:.4f}", flush=True)
        if m["auroc"] > best["auroc"]:
            best = {"auroc": m["auroc"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif ep - best["epoch"] >= int(cfg["train"]["patience"]):
            break
    if best["state"] is None:
        best = {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": int(cfg["train"]["relation_epochs"]) - 1, "auroc": float("nan")}
    model.load_state_dict(best["state"])
    torch.save({"state": best["state"], "n_pred": n_pred,
                "pred_dim": int(cfg["model"]["pred_dim"]), "hidden": int(cfg["model"]["hidden"]),
                "state_dim": int(cfg["model"]["state_dim"]), "pool_mode": pool_mode,
                "pos_weight": float(pos_weight), "best_epoch": best["epoch"]}, out_path)
    save_json({"history": hist, "best_epoch": best["epoch"], "pos_weight": float(pos_weight),
               "pool_mode": pool_mode, "n_train": int(len(tr)), "n_dev": int(len(dv))},
              out_path.replace(".pt", "_history.json"))
    print(f"[relation] saved {out_path} best_epoch={best['epoch']} auroc={best['auroc']:.4f}")
    return {"model": model, "best_auroc": best["auroc"], "history": hist, "state_path": out_path}


def train_prior_baseline(cache: QueryCache, train_mask, dev_mask, out_path: str, seed: int = 0) -> dict:
    """Q0: predicate + geometry only (no image) logistic baseline."""
    from sklearn.linear_model import LogisticRegression

    def feats(idx):
        g = cache.geom.numpy()[idx]
        p = cache.pred.numpy()[idx]
        one = np.eye(int(cache.pred.max()) + 1, dtype=np.float32)[p]
        return np.concatenate([g, one], 1)

    tr, dv = np.nonzero(train_mask)[0], np.nonzero(dev_mask)[0]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(feats(tr), cache.label.numpy()[tr])
    pv = clf.predict_proba(feats(dv))[:, 1]
    m = binary_metrics(cache.label.numpy()[dv], pv)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    save_json({"dev": m}, out_path.replace(".pkl", "_dev.json"))
    print(f"[prior] dev {m}")
    return {"clf": clf, "dev": m}


def _crop_resize(img, box, size=224):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = int(np.floor(max(0, x1))), int(np.floor(max(0, y1)))
    x2, y2 = int(np.ceil(min(w, x2))), int(np.ceil(min(h, y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def precompute_teacher_fine(cache: QueryCache, cfg: dict, out_path: str) -> dict:
    """Fine embeddings for every TRAIN_TEACHER query x 6 actions (model-independent).

    This is offline training compute and is never counted as lazy inference saving.
    """
    idx = np.nonzero(cache.split_mask("train_teacher"))[0]
    images = cache.image_uid[idx]
    uniq = np.unique(images)
    device = torch.device(cfg["hardware"].get("train_device", "cuda:0"))
    backbone = build_backbone(cfg).to(device).eval()
    fine = np.zeros((len(idx), 6, 512), np.float16)
    legal = np.zeros((len(idx), 6), bool)
    image_hw = np.zeros((len(idx), 2), np.int32)
    pos_of = {u: [] for u in uniq}
    for k, u in enumerate(images):
        pos_of[u].append(k)
    t0 = time.time()
    for n, u in enumerate(uniq):
        rows = pos_of[u]
        img_path = os.path.join(cfg["paths"]["images"], os.path.basename(str(cache.file_name[idx[rows[0]]])))
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        crops, meta, valid = [], [], set()
        for k in rows:
            image_hw[k] = (h, w)
            for a in range(6):
                c = _crop_resize(img, cache.actions[idx[k], a].numpy(), int(cfg["model"]["fine_size"]))
                if c is not None:
                    meta.append((k, a))
                    crops.append(c)
                    valid.add((k, a))
        if crops:
            assert len(meta) == len(crops)
            x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(device)
            x = backbone.preprocess(x)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=bool(cfg["hardware"]["amp"]) and device.type == "cuda",
                                dtype=torch.float16):
                uu = backbone.forward_fine(x)
            uu = uu.float().cpu().numpy().astype(np.float16)
            for (k, a), feat in zip(meta, uu):
                fine[k, a] = feat
        for k in rows:
            hh, ww = image_hw[k]
            for a in range(6):
                x1, y1, x2, y2 = cache.actions[idx[k], a].numpy()
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (hh * ww)
                legal[k, a] = ((x2 - x1 >= 16 and y2 - y1 >= 16
                                and area >= cfg["runtime"]["min_area_frac"]
                                and area <= cfg["runtime"]["max_area_frac"])
                               and (k, a) in valid)
        if (n + 1) % 200 == 0:
            print(f"[fine] {n+1}/{len(uniq)} images, {time.time()-t0:.0f}s", flush=True)
    np.savez(out_path, qid=cache.qid[idx], image_uid=images, fine=fine, legal=legal, image_hw=image_hw)
    stats = {"n_queries": int(len(idx)), "n_images": int(len(uniq)), "elapsed_sec": round(time.time() - t0, 1),
             "legal_per_action": legal.mean(0).tolist()}
    save_json(stats, out_path.replace(".npz", "_stats.json"))
    print("[fine]", stats)
    return stats


@torch.no_grad()
def compute_teacher_gains(cache: QueryCache, fine_path: str, relation_pt: str, cfg: dict,
                          subset_mask: np.ndarray, out_path: str) -> dict:
    """Out-of-sample teacher: scorer trained on train_core, gains on train_teacher."""
    device = torch.device(cfg["hardware"].get("train_device", "cuda:0"))
    ck = torch.load(relation_pt, map_location="cpu", weights_only=False)
    model = RelationScorer(pool_dim=4 * 768, geom_dim=24, n_pred=ck["n_pred"],
                           pred_dim=ck["pred_dim"], hidden=ck["hidden"],
                           state_dim=ck["state_dim"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    f = np.load(fine_path)
    f_qid, f_legal, f_hw = f["qid"][:], f["legal"][:], f["image_hw"][:]
    f_fine = f["fine"][:]
    qid_to_row = {q: i for i, q in enumerate(f_qid)}
    idx = np.nonzero(subset_mask & cache.split_mask("train_teacher"))[0]
    idx = np.array([i for i in idx if cache.qid[i] in qid_to_row], dtype=np.int64)
    if len(idx) == 0:
        raise RuntimeError("empty teacher gain split; check split_teacher_images / fine cache coverage")
    rows = np.array([qid_to_row[cache.qid[i]] for i in idx], dtype=np.int64)
    ys = cache.label[idx].float().to(device)
    h, logit0 = model(cache.pool[idx].to(device), cache.geom[idx].to(device), cache.pred[idx].to(device))
    p0 = torch.sigmoid(logit0)
    loss0 = F.binary_cross_entropy_with_logits(logit0, ys, reduction="none")
    FEAT = np.zeros((len(idx), 6, ACTION_FEAT_DIM), np.float32)
    DELTA = np.full((len(idx), 6), -1.0, np.float32)
    LEGAL = f_legal[rows].copy()
    for a in range(6):
        u = torch.from_numpy(f_fine[rows, a].astype(np.float32)).to(device)
        af = torch.cat([
            action_features(cache.actions[i, a:a + 1], cache.boxes[i],
                            tuple(f_hw[rows[j]]), torch.zeros(1), torch.tensor([0.49]),
                            [ACTION_TYPES[a]])
            for j, i in enumerate(idx)], 0).to(device)
        h2, logit1 = model.update(h, u, cache.pred[idx].to(device), af, None)
        loss1 = F.binary_cross_entropy_with_logits(logit1, ys, reduction="none")
        d = (loss0 - loss1).float().cpu().numpy()
        FEAT[:, a] = af.cpu().numpy()
        m = LEGAL[:, a]
        DELTA[m, a] = d[m]
    np.savez(out_path, qid=cache.qid[idx], image_uid=cache.image_uid[idx], pred=cache.pred[idx].numpy(),
             label=cache.label[idx].numpy(), h=h.float().cpu().numpy(), p0=p0.cpu().numpy(),
             act_feat=FEAT, delta=DELTA, legal=LEGAL, actions=cache.actions[idx].numpy())
    dmeans, dpos = [], []
    for a in range(6):
        vals = DELTA[LEGAL[:, a], a]
        dmeans.append(float(vals.mean()) if vals.size else None)
        dpos.append(float((vals > 0).mean()) if vals.size else None)
    stats = {"n_queries": int(len(idx)), "n_images": int(len(np.unique(cache.image_uid[idx]))),
             "delta_mean_by_action_legal_only": dmeans,
             "frac_positive_delta_legal_only": dpos,
             "legal_rate_per_action": LEGAL.mean(0).tolist()}
    save_json(stats, out_path.replace(".npz", "_stats.json"))
    print("[teacher]", json.dumps(stats))
    return stats


def _value_scores(model, h, pred, act_feat, p, entropy, idx, device, batch=16384):
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            ii = torch.as_tensor(idx[s:s + batch])
            B = len(ii)
            A = act_feat.shape[1]
            hh = h[ii].to(device)[:, None, :].expand(B, A, -1).reshape(B * A, -1)
            pp = pred[ii].to(device)[:, None].expand(B, A).reshape(-1)
            aa = act_feat[ii].to(device).reshape(B * A, -1)
            pv = p[ii].to(device)[:, None].expand(B, A).reshape(-1)
            ev = entropy[ii].to(device)[:, None].expand(B, A).reshape(-1)
            out.append(model(hh, pp, aa, pv, ev, torch.zeros(B * A, device=device)).view(B, A).cpu())
    return torch.cat(out, 0) if out else torch.zeros((0, act_feat.shape[1]))


def split_teacher_images(cache: QueryCache, teacher_mask: np.ndarray, seed: int,
                         update_frac: float = 0.7):
    """Image-level split of train_teacher: update-module fitting vs teacher/value."""
    idx = np.nonzero(teacher_mask)[0]
    uniq = np.unique(cache.image_uid[idx])
    if len(uniq) < 2:
        raise RuntimeError("train_teacher needs at least 2 images for updater/teacher split")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    cut = max(1, int(round(update_frac * len(uniq))))
    upd_img = set(uniq[:cut].tolist())
    upd = teacher_mask & np.array([u in upd_img for u in cache.image_uid])
    gain = teacher_mask & ~np.array([u in upd_img for u in cache.image_uid])
    return upd, gain


@torch.no_grad()
def _teacher_action_table(cache, f_legal, f_hw, f_fine, rows, idx):
    feats, us = [], []
    for j, i in enumerate(idx):
        for a in range(6):
            if not f_legal[rows[j], a]:
                continue
            af = action_features(cache.actions[i, a:a + 1], cache.boxes[i],
                                 tuple(f_hw[rows[j]]), torch.zeros(1),
                                 torch.tensor([0.49]), [ACTION_TYPES[a]])
            feats.append(af[0])
            us.append(torch.from_numpy(f_fine[rows[j], a].astype(np.float32)))
    return torch.stack(feats), torch.stack(us)


def train_evidence_update(cache: QueryCache, fine_path: str, relation_pt: str, cfg: dict,
                          update_mask: np.ndarray, out_path: str, seed: int | None = None) -> dict:
    """Fit only the gated fine-evidence update against train_teacher labels.

    The coarse scorer and its head stay frozen (S2 in the protocol); this prevents
    the coarse decision from drifting and keeps the attribution of any gain to
    actually acquiring a crop.
    """
    seed_everything(int(seed if seed is not None else cfg["seed"]))
    device = torch.device(cfg["hardware"].get("train_device", "cuda:0"))
    ck = torch.load(relation_pt, map_location="cpu", weights_only=False)
    model = RelationScorer(pool_dim=4 * 768, geom_dim=24, n_pred=ck["n_pred"],
                           pred_dim=ck["pred_dim"], hidden=ck["hidden"],
                           state_dim=ck["state_dim"]).to(device)
    model.load_state_dict(ck["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    for p in list(model.update_gate.parameters()) + list(model.update_val.parameters()):
        p.requires_grad_(True)
    f = np.load(fine_path)
    f_qid, f_legal, f_hw, f_fine = f["qid"][:], f["legal"][:], f["image_hw"][:], f["fine"][:]
    q2r = {q: i for i, q in enumerate(f_qid)}
    idx = np.array([i for i in np.nonzero(update_mask)[0] if cache.qid[i] in q2r], dtype=np.int64)
    if len(idx) == 0:
        raise RuntimeError("empty evidence-update split; check train_teacher images")
    rows = np.array([q2r[cache.qid[i]] for i in idx], dtype=np.int64)
    ys = cache.label[idx].float().to(device)
    h0 = model.encode(cache.pool[idx].to(device), cache.geom[idx].to(device), cache.pred[idx].to(device))
    feats, us = _teacher_action_table(cache, f_legal, f_hw, f_fine, rows, idx)
    feats, us = feats.to(device), us.to(device)
    # repeat query-level tensors for each legal action
    rep = []
    for j, i in enumerate(idx):
        rep.extend([j] * int(f_legal[rows[j]].sum()))
    rep = torch.as_tensor(rep, device=device)
    h_rep, pred_rep, y_rep = h0[rep], cache.pred[idx].to(device)[rep], ys[rep]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(cfg["train"].get("update_lr", 0.001)),
                            weight_decay=float(cfg["train"].get("value_wd", 0.01)))
    bs = int(cfg["train"].get("update_batch", 4096))
    best = {"auroc": -1, "state": None, "epoch": -1}
    for ep in range(int(cfg["train"].get("update_epochs", 8))):
        model.train()
        perm = torch.randperm(len(h_rep), device=device)
        losses = []
        for s in range(0, len(perm), bs):
            ii = perm[s:s + bs]
            h1, logit = model.update(h_rep[ii], us[ii], pred_rep[ii], feats[ii], None)
            loss = F.binary_cross_entropy_with_logits(logit, y_rep[ii])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            h1, logit = model.update(h_rep, us, pred_rep, feats, None)
            m = binary_metrics(y_rep.cpu().numpy(), torch.sigmoid(logit).float().cpu().numpy())
        print(f"[update] ep={ep} loss={np.mean(losses):.4f} action_auroc={m['auroc']:.4f}", flush=True)
        if m["auroc"] > best["auroc"]:
            best = {"auroc": m["auroc"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif ep - best["epoch"] >= int(cfg["train"].get("patience", 8)):
            break
    if best["state"] is None:
        best = {"auroc": float("nan"), "epoch": -1,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    torch.save({"state": best["state"], "n_pred": ck["n_pred"], "pred_dim": ck["pred_dim"],
                "hidden": ck["hidden"], "state_dim": ck["state_dim"], "pool_mode": "full",
                "best_epoch": best["epoch"], "action_auroc": best["auroc"]}, out_path)
    save_json({"best_epoch": best["epoch"], "action_auroc": best["auroc"],
               "n_queries": int(len(idx)), "n_action_samples": int(len(h_rep))},
              out_path.replace(".pt", "_history.json"))
    print(f"[update] saved {out_path} best_epoch={best['epoch']} auroc={best['auroc']:.4f}")
    return {"state_path": out_path, "auroc": best["auroc"], "epoch": best["epoch"]}


def train_value(gains_path: str, cfg: dict, out_path: str, seed: int | None = None,
                n_pred: int | None = None) -> dict:
    seed_everything(int(seed if seed is not None else cfg["seed"]))
    z = np.load(gains_path)
    delta, legal, act_feat = z["delta"], z["legal"], z["act_feat"]
    h, pred, p0 = z["h"], z["pred"], z["p0"]
    image_uid = z["image_uid"]
    device = torch.device(cfg["hardware"].get("train_device", "cuda:0"))
    uniq = np.unique(image_uid)
    rng = np.random.default_rng(int(seed if seed is not None else cfg["seed"]))
    rng.shuffle(uniq)
    cut = max(1, int(0.9 * len(uniq)))
    tr_img = set(uniq[:cut])
    tr = np.array([u in tr_img for u in image_uid])
    va = ~tr
    n_pred = int(n_pred or max(int(pred.max()) + 1, len(cfg["data"]["predicates"])))
    model = ValueHead(n_pred=n_pred, state_dim=int(cfg["model"]["state_dim"]),
                      pred_dim=int(cfg["model"]["pred_dim"]), act_dim=ACTION_FEAT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["value_lr"]),
                            weight_decay=float(cfg["train"]["value_wd"]))
    legal_t, delta_t, act_t = torch.from_numpy(legal), torch.from_numpy(delta), torch.from_numpy(act_feat)
    h_t, pred_t, p_t = torch.from_numpy(h), torch.from_numpy(pred).long(), torch.from_numpy(p0)
    entropy = -(p_t * torch.log(p_t.clamp(min=1e-6)) + (1 - p_t) * torch.log((1 - p_t).clamp(min=1e-6)))
    best = {"rho": -1, "state": None, "epoch": -1}
    bs, hist = int(cfg["train"]["value_batch"]), []
    tr_idx = np.nonzero(tr)[0]
    va_idx = np.nonzero(va)[0]
    can_validate = len(va_idx) > 0 and int(legal_t[va].sum()) >= 2
    for ep in range(int(cfg["train"]["value_epochs"])):
        model.train()
        perm = rng.permutation(tr_idx)
        losses = []
        for s in range(0, len(perm), bs):
            ii = torch.as_tensor(perm[s:s + bs])
            m = legal_t[ii]
            if not m.any():
                continue
            B, A = m.shape
            hh = h_t[ii].to(device)[:, None, :].expand(B, A, -1).reshape(B * A, -1)
            pp = pred_t[ii].to(device)[:, None].expand(B, A).reshape(-1)
            aa = act_t[ii].to(device).reshape(B * A, -1)
            pv = p_t[ii].to(device)[:, None].expand(B, A).reshape(-1)
            ev = entropy[ii].to(device)[:, None].expand(B, A).reshape(-1)
            predv = model(hh, pp, aa, pv, ev, torch.zeros(B * A, device=device)).view(B, A)
            md = m.to(device)
            target = delta_t[ii].to(device).masked_fill(~md, 0.0)
            loss = F.smooth_l1_loss(predv[md], target[md], beta=0.05)
            diff = target[:, :, None] - target[:, None, :]
            pair = (diff > 0.01) & md[:, :, None] & md[:, None, :]
            if pair.any():
                rank = (predv[:, :, None] - predv[:, None, :])
                loss = loss + 0.5 * F.softplus(-rank[pair]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        if can_validate:
            outv = _value_scores(model, h_t, pred_t, act_t, p_t, entropy, va_idx, device)
            vmask = legal_t[va]
            from scipy.stats import spearmanr
            rho = float(spearmanr(outv[vmask].numpy(), delta_t[va][vmask].numpy()).correlation)
            if not np.isfinite(rho):
                rho = -1.0
        else:
            rho = -1.0
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)) if losses else None, "val_spearman": rho})
        print(f"[value] ep={ep} loss={np.mean(losses) if losses else 0:.4f} val_rho={rho:.4f}", flush=True)
        if rho > best["rho"]:
            best = {"rho": rho, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif ep - best["epoch"] >= int(cfg["train"].get("patience", 8)):
            break
    if best["state"] is None:
        best = {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": int(cfg["train"]["value_epochs"]) - 1, "rho": float("nan")}
    model.load_state_dict(best["state"])
    torch.save({"state": best["state"], "n_pred": n_pred, "state_dim": int(cfg["model"]["state_dim"]),
                "pred_dim": int(cfg["model"]["pred_dim"]), "act_dim": ACTION_FEAT_DIM,
                "best_epoch": best["epoch"], "val_spearman": best["rho"],
                "val_images": uniq[cut:] if can_validate else np.array([], dtype=object)}, out_path)
    save_json({"history": hist, "best_epoch": best["epoch"], "val_spearman": best["rho"]},
              out_path.replace(".pt", "_history.json"))
    print(f"[value] saved best_epoch={best['epoch']} rho={best['rho']:.4f}")
    return {"model": model, "rho": best["rho"], "history": hist, "state_path": out_path}


def select_lambda(gains_path: str, value_pt: str, cfg: dict) -> dict:
    z = np.load(gains_path)
    ck = torch.load(value_pt, map_location="cpu", weights_only=False)
    model = ValueHead(n_pred=ck["n_pred"], state_dim=ck["state_dim"], pred_dim=ck["pred_dim"],
                      act_dim=ck["act_dim"])
    model.load_state_dict(ck["state"])
    h, pred, af, delta, legal, p0 = (z["h"], z["pred"], z["act_feat"], z["delta"], z["legal"], z["p0"])
    entropy = -(p0 * np.log(np.clip(p0, 1e-6, 1)) + (1 - p0) * np.log(np.clip(1 - p0, 1e-6, 1)))
    val_imgs = set(ck.get("val_images", z["image_uid"]).tolist())
    sel = np.array([u in val_imgs for u in z["image_uid"]])
    out = _value_scores(model, torch.from_numpy(h), torch.from_numpy(pred).long(),
                        torch.from_numpy(af), torch.from_numpy(p0), torch.from_numpy(entropy),
                        np.arange(len(h)), torch.device("cpu")).numpy()
    cost = float(cfg["runtime"].get("fine_cost", 0.49))
    idx_all = np.arange(len(out))
    # Absolute utility calibration is fragile (a constant offset preserves ranking
    # but makes a fixed 0 threshold meaningless), so select lambda AND the stop
    # score threshold on the value-validation images.
    best, rows = None, []
    for lam in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        net = np.where(legal, out - lam * cost, -1e9)
        best_a = net.argmax(1)
        best_score = net[idx_all, best_a]
        ref = best_score[sel] if sel.any() else best_score
        thresholds = np.quantile(ref, np.linspace(0.0, 0.95, 20)) if len(ref) else np.array([0.0])
        for thr in thresholds:
            take = (best_score > thr) & sel
            gain = np.where(take, delta[idx_all, best_a] - lam * cost, 0.0)
            val = float(gain[sel].mean()) if sel.any() else float(gain.mean())
            row = {"lambda": float(lam), "stop_threshold": float(thr),
                   "mean_net_gain": val,
                   "take_rate": float(take[sel].mean()) if sel.any() else 0.0}
            rows.append(row)
            if best is None or val > best["mean_net_gain"]:
                best = row
    print("[lambda]", json.dumps(best))
    save_json({"grid": rows, "best": best}, os.path.join(os.path.dirname(value_pt), "lambda_selection.json"))
    return best
