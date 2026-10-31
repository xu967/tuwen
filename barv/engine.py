"""Lazy evidence-acquisition runtime and baselines.

Every fine crop is cut from the ORIGINAL image only when a policy selects it,
and the fine encoder is invoked only for selected crops. Teacher/fine caches are
never used here, so reported fine-call counts reflect actual lazy execution.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import ACTION_TYPES, RelationScorer, ValueHead, action_features, build_backbone

FINE_COST = 0.49  # (224/320)^2 coarse-FLOPs units


def full_pool_rows(cache) -> np.ndarray:
    rows = np.full(len(cache), -1, np.int64)
    rows[cache.full_index.numpy()] = np.arange(len(cache.full_index))
    return rows


def load_relation(path: str, device="cuda:0") -> RelationScorer:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RelationScorer(pool_dim=4 * 768, geom_dim=24, n_pred=ck["n_pred"],
                       pred_dim=ck["pred_dim"], hidden=ck["hidden"], state_dim=ck["state_dim"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def load_value(path: str, device="cuda:0") -> ValueHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ValueHead(n_pred=ck["n_pred"], state_dim=ck["state_dim"], pred_dim=ck["pred_dim"],
                  act_dim=ck["act_dim"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def legal_actions(actions: torch.Tensor, h: int, w: int, cfg: dict) -> torch.Tensor:
    """actions [Q,6,4] -> bool [Q,6]; bounds, minimum size, area budget."""
    x1, y1, x2, y2 = actions.unbind(-1)
    area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) / float(h * w)
    return ((x2 - x1) >= 16) & ((y2 - y1) >= 16) & (area >= cfg["runtime"]["min_area_frac"]) & \
           (area <= cfg["runtime"]["max_area_frac"]) & (x1 >= 0) & (y1 >= 0) & (x2 <= w) & (y2 <= h)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best = (1e9, 1.0)
    for t in np.linspace(0.4, 3.0, 53):
        p = 1 / (1 + np.exp(-logits / t))
        nll = -np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
        if nll < best[0]:
            best = (float(nll), float(t))
    return best[1]


def choose_margin(p: np.ndarray, coverage_target: float, threshold: float = 0.5) -> float:
    conf = np.abs(p - threshold)
    return float(np.quantile(conf, 1.0 - coverage_target))


def choose_threshold(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    best = (-1.0, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best[0] + 1e-9 or (abs(b - best[0]) < 1e-9 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (float(b), float(t))
    return best[1]


class LazyEngine:
    def __init__(self, cache, cfg: dict, relation: RelationScorer, value: ValueHead | None,
                 lambda_cost: float, device="cuda:0"):
        self.cache, self.cfg, self.relation = cache, cfg, relation
        self.value, self.lambda_cost, self.device = value, float(lambda_cost), device
        self.dev_type = str(device).split(":")[0]
        self._backbone = None
        self.full_rows = full_pool_rows(cache)
        self.fine_size = int(cfg["model"]["fine_size"])

    @torch.no_grad()
    def initial_states(self, idx):
        pool = self.cache.pool[idx].to(self.device)
        geom = self.cache.geom[idx].to(self.device)
        pred = self.cache.pred[idx].to(self.device)
        h = self.relation.encode(pool, geom, pred)
        return h, self.relation.logits(h)

    @torch.no_grad()
    def full_states(self, idx):
        pos = torch.as_tensor(self.full_rows[idx], dtype=torch.long)
        assert (pos >= 0).all(), "full-res cache missing for some evaluation queries"
        pool = self.cache.pool_full[pos].to(self.device)
        geom = self.cache.geom[idx].to(self.device)
        pred = self.cache.pred[idx].to(self.device)
        h = self.relation.encode(pool, geom, pred)
        return h, self.relation.logits(h)

    def _act_feats(self, actions, boxes, hw, seen):
        return torch.stack([
            action_features(actions[q], boxes[q], hw, seen[q],
                            torch.full((actions.shape[1],), FINE_COST), ACTION_TYPES)
            for q in range(actions.shape[0])])

    def _value_scores(self, h, pred, act_feat, p, seen):
        Q, A = act_feat.shape[:2]
        act_feat = act_feat.to(self.device)
        seen = seen.to(self.device)
        ent = -(p * torch.log(p.clamp(min=1e-6)) + (1 - p) * torch.log((1 - p).clamp(min=1e-6)))
        with torch.no_grad():
            out = self.value(
                h[:, None, :].expand(Q, A, -1).reshape(Q * A, -1),
                pred[:, None].expand(Q, A).reshape(-1),
                act_feat.reshape(Q * A, -1),
                p[:, None].expand(Q, A).reshape(-1),
                ent[:, None].expand(Q, A).reshape(-1),
                seen.reshape(-1).float())
        return out.view(Q, A)

    def _get_backbone(self):
        if self._backbone is None:
            self._backbone = build_backbone(self.cfg).to(self.device).eval()
        return self._backbone

    def _fine_batch(self, img, crops):
        xs = []
        for box in crops:
            x1, y1, x2, y2 = box
            h, w = img.shape[:2]
            x1, y1 = int(np.floor(max(0, x1))), int(np.floor(max(0, y1)))
            x2, y2 = int(np.ceil(min(w, x2))), int(np.ceil(min(h, y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                xs.append(None)
                continue
            xs.append(cv2.resize(img[y1:y2, x1:x2], (self.fine_size, self.fine_size),
                                 interpolation=cv2.INTER_LINEAR))
        ok = [i for i, x in enumerate(xs) if x is not None]
        u = torch.zeros((len(crops), 512), device=self.device)
        if ok:
            x = torch.from_numpy(np.stack([xs[i] for i in ok])).permute(0, 3, 1, 2).float().to(self.device)
            x = self._get_backbone().preprocess(x)
            with torch.autocast(device_type=self.dev_type,
                                enabled=bool(self.cfg["hardware"]["amp"]) and self.dev_type == "cuda",
                                dtype=torch.float16):
                uu = self._get_backbone().forward_fine(x)
            u[ok] = uu.float()
        return u

    def _execute(self, img, h, p, pred, actions, boxes, seen, act_feat, selections):
        """selections: list of (query_pos, action_idx). Returns updated h,p and stats."""
        if not selections:
            return h, p, {"fine_calls": 0, "cache_hits": 0, "cost": 0.0, "latency_ms": 0.0}
        act_feat = act_feat.to(self.device)
        keys, groups = {}, {}
        for q, a in selections:
            box = actions[q, a].numpy()
            key = tuple(int(round(v)) for v in box)
            keys[(q, a)] = key
            groups.setdefault(key, []).append((q, a))
        unique = list(groups.keys())
        t0 = time.perf_counter()
        u = self._fine_batch(img, [np.array(k, dtype=np.float32) for k in unique])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fine_ms = (time.perf_counter() - t0) * 1000
        h2, p2 = h.clone(), p.clone()
        for k, (key, sel) in enumerate(groups.items()):
            for q, a in sel:
                h2[q], logit = self.relation.update(
                    h2[q:q + 1], u[k:k + 1], pred[q:q + 1], act_feat[q, a:a + 1], None)
                p2[q] = torch.sigmoid(logit[0])
                seen[q, a] = True
        share = FINE_COST / max(1, len(selections))
        cost = FINE_COST * len(unique)
        return h2, p2, {"fine_calls": len(unique), "cache_hits": len(selections) - len(unique),
                        "cost": float(cost), "cost_per_selector": share, "latency_ms": fine_ms}

    def run_image(self, idx: np.ndarray, img: np.ndarray, method: str) -> dict:
        Q = len(idx)
        pred = self.cache.pred[torch.as_tensor(idx)].to(self.device)
        h, logit = self.initial_states(idx)
        p = torch.sigmoid(logit)
        actions = self.cache.actions[torch.as_tensor(idx)]
        boxes = self.cache.boxes[torch.as_tensor(idx)]
        hw = img.shape[:2]
        legal0 = legal_actions(actions, hw[0], hw[1], self.cfg)
        seen = torch.zeros(Q, len(ACTION_TYPES), dtype=torch.bool)
        cost_q = torch.zeros(Q)
        fine_calls = cache_hits = 0
        crop_pixels = 0
        if method == "coarse":
            pass
        elif method == "full":
            h, logit = self.full_states(idx)
            p = torch.sigmoid(logit)
            cost_q += 4.0 / Q
        elif method.startswith("fixed_"):
            a = ACTION_TYPES.index(method.split("_", 1)[1])
            sel = [(q, a) for q in range(Q) if legal0[q, a]]
            act_feat = self._act_feats(actions, boxes, hw, seen)
            h, p, st = self._execute(img, h, p, pred, actions, boxes, seen, act_feat, sel)
            fine_calls += st["fine_calls"]; cache_hits += st.get("cache_hits", 0)
            for q, _ in sel:
                cost_q[q] += st["cost"] / max(1, len(sel))
        elif method == "random":
            rng = np.random.default_rng(int(self.cfg["runtime"].get("random_seed", 7)) + int(idx[0]))
            sel = []
            for q in range(Q):
                legal = np.nonzero((legal0[q] & ~seen[q]).numpy())[0]
                if len(legal):
                    sel.append((q, int(rng.choice(legal))))
            act_feat = self._act_feats(actions, boxes, hw, seen)
            h, p, st = self._execute(img, h, p, pred, actions, boxes, seen, act_feat, sel)
            fine_calls += st["fine_calls"]; cache_hits += st.get("cache_hits", 0)
            for q, _ in sel:
                cost_q[q] += st["cost"] / max(1, len(sel))
        elif method == "entropy":
            thresh = float(self.cfg["runtime"].get("entropy_threshold", 0.5))
            for _ in range(int(self.cfg["runtime"]["max_steps"])):
                unc = 1.0 - torch.abs(2 * p - 1)
                active = unc > thresh
                if not active.any():
                    break
                act_feat = self._act_feats(actions, boxes, hw, seen)
                sel = []
                for q in torch.nonzero(active).flatten().tolist():
                    order = [0, 3, 1, 2, 4, 5]
                    for a in order:
                        if legal0[q, a] and not seen[q, a]:
                            sel.append((q, a)); break
                if not sel:
                    break
                h, p, st = self._execute(img, h, p, pred, actions, boxes, seen, act_feat, sel)
                fine_calls += st["fine_calls"]; cache_hits += st.get("cache_hits", 0)
                for q, _ in sel:
                    cost_q[q] += st["cost"] / max(1, len(sel))
        elif method in ("mve", "mve_abstain"):
            assert self.value is not None
            for _ in range(int(self.cfg["runtime"]["max_steps"])):
                act_feat = self._act_feats(actions, boxes, hw, seen)
                valid = (legal0 & ~seen).to(self.device)
                scores = self._value_scores(h, pred, act_feat, p, seen) - self.lambda_cost * FINE_COST
                scores = scores.masked_fill(~valid, -1e9)
                best_score, best_a = scores.max(1)
                active = best_score > float(self.cfg["runtime"]["stop_margin"])
                if not active.any():
                    break
                sel = [(int(q), int(best_a[q])) for q in torch.nonzero(active).flatten().tolist()]
                h, p, st = self._execute(img, h, p, pred, actions, boxes, seen, act_feat, sel)
                fine_calls += st["fine_calls"]; cache_hits += st.get("cache_hits", 0)
                for q, _ in sel:
                    cost_q[q] += st["cost"] / max(1, len(sel))
        else:
            raise ValueError(method)
        if method != "full":
            cost_q += 1.0 / Q  # shared coarse encode (full-res replaces it)
        logit_final = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        idx_np = np.asarray(idx)
        return {"qid": np.atleast_1d(self.cache.qid[idx_np]), "y": np.atleast_1d(self.cache.label[torch.as_tensor(idx)].numpy()),
                "p": p.detach().cpu().numpy(), "logit": logit_final.detach().cpu().numpy(),
                "cost": cost_q.numpy(), "fine_calls": fine_calls, "cache_hits": cache_hits,
                "image_uid": np.atleast_1d(self.cache.image_uid[idx_np])}


def load_images(cache, idx: np.ndarray, cfg: dict) -> dict:
    out = {}
    for i in idx:
        uid = cache.image_uid[i]
        if uid in out:
            continue
        path = os.path.join(cfg["paths"]["images"], os.path.basename(str(cache.file_name[i])))
        bgr = cv2.imread(path)
        out[uid] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    return out


def run_method(cache, cfg: dict, relation: RelationScorer, value: ValueHead | None, method: str,
               indices: np.ndarray, images: dict, lambda_cost: float, device="cuda:0") -> dict:
    idx_by_image = {}
    for i in indices:
        idx_by_image.setdefault(cache.image_uid[i], []).append(i)
    engine = LazyEngine(cache, cfg, relation, value, lambda_cost, device)
    rows = []
    dropped_images = dropped_queries = 0
    t0 = time.perf_counter()
    for uid, idx in idx_by_image.items():
        img = images.get(uid)
        if img is None:
            dropped_images += 1
            dropped_queries += len(idx)
            continue
        r = engine.run_image(np.asarray(idx), img, method)
        rows.append(r)
    wall = time.perf_counter() - t0
    if not rows:
        raise RuntimeError("no images")
    out = {k: np.concatenate([r[k] for r in rows]) for k in ("qid", "y", "p", "logit", "cost", "image_uid") }
    out["fine_calls"] = int(sum(r["fine_calls"] for r in rows))
    out["cache_hits"] = int(sum(r["cache_hits"] for r in rows))
    out["dropped_images"] = int(dropped_images)
    out["dropped_queries"] = int(dropped_queries)
    out["lambda"] = float(lambda_cost)
    out["method"] = method
    out["wall_sec"] = float(wall)
    out["ms_per_query"] = float(wall * 1000 / len(out["qid"]))
    return out
