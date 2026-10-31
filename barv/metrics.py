"""Binary relation-verification metrics with image-clustered bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(y.size),
        "pos_rate": float(y.mean()) if y.size else float("nan"),
        "accuracy": float((pred == y).mean()) if y.size else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro")) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": ece(y, p),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def per_predicate_metrics(y, p, pred, names, threshold=0.5) -> dict:
    out = {}
    for i, name in enumerate(names):
        m = pred == i
        if m.sum() == 0:
            continue
        out[name] = binary_metrics(y[m], p[m], threshold)
    return out


def risk_coverage(y, p, abstain: np.ndarray, threshold: float = 0.5) -> dict:
    accept = ~abstain
    risk = float((((p >= threshold).astype(int) != y) & accept).sum() / max(1, accept.sum()))
    coverage = float(accept.mean())
    return {"coverage": coverage, "selective_risk": risk,
            "accepted_acc": 1.0 - risk if accept.sum() else float("nan")}


def aurc(y, p, threshold: float = 0.5) -> float:
    """Area under risk-coverage curve (lower is better), confidence = |p-threshold|."""
    conf = np.abs(p - threshold)
    order = np.argsort(-conf)
    y, p = y[order], p[order]
    err = ((p >= threshold).astype(int) != y).astype(float)
    cum_err = np.cumsum(err)
    risks = cum_err / (np.arange(len(y)) + 1)
    return float(np.mean(risks))


def cluster_bootstrap(y, p, groups, n_boot: int = 500, seed: int = 0, threshold: float = 0.5) -> dict:
    """Bootstrap at the image level (queries from one image are correlated)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], threshold)
        vals.append([m["accuracy"], m["balanced_acc"], m["macro_f1"], m["auroc"], m["auprc"]])
    if not vals:
        return {}
    vals = np.asarray(vals)
    keys = ["accuracy", "balanced_acc", "macro_f1", "auroc", "auprc"]
    point = binary_metrics(y, p, threshold)
    return {k: {"point": point[k], "lo": float(np.percentile(vals[:, i], 2.5)),
                "hi": float(np.percentile(vals[:, i], 97.5)),
                "boot_mean": float(vals[:, i].mean()), "boot_std": float(vals[:, i].std())}
            for i, k in enumerate(keys)}


def paired_bootstrap(y, p_a, p_b, groups, n_boot: int = 500, seed: int = 0) -> dict:
    """CI of metric(A) - metric(B) with shared image resamples."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(idx) == 0 or len(np.unique(y[idx])) < 2:
            continue
        ma, mb = binary_metrics(y[idx], p_a[idx]), binary_metrics(y[idx], p_b[idx])
        diffs.append([ma["balanced_acc"] - mb["balanced_acc"], ma["macro_f1"] - mb["macro_f1"],
                      ma["auroc"] - mb["auroc"]])
    d = np.asarray(diffs)
    if not d.size:
        return {}
    return {k: {"mean_diff": float(d[:, i].mean()),
                "lo": float(np.percentile(d[:, i], 2.5)),
                "hi": float(np.percentile(d[:, i], 97.5)),
                "p_one_sided_ge0": float((d[:, i] >= 0).mean()),
                "p_one_sided_le0": float((d[:, i] <= 0).mean())}
            for i, k in enumerate(["balanced_acc", "macro_f1", "auroc"])}
