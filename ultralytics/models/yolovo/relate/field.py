"""GLP predicate field: predicate-conditioned anisotropic Gaussian mixture read.

V1 (``local_frame=False``) builds the field in image coordinates from the
predicate embedding, geometry features and the four pair anchors.  V2
(``local_frame=True, use_geo=False``) builds it in the subject frame from the
predicate embedding alone and maps it back with a closed-form similarity
transform.  V2 is therefore object-box independent and translation/scale
equivariant; V1 is the negative control.

One core: ``e_p -> (mu, sigma, logw[, beta])`` plus grid-sample expectation
read.  No extra tasks, datasets or gating modules.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GEO = 19
N_ANCHORS = 4  # subject, object, union, contact (V1 only)


@dataclass
class FieldConfig:
    """Configuration shared by train/eval/checkpoint restore."""

    mode: str = "glp"           # off | glp | inject | mlp
    # top-M predicate candidates per pair。原来是 4：检索头修好之前 GT 永远在 top-4 之外，
    # 场头没机会重排；修好后测到 top-5 召回只有 0.43，M=4 会把上限卡死在四成。
    # 16 在 batch=32 下 backward 会 OOM（实测要 19.25+4 GiB），8 是这张 22 GiB 卡上的上限。
    n_candidates: int = 8
    n_kernels: int = 8          # kernels per anchor (V1 A*G = 32)
    hidden: int = 64
    sigma_min: float = 0.02
    sigma_max: float = 0.6
    sigma_local_min: float = 0.1
    sigma_local_max: float = 2.0
    eta: float = 0.5            # V1 one-step mean shift
    lam: float = 1.0            # V1 log-omega weight
    # V2 switches
    local_frame: bool = False
    use_null_bin: bool = False
    use_geo: bool = True        # V2 sets False: no object-box leakage
    compose: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_any(cls, value) -> "FieldConfig":
        if value is None:
            return cls(mode="off")
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            if value == "v2":
                return cls(mode="glp", local_frame=True, use_null_bin=True, use_geo=False)
            return cls(mode=value)
        if isinstance(value, dict):
            return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})
        raise TypeError(f"unsupported FieldConfig value: {type(value)!r}")


def to_image_frame(mu_local: torch.Tensor, sigma_local: torch.Tensor,
                   subject_box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map local-frame kernels to image coordinates (pure, no parameters).

    ``mu_local``/``sigma_local`` are ``[..., 2]`` in subject half-extent units;
    ``subject_box`` is ``[..., 4]`` normalised cxcywh.  ``sigma_local`` is the
    diagonal standard deviation.  ``R_s`` is the identity until a subject
    orientation estimator exists (documented V2 limitation).
    """
    cx, cy, w, h = subject_box.unbind(-1)
    scale = torch.stack([w, h], -1).clamp_min(1e-4)
    centre = torch.stack([cx, cy], -1)
    while centre.dim() < mu_local.dim():
        centre, scale = centre.unsqueeze(-2), scale.unsqueeze(-2)
    mu_abs = centre + scale * mu_local
    sigma_abs = scale * sigma_local
    return mu_abs, sigma_abs


def _log_components(q: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log N(q; mu, diag(sigma^2)) up to a q-independent constant."""
    var = sigma.clamp_min(1e-4).pow(2)
    diff = q - mu
    return -0.5 * (diff.pow(2) / var).sum(-1) - sigma.clamp_min(1e-4).log().sum(-1)


@torch.no_grad()
def field_grid(field: Dict[str, torch.Tensor], k: int = 0, m: int = 0,
               hw: Tuple[int, int] = (64, 64), normalize: bool = True) -> torch.Tensor:
    """Evaluate one predicate field on a grid; returns ``[h, w]``."""
    mu = field["mu"][k, m].float()                 # [K,2]
    sigma = field["sigma"][k, m].float()           # [K,2]
    logw = field["logw"][k, m].float()             # [K]
    h, w = hw
    ys = (torch.arange(h, device=mu.device, dtype=mu.dtype) + 0.5) / h
    xs = (torch.arange(w, device=mu.device, dtype=mu.dtype) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([xx, yy], -1).reshape(-1, 2)
    logp = _log_components(pts.unsqueeze(1), mu.unsqueeze(0), sigma.unsqueeze(0))
    log_omega = torch.logsumexp(logw.unsqueeze(0) + logp, dim=-1).reshape(h, w)
    omega = log_omega.exp()
    if normalize:
        omega = omega / omega.max().clamp_min(1e-12)
    return omega


@torch.no_grad()
def field_stats(field: Dict[str, torch.Tensor], anchors: torch.Tensor,
                boxes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Mass/peak/entropy/alignment diagnostics plus V2 background mass.

    This is the single-pair (k=0, m=0) diagnostic; use the analytic
    ``field_metrics.field_mass_in_box`` for dataset-wide averages.
    """
    m0_mu, m0_sigma, m0_w = field["mu"][0, 0], field["sigma"][0, 0], field["logw"][0, 0]
    K = m0_mu.shape[0]
    hw = 32
    ys = (torch.arange(hw, device=m0_mu.device, dtype=m0_mu.dtype) + 0.5) / hw
    ys, xs = torch.meshgrid(ys, ys, indexing="ij")
    pts = torch.stack([xs, ys], -1).reshape(-1, 2)
    logp = _log_components(pts.unsqueeze(1), m0_mu.unsqueeze(0), m0_sigma.unsqueeze(0))
    omega = torch.logsumexp(m0_w.unsqueeze(0) + logp, dim=-1).exp().reshape(hw, hw)
    omega = omega / omega.sum().clamp_min(1e-12)

    def _mass_in(box: torch.Tensor) -> torch.Tensor:
        cx, cy, bw, bh = box
        inside = ((xs - cx).abs() <= bw * 0.5) & ((ys - cy).abs() <= bh * 0.5)
        return (omega * inside).sum()

    a = anchors[0]  # [4,4]
    mass_sub, mass_obj = _mass_in(a[0]), _mass_in(a[1])
    mass_contact = _mass_in(a[3])
    union = _mass_in(a[2])
    w = m0_w.exp()
    entropy = -(w * w.clamp_min(1e-9).log()).sum() / math.log(max(K, 2))
    peak = omega.max()
    inside_union = ((m0_mu[..., 0] - a[2, 0]).abs() <= a[2, 2] * 0.5) & \
                   ((m0_mu[..., 1] - a[2, 1]).abs() <= a[2, 3] * 0.5)
    alignment = (w * inside_union.float()).sum()
    out = {
        "mass_in_sub": mass_sub.detach(),
        "mass_in_obj": mass_obj.detach(),
        "mass_in_contact": mass_contact.detach(),
        "mass_in_union": union.detach(),
        "mass_outside": (1.0 - union).clamp_min(0.0).detach(),
        "peak": peak.detach(),
        "kernel_entropy": entropy.detach(),
        "alignment": alignment.detach(),
    }
    if "w_bg" in field:
        out["w_bg"] = field["w_bg"][:, 0].mean().detach()
    return out


class PredicateFieldRead(nn.Module):
    """One predicate field core used by V1 (absolute) and V2 (subject frame)."""

    def __init__(self, d_model: int = 512, geo_dim: int = NUM_GEO,
                 cfg: Optional[FieldConfig] = None):
        super().__init__()
        self.cfg = cfg or FieldConfig()
        self.d_model, self.geo_dim = d_model, geo_dim
        self.use_geo = bool(self.cfg.use_geo)
        # V2 drops the object/union/contact anchors and the geometry vector:
        # the field must depend on (predicate, subject) only.
        self.n_anchors = N_ANCHORS if self.use_geo else 1
        self.n_kernels = int(self.cfg.n_kernels)
        self.n_kernels_total = self.n_anchors * self.n_kernels
        hidden = int(self.cfg.hidden)
        in_dim = d_model + (geo_dim + 16 if self.use_geo else 0)
        self.per_anchor = self.n_kernels * 5 + (1 if self.cfg.use_null_bin else 0)

        self.hyper = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.per_anchor))
        self.anchor_emb = nn.Embedding(N_ANCHORS, 16) if self.use_geo else None
        if self.anchor_emb is not None:
            nn.init.normal_(self.anchor_emb.weight, std=0.02)

        # V1-only aggregation / residual gate (V2 deletes both)
        self.agg = nn.Linear(d_model, self.n_kernels_total) if not self.cfg.local_frame else None
        self.e_proj = None
        if self.agg is not None:
            self.e_proj = nn.Sequential(
                nn.Linear(d_model, max(hidden // 2, 16)), nn.GELU(),
                nn.Linear(max(hidden // 2, 16), d_model))
        self.gamma = nn.Parameter(torch.zeros(d_model)) if not self.cfg.local_frame else None
        # V2 zero-init read-out
        self.read_out = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.register_buffer("sigma_weights",
                             torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]))
        self._ring_init()
        self._zero_init_last()
        self.capture = False
        self.last_field: Optional[Dict[str, torch.Tensor]] = None

    def _ring_init(self) -> None:
        """Hyper bias: ring offsets, zero sigma/logit (MD §2.3)."""
        G = self.n_kernels
        bias = torch.zeros(self.per_anchor)
        for j in range(G):
            theta = 2.0 * math.pi * j / max(G, 1)
            r = 0.35 * (j + 1)
            base = j * 5
            bias[base + 0] = math.cos(theta) * r
            bias[base + 1] = math.sin(theta) * r
        with torch.no_grad():
            self.hyper[-1].bias.copy_(bias)

    def _zero_init_last(self) -> None:
        nn.init.zeros_(self.hyper[-1].weight)
        for m in (self.agg, self.e_proj):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)

    def _local_geo(self, geo: torch.Tensor) -> torch.Tensor:
        """Drop absolute position/scale cues when a (V1-frame) field is localised."""
        if not self.cfg.local_frame:
            return geo
        g = geo.clone()
        g[..., 5] = 0.0
        g[..., 6] = 0.0
        g[..., 14] = 0.0
        return g

    def forward(self, queries: torch.Tensor, scene: torch.Tensor,
                anchors: torch.Tensor, geo: torch.Tensor,
                e_cand: torch.Tensor, return_field: bool = False):
        """Read predicate-conditioned evidence for every pair/candidate.

        queries ``[B,K,d]``; scene ``[B,d,h,w]``; anchors ``[B,K,4,4]``
        (normalised cxcywh, order sub/obj/union/contact); geo ``[B,K,19]``;
        e_cand ``[B,K,M,d]``.  Returns ``r_field [B,K,M,d]`` and the field dict.
        """
        cfg = self.cfg
        B, K, d = queries.shape
        M = e_cand.shape[2]
        A, G = self.n_anchors, self.n_kernels
        AG = A * G

        if self.use_geo:
            geo_in = self._local_geo(geo)
            e = e_cand
            geo_rep = geo_in[:, :, None, None, :].expand(B, K, M, A, self.geo_dim)
            aemb = self.anchor_emb.weight.view(1, 1, 1, A, 16).expand(B, K, M, A, 16)
            hin = torch.cat([e.unsqueeze(3).expand(B, K, M, A, d), geo_rep, aemb], dim=-1)
        else:
            hin = e_cand.unsqueeze(3)                    # [B,K,M,1,d]

        raw = self.hyper(hin).view(B, K, M, A, self.per_anchor)
        raw5 = raw[..., :G * 5].reshape(B, K, M, A, G, 5)
        delta = 2.0 * torch.tanh(raw5[..., :2])          # [B,K,M,A,G,2]
        sig_raw = raw5[..., 2:4]
        sig_local = cfg.sigma_local_min + (cfg.sigma_local_max - cfg.sigma_local_min) * \
            torch.sigmoid(sig_raw)
        logit = raw5[..., 4]                             # [B,K,M,A,G]
        beta = raw[..., G * 5] if cfg.use_null_bin else None

        anchors_eff = anchors[:, :, :A]                  # V2: subject only
        anchor_c = anchors_eff[..., :2]
        half = anchors_eff[..., 2:].clamp_min(0.025) * 0.5
        if cfg.local_frame:
            subject = anchors[:, :, 0]
            mu, sigma = to_image_frame(delta, sig_local, subject)
            mu = mu.reshape(B, K, M, AG, 2)
            # No absolute clamp here: Σ_abs = A Σ̃ Aᵀ must scale with the
            # subject (V2 §2.2/§3.2).  σ̃ is bounded by the 0.1..2.0 squashing.
            sigma = sigma.reshape(B, K, M, AG, 2)
            mu_local_f = delta.reshape(B, K, M, AG, 2)
            sigma_local_f = sig_local.reshape(B, K, M, AG, 2)
        else:
            half6 = half.unsqueeze(2).unsqueeze(4).expand(B, K, M, A, G, 2)
            centre6 = anchor_c.unsqueeze(2).unsqueeze(4).expand(B, K, M, A, G, 2)
            mu = (centre6 + half6 * delta).reshape(B, K, M, AG, 2)
            sigma = (sig_local * half6).clamp(cfg.sigma_min, cfg.sigma_max).reshape(
                B, K, M, AG, 2)
            mu_local_f = sigma_local_f = None

        logw = F.log_softmax(logit.reshape(B, K, M, AG), dim=-1)
        if cfg.use_null_bin:
            b = beta.mean(-1)                            # [B,K,M]
            joint = F.log_softmax(torch.cat([logw, b.unsqueeze(-1)], dim=-1), dim=-1)
            logw, w_bg = joint[..., :-1], joint[..., -1].exp()
        else:
            w_bg = None
        field_extra = {}

        if cfg.local_frame:
            # V2: unscented 5-point expectation read, one centre + 4 axial points
            offs = torch.tensor([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
                                device=mu.device, dtype=mu.dtype)
            off = offs.view(1, 1, 1, 1, 5, 1, 2)
            sig = sigma.unsqueeze(-2).unsqueeze(-2)      # [B,K,M,AG,1,1,2]
            base = mu.unsqueeze(-2).unsqueeze(-2)        # [B,K,M,AG,1,1,2]
            q_pts = (base + off * sig).squeeze(-2)       # [B,K,M,AG,5,2]
            h, w = scene.shape[-2:]
            grid = (q_pts.reshape(B, K * M * AG * 5, 1, 2) * 2.0 - 1.0)
            vals = F.grid_sample(scene, grid, mode="bilinear", align_corners=False,
                                 padding_mode="zeros")
            vals = vals.view(B, d, K, M, AG, 5).permute(0, 2, 3, 4, 5, 1)
            sw = self.sigma_weights.to(vals.dtype).view(1, 1, 1, 1, 5, 1)
            z = (vals * sw).sum(-2)                          # [B,K,M,AG,d]
            w_agg = logw.exp()                           # pure field weights (sum=1-w_bg)
            w_norm = w_agg / w_agg.sum(-1, keepdim=True).clamp_min(1e-6)
            z = (z * w_norm.unsqueeze(-1)).sum(-2)
            z = self.out_norm(z)
            # Apply the background leak AFTER LayerNorm: LN is scale-invariant,
            # so scaling z before it would make w_bg a no-op (and beta grad 0).
            if w_bg is not None:
                z = z * (1.0 - w_bg).unsqueeze(-1)
            r_field = queries.unsqueeze(2) + self.read_out(z)
            q_center = mu
        else:
            # V1: one-step mode seeking (q0 == mu -> mean-shift term is zero)
            q0 = mu
            comp = _log_components(q0, mu, sigma)
            pi = F.softmax(logw + comp, dim=-1)
            grad = (pi.unsqueeze(-1) * (mu - q0) / sigma.clamp_min(1e-4).pow(2)).sum(-2)
            hstep = half.unsqueeze(2).unsqueeze(4).expand(B, K, M, A, G, 2).reshape(
                B, K, M, AG, 2)
            step = cfg.eta * hstep * grad.unsqueeze(-2)
            cap = 0.25 * hstep
            q0 = (q0 + step.clamp(-cap, cap)).clamp(0.0, 1.0)

            log_omega = logw + _log_components(q0, mu, sigma)
            log_omega = log_omega - torch.logsumexp(log_omega, dim=-1, keepdim=True)
            h, w = scene.shape[-2:]
            grid = (q0.reshape(B, K * M * AG, 1, 2) * 2.0 - 1.0)
            vals = F.grid_sample(scene, grid, mode="bilinear", align_corners=False,
                                 padding_mode="zeros")
            vals = vals.view(B, d, K, M, AG).permute(0, 2, 3, 4, 1)
            if cfg.mode in ("glp", "mlp") and not cfg.use_null_bin:
                score = self.agg(self.out_norm(queries).unsqueeze(2) + self.e_proj(e_cand))
                if cfg.mode == "glp":
                    score = score + cfg.lam * log_omega
                w_agg = F.softmax(score, dim=-1)
            elif cfg.mode == "inject":
                w_agg = F.softmax(
                    self.agg(self.out_norm(queries).unsqueeze(2) + self.e_proj(e_cand)), dim=-1)
            else:
                w_agg = logw.exp()
            z = (vals * w_agg.unsqueeze(-1)).sum(-2)
            z = self.out_norm(z)
            r_field = queries.unsqueeze(2) + self.gamma * z
            q_center = q0
            field_extra = {"log_omega": log_omega.detach()}

        field = {
            "mu": mu.detach(), "sigma": sigma.detach(), "logw": logw.detach(),
            "q": q_center.detach(), "anchors": anchors.detach(),
            "w_agg": w_agg.detach(),
        }
        field.update(field_extra)
        if w_bg is not None:
            field["w_bg"] = w_bg.detach()
        if cfg.local_frame:
            field["mu_local"] = mu_local_f.detach()
            field["sigma_local"] = sigma_local_f.detach()
            field["subject_box"] = anchors[:, :, 0].detach()
        if return_field:
            return r_field, field
        return r_field

    def reset_parameters(self) -> None:
        self._ring_init()
        self._zero_init_last()


# ----------------------------------------------------------------------
# V2 closed-form algebra (no parameters, no gates)
# ----------------------------------------------------------------------

def _pair_mixture(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor],
                  op: str) -> Dict[str, torch.Tensor]:
    """Combine two diagonal-Gaussian mixtures; keeps at most K kernels."""
    mu_a, s_a = a["mu"], a["sigma"]
    mu_b, s_b = b["mu"], b["sigma"]
    Ka, Kb = mu_a.shape[-2], mu_b.shape[-2]
    lead = torch.broadcast_shapes(mu_a.shape[:-2], mu_b.shape[:-2])
    mu_a = mu_a.expand(*lead, Ka, 2)
    mu_b = mu_b.expand(*lead, Kb, 2)
    s_a = s_a.expand(*lead, Ka, 2)
    s_b = s_b.expand(*lead, Kb, 2)
    w_a = a["logw"].exp().expand(*lead, Ka)
    w_b = b["logw"].exp().expand(*lead, Kb)

    if op == "or":
        mu = torch.cat([mu_a, mu_b], dim=-2)
        sigma = torch.cat([s_a, s_b], dim=-2)
        logw = torch.log(torch.cat([w_a, w_b], dim=-1).clamp_min(1e-12))
    else:
        ma = mu_a.unsqueeze(-2)                          # [...,Ka,1,2]
        mb = mu_b.unsqueeze(-3)                          # [...,1,Kb,2]
        sa = s_a.unsqueeze(-2)
        sb = s_b.unsqueeze(-3)
        wa = w_a.unsqueeze(-1)
        wb = w_b.unsqueeze(-2)
        if op == "and":
            prec = 1.0 / sa.pow(2) + 1.0 / sb.pow(2)
            mu = (ma / sa.pow(2) + mb / sb.pow(2)) / prec
            sigma = prec.rsqrt()
            # normalised product weight: det(2pi(Sa+Sb)) and overlap exponent
            logw = (wa * wb).clamp_min(1e-12).log() \
                - 0.5 * (sa.pow(2) + sb.pow(2)).log().sum(-1) \
                - 0.5 * ((ma - mb).pow(2) / (sa.pow(2) + sb.pow(2))).sum(-1)
        elif op == "transitive":
            mu = ma + mb
            sigma = (sa.pow(2) + sb.pow(2)).sqrt()
            logw = (wa * wb).clamp_min(1e-12).log()      # no overlap exponent
        else:
            raise ValueError(f"unknown compose op: {op!r}")
        mu = mu.reshape(*lead, Ka * Kb, 2)
        sigma = sigma.reshape(*lead, Ka * Kb, 2)
        logw = logw.reshape(*lead, Ka * Kb)

    K = max(Ka, Kb)
    if logw.shape[-1] > K:
        idx = logw.topk(K, dim=-1).indices
        idx2 = idx.unsqueeze(-1).expand(*idx.shape, 2)
        mu = torch.gather(mu, -2, idx2)
        sigma = torch.gather(sigma, -2, idx2)
        logw = torch.gather(logw, -1, idx)
    logw = F.log_softmax(logw, dim=-1)
    # Drop stale metadata (w_bg / mu_local / subject_box / log_omega) from the
    # left operand: it no longer describes the composed field.
    return {"mu": mu, "sigma": sigma, "logw": logw, "op": op}


def compose_fields(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor],
                   op: str = "and") -> Dict[str, torch.Tensor]:
    """Closed-form AND / OR / transitive composition of two fields.

    ``and`` multiplies densities (precision addition), ``or`` normalises the
    concatenated mixture, ``transitive`` convolves means/variances.  ``negate``
    is a calibrated mass complement and returns a diagnostic field, not a new
    Gaussian mixture.
    """
    if op == "negate":
        # 1 - Ω is not a Gaussian mixture; keep it as a diagnostic only.
        # The total complement mass is reported explicitly (K=1 -> ~0).
        out = {k: a[k] for k in ("mu", "sigma") if k in a}
        out["logw"] = torch.log((1.0 - a["logw"].exp()).clamp_min(1e-6))
        out["mass"] = (1.0 - a["logw"].exp().sum(-1)).clamp_min(0.0)
        out["negated"] = True
        return out
    return _pair_mixture(a, b, op)


def gate(field_p: Dict[str, torch.Tensor], field_r: Dict[str, torch.Tensor],
         tau: float = 1.0) -> Dict[str, torch.Tensor]:
    """Ω_p · (Ω_r / max Ω_r)^τ, implemented as a kernel-weight reweighting."""
    log_r = field_r.get("log_omega")
    if log_r is None:
        raise ValueError("gate() needs a field dict carrying log_omega")
    out = {k: field_p[k] for k in ("mu", "sigma") if k in field_p}
    out["logw"] = field_p["logw"] + tau * (log_r - log_r.max(-1, keepdim=True).values)
    out["logw"] = F.log_softmax(out["logw"], dim=-1)
    out["op"] = "gate"
    return out
