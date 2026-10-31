"""Frozen visual encoders and small trained heads for BARV."""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import PROJECT_ROOT, normalize_imagenet

ACTION_TYPES = ["contact", "subject", "object", "union", "context", "revisit"]


class ResNetBackbone(nn.Module):
    """Frozen ImageNet ResNet; coarse uses layer3+layer4, fine uses pooled layer4."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        m = getattr(torchvision.models, name)(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.out_dim = 512 if name in ("resnet18", "resnet34") else 2048
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_coarse(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        return f3, f4

    @torch.no_grad()
    def forward_fine(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """x is float in [0, 255] from the uint8 letterbox/crop arrays."""
        return normalize_imagenet(x / 255.0)


class MobileCLIPBackbone(nn.Module):
    """Frozen MobileCLIP-S0 image tower (the checkpoint's text tower is dropped).

    Coarse maps are stride 16 (256 ch) and stride 32 (512 ch); fine is the
    global CLIP image embedding (512). The 4-region pooling therefore stays at
    4 * (256 + 512) = 3072, the same shape as the ResNet-18 path.
    """

    def __init__(self, weight_path: str):
        super().__init__()
        vendor = PROJECT_ROOT / "ultralytics" / "models" / "yolovo" / "text_tower" / "mobileclip" / "_vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from mobileclip.clip import CLIP  # vendored Apple source

        cfg = json.load(open(vendor / "mobileclip" / "configs" / "mobileclip_s0.json"))
        clip = CLIP(cfg=cfg)
        clip.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model = clip.image_encoder.model
        del clip  # drop the text tower and logit scale
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """MobileCLIP's official transform is ToTensor only (no mean/std)."""
        return x / 255.0

    @torch.no_grad()
    def forward_coarse(self, x: torch.Tensor):
        f = self.model.forward_embeddings(self.preprocess(x))
        f3 = None
        target = x.shape[-1] // 16
        for blk in self.model.network:
            f = blk(f)
            if f.shape[-1] == target:
                f3 = f  # keep the last stride-16 map, i.e. stage-3 output
        return f3, f

    @torch.no_grad()
    def forward_fine(self, x: torch.Tensor):
        f = self.model.forward_embeddings(self.preprocess(x))
        f = self.model.forward_tokens(f)
        return self.model.head(self.model.conv_exp(f))


def build_backbone(cfg: dict) -> nn.Module:
    m = cfg["model"]
    name = m.get("backbone", "resnet18")
    if name == "mobileclip_s0":
        path = m.get("backbone_path")
        if not path:
            raise ValueError("model.backbone_path is required for mobileclip_s0")
        path = path if os.path.isabs(path) else str(PROJECT_ROOT / path)
        return MobileCLIPBackbone(path)
    return ResNetBackbone(name, pretrained=bool(m.get("pretrained", True)))


def pool_regions(f3: torch.Tensor, f4: torch.Tensor, m3: torch.Tensor, m4: torch.Tensor) -> torch.Tensor:
    """m3 [R,4,g3,g3], m4 [R,4,g4,g4] -> [R,4,C3+C4]."""
    def pool(f, m):
        w = m.to(f.dtype)
        num = torch.einsum("rqhw,chw->rqc", w, f)
        den = w.sum(dim=(-1, -2)).clamp(min=1.0).unsqueeze(-1)
        return num / den

    return torch.cat([pool(f3, m3), pool(f4, m4)], dim=-1)


def action_features(boxes: torch.Tensor, query_boxes: torch.Tensor, img_hw, seen: torch.Tensor,
                    cost: torch.Tensor, types: list[str]) -> torch.Tensor:
    """Label-free action identity features.

    Deliberately excludes crop coordinates/IoU/contact distance: those are derived
    from mask geometry, and our geometric negative rule also uses mask gaps, so
    feeding them to the policy would leak the label rule. The executor still uses
    the boxes to cut pixels; the learned policy only sees which action type it is.
    """
    onehot = torch.zeros((len(types), len(ACTION_TYPES)), device=seen.device)
    for i, t in enumerate(types):
        onehot[i, ACTION_TYPES.index(t)] = 1.0
    return torch.cat([onehot, seen.float().unsqueeze(1), cost.unsqueeze(1)], dim=1)


ACTION_FEAT_DIM = len(ACTION_TYPES) + 2  # action identity + seen + cost


class RelationScorer(nn.Module):
    def __init__(self, pool_dim: int = 4 * 768, geom_dim: int = 24, n_pred: int = 9,
                 pred_dim: int = 64, hidden: int = 512, state_dim: int = 256):
        super().__init__()
        self.pred_emb = nn.Embedding(n_pred, pred_dim)
        nn.init.normal_(self.pred_emb.weight, std=0.02)
        self.visual = nn.Sequential(
            nn.LayerNorm(pool_dim), nn.Linear(pool_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.geo = nn.Sequential(nn.Linear(geom_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.state = nn.Sequential(
            nn.Linear(hidden + 64 + pred_dim, hidden), nn.GELU(),
            nn.Linear(hidden, state_dim), nn.LayerNorm(state_dim))
        self.head = nn.Linear(state_dim, 1)
        self.update_gate = nn.Linear(state_dim + 512 + pred_dim + ACTION_FEAT_DIM, 1)
        self.update_val = nn.Linear(512, state_dim)
        self.act = nn.Sequential(nn.Linear(ACTION_FEAT_DIM, 64), nn.GELU(), nn.Linear(64, 64))

    def encode(self, pool: torch.Tensor, geom: torch.Tensor, pred: torch.Tensor):
        h = self.state(torch.cat([self.visual(pool.flatten(1)), self.geo(geom), self.pred_emb(pred)], -1))
        return h

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)

    def forward(self, pool, geom, pred):
        h = self.encode(pool, geom, pred)
        return h, self.logits(h)

    def update(self, h, fine_u, pred, act_feat, act_type_ids):
        e = self.pred_emb(pred)
        gate = torch.sigmoid(self.update_gate(torch.cat([h, fine_u, e, act_feat], -1)))
        h2 = F.layer_norm(h + gate * self.update_val(fine_u), h.shape[-1:])
        return h2, self.logits(h2)


class ValueHead(nn.Module):
    def __init__(self, n_pred: int = 9, state_dim: int = 256, pred_dim: int = 64,
                 act_dim: int = ACTION_FEAT_DIM, hidden: int = 256):
        super().__init__()
        self.n_pred, self.state_dim, self.pred_dim, self.act_dim = n_pred, state_dim, pred_dim, act_dim
        self.pred_emb = nn.Embedding(n_pred, pred_dim)
        nn.init.normal_(self.pred_emb.weight, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(state_dim + pred_dim + act_dim + 3, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, h, pred, act_feat, prob, entropy, seen):
        x = torch.cat([h, self.pred_emb(pred), act_feat, prob.unsqueeze(-1), entropy.unsqueeze(-1),
                       seen.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


def bce_with_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
