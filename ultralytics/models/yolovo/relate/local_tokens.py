"""Diagnostic extraction of local subject and object feature tokens."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairLocalTokens(NamedTuple):
    """Local tokens, normalized sample positions, and per-token validity masks."""

    subject: torch.Tensor
    object: torch.Tensor
    subject_positions: torch.Tensor
    object_positions: torch.Tensor
    subject_valid: torch.Tensor
    object_valid: torch.Tensor


class PairLocalTokenExtractor(nn.Module):
    """Sample a regular grid of scene features inside each subject and object box.

    Boxes use normalized ``cxcywh`` coordinates. ``valid_canvas`` uses normalized
    ``xyxy`` coordinates. Returned tokens are float32 with shape ``[B,K,T,C]``.
    """

    def __init__(self, grid_size: int = 4):
        super().__init__()
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")
        self.grid_size = int(grid_size)

    def forward(
        self,
        scene: torch.Tensor,
        subject_boxes: torch.Tensor,
        object_boxes: torch.Tensor,
        valid_canvas: torch.Tensor | None = None,
        pair_valid: torch.Tensor | None = None,
    ) -> PairLocalTokens:
        """Extract ``grid_size ** 2`` tokens for each role in each pair."""
        if scene.ndim != 4 or min(scene.shape) < 1:
            raise ValueError("scene must have nonempty shape [B,C,H,W]")
        if subject_boxes.ndim != 3 or subject_boxes.shape[-1] != 4:
            raise ValueError("subject_boxes must have shape [B,K,4]")
        if object_boxes.shape != subject_boxes.shape:
            raise ValueError("object_boxes must match subject_boxes shape")

        batch = scene.shape[0]
        if subject_boxes.shape[0] != batch:
            raise ValueError("scene and boxes must have the same batch size")
        pairs = subject_boxes.shape[1]
        if valid_canvas is not None and valid_canvas.shape != (batch, 4):
            raise ValueError("valid_canvas must have shape [B,4] in normalized xyxy")
        if pair_valid is not None and pair_valid.shape != (batch, pairs):
            raise ValueError("pair_valid must have shape [B,K]")

        token_count = self.grid_size * self.grid_size
        channels = scene.shape[1]
        if pairs == 0:
            empty_tokens = torch.zeros(batch, 0, token_count, channels,
                                       device=scene.device, dtype=torch.float32)
            empty_positions = torch.zeros(batch, 0, token_count, 2,
                                          device=scene.device, dtype=torch.float32)
            empty_valid = torch.zeros(batch, 0, token_count,
                                      device=scene.device, dtype=torch.bool)
            return PairLocalTokens(
                empty_tokens, empty_tokens, empty_positions, empty_positions,
                empty_valid, empty_valid,
            )

        boxes = torch.stack((subject_boxes, object_boxes), dim=2).to(
            device=scene.device, dtype=torch.float32)
        axis = (torch.arange(self.grid_size, device=scene.device, dtype=torch.float32) + 0.5)
        axis = axis / self.grid_size - 0.5
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(1, 1, 1, token_count, 2)
        positions = boxes[..., :2].unsqueeze(-2) + offsets * boxes[..., 2:].unsqueeze(-2)

        finite_positions = torch.isfinite(positions).all(dim=-1)
        valid = finite_positions & (positions >= 0.0).all(dim=-1) & (positions <= 1.0).all(dim=-1)
        valid &= torch.isfinite(boxes).all(dim=-1).unsqueeze(-1)
        valid &= (boxes[..., 2:] > 0.0).all(dim=-1).unsqueeze(-1)
        if pair_valid is not None:
            valid &= pair_valid.to(device=scene.device, dtype=torch.bool)[:, :, None, None]
        if valid_canvas is not None:
            canvas = valid_canvas.to(device=scene.device, dtype=torch.float32)[:, None, None, None, :]
            valid &= ((positions[..., 0] >= canvas[..., 0])
                      & (positions[..., 1] >= canvas[..., 1])
                      & (positions[..., 0] <= canvas[..., 2])
                      & (positions[..., 1] <= canvas[..., 3]))

        safe_positions = torch.where(torch.isfinite(positions), positions, 0.0)
        sample_grid = (safe_positions * 2.0 - 1.0).reshape(batch, 2 * pairs * token_count, 1, 2)
        sampled = F.grid_sample(
            scene.float(), sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        tokens = sampled.squeeze(-1).transpose(1, 2).reshape(
            batch, pairs, 2, token_count, channels
        )
        tokens = tokens * valid.unsqueeze(-1)

        return PairLocalTokens(
            tokens[:, :, 0], tokens[:, :, 1],
            positions[:, :, 0], positions[:, :, 1],
            valid[:, :, 0], valid[:, :, 1],
        )
