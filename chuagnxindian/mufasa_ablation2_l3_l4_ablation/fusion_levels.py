"""MUFASA-style adjacent slot fusion for two or three visual levels."""

import torch
import torch.nn as nn


class MFusion(nn.Module):
    """Fuse adjacent level pairs with the v1.1 two-layer MLP rule."""

    def __init__(self, num_levels, slot_dim=512):
        super().__init__()
        if num_levels not in {2, 3}:
            raise ValueError("MFusion supports exactly two or three levels")
        self.num_levels = num_levels
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim * (num_levels - 1), slot_dim * num_levels),
            nn.GELU(),
            nn.Linear(slot_dim * num_levels, slot_dim),
        )

    def forward(self, slots):
        if len(slots) != self.num_levels:
            raise ValueError(
                f"MFusion expects {self.num_levels} levels, got {len(slots)}"
            )
        adjacent_pairs = [
            (left + right) / 2
            for left, right in zip(slots[:-1], slots[1:])
        ]
        return self.mlp(torch.cat(adjacent_pairs, dim=-1))
