"""MUFASA-inspired slot and attention fusion used by MUFASA-JSA v1."""

import torch
import torch.nn as nn


class MFusion(nn.Module):
    """Fuse three aligned visual slot tensors through adjacent-pair features."""

    def __init__(self, slot_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim * 3),
            nn.GELU(),
            nn.Linear(slot_dim * 3, slot_dim),
        )

    def forward(self, slots):
        if len(slots) != 3:
            raise ValueError(f"MFusion expects three levels, got {len(slots)}")
        pair23 = (slots[0] + slots[1]) / 2
        pair34 = (slots[1] + slots[2]) / 2
        return self.mlp(torch.cat([pair23, pair34], dim=-1))


class LearnedAttentionFusion(nn.Module):
    """Fuse three same-resolution maps with shared learned adjacent-pair weights."""

    def __init__(self):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(2))

    def normalized_weights(self):
        return torch.softmax(self.layer_weights, dim=0)

    def forward(self, attention_maps):
        if len(attention_maps) != 3:
            raise ValueError(
                f"LearnedAttentionFusion expects three levels, got "
                f"{len(attention_maps)}"
            )
        if not (
            attention_maps[0].shape
            == attention_maps[1].shape
            == attention_maps[2].shape
        ):
            raise ValueError("All attention maps must have the same shape")

        pair23 = (attention_maps[0] + attention_maps[1]) / 2
        pair34 = (attention_maps[1] + attention_maps[2]) / 2
        weights = self.normalized_weights()
        return weights[0] * pair23 + weights[1] * pair34
