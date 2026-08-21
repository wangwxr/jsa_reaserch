"""Independent spatial Slot Attention initialized by frozen semantic L4 slots."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialSlotAttention(nn.Module):
    """JSA visual Slot branch with external semantic initialization.

    The returned ownership is the final-iteration softmax over the slot axis,
    before the token renormalization used only for the recurrent slot update.
    """

    def __init__(self, slot_dim: int = 512, num_slots: int = 2, iters: int = 5):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = 1e-8
        self.scale = slot_dim**-0.5

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim)
        self.to_v = nn.Linear(slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

    def forward(
        self, tokens: torch.Tensor, semantic_initial_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, channels = tokens.shape
        expected_slots = (batch_size, self.num_slots, self.slot_dim)
        if tokens.shape[-1] != self.slot_dim:
            raise ValueError(f"Expected token dim {self.slot_dim}, got {tokens.shape}")
        if tuple(semantic_initial_slots.shape) != expected_slots:
            raise ValueError(
                f"Expected semantic slots {expected_slots}, got {semantic_initial_slots.shape}"
            )

        normalized_tokens = self.norm_input(tokens)
        keys = self.to_k(normalized_tokens)
        values = self.to_v(normalized_tokens)
        slots = semantic_initial_slots
        ownership = None

        for _ in range(self.iters):
            previous_slots = slots
            queries = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bsd,bnd->bsn", queries, keys) * self.scale
            ownership = logits.softmax(dim=1)

            # Match the existing JSA branch update exactly. This normalized
            # tensor is not exposed as object ownership.
            update_weights = ownership + self.eps
            update_weights = update_weights / update_weights.sum(
                dim=-1, keepdim=True
            )
            updates = torch.einsum("bnd,bsn->bsd", values, update_weights)
            slots = self.gru(
                updates.reshape(-1, channels),
                previous_slots.reshape(-1, channels),
            ).reshape(batch_size, self.num_slots, channels)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        if ownership is None or ownership.shape != (
            batch_size,
            self.num_slots,
            token_count,
        ):
            raise RuntimeError("Spatial ownership was not constructed correctly")
        return slots, ownership

