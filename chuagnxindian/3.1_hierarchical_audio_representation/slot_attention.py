"""L3+L4 visual slots with hierarchical A3+A4 audio semantics."""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion_levels import MFusion
from multi_layer_slot_attention import AudioSlotBranch, VisualSlotBranch


class AudioHierarchicalFusion(nn.Module):
    """A4-anchored residual fusion initialized to the exact A4 solution."""

    def __init__(self, slot_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, audio_slots_a3, audio_slots_a4):
        delta = self.net(torch.cat([audio_slots_a3, audio_slots_a4], dim=-1))
        return audio_slots_a4 + delta, delta


class HierarchicalAudioL3L4SlotAttention(nn.Module):
    """Unchanged visual hierarchy with independent A3 and A4 audio branches."""

    def __init__(
        self,
        num_slots,
        infer_sharpening,
        mask_ratio,
        iters,
        slot_dim=512,
        slot_alignment="none",
    ):
        super().__init__()
        if slot_alignment != "none":
            raise ValueError("Experiment 3.1 requires slot_alignment='none'")

        self.num_slots = num_slots
        self.infer_sharpening = infer_sharpening
        self.mask_ratio = mask_ratio
        self.slot_dim = slot_dim
        self.eps = 1e-8
        self.scale = slot_dim**-0.5

        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.mask_token_img = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.mask_token_aud = nn.Parameter(torch.randn(1, 1, slot_dim))

        self.visual_branches = nn.ModuleList(
            [VisualSlotBranch(slot_dim=slot_dim, iters=iters) for _ in range(2)]
        )
        # Preserve the formal baseline name for the A4 branch.
        self.audio_branch = AudioSlotBranch(slot_dim=slot_dim, iters=iters)
        self.slot_fusion = MFusion(num_levels=2, slot_dim=slot_dim)

        self.audio_branch_a3 = AudioSlotBranch(slot_dim=slot_dim, iters=iters)
        self.audio_hierarchical_fusion = AudioHierarchicalFusion(slot_dim=slot_dim)

    @property
    def audio_branch_a4(self):
        return self.audio_branch

    def _masked(self, tokens, mask_token):
        if not self.training:
            return tokens
        batch_size, num_tokens, channels = tokens.shape
        mask = torch.rand(
            batch_size, num_tokens, 1, device=tokens.device
        ) < self.mask_ratio
        mask = mask.to(dtype=tokens.dtype)
        expanded = mask_token.expand(batch_size, num_tokens, channels)
        return tokens * (1 - mask) + expanded * mask

    def _attention(self, query, key, scale_multiplier=1.0):
        dots = (
            torch.einsum("bid,bjd->bij", query, key)
            * scale_multiplier
            * self.scale
        )
        attention = dots.softmax(dim=1) + self.eps
        return attention / attention.sum(dim=-1, keepdim=True)

    def _encode(self, image_levels, audio_tokens_a3, audio_tokens_a4):
        if len(image_levels) != 2:
            raise ValueError(f"Expected L3 and L4 visual levels, got {len(image_levels)}")
        if any(level.shape[-1] != self.slot_dim for level in image_levels):
            raise ValueError("Every visual level must have dimension 512")
        if image_levels[0].shape[1] != image_levels[1].shape[1]:
            raise ValueError("L3 and L4 must have the same token count")

        batch_size = image_levels[0].shape[0]
        initial_slots = self.slots.expand(batch_size, -1, -1)

        masked_a3 = self._masked(audio_tokens_a3, self.mask_token_aud)
        masked_a4 = self._masked(audio_tokens_a4, self.mask_token_aud)
        audio_slots_a3, audio_query_a3, audio_keys_a3 = self.audio_branch_a3(
            masked_a3, initial_slots
        )
        audio_slots_a4, audio_query_a4, audio_keys_a4 = self.audio_branch_a4(
            masked_a4, initial_slots
        )
        fused_audio_slots, audio_delta = self.audio_hierarchical_fusion(
            audio_slots_a3, audio_slots_a4
        )

        visual_slots = []
        visual_queries = []
        visual_keys = []
        for branch, level in zip(self.visual_branches, image_levels):
            masked_level = self._masked(level, self.mask_token_img)
            slots, query, keys = branch(masked_level, initial_slots)
            visual_slots.append(slots)
            visual_queries.append(query)
            visual_keys.append(keys)

        return {
            "visual_slots": visual_slots,
            "visual_queries": visual_queries,
            "visual_keys": visual_keys,
            "audio_slots_a3": audio_slots_a3,
            "audio_slots_a4": audio_slots_a4,
            "fused_audio_slots": fused_audio_slots,
            "audio_delta": audio_delta,
            "audio_query_a3": audio_query_a3,
            "audio_query_a4": audio_query_a4,
            "audio_keys_a3": audio_keys_a3,
            "audio_keys_a4": audio_keys_a4,
        }

    def _l4_attentions(self, encoded, scale_multiplier):
        image_query = encoded["visual_queries"][-1]
        image_keys = encoded["visual_keys"][-1]
        audio_query_a4 = encoded["audio_query_a4"]
        audio_keys_a4 = encoded["audio_keys_a4"]
        return {
            "imgq_imgk_attn": self._attention(
                image_query, image_keys, scale_multiplier
            ),
            "imgq_audk_attn": self._attention(
                image_query, audio_keys_a4, scale_multiplier
            ),
            "audq_imgk_attn": self._attention(
                audio_query_a4, image_keys, scale_multiplier
            ),
            "audq_audk_attn": self._attention(
                audio_query_a4, audio_keys_a4, scale_multiplier
            ),
        }

    def forward(self, image_levels, audio_tokens_a3, audio_tokens_a4):
        encoded = self._encode(image_levels, audio_tokens_a3, audio_tokens_a4)
        output = {
            "img_slots": self.slot_fusion(encoded["visual_slots"]),
            "aud_slots": encoded["fused_audio_slots"],
            "aud_slots_a3": encoded["audio_slots_a3"],
            "aud_slots_a4": encoded["audio_slots_a4"],
            "audio_query_a3": encoded["audio_query_a3"],
            "audio_query_a4": encoded["audio_query_a4"],
            "audio_delta": encoded["audio_delta"],
        }
        output.update(self._l4_attentions(encoded, scale_multiplier=1.0))
        return output

    def get_eval_attentions(self, image_levels, audio_tokens_a3, audio_tokens_a4):
        encoded = self._encode(image_levels, audio_tokens_a3, audio_tokens_a4)
        attentions = self._l4_attentions(
            encoded, scale_multiplier=self.infer_sharpening
        )
        return attentions["imgq_imgk_attn"], attentions["audq_imgk_attn"]

    def get_a3_query_eval_attention(
        self, image_levels, audio_tokens_a3, audio_tokens_a4
    ):
        encoded = self._encode(image_levels, audio_tokens_a3, audio_tokens_a4)
        return self._attention(
            encoded["audio_query_a3"],
            encoded["visual_keys"][-1],
            self.infer_sharpening,
        )

    def get_representations(self, image_levels, audio_tokens_a3, audio_tokens_a4):
        encoded = self._encode(image_levels, audio_tokens_a3, audio_tokens_a4)
        return {
            **encoded,
            "fused_visual_slots": self.slot_fusion(encoded["visual_slots"]),
        }

    def get_slots(self, image_levels, audio_tokens_a3, audio_tokens_a4):
        encoded = self.get_representations(
            image_levels, audio_tokens_a3, audio_tokens_a4
        )
        return encoded["fused_visual_slots"], encoded["fused_audio_slots"]
