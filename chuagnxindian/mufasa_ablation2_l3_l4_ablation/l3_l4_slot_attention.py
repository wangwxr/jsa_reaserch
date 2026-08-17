"""Two visual Slot Attention branches with L4-only spatial attention."""

import torch
import torch.nn as nn

from fusion_levels import MFusion
from multi_layer_slot_attention import AudioSlotBranch, VisualSlotBranch


class L3L4JointSlotAttention(nn.Module):
    """Independent L3/L4 visual SA branches and one shared audio branch."""

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
            raise ValueError("L3+L4 ablation supports slot_alignment='none'")

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
        self.audio_branch = AudioSlotBranch(slot_dim=slot_dim, iters=iters)
        self.slot_fusion = MFusion(num_levels=2, slot_dim=slot_dim)

    def _masked(self, tokens, mask_token):
        if not self.training:
            return tokens
        batch_size, num_tokens, channels = tokens.shape
        mask = torch.rand(
            batch_size, num_tokens, 1, device=tokens.device
        ) < self.mask_ratio
        mask = mask.to(dtype=tokens.dtype)
        expanded_token = mask_token.expand(
            batch_size, num_tokens, channels
        )
        return tokens * (1 - mask) + expanded_token * mask

    def _attention(self, query, key, scale_multiplier=1.0):
        dots = (
            torch.einsum("bid,bjd->bij", query, key)
            * scale_multiplier
            * self.scale
        )
        attention = dots.softmax(dim=1) + self.eps
        return attention / attention.sum(dim=-1, keepdim=True)

    def _encode(self, image_levels, audio_tokens):
        if len(image_levels) != 2:
            raise ValueError(
                f"Expected L3 and L4 visual levels, got {len(image_levels)}"
            )
        if any(level.shape[-1] != self.slot_dim for level in image_levels):
            raise ValueError("Every visual level must have dimension 512")
        if image_levels[0].shape[1] != image_levels[1].shape[1]:
            raise ValueError("L3 and L4 must have the same token count")

        batch_size = image_levels[0].shape[0]
        initial_slots = self.slots.expand(batch_size, -1, -1)

        masked_audio = self._masked(audio_tokens, self.mask_token_aud)
        audio_slots, audio_query, audio_keys = self.audio_branch(
            masked_audio, initial_slots
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
            "audio_slots": audio_slots,
            "audio_query": audio_query,
            "audio_keys": audio_keys,
        }

    def _l4_attentions(self, encoded, scale_multiplier):
        image_query = encoded["visual_queries"][-1]
        image_keys = encoded["visual_keys"][-1]
        audio_query = encoded["audio_query"]
        audio_keys = encoded["audio_keys"]
        return {
            "imgq_imgk_attn": self._attention(
                image_query, image_keys, scale_multiplier
            ),
            "imgq_audk_attn": self._attention(
                image_query, audio_keys, scale_multiplier
            ),
            "audq_imgk_attn": self._attention(
                audio_query, image_keys, scale_multiplier
            ),
            "audq_audk_attn": self._attention(
                audio_query, audio_keys, scale_multiplier
            ),
        }

    def forward(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        output = {
            "img_slots": self.slot_fusion(encoded["visual_slots"]),
            "aud_slots": encoded["audio_slots"],
        }
        output.update(self._l4_attentions(encoded, scale_multiplier=1.0))
        return output

    def get_eval_attentions(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        attentions = self._l4_attentions(
            encoded, scale_multiplier=self.infer_sharpening
        )
        return attentions["imgq_imgk_attn"], attentions["audq_imgk_attn"]

    def get_slots(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        return self.slot_fusion(encoded["visual_slots"]), encoded["audio_slots"]
