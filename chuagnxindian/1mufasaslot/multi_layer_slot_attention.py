"""Multi-layer visual / single-branch audio Joint Slot Attention."""

import torch
import torch.nn as nn

from fusion import LearnedAttentionFusion, MFusion


class VisualSlotBranch(nn.Module):
    """One independent JSA visual Slot Attention branch."""

    def __init__(self, slot_dim=512, iters=5):
        super().__init__()
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = 1e-8
        self.scale = slot_dim**-0.5

        self.img_to_q = nn.Linear(slot_dim, slot_dim)
        self.img_to_k = nn.Linear(slot_dim, slot_dim)
        self.img_to_v = nn.Linear(slot_dim, slot_dim)
        self.img_gru = nn.GRUCell(slot_dim, slot_dim)
        self.img_mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.img_norm_input = nn.LayerNorm(slot_dim)
        self.img_norm_slots = nn.LayerNorm(slot_dim)
        self.img_norm_pre_ff = nn.LayerNorm(slot_dim)

    def _attention(self, query, key):
        dots = torch.einsum("bid,bjd->bij", query, key) * self.scale
        attention = dots.softmax(dim=1) + self.eps
        return attention / attention.sum(dim=-1, keepdim=True)

    def forward(self, image_tokens, initial_slots):
        batch_size, _, channels = image_tokens.shape
        image_tokens = self.img_norm_input(image_tokens)
        image_keys = self.img_to_k(image_tokens)
        image_values = self.img_to_v(image_tokens)

        image_slots = initial_slots
        image_query = None
        for _ in range(self.iters):
            previous_slots = image_slots
            normalized_slots = self.img_norm_slots(image_slots)
            image_query = self.img_to_q(normalized_slots)
            attention = self._attention(image_query, image_keys)
            updates = torch.einsum("bjd,bij->bid", image_values, attention)
            image_slots = self.img_gru(
                updates.reshape(-1, channels),
                previous_slots.reshape(-1, channels),
            )
            image_slots = image_slots.reshape(batch_size, -1, channels)
            image_slots = image_slots + self.img_mlp(
                self.img_norm_pre_ff(image_slots)
            )

        return image_slots, image_query, image_keys


class AudioSlotBranch(nn.Module):
    """The single audio Slot Attention branch shared by all visual levels."""

    def __init__(self, slot_dim=512, iters=5):
        super().__init__()
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = 1e-8
        self.scale = slot_dim**-0.5

        self.aud_to_q = nn.Linear(slot_dim, slot_dim)
        self.aud_to_k = nn.Linear(slot_dim, slot_dim)
        self.aud_to_v = nn.Linear(slot_dim, slot_dim)
        self.aud_gru = nn.GRUCell(slot_dim, slot_dim)
        self.aud_mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.aud_norm_input = nn.LayerNorm(slot_dim)
        self.aud_norm_slots = nn.LayerNorm(slot_dim)
        self.aud_norm_pre_ff = nn.LayerNorm(slot_dim)

    def _attention(self, query, key):
        dots = torch.einsum("bid,bjd->bij", query, key) * self.scale
        attention = dots.softmax(dim=1) + self.eps
        return attention / attention.sum(dim=-1, keepdim=True)

    def forward(self, audio_tokens, initial_slots):
        batch_size, _, channels = audio_tokens.shape
        audio_tokens = self.aud_norm_input(audio_tokens)
        audio_keys = self.aud_to_k(audio_tokens)
        audio_values = self.aud_to_v(audio_tokens)

        audio_slots = initial_slots
        audio_query = None
        for _ in range(self.iters):
            previous_slots = audio_slots
            normalized_slots = self.aud_norm_slots(audio_slots)
            audio_query = self.aud_to_q(normalized_slots)
            attention = self._attention(audio_query, audio_keys)
            updates = torch.einsum("bjd,bij->bid", audio_values, attention)
            audio_slots = self.aud_gru(
                updates.reshape(-1, channels),
                previous_slots.reshape(-1, channels),
            )
            audio_slots = audio_slots.reshape(batch_size, -1, channels)
            audio_slots = audio_slots + self.aud_mlp(
                self.aud_norm_pre_ff(audio_slots)
            )

        return audio_slots, audio_query, audio_keys


class MultiLayerJointSlotAttention(nn.Module):
    """Three independent visual SA branches plus exactly one audio SA branch."""

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
            raise ValueError("MUFASA-JSA v1 only supports slot_alignment='none'")

        self.num_slots = num_slots
        self.infer_sharpening = infer_sharpening
        self.mask_ratio = mask_ratio
        self.slot_dim = slot_dim
        self.slot_alignment = slot_alignment
        self.eps = 1e-8
        self.scale = slot_dim**-0.5

        # Shared initialization preserves JSA's target/off-target slot semantics.
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.mask_token_img = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.mask_token_aud = nn.Parameter(torch.randn(1, 1, slot_dim))

        self.visual_branches = nn.ModuleList(
            [VisualSlotBranch(slot_dim=slot_dim, iters=iters) for _ in range(3)]
        )
        self.audio_branch = AudioSlotBranch(slot_dim=slot_dim, iters=iters)
        self.slot_fusion = MFusion(slot_dim=slot_dim)
        self.attention_fusion = LearnedAttentionFusion()

        # TODO: add Hungarian cross-layer slot alignment ablation.

    def _masked(self, tokens, mask_token):
        if not self.training:
            return tokens
        batch_size, num_tokens, channels = tokens.shape
        mask = torch.rand(
            batch_size, num_tokens, 1, device=tokens.device
        ) < self.mask_ratio
        mask = mask.to(dtype=tokens.dtype)
        expanded_token = mask_token.expand(batch_size, num_tokens, channels)
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
        if len(image_levels) != 3:
            raise ValueError(f"Expected three visual levels, got {len(image_levels)}")
        if any(level.shape[-1] != self.slot_dim for level in image_levels):
            raise ValueError("Every visual level must have feature dimension 512")
        if any(level.shape[1] != image_levels[0].shape[1] for level in image_levels):
            raise ValueError("All visual levels must have the same token count")

        batch_size = image_levels[0].shape[0]
        initial_slots = self.slots.expand(batch_size, -1, -1)

        # Audio masking and audio Slot Attention are each executed exactly once.
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

    def _fused_attentions(self, encoded, scale_multiplier):
        imgq_imgk = [
            self._attention(query, key, scale_multiplier)
            for query, key in zip(
                encoded["visual_queries"], encoded["visual_keys"]
            )
        ]
        imgq_audk = [
            self._attention(query, encoded["audio_keys"], scale_multiplier)
            for query in encoded["visual_queries"]
        ]
        audq_imgk = [
            self._attention(encoded["audio_query"], key, scale_multiplier)
            for key in encoded["visual_keys"]
        ]
        audq_audk = self._attention(
            encoded["audio_query"], encoded["audio_keys"], scale_multiplier
        )

        return {
            "imgq_imgk_attn": self.attention_fusion(imgq_imgk),
            "imgq_audk_attn": self.attention_fusion(imgq_audk),
            "audq_imgk_attn": self.attention_fusion(audq_imgk),
            "audq_audk_attn": audq_audk,
        }

    def forward(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        output = {
            "img_slots": self.slot_fusion(encoded["visual_slots"]),
            "aud_slots": encoded["audio_slots"],
        }
        output.update(self._fused_attentions(encoded, scale_multiplier=1.0))
        return output

    def get_eval_attentions(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        attentions = self._fused_attentions(
            encoded, scale_multiplier=self.infer_sharpening
        )
        return attentions["imgq_imgk_attn"], attentions["audq_imgk_attn"]

    def get_l4_eval_attentions(self, image_levels, audio_tokens):
        if len(image_levels) != 3:
            raise ValueError(f"Expected three visual levels, got {len(image_levels)}")

        batch_size = image_levels[-1].shape[0]
        initial_slots = self.slots.expand(batch_size, -1, -1)
        _, audio_query, _ = self.audio_branch(audio_tokens, initial_slots)
        _, image_query, image_keys = self.visual_branches[-1](
            image_levels[-1], initial_slots
        )

        img_attn = self._attention(
            image_query, image_keys, self.infer_sharpening
        )
        cross_attn = self._attention(
            audio_query, image_keys, self.infer_sharpening
        )
        return img_attn, cross_attn

    def get_fused_query_l4_eval_attentions(self, image_levels, audio_tokens):
        """Evaluate fused visual slots as queries against the L4 visual keys."""
        encoded = self._encode(image_levels, audio_tokens)
        fused_img_slots = self.slot_fusion(encoded["visual_slots"])

        l4_branch = self.visual_branches[-1]
        fused_image_query = l4_branch.img_to_q(
            l4_branch.img_norm_slots(fused_img_slots)
        )
        l4_image_keys = encoded["visual_keys"][-1]

        img_attn = self._attention(
            fused_image_query, l4_image_keys, self.infer_sharpening
        )
        cross_attn = self._attention(
            encoded["audio_query"], l4_image_keys, self.infer_sharpening
        )
        return img_attn, cross_attn

    def get_slots(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        return self.slot_fusion(encoded["visual_slots"]), encoded["audio_slots"]
