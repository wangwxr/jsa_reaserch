"""MUFASA-JSA v1.1 Slot Attention allowing a native 196-token L3 branch."""

import torch

from multi_layer_slot_attention_v1_1 import MultiLayerJointSlotAttentionV11


class NativeL3MultiLayerJointSlotAttentionV11(MultiLayerJointSlotAttentionV11):
    """Remove only the unnecessary equal-token-count constraint from v1.1."""

    def _encode(self, image_levels, audio_tokens):
        if len(image_levels) != 3:
            raise ValueError(f"Expected three visual levels, got {len(image_levels)}")
        if any(level.ndim != 3 for level in image_levels):
            raise ValueError("Every visual level must have shape [B,N,C]")
        if any(level.shape[-1] != self.slot_dim for level in image_levels):
            raise ValueError(
                f"Every visual level must have feature dimension {self.slot_dim}"
            )
        batch_size = image_levels[0].shape[0]
        if any(level.shape[0] != batch_size for level in image_levels):
            raise ValueError("Every visual level must have the same batch size")
        if audio_tokens.ndim != 3 or audio_tokens.shape[-1] != self.slot_dim:
            raise ValueError("Audio tokens must have shape [B,N,512]")
        if audio_tokens.shape[0] != batch_size:
            raise ValueError("Visual and audio batches must match")

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
            if slots.shape != (batch_size, self.num_slots, self.slot_dim):
                raise RuntimeError(f"Unexpected visual slot output: {slots.shape}")
            visual_slots.append(slots)
            visual_queries.append(query)
            visual_keys.append(keys)

        if audio_slots.shape != (batch_size, self.num_slots, self.slot_dim):
            raise RuntimeError(f"Unexpected audio slot output: {audio_slots.shape}")
        return {
            "visual_slots": visual_slots,
            "visual_queries": visual_queries,
            "visual_keys": visual_keys,
            "audio_slots": audio_slots,
            "audio_query": audio_query,
            "audio_keys": audio_keys,
        }

    def native_ownership(self, encoded):
        """Return final logits and pre-update token ownership for L3 and L4."""
        output = {}
        for name, index, expected_tokens in (("L3", 1, 196), ("L4", 2, 49)):
            query = encoded["visual_queries"][index]
            key = encoded["visual_keys"][index]
            logits = torch.einsum("bsd,bnd->bsn", query, key) * self.scale
            if logits.shape[1:] != (self.num_slots, expected_tokens):
                raise RuntimeError(f"Unexpected {name} final logits: {logits.shape}")
            ownership = logits.softmax(dim=1)
            spatial_size = 14 if name == "L3" else 7
            output[f"LOGITS_{name}"] = logits
            output[f"OWNERSHIP_{name}"] = ownership
            map_name = "SLOT_L3_NATIVE" if name == "L3" else "SLOT_L4"
            output[map_name] = ownership[:, 0].reshape(
                logits.shape[0], 1, spatial_size, spatial_size
            )
        return output

    def get_eval_outputs(self, image_levels, audio_tokens):
        encoded = self._encode(image_levels, audio_tokens)
        attentions = self._l4_attentions(
            encoded, scale_multiplier=self.infer_sharpening
        )
        output = {
            "IMG_L4_ALL": attentions["imgq_imgk_attn"],
            "AUD_L4_ALL": attentions["audq_imgk_attn"],
            "VISUAL_SLOTS": encoded["visual_slots"],
        }
        output.update(self.native_ownership(encoded))
        return output
