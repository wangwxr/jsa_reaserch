"""MUFASA-JSA v1.1: fused slots with L4-only spatial attention."""

from multi_layer_slot_attention import MultiLayerJointSlotAttention


class MultiLayerJointSlotAttentionV11(MultiLayerJointSlotAttention):
    """Keep multi-layer slots/M-Fusion, but use only L4 attention maps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v1.1 has no learned spatial-attention fusion parameter or path.
        del self.attention_fusion

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
