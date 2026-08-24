"""Loss-factorized L3+L4 teacher used only by Experiment 6.1.

The architecture and parameter names intentionally remain identical to the
formal L3+L4 ablation.  Only diagnostic return values and independently
weighted loss composition are added here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from model_l3_l4 import MUFASAL3L4


@dataclass
class LossWeights:
    rec_img: float = 0.1
    rec_aud: float = 0.1
    div_img: float = 0.1
    div_aud: float = 0.1
    att_space: float = 100.0
    att_time: float = 100.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "rec_img": self.rec_img,
            "rec_aud": self.rec_aud,
            "div_img": self.div_img,
            "div_aud": self.div_aud,
            "att_space": self.att_space,
            "att_time": self.att_time,
        }


def compose_total(losses: Dict[str, torch.Tensor], weights: LossWeights):
    """Compose the factorized loss, preserving the formal Full grouping."""
    if weights == LossWeights():
        # This is the exact parenthesization used by train_slot.py with
        # recon/div/attention aggregates returned by the formal model.
        return (
            losses["info"]
            + 0.1 * (losses["rec_img"] + losses["rec_aud"])
            + 0.1 * (losses["div_img"] + losses["div_aud"])
            + 100.0 * (losses["att_space"] + losses["att_time"])
        )
    return (
        losses["info"]
        + weights.rec_img * losses["rec_img"]
        + weights.rec_aud * losses["rec_aud"]
        + weights.div_img * losses["div_img"]
        + weights.div_aud * losses["div_aud"]
        + weights.att_space * losses["att_space"]
        + weights.att_time * losses["att_time"]
    )


class LossFactorizedL3L4(MUFASAL3L4):
    """Numerically faithful L3+L4 model with six exposed auxiliary losses."""

    def forward_train_detailed(self, frame, spec, weights=None):
        weights = weights or LossWeights()

        image_levels = self.imgnet(frame)
        audio_tokens = self._audio_tokens(self.audnet(spec))
        output = self.slot_attn(image_levels, audio_tokens)

        fused_img_slots = output["img_slots"]
        aud_slots = output["aud_slots"]
        img_recon = self.img_decoder(fused_img_slots)
        aud_recon = self.aud_decoder(aud_slots).flatten(start_dim=2)
        layer4_target = image_levels[-1]

        losses = {
            "rec_img": self.MSELoss(img_recon, layer4_target.detach()),
            "rec_aud": self.MSELoss(aud_recon, audio_tokens.detach()),
            "att_space": self.MSELoss(
                output["audq_imgk_attn"][:, 0, :],
                output["imgq_imgk_attn"][:, 0, :].detach(),
            ),
            "att_time": self.MSELoss(
                output["imgq_audk_attn"][:, 0, :],
                output["audq_audk_attn"][:, 0, :].detach(),
            ),
        }

        normalized_img_slots = F.normalize(fused_img_slots, dim=2)
        normalized_aud_slots = F.normalize(aud_slots, dim=2)
        info_loss, old_div_loss = self.calculate(
            normalized_img_slots, normalized_aud_slots
        )
        losses["info"] = info_loss
        losses["div_img"] = self.cosine_loss(normalized_img_slots)
        losses["div_aud"] = self.cosine_loss(normalized_aud_slots)
        losses["old_div"] = old_div_loss
        losses["old_recon"] = losses["rec_img"] + losses["rec_aud"]
        losses["old_att"] = losses["att_space"] + losses["att_time"]
        losses["total"] = compose_total(losses, weights)

        weighted = {"info": losses["info"]}
        for name, coefficient in weights.as_dict().items():
            weighted[name] = losses[name] * coefficient

        return {
            "losses": losses,
            "weighted_losses": weighted,
            "image_levels": image_levels,
            "audio_tokens": audio_tokens,
            "fused_img_slots": fused_img_slots,
            "audio_slots": aud_slots,
            "img_recon": img_recon,
            "aud_recon": aud_recon,
            "img_raw": output["imgq_imgk_attn"][:, 0, :],
            "aud_raw": output["audq_imgk_attn"][:, 0, :],
            "attentions": output,
        }

    def forward_train(self, frame, spec):
        detailed = self.forward_train_detailed(frame, spec, LossWeights())
        losses = detailed["losses"]
        return (
            losses["info"],
            losses["old_recon"],
            losses["old_div"],
            losses["old_att"],
        )

    def forward_eval_detailed(self, image, audio, run_decoder=True):
        """Expose exactly the formal L4-only evaluator maps and decoder alpha."""
        image_levels = self.imgnet(image)
        audio_tokens = self._audio_tokens(self.audnet(audio))
        encoded = self.slot_attn._encode(image_levels, audio_tokens)
        attentions = self.slot_attn._l4_attentions(
            encoded, scale_multiplier=self.infer_sharpening
        )
        fused_img_slots = self.slot_attn.slot_fusion(encoded["visual_slots"])

        decoder_alpha = None
        hook = None
        if run_decoder:
            captured = {}

            def capture_alpha(_module, _inputs, output):
                captured["alpha"] = output

            hook = self.img_decoder.alpha_holder.register_forward_hook(
                capture_alpha
            )
            self.img_decoder(fused_img_slots)
            decoder_alpha = captured["alpha"]
            hook.remove()

        batch_size = image.shape[0]
        img_all = attentions["imgq_imgk_attn"].reshape(
            batch_size, self.num_slots, 7, 7
        )
        aud_all = attentions["audq_imgk_attn"].reshape(
            batch_size, self.num_slots, 7, 7
        )
        return {
            "img": img_all[:, 0:1],
            "aud": aud_all[:, 0:1],
            "img_all": img_all,
            "aud_all": aud_all,
            "decoder_alpha": decoder_alpha,
            "fused_img_slots": fused_img_slots,
            "audio_slots": encoded["audio_slots"],
            "image_levels": image_levels,
            "audio_tokens": audio_tokens,
        }


mymodel = LossFactorizedL3L4
