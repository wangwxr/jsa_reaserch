"""Experiment 3.1 Stage1 model with A3+A4 audio semantics and A4 localization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_slot

from audio_multilevel import audio_resnet18_multilevel
from model_mufasa_jsa import MUFASAJSA
from slot_attention import HierarchicalAudioL3L4SlotAttention
from two_level_resnet import resnet18_l3_l4


def temporal_token_diagnostics(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    normalized = F.normalize(tokens, dim=-1)
    adjacent = F.cosine_similarity(
        normalized[:, :-1], normalized[:, 1:], dim=-1
    ).mean()
    pairwise = torch.einsum("btd,bsd->bts", normalized, normalized)
    token_count = tokens.shape[1]
    off_diagonal = ~torch.eye(
        token_count, dtype=torch.bool, device=tokens.device
    ).unsqueeze(0)
    pairwise_mean = pairwise.masked_select(off_diagonal).mean()
    temporal_variance = tokens.var(dim=1, unbiased=False).mean()
    return {
        "adjacent_cosine": adjacent,
        "pairwise_cosine": pairwise_mean,
        "temporal_feature_variance": temporal_variance,
    }


class HierarchicalAudioStage1(MUFASAJSA):
    """Formal L3+L4 Stage1 with hierarchy only in the semantic audio path."""

    eval_audio_query_source = "A4"

    def __init__(self, args):
        nn.Module.__init__(self)
        if args.out_dim != 512:
            raise ValueError("Experiment 3.1 requires --out_dim 512")

        self.tau = args.tau
        self.k = args.reciprocal_k
        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening

        self.imgnet = resnet18_l3_l4(pretrained=True, output_dim=args.out_dim)
        self.audnet = audio_resnet18_multilevel(
            output_dim=args.out_dim,
            fourth_stride=2,
        )
        self.slot_attn = HierarchicalAudioL3L4SlotAttention(
            num_slots=args.num_slots,
            infer_sharpening=args.infer_sharpening,
            mask_ratio=args.mask_ratio,
            iters=args.iters,
            slot_dim=args.out_dim,
            slot_alignment="none",
        )

        self.img_decoder = model_slot.MlpDecoder(7, 7, 512, 512)
        self.aud_decoder = model_slot.MlpDecoder(1, 16, 512, 512)
        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()

    @staticmethod
    def _audio_tokens(audio_feature):
        return audio_feature.permute(0, 2, 1)

    def extract_features(self, frame, spec):
        image_levels = self.imgnet(frame)
        audio = self.audnet(spec)
        audio["a3_tokens"] = self._audio_tokens(audio["a3_feature"])
        audio["a4_tokens"] = self._audio_tokens(audio["a4_feature"])
        return image_levels, audio

    @staticmethod
    def representation_diagnostics(output, audio):
        a3_slots = output["aud_slots_a3"]
        a4_slots = output["aud_slots_a4"]
        fused_slots = output["aud_slots"]
        delta = output["audio_delta"]
        diagnostics = {
            "cos_a3_a4_slot0": F.cosine_similarity(
                a3_slots[:, 0], a4_slots[:, 0], dim=-1
            ).mean(),
            "cos_a3_a4_slot1": F.cosine_similarity(
                a3_slots[:, 1], a4_slots[:, 1], dim=-1
            ).mean(),
            "cos_fused_a4_slot0": F.cosine_similarity(
                fused_slots[:, 0], a4_slots[:, 0], dim=-1
            ).mean(),
            "cos_fused_a4_slot1": F.cosine_similarity(
                fused_slots[:, 1], a4_slots[:, 1], dim=-1
            ).mean(),
            "delta_norm_over_a4_norm": (
                delta.norm(dim=-1)
                / a4_slots.norm(dim=-1).clamp_min(1e-8)
            ).mean(),
        }
        for level in ("a3", "a4"):
            temporal = temporal_token_diagnostics(audio[f"{level}_tokens"])
            diagnostics.update(
                {f"{level}_{name}": value for name, value in temporal.items()}
            )
        return diagnostics

    def forward_train_detailed(self, frame, spec):
        image_levels, audio = self.extract_features(frame, spec)
        output = self.slot_attn(
            image_levels,
            audio["a3_tokens"],
            audio["a4_tokens"],
        )
        fused_img_slots = output["img_slots"]
        fused_audio_slots = output["aud_slots"]
        audio_slots_a4 = output["aud_slots_a4"]

        img_recon = self.img_decoder(fused_img_slots)
        aud_recon = self.aud_decoder(audio_slots_a4).flatten(start_dim=2)
        recon_loss = self.MSELoss(img_recon, image_levels[-1].detach())
        recon_loss = recon_loss + self.MSELoss(
            aud_recon, audio["a4_tokens"].detach()
        )

        att_loss = self.MSELoss(
            output["audq_imgk_attn"][:, 0, :],
            output["imgq_imgk_attn"][:, 0, :].detach(),
        )
        att_loss = att_loss + self.MSELoss(
            output["imgq_audk_attn"][:, 0, :],
            output["audq_audk_attn"][:, 0, :].detach(),
        )

        normalized_img_slots = F.normalize(fused_img_slots, dim=2)
        normalized_audio_slots = F.normalize(fused_audio_slots, dim=2)
        info_loss, div_loss = self.calculate(
            normalized_img_slots, normalized_audio_slots
        )
        diagnostics = self.representation_diagnostics(output, audio)
        return {
            "losses": (info_loss, recon_loss, div_loss, att_loss),
            "diagnostics": diagnostics,
            "features": audio,
            "representations": {
                "a3_slots": output["aud_slots_a3"],
                "a4_slots": output["aud_slots_a4"],
                "fused_audio_slots": fused_audio_slots,
                "a3_query": output["audio_query_a3"],
                "a4_query": output["audio_query_a4"],
                "fused_visual_slots": fused_img_slots,
            },
        }

    def forward_train(self, frame, spec):
        return self.forward_train_detailed(frame, spec)["losses"]

    def forward_eval(self, image, audio_input):
        with torch.no_grad():
            image_levels, audio = self.extract_features(image, audio_input)
            img_attn, cross_attn = self.slot_attn.get_eval_attentions(
                image_levels,
                audio["a3_tokens"],
                audio["a4_tokens"],
            )
        batch_size = image.size(0)
        img_attn = img_attn.reshape(batch_size, self.num_slots, 7, 7)
        cross_attn = cross_attn.reshape(batch_size, self.num_slots, 7, 7)
        return img_attn[:, 0].unsqueeze(1), cross_attn[:, 0].unsqueeze(1)

    @torch.no_grad()
    def forward_a3_query_eval(self, image, audio_input):
        image_levels, audio = self.extract_features(image, audio_input)
        attention = self.slot_attn.get_a3_query_eval_attention(
            image_levels,
            audio["a3_tokens"],
            audio["a4_tokens"],
        )
        return attention[:, 0].reshape(image.shape[0], 1, 7, 7)

    @torch.no_grad()
    def diagnostic_representations(self, image, audio_input):
        image_levels, audio = self.extract_features(image, audio_input)
        representations = self.slot_attn.get_representations(
            image_levels,
            audio["a3_tokens"],
            audio["a4_tokens"],
        )
        return audio, representations

    def forward(self, frame, spec):
        if self.training:
            return self.forward_train(frame, spec)
        return self.forward_eval(frame, spec)


mymodel = HierarchicalAudioStage1
