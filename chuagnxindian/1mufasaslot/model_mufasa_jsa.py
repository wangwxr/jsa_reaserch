"""MUFASA-inspired Multi-Layer JSA v1 model."""

import torch
import torch.nn as nn

import model_slot
import resnet
import utils

from multi_layer_slot_attention import MultiLayerJointSlotAttention
from multi_level_resnet import resnet18_multilevel


class MUFASAJSA(nn.Module):
    """JSA with multi-level visual slots and a single unchanged audio branch."""

    def __init__(self, args):
        super().__init__()
        if args.out_dim != 512:
            raise ValueError("MUFASA-JSA v1 requires --out_dim 512")

        self.tau = args.tau
        self.k = args.reciprocal_k
        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening
        self.eval_attention_mode = getattr(args, "eval_attention_mode", "fused")
        if self.eval_attention_mode not in {
            "fused",
            "l4_only",
            "fused_query_l4",
        }:
            raise ValueError(
                f"Unknown eval_attention_mode={self.eval_attention_mode!r}"
            )

        self.imgnet = resnet18_multilevel(pretrained=True, output_dim=args.out_dim)
        self.audnet = resnet.resnet(
            pretrained=False,
            modal="audio",
            dropout_rate=0.0,
            output_dim=args.out_dim,
            fourth_stride=2,
        )
        self.slot_attn = MultiLayerJointSlotAttention(
            num_slots=args.num_slots,
            infer_sharpening=args.infer_sharpening,
            mask_ratio=args.mask_ratio,
            iters=args.iters,
            slot_dim=args.out_dim,
            slot_alignment="none",
        )

        # Identical decoder targets and dimensions to the original JSA.
        self.img_decoder = model_slot.MlpDecoder(7, 7, 512, 512)
        self.aud_decoder = model_slot.MlpDecoder(1, 16, 512, 512)

        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()

    def cosine_loss(self, slots):
        batch_size, num_slots, _ = slots.size()
        mask = (1 - torch.eye(num_slots)).unsqueeze(0)
        mask = mask.expand(batch_size, -1, -1).to(slots.device)
        similarity = torch.einsum("bic,bjc->bij", slots, slots)
        positive_similarity = torch.relu(similarity * mask)
        return positive_similarity.sum() / (
            batch_size * num_slots * (num_slots - 1)
        )

    def calculate(self, img_slots, aud_slots):
        batch_size, _, channels = img_slots.size()
        aud_batch_size, _, aud_channels = aud_slots.size()
        if batch_size != aud_batch_size or channels != aud_channels:
            raise ValueError("Visual and audio slots must have matching batches/dims")

        labels = torch.arange(batch_size, device=img_slots.device).long()
        similarity = torch.einsum("nic,mjc->nmij", img_slots, aud_slots)
        target_similarity = similarity[:, :, 0, 0]

        _, _, reciprocal = utils.get_potential_false_negative(
            img_slots.detach(), aud_slots.detach(), k=self.k
        )
        target_similarity = target_similarity.masked_fill(
            reciprocal == False, -float("inf")
        )
        info_loss = self.CELoss(target_similarity / self.tau, labels)
        info_loss = info_loss + self.CELoss(
            target_similarity.permute(1, 0) / self.tau, labels
        )
        div_loss = self.cosine_loss(img_slots) + self.cosine_loss(aud_slots)
        return info_loss, div_loss

    @staticmethod
    def _audio_tokens(audio_feature):
        return audio_feature.permute(0, 2, 1)

    def forward_train(self, frame, spec):
        image_levels = self.imgnet(frame)
        audio_tokens = self._audio_tokens(self.audnet(spec))

        output = self.slot_attn(image_levels, audio_tokens)
        fused_img_slots = output["img_slots"]
        aud_slots = output["aud_slots"]

        img_recon = self.img_decoder(fused_img_slots)
        aud_recon = self.aud_decoder(aud_slots).flatten(start_dim=2)

        # Only the final projected visual representation is reconstructed.
        layer4_target = image_levels[-1]
        recon_loss = self.MSELoss(img_recon, layer4_target.detach())
        recon_loss = recon_loss + self.MSELoss(aud_recon, audio_tokens.detach())

        att_loss = self.MSELoss(
            output["audq_imgk_attn"][:, 0, :],
            output["imgq_imgk_attn"][:, 0, :].detach(),
        )
        att_loss = att_loss + self.MSELoss(
            output["imgq_audk_attn"][:, 0, :],
            output["audq_audk_attn"][:, 0, :].detach(),
        )

        normalized_img_slots = nn.functional.normalize(fused_img_slots, dim=2)
        normalized_aud_slots = nn.functional.normalize(aud_slots, dim=2)
        info_loss, div_loss = self.calculate(
            normalized_img_slots, normalized_aud_slots
        )
        return info_loss, recon_loss, div_loss, att_loss

    def forward_eval(self, image, audio):
        with torch.no_grad():
            image_levels = self.imgnet(image)
            audio_tokens = self._audio_tokens(self.audnet(audio))
            if self.eval_attention_mode == "l4_only":
                img_attn, cross_attn = self.slot_attn.get_l4_eval_attentions(
                    image_levels, audio_tokens
                )
            elif self.eval_attention_mode == "fused_query_l4":
                img_attn, cross_attn = (
                    self.slot_attn.get_fused_query_l4_eval_attentions(
                        image_levels, audio_tokens
                    )
                )
            else:
                img_attn, cross_attn = self.slot_attn.get_eval_attentions(
                    image_levels, audio_tokens
                )

        batch_size = image.size(0)
        img_attn = img_attn.reshape(batch_size, self.num_slots, 7, 7)
        cross_attn = cross_attn.reshape(batch_size, self.num_slots, 7, 7)
        return img_attn[:, 0].unsqueeze(1), cross_attn[:, 0].unsqueeze(1)

    def get_slot(self, frame, spec):
        with torch.no_grad():
            image_levels = self.imgnet(frame)
            audio_tokens = self._audio_tokens(self.audnet(spec))
            return self.slot_attn.get_slots(image_levels, audio_tokens)

    def forward(self, frame, spec):
        if self.training:
            return self.forward_train(frame, spec)
        return self.forward_eval(frame, spec)


# The baseline entry points instantiate model_slot.mymodel(args).
mymodel = MUFASAJSA
