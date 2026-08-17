"""MUFASA-inspired Multi-Layer JSA v1.1 model."""

import torch
import torch.nn as nn

import model_slot
import resnet

from model_mufasa_jsa import MUFASAJSA
from multi_layer_slot_attention_v1_1 import MultiLayerJointSlotAttentionV11
from multi_level_resnet import resnet18_multilevel


class MUFASAJSA11(MUFASAJSA):
    """v1 slots/losses with L4-only training and evaluation attentions."""

    def __init__(self, args):
        nn.Module.__init__(self)
        if args.out_dim != 512:
            raise ValueError("MUFASA-JSA v1.1 requires --out_dim 512")

        self.tau = args.tau
        self.k = args.reciprocal_k
        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening

        self.imgnet = resnet18_multilevel(pretrained=True, output_dim=args.out_dim)
        self.audnet = resnet.resnet(
            pretrained=False,
            modal="audio",
            dropout_rate=0.0,
            output_dim=args.out_dim,
            fourth_stride=2,
        )
        self.slot_attn = MultiLayerJointSlotAttentionV11(
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

    def forward_eval(self, image, audio):
        with torch.no_grad():
            image_levels = self.imgnet(image)
            audio_tokens = self._audio_tokens(self.audnet(audio))
            img_attn, cross_attn = self.slot_attn.get_eval_attentions(
                image_levels, audio_tokens
            )

        batch_size = image.size(0)
        img_attn = img_attn.reshape(batch_size, self.num_slots, 7, 7)
        cross_attn = cross_attn.reshape(batch_size, self.num_slots, 7, 7)
        return img_attn[:, 0].unsqueeze(1), cross_attn[:, 0].unsqueeze(1)


mymodel = MUFASAJSA11
