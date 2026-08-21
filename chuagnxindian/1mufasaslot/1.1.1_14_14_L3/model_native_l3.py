"""Strict single-variable native-L3 ablation of MUFASA-JSA v1.1."""

import torch
import torch.nn as nn

import model_slot
import resnet

from model_mufasa_jsa_v1_1 import MUFASAJSA11
from native_l3_resnet import resnet18_native_l3
from native_l3_slot_attention import NativeL3MultiLayerJointSlotAttentionV11


class MUFASAJSA11NativeL3(MUFASAJSA11):
    """v1.1 with 196 L3 tokens; all remaining modules and losses are unchanged."""

    def __init__(self, args):
        nn.Module.__init__(self)
        if args.out_dim != 512:
            raise ValueError("1.1.1_14_14_L3 requires --out_dim 512")

        self.tau = args.tau
        self.k = args.reciprocal_k
        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening

        # Construction order intentionally matches formal v1.1 exactly.
        self.imgnet = resnet18_native_l3(pretrained=True, output_dim=args.out_dim)
        self.audnet = resnet.resnet(
            pretrained=False,
            modal="audio",
            dropout_rate=0.0,
            output_dim=args.out_dim,
            fourth_stride=2,
        )
        self.slot_attn = NativeL3MultiLayerJointSlotAttentionV11(
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

    def forward_eval_with_ownership(self, image, audio):
        image_levels = self.imgnet(image)
        audio_tokens = self._audio_tokens(self.audnet(audio))
        output = self.slot_attn.get_eval_outputs(image_levels, audio_tokens)
        batch_size = image.shape[0]
        output["IMG_QUERY"] = output["IMG_L4_ALL"][:, 0].reshape(
            batch_size, 1, 7, 7
        )
        output["AUD"] = output["AUD_L4_ALL"][:, 0].reshape(
            batch_size, 1, 7, 7
        )
        output["IMAGE_LEVELS"] = image_levels
        return output

    def forward_eval(self, image, audio):
        with torch.no_grad():
            output = self.forward_eval_with_ownership(image, audio)
        return output["IMG_QUERY"], output["AUD"]


mymodel = MUFASAJSA11NativeL3
