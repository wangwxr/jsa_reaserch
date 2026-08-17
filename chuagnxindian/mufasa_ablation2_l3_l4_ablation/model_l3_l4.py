"""L3+L4 two-level ablation for MUFASA-JSA v1.1."""

import torch.nn as nn

import model_slot
import resnet

from model_mufasa_jsa_v1_1 import MUFASAJSA11
from l3_l4_slot_attention import L3L4JointSlotAttention
from two_level_resnet import resnet18_l3_l4


class MUFASAL3L4(MUFASAJSA11):
    """L3/L4 slots fused for losses; L4 alone supplies spatial attention."""

    def __init__(self, args):
        nn.Module.__init__(self)
        if args.out_dim != 512:
            raise ValueError("L3+L4 ablation requires --out_dim 512")

        self.tau = args.tau
        self.k = args.reciprocal_k
        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening

        self.imgnet = resnet18_l3_l4(
            pretrained=True, output_dim=args.out_dim
        )
        self.audnet = resnet.resnet(
            pretrained=False,
            modal="audio",
            dropout_rate=0.0,
            output_dim=args.out_dim,
            fourth_stride=2,
        )
        self.slot_attn = L3L4JointSlotAttention(
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


mymodel = MUFASAL3L4
