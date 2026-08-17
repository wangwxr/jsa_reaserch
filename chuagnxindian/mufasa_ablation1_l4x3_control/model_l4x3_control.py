"""L4x3 parameter-control ablation for MUFASA-JSA v1.1."""

import torch.nn as nn

import model_slot
import resnet

from model_mufasa_jsa_v1_1 import MUFASAJSA11
from multi_layer_slot_attention_v1_1 import MultiLayerJointSlotAttentionV11
from multi_level_resnet import resnet18_multilevel


class L4TripleResNet18(nn.Module):
    """Return the same projected L4 tokens to three independent SA branches."""

    def __init__(self, output_dim=512):
        super().__init__()
        self.backbone = resnet18_multilevel(
            pretrained=True, output_dim=output_dim
        )
        del self.backbone.proj2
        del self.backbone.proj3

    def forward(self, image):
        _, _, layer4 = self.backbone.forward_feature_maps(image)
        layer4 = self.backbone._pool_to_7x7(
            self.backbone.proj4(layer4)
        )
        layer4_tokens = self.backbone._to_tokens(layer4)
        return layer4_tokens, layer4_tokens, layer4_tokens


class MUFASAL4x3Control(MUFASAJSA11):
    """Three independent visual SA branches receiving identical L4 tokens."""

    def __init__(self, args):
        nn.Module.__init__(self)
        if args.out_dim != 512:
            raise ValueError("L4x3 control requires --out_dim 512")

        self.tau = args.tau
        self.k = args.reciprocal_k
        self.num_slots = args.num_slots
        self.infer_sharpening = args.infer_sharpening

        self.imgnet = L4TripleResNet18(output_dim=args.out_dim)
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

        self._print_parameter_report()

    def _print_parameter_report(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        # v1.1 additionally has active L2/L3 projection layers.
        omitted_projection_parameters = (
            128 * 512 + 512
            + 256 * 512 + 512
        )
        reference_total = total + omitted_projection_parameters
        reference_trainable = trainable + omitted_projection_parameters

        print(f"L4x3 control total params: {total:,}")
        print(f"L4x3 control trainable params: {trainable:,}")
        print(f"MUFASA-JSA v1.1 total params: {reference_total:,}")
        print(
            "MUFASA-JSA v1.1 trainable params: "
            f"{reference_trainable:,}"
        )
        print(
            "Parameter difference vs v1.1: "
            f"{total - reference_total:+,} "
            f"({100.0 * (total - reference_total) / reference_total:+.3f}%)"
        )


mymodel = MUFASAL4x3Control
