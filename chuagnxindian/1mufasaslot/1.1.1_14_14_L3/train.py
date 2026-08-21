#!/usr/bin/env python3
"""Formal training entry for 1.1.1_14_14_L3."""

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
V11_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, V11_ROOT, HERE):
    sys.path.insert(0, str(path))

import train_slot  # noqa: E402
from model_native_l3 import MUFASAJSA11NativeL3  # noqa: E402


def main():
    train_slot.model_slot.mymodel = MUFASAJSA11NativeL3
    args = train_slot.get_arguments()
    if args.model != "jsa":
        raise ValueError("1.1.1_14_14_L3 training requires --model jsa")
    args.architecture = "1.1.1_14_14_L3"
    args.visual_levels = ["layer2_7x7", "layer3_native_14x14", "layer4_7x7"]
    args.visual_token_counts = [49, 196, 49]
    args.slot_alignment = "none"
    args.spatial_attention = "l4_only"
    args.reconstruction_target = "projected_layer4_7x7"
    train_slot.main(args)


if __name__ == "__main__":
    main()
