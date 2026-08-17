#!/usr/bin/env python3
"""Training entry for the L4x3 parameter-control ablation."""

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V11_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import train_slot  # noqa: E402
from model_l4x3_control import MUFASAL4x3Control  # noqa: E402


def main():
    train_slot.model_slot.mymodel = MUFASAL4x3Control
    args = train_slot.get_arguments()
    if args.model != "jsa":
        raise ValueError("L4x3 control training requires --model jsa")
    args.architecture = "mufasa_ablation1_l4x3_control"
    args.visual_levels = ["layer4", "layer4", "layer4"]
    args.slot_alignment = "none"
    args.spatial_attention = "l4_only"
    args.checkpoint_selection = "IQR_cIoU"
    train_slot.main(args)


if __name__ == "__main__":
    main()
