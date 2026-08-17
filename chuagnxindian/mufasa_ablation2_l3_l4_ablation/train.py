#!/usr/bin/env python3
"""Training entry for the L3+L4 two-level ablation."""

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V11_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import train_slot  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402


def main():
    train_slot.model_slot.mymodel = MUFASAL3L4
    args = train_slot.get_arguments()
    if args.model != "jsa":
        raise ValueError("L3+L4 ablation training requires --model jsa")
    args.architecture = "mufasa_ablation2_l3_l4_ablation"
    args.visual_levels = ["layer3", "layer4"]
    args.slot_alignment = "none"
    args.spatial_attention = "l4_only"
    args.checkpoint_selection = "IQR_cIoU"
    train_slot.main(args)


if __name__ == "__main__":
    main()
