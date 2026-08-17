#!/usr/bin/env python3
"""Training entry for MUFASA-JSA v1.1."""

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import train_slot  # noqa: E402
from model_mufasa_jsa_v1_1 import MUFASAJSA11  # noqa: E402


def main():
    train_slot.model_slot.mymodel = MUFASAJSA11
    args = train_slot.get_arguments()
    if args.model != "jsa":
        raise ValueError("MUFASA-JSA v1.1 training requires --model jsa")
    args.architecture = "mufasa_jsa_v1_1"
    args.visual_levels = ["layer2", "layer3", "layer4"]
    args.slot_alignment = "none"
    args.spatial_attention = "l4_only"
    train_slot.main(args)


if __name__ == "__main__":
    main()
