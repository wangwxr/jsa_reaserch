#!/usr/bin/env python3
"""Test entry for the L4x3 parameter-control ablation."""

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V11_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import test_model  # noqa: E402
from model_l4x3_control import MUFASAL4x3Control  # noqa: E402


def main():
    test_model.model_slot.mymodel = MUFASAL4x3Control
    args = test_model.get_arguments()
    architecture = getattr(args, "architecture", None)
    if architecture != "mufasa_ablation1_l4x3_control":
        raise RuntimeError(
            "This entry only tests L4x3 control checkpoints; "
            f"configs.json has architecture={architecture!r}"
        )
    if getattr(args, "model", "jsa") != "jsa":
        raise ValueError("L4x3 control testing requires model='jsa'")
    test_model.main(args)


if __name__ == "__main__":
    main()
