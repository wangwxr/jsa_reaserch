#!/usr/bin/env python3
"""Test entry for MUFASA-JSA v1.1 using the unchanged JSA evaluator."""

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import test_model  # noqa: E402
from model_mufasa_jsa_v1_1 import MUFASAJSA11  # noqa: E402


def main():
    test_model.model_slot.mymodel = MUFASAJSA11
    args = test_model.get_arguments()
    architecture = getattr(args, "architecture", None)
    if architecture != "mufasa_jsa_v1_1":
        raise RuntimeError(
            "This entry only tests MUFASA-JSA v1.1 checkpoints; "
            f"configs.json has architecture={architecture!r}"
        )
    if getattr(args, "model", "jsa") != "jsa":
        raise ValueError("MUFASA-JSA v1.1 testing requires model='jsa'")
    test_model.main(args)


if __name__ == "__main__":
    main()
