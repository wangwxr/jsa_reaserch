#!/usr/bin/env python3
"""Run the verified 8.7 C3 Stage-1 trainer against the fixed 10k assets."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MEAN = HERE.parent / "8.7_counterfactual_relative_attention_matching"
HARD = MEAN / "8.7_hard_negative_cram"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mean", "hard_default"), required=True)
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.environ["CRAM10K_METHOD"] = args.method
    if args.method == "hard_default":
        os.environ["CRAM_NEGATIVE_AGGREGATION"] = "softmin"
        os.environ["CRAM_SOFTMIN_TAU"] = {
            "vggss": "2.6128406021282223e-05",
            "flickr": "8.700235267679427e-06",
        }[args.dataset]
    else:
        os.environ["CRAM_NEGATIVE_AGGREGATION"] = "mean"
        os.environ.pop("CRAM_SOFTMIN_TAU", None)
    # The 144k acceleration audit established that this objective is exactly
    # equivalent for Mean when aggregation=mean: it only reuses source-audio
    # tokens before applying the *same* individual D_neg distances.  The
    # current verified trainer calls that reuse interface for both methods;
    # the only method variable remains the aggregation selected above.
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(HARD))
    sys.argv = [str(MEAN / "train.py"), "--dataset", args.dataset, "--group", "C3",
                "--seed", "12345", "--gpu", str(args.gpu)]
    runpy.run_path(str(MEAN / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
