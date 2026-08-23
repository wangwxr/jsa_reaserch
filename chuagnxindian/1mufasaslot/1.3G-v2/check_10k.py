#!/usr/bin/env python3
"""Fail closed before 144k only on a clear 10k numerical/training disaster."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = {
    "vggss_10k": ("1.3G-v2_vggss_10k", 0.4112059),
    "flickr_10k": ("1.3G-v2_flickr_10k_frame8_center5", 0.8040),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=sorted(REFERENCE),
        help="Check one dataset pipeline; omit to check both.",
    )
    parser.add_argument(
        "--maximum-allowed-ciou-drop",
        type=float,
        default=0.05,
        help="Only a clear absolute cIoU collapse blocks 144k; no improvement is required.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = {}
    selected = (
        {arguments.experiment: REFERENCE[arguments.experiment]}
        if arguments.experiment
        else REFERENCE
    )
    for key, (experiment, original_ciou) in selected.items():
        directory = PROJECT_ROOT / "checkpoints" / experiment
        metrics_path = directory / "best_test_metrics.json"
        history_path = directory / "epoch_metrics.csv"
        sanity_path = directory / "sanity_checks.json"
        for path in (metrics_path, history_path, sanity_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = [float(value) for row in rows for value in row.values()]
        current = float(metrics["AUD_FINE"]["cIoU"])
        passed = (
            bool(rows)
            and all(math.isfinite(value) for value in values)
            and not sanity["teacher_has_any_gradient"]
            and current >= original_ciou - arguments.maximum_allowed_ciou_drop
        )
        report[key] = {
            "experiment": experiment,
            "epochs_completed": len(rows),
            "original_G_AUD_FINE_cIoU": original_ciou,
            "v2_AUD_FINE_cIoU": current,
            "minimum_non_disaster_cIoU": original_ciou - arguments.maximum_allowed_ciou_drop,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("10k non-disaster gate passed; 144k is authorized by the staged script.")


if __name__ == "__main__":
    main()
