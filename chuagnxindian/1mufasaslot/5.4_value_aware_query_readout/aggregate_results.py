#!/usr/bin/env python3
"""Aggregate formal VGGSoundSS/Flickr Experiment 5.4 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = ("vggss_144k", "flickr_144k")
METHODS = (
    "AUD_FINE",
    "IMG_L4",
    "IMG_FINE",
    "IMG_VALUE",
    "IMG_VALUE_RES",
    "IQR_FINE",
    "IQR_VALUE",
    "IQR_VALUE_RES",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    summaries = {
        experiment: json.loads(
            (arguments.result_root / experiment / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for experiment in EXPERIMENTS
    }
    rows = []
    for method in METHODS:
        row = {"method": method}
        for experiment in EXPERIMENTS:
            values = summaries[experiment]["metrics"][method]
            row[f"{experiment}_cIoU"] = values["cIoU"]
            row[f"{experiment}_AUC"] = values["AUC"]
        rows.append(row)
    output = arguments.result_root / "combined_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (arguments.result_root / "combined_summary.json").write_text(
        json.dumps({"experiment": "5.4_value_aware_query_readout", "datasets": summaries}, indent=2),
        encoding="utf-8",
    )

    print("\nMethod                    VGGSS-144k cIoU/AUC     Flickr-144k cIoU/AUC")
    for row in rows:
        print(
            f"{row['method']:<25} "
            f"{row['vggss_144k_cIoU']:.4f}/{row['vggss_144k_AUC']:.4f}          "
            f"{row['flickr_144k_cIoU']:.4f}/{row['flickr_144k_AUC']:.4f}"
        )
    for experiment in EXPERIMENTS:
        summary = summaries[experiment]
        print(f"\n{experiment}:")
        print("  deltas:", json.dumps(summary["metric_deltas"], sort_keys=True))
        print("  per_sample:", json.dumps(summary["per_sample_analysis"], sort_keys=True))
        print("  audit_passed:", summary["audit"]["passed"])
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
