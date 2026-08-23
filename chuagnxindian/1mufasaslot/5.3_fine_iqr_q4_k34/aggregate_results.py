#!/usr/bin/env python3
"""Aggregate the two formal Experiment 5.3 zero-training evaluations."""

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
    "IQR_OLD",
    "IQR_FINE",
    "IQR_FINE_EVALSPACE",
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
            metric = summaries[experiment]["metrics"][method]
            row[f"{experiment}_cIoU"] = metric["cIoU"]
            row[f"{experiment}_AUC"] = metric["AUC"]
        rows.append(row)

    output_csv = arguments.result_root / "combined_metrics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    combined = {
        "experiment": "5.3_fine_iqr_q4_k34",
        "datasets": summaries,
    }
    (arguments.result_root / "combined_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
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
        print(f"\n{experiment} per-sample changes:")
        for comparison, counts in summary["per_sample_analysis"].items():
            print(f"  {comparison}: {counts}")
        print(f"  audit_passed={summary['audit']['passed']}")
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
