#!/usr/bin/env python3
"""Aggregate Experiment 2.1R VGG/Flickr summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = ("vggss_144k", "flickr_144k")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    summaries = {dataset: json.loads((args.result_root / dataset / "summary.json").read_text()) for dataset in DATASETS}
    features = [row["feature"] for row in summaries[DATASETS[0]]["reliability_auroc"]]
    auroc_rows = []
    for feature in features:
        row = {"feature": feature}
        for dataset in DATASETS:
            lookup = {item["feature"]: item for item in summaries[dataset]["reliability_auroc"]}
            row[f"{dataset}_AUROC"] = lookup[feature]["AUROC"]
        auroc_rows.append(row)
    label_rows = [{"dataset": dataset, **summary["fixed_label_summary"]} for dataset, summary in summaries.items()]
    distribution_rows = [
        {"dataset": dataset, **row}
        for dataset, summary in summaries.items()
        for row in summary["feature_group_statistics"]
    ]
    write_csv(args.result_root / "combined_fixed_reliability_auroc.csv", auroc_rows)
    write_csv(args.result_root / "combined_fixed_label_summary.csv", label_rows)
    write_csv(args.result_root / "combined_feature_group_statistics.csv", distribution_rows)
    combined = {"experiment": "2.1R Fixed-Slot Reliability Recheck", "zero_training": True, "datasets": summaries}
    (args.result_root / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("Fixed-slot labels")
    for row in label_rows:
        print(f"{row['dataset']}: Rescue={row['rescue']} Hurt={row['hurt']} Net={row['net_rescue']}")
    print("\nFixed-slot Rescue-vs-Hurt AUROC")
    print(f"{'Feature':<30} {'VGG':>10} {'Flickr':>10}")
    for row in auroc_rows:
        print(f"{row['feature']:<30} {row['vggss_144k_AUROC']:>10.4f} {row['flickr_144k_AUROC']:>10.4f}")


if __name__ == "__main__":
    main()
