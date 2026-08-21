#!/usr/bin/env python3
"""Aggregate the VGG/Flickr Experiment 2.2 tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = ("vggss_144k", "flickr_144k")
DISPLAY = {"vggss_144k": "VGG", "flickr_144k": "Flickr"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    summaries = {dataset: json.loads((args.result_root / dataset / "summary.json").read_text()) for dataset in DATASETS}
    metric_rows = []
    methods = [row["method"] for row in summaries[DATASETS[0]]["method_metrics"]]
    for method in methods:
        row = {"method": method}
        for dataset in DATASETS:
            lookup = {item["method"]: item for item in summaries[dataset]["method_metrics"]}
            row[f"{dataset}_cIoU"] = lookup[method]["cIoU"]
            row[f"{dataset}_AUC"] = lookup[method]["AUC"]
        metric_rows.append(row)
    rescue_rows = [
        {"dataset": dataset, **row}
        for dataset, summary in summaries.items()
        for row in summary["rescue_hurt_oracle"]
    ]
    reliability_rows = []
    for dataset, summary in summaries.items():
        for row in summary["reliability_auroc"]:
            reliability_rows.append({"dataset": dataset, **row})
    alpha_rows = [
        {"dataset": dataset, **row}
        for dataset, summary in summaries.items()
        for row in summary["alpha_sweep"]
    ]
    write_csv(args.result_root / "combined_method_metrics.csv", metric_rows)
    write_csv(args.result_root / "combined_rescue_hurt_oracle.csv", rescue_rows)
    write_csv(args.result_root / "combined_reliability_auroc.csv", reliability_rows)
    write_csv(args.result_root / "combined_alpha_sweep.csv", alpha_rows)
    combined = {"experiment": "2.2 High-Resolution Internal Slot Ownership Probe", "zero_training": True, "datasets": summaries}
    (args.result_root / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print("Method                         VGG cIoU/AUC       Flickr cIoU/AUC")
    for row in metric_rows:
        print(f"{row['method']:<30} {row['vggss_144k_cIoU']:.4f}/{row['vggss_144k_AUC']:.4f}       {row['flickr_144k_cIoU']:.4f}/{row['flickr_144k_AUC']:.4f}")
    print("\nCandidate          Rescue Hurt Net   Oracle cIoU/AUC")
    for row in rescue_rows:
        print(f"{DISPLAY[row['dataset']]} {row['candidate']:<10} {row['rescue']:>5} {row['hurt']:>4} {row['net_rescue']:>5}   {row['oracle_cIoU']:.4f}/{row['oracle_AUC']:.4f}")
    print("\nReliability                    VGG 7/HR       Flickr 7/HR")
    features = ["raw_seed_top20", "eval_seed_top20", "js_divergence", "centroid_distance"]
    lookup = {(row['dataset'], row['candidate'], row['feature']): row['AUROC'] for row in reliability_rows}
    for feature in features:
        print(
            f"{feature:<30} "
            f"{lookup[('vggss_144k','7x7',feature)]:.4f}/{lookup[('vggss_144k','HR14',feature)]:.4f}       "
            f"{lookup[('flickr_144k','7x7',feature)]:.4f}/{lookup[('flickr_144k','HR14',feature)]:.4f}"
        )


if __name__ == "__main__":
    main()
