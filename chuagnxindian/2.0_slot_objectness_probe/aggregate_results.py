#!/usr/bin/env python3
"""Aggregate the two formal 144k zero-training probe results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATASETS = ("vggss_144k", "flickr_144k")
METHODS = (
    "AUD_FINE",
    "IMG_QUERY",
    "SLOT_L3",
    "SLOT_L4",
    "AUD_SLOT_L3",
    "AUD_SLOT_L4",
    "OBJ_PRIOR",
    "OGL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "results")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in summary["official_metrics"]}


def main() -> None:
    arguments = parse_args()
    summaries = {
        dataset: json.loads(
            (arguments.result_root / dataset / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for dataset in DATASETS
    }
    metrics = {dataset: metric_lookup(summary) for dataset, summary in summaries.items()}

    metric_rows = []
    for method in METHODS:
        metric_rows.append(
            {
                "method": method,
                "vggss_144k_cIoU": metrics["vggss_144k"][method]["cIoU"],
                "vggss_144k_AUC": metrics["vggss_144k"][method]["AUC"],
                "flickr_144k_cIoU": metrics["flickr_144k"][method]["cIoU"],
                "flickr_144k_AUC": metrics["flickr_144k"][method]["AUC"],
            }
        )
    write_csv(arguments.result_root / "combined_metrics.csv", metric_rows)

    alpha_rows = []
    for dataset, summary in summaries.items():
        for row in summary["alpha_sweep"]:
            alpha_rows.append({"dataset": dataset, **row})
    write_csv(arguments.result_root / "combined_alpha_sweep.csv", alpha_rows)

    rescue_rows = []
    for dataset, summary in summaries.items():
        for row in summary["rescue_hurt"]:
            rescue_rows.append({"dataset": dataset, **row})
    write_csv(arguments.result_root / "combined_rescue_hurt.csv", rescue_rows)

    complementarity_rows = []
    for dataset, summary in summaries.items():
        for row in summary["map_complementarity"]:
            complementarity_rows.append({"dataset": dataset, **row})
    write_csv(
        arguments.result_root / "combined_map_complementarity.csv",
        complementarity_rows,
    )

    combined = {
        "experiment": "2.0 Internal Slot Objectness Probe",
        "zero_training": True,
        "datasets": summaries,
        "metric_table": metric_rows,
        "alpha_sweep": alpha_rows,
        "rescue_hurt": rescue_rows,
        "map_complementarity": complementarity_rows,
    }
    (arguments.result_root / "combined_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    print("\nFixed alpha=0.6 (cIoU / AUC)")
    print(f"{'Method':<20} {'VGGSoundSS-144k':>22} {'Flickr-144k':>22}")
    for row in metric_rows:
        print(
            f"{row['method']:<20} "
            f"{row['vggss_144k_cIoU']:.4f} / {row['vggss_144k_AUC']:.4f}"
            f"{row['flickr_144k_cIoU']:>13.4f} / {row['flickr_144k_AUC']:.4f}"
        )

    print("\nRescue / hurt at IoU>=0.5")
    fields = ("rescue_count", "hurt_count", "net_rescue")
    for dataset in DATASETS:
        print(f"[{dataset}]")
        for row in summaries[dataset]["rescue_hurt"]:
            values = "  ".join(f"{field}={row[field]}" for field in fields)
            print(f"  {row['method']:<18} {values}")

    print(f"\nSaved combined results: {arguments.result_root.resolve()}")


if __name__ == "__main__":
    main()
