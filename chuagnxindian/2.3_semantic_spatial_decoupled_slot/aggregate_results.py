#!/usr/bin/env python3
"""Aggregate the two formal 144k Experiment 2.3 summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import runtime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=runtime.PROJECT_ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main():
    arguments = parse_args()
    summaries = {}
    for experiment in runtime.EXPERIMENTS:
        name = runtime.DEFAULT_NAMES[experiment]
        path = arguments.model_dir / name / "summary.json"
        summaries[experiment] = json.loads(path.read_text(encoding="utf-8"))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    rows = []
    for experiment, summary in summaries.items():
        best = summary["best_results"]
        for method, metric in best["metrics"].items():
            rows.append(
                {
                    "dataset": experiment,
                    "best_epoch": summary["best_epoch"],
                    "method": method,
                    **metric,
                }
            )
    with (arguments.output_dir / "combined_method_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for experiment, summary in summaries.items():
        best = summary["best_results"]
        metrics = best["metrics"]
        counts = best["rescue_hurt"]
        print(
            f"{experiment}: best epoch {summary['best_epoch']} | "
            f"AUD {metrics['AUD_FINE']['cIoU']:.4f}/{metrics['AUD_FINE']['AUC']:.4f} | "
            f"Slot {metrics['SPATIAL_SLOT0']['cIoU']:.4f}/{metrics['SPATIAL_SLOT0']['AUC']:.4f} | "
            f"Fusion {metrics['AUD_SPATIAL']['cIoU']:.4f}/{metrics['AUD_SPATIAL']['AUC']:.4f} | "
            f"OGL {metrics['OGL']['cIoU']:.4f}/{metrics['OGL']['AUC']:.4f} | "
            f"R/H/N {counts['new']['rescue']}/{counts['new']['hurt']}/{counts['new']['net']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

