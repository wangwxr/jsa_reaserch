#!/usr/bin/env python3
"""Aggregate VGGSoundSS/Flickr 144k Experiment 2.1 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATASETS = ("vggss_144k", "flickr_144k")
DISPLAY = {"vggss_144k": "VGG", "flickr_144k": "Flickr"}
METHODS = (
    "AUD",
    "SLOT_L4_FIXED_SLOT0",
    "SLOT_L4_AUDIO_SELECTED",
    "AUD_FIXED_SLOT0",
    "AUD_AUDIO_SELECTED_SLOT",
    "OGL",
    "ORACLE_AUD_VS_FIXED_SLOT0",
    "ORACLE_AUD_VS_AUDIO_SELECTED_SLOT",
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


def lookup(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in rows}


def main() -> None:
    args = parse_args()
    summaries = {
        dataset: json.loads(
            (args.result_root / dataset / "summary.json").read_text(encoding="utf-8")
        )
        for dataset in DATASETS
    }
    metrics = {
        dataset: lookup(summary["method_metrics"], "method")
        for dataset, summary in summaries.items()
    }
    aurocs = {
        dataset: lookup(summary["reliability_auroc"], "feature")
        for dataset, summary in summaries.items()
    }

    metric_rows = []
    for method in METHODS:
        row: dict[str, Any] = {"method": method}
        for dataset in DATASETS:
            row[f"{dataset}_cIoU"] = metrics[dataset][method]["cIoU"]
            row[f"{dataset}_AUC"] = metrics[dataset][method]["AUC"]
        metric_rows.append(row)
    write_csv(args.result_root / "combined_method_metrics.csv", metric_rows)

    auroc_rows = []
    for feature in aurocs[DATASETS[0]]:
        auroc_rows.append(
            {
                "feature": feature,
                "score_direction": aurocs[DATASETS[0]][feature]["score_direction"],
                "vggss_144k_AUROC": aurocs["vggss_144k"][feature]["AUROC"],
                "flickr_144k_AUROC": aurocs["flickr_144k"][feature]["AUROC"],
            }
        )
    write_csv(args.result_root / "combined_reliability_auroc.csv", auroc_rows)

    selection_rows = [
        {"dataset": dataset, **summary["selection_summary"]}
        for dataset, summary in summaries.items()
    ]
    write_csv(args.result_root / "combined_selection_summary.csv", selection_rows)

    distribution_rows = []
    for dataset, summary in summaries.items():
        distribution_rows.extend(
            {"dataset": dataset, **row}
            for row in summary["feature_group_statistics"]
        )
    write_csv(args.result_root / "combined_feature_group_statistics.csv", distribution_rows)

    combined = {
        "experiment": "2.1 Audio-Guided Slot Reliability Probe",
        "zero_training": True,
        "method_table": metric_rows,
        "reliability_auroc_table": auroc_rows,
        "selection_table": selection_rows,
        "datasets": summaries,
    }
    (args.result_root / "combined_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    print("\nMethod results (cIoU / AUC)")
    print(f"{'Method':<38} {'VGG':>18} {'Flickr':>18}")
    for row in metric_rows:
        print(
            f"{row['method']:<38} "
            f"{row['vggss_144k_cIoU']:.4f}/{row['vggss_144k_AUC']:.4f}"
            f"{row['flickr_144k_cIoU']:>11.4f}/{row['flickr_144k_AUC']:.4f}"
        )

    print("\nRescue-vs-Hurt AUROC")
    print(f"{'Feature':<32} {'VGG':>10} {'Flickr':>10}")
    for row in auroc_rows:
        print(
            f"{row['feature']:<32} {row['vggss_144k_AUROC']:>10.4f} "
            f"{row['flickr_144k_AUROC']:>10.4f}"
        )

    print("\nSelection / completion counts")
    for row in selection_rows:
        print(
            f"{DISPLAY[row['dataset']]}: rescue={row['selected_rescue']}, "
            f"hurt={row['selected_hurt']}, net={row['selected_net_rescue']}, "
            f"changed={row['slot_selection_changed']} "
            f"({row['slot_selection_changed_ratio']:.2%}), "
            f"changed improves/worsens/ties={row['changed_selection_improves']}/"
            f"{row['changed_selection_worsens']}/{row['changed_selection_ties']}"
        )
    print(f"\nSaved: {args.result_root.resolve()}")


if __name__ == "__main__":
    main()
