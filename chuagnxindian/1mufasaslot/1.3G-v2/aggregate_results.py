#!/usr/bin/env python3
"""Aggregate v2 metrics and formal 144k per-sample comparisons to G/5.3."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULTS = HERE / "results"
EXPERIMENTS = {
    "VGGSS-10k": "1.3G-v2_vggss_10k",
    "Flickr-10k": "1.3G-v2_flickr_10k_frame8_center5",
    "VGGSS-144k": "1.3G-v2_vggss_144k",
    "Flickr-144k": "1.3G-v2_flickr_144k_frame8_center5",
}
FORMAL = {
    "VGGSS-144k": "vggss_144k",
    "Flickr-144k": "flickr_144k",
}
REFERENCES = {
    "VGGSS-144k": {
        "G_AUD_FINE": {"cIoU": 0.42690965490500193, "AUC": 0.42295463357890656},
        "5.3_IMG_FINE": {"cIoU": 0.42303218301667313, "AUC": 0.42598875533152387},
    },
    "Flickr-144k": {
        "G_AUD_FINE": {"cIoU": 0.812, "AUC": 0.6356},
        "5.3_IMG_FINE": {"cIoU": 0.816, "AUC": 0.6408},
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(
    new_rows: list[dict[str, str]],
    old_rows: list[dict[str, str]],
    new_field: str,
    old_field: str,
) -> dict[str, Any]:
    old = {row["sample_id"]: float(row[old_field]) for row in old_rows}
    deltas = [float(row[new_field]) - old[row["sample_id"]] for row in new_rows]
    positive = [value for value in deltas if value >= 0.01]
    negative = [value for value in deltas if value <= -0.01]
    unchanged = [value for value in deltas if -0.01 < value < 0.01]
    return {
        "improved": len(positive),
        "hurt": len(negative),
        "unchanged": len(unchanged),
        "mean_positive_gain": sum(positive) / len(positive) if positive else 0.0,
        "mean_negative_gain": sum(negative) / len(negative) if negative else 0.0,
        "total_net_IoU_gain": sum(deltas),
        "mean_net_IoU_gain": sum(deltas) / len(deltas),
        "num_samples": len(deltas),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"experiments": {}, "formal_per_sample": {}}
    table_rows = []
    for label, experiment in EXPERIMENTS.items():
        directory = PROJECT_ROOT / "checkpoints" / experiment
        metrics_path = directory / "best_test_metrics.json"
        full_path = directory / "best_full_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        full = json.loads(full_path.read_text(encoding="utf-8")) if full_path.is_file() else None
        history = read_csv(directory / "epoch_metrics.csv")
        checkpoint_name = "vggss_best.pth" if label.startswith("VGGSS") else "flickr_best.pth"
        import torch

        checkpoint = torch.load(
            directory / checkpoint_name, map_location="cpu", weights_only=False
        )
        cosines = [float(row["grad_cosine"]) for row in history]
        ratios = [float(row["grad_norm_ratio"]) for row in history]
        negatives = [float(row["negative_grad_cosine_ratio"]) for row in history]
        gradient_summary = {
            "mean_grad_cosine": statistics.fmean(cosines),
            "std_grad_cosine": statistics.pstdev(cosines),
            "mean_grad_norm_ratio_img_over_aud": statistics.fmean(ratios),
            "mean_negative_grad_cosine_ratio": statistics.fmean(negatives),
            "late_10_epoch_mean_grad_cosine": statistics.fmean(cosines[-10:]),
            "late_10_epoch_negative_ratio": statistics.fmean(negatives[-10:]),
        }
        summary["experiments"][label] = {
            "experiment": experiment,
            "best_epoch": int(checkpoint["epoch"]),
            "metrics": metrics,
            "full_metrics": full,
            "gradient_conflict": gradient_summary,
            "losses": {
                "initial": {
                    key: float(history[0][key])
                    for key in ("loss_aud_coarse", "loss_aud_equiv", "loss_img_coarse", "loss_total")
                },
                "final": {
                    key: float(history[-1][key])
                    for key in ("loss_aud_coarse", "loss_aud_equiv", "loss_img_coarse", "loss_total")
                },
            },
        }
        for method in ("AUD_L4", "AUD_FINE", "IMG_L4", "IMG_FINE", "IQR_FINE"):
            table_rows.append(
                {
                    "experiment": label,
                    "method": method,
                    "cIoU": metrics[method]["cIoU"],
                    "AUC": metrics[method]["AUC"],
                }
            )

    old_root = HERE.parent / "5.3_fine_iqr_q4_k34" / "results"
    for label, old_key in FORMAL.items():
        experiment = EXPERIMENTS[label]
        new_rows = read_csv(
            PROJECT_ROOT / "checkpoints" / experiment / "best_per_sample_iou.csv"
        )
        old_rows = read_csv(old_root / old_key / "per_sample_iou.csv")
        if [row["sample_id"] for row in new_rows] != [row["sample_id"] for row in old_rows]:
            raise RuntimeError(f"Sample order/IDs differ for {label}")
        summary["formal_per_sample"][label] = {
            "AUD_FINE_v2_vs_G": compare(
                new_rows, old_rows, "IoU_AUD_FINE", "IoU_AUD_FINE"
            ),
            "IMG_FINE_v2_vs_5.3": compare(
                new_rows, old_rows, "IoU_IMG_FINE", "IoU_IMG_FINE"
            ),
        }
        current = summary["experiments"][label]["metrics"]
        reference = REFERENCES[label]
        summary["experiments"][label]["formal_deltas"] = {
            "AUD_FINE_vs_G": {
                metric: current["AUD_FINE"][metric] - reference["G_AUD_FINE"][metric]
                for metric in ("cIoU", "AUC")
            },
            "IMG_FINE_vs_5.3": {
                metric: current["IMG_FINE"][metric] - reference["5.3_IMG_FINE"][metric]
                for metric in ("cIoU", "AUC")
            },
        }

    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "method", "cIoU", "AUC"))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved aggregate results under {RESULTS}")


if __name__ == "__main__":
    main()
