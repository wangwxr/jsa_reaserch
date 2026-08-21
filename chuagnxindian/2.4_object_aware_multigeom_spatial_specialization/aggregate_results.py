#!/usr/bin/env python3
"""Aggregate the two formal Experiment 2.4 summaries and histories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import EXPERIMENTS, PROJECT_ROOT


HERE = Path(__file__).resolve().parent


def main() -> None:
    summaries = {}
    method_rows = []
    transition_rows = []
    collapse_rows = []
    for experiment, registry in EXPERIMENTS.items():
        model_dir = PROJECT_ROOT / "checkpoints" / registry["default_experiment"]
        summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
        summaries[experiment] = summary
        comparison = summary["comparison"]
        for method, metric in comparison["new_primary"].items():
            if not isinstance(metric, dict) or "cIoU" not in metric:
                continue
            method_rows.append(
                {
                    "dataset": experiment,
                    "best_epoch": summary["best_AUD_OBJ_epoch"],
                    "method": method,
                    **metric,
                }
            )
        transition = comparison["new_primary"]["rescue_hurt"]
        transition_rows.append({"dataset": experiment, **transition})

        with (model_dir / "epoch_metrics.csv").open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        for row in history:
            collapse_rows.append(
                {
                    "dataset": experiment,
                    "epoch": row["epoch"],
                    "own7_slot0_mass": row["own7_slot0_mass"],
                    "own14_slot0_mass": row["own14_slot0_mass"],
                    "own7_entropy": row["own7_entropy"],
                    "own14_entropy": row["own14_entropy"],
                    "pooled_ownership_mae": row["pooled_ownership_mae"],
                    "loss_audio_coarse": row["loss_audio_coarse"],
                    "loss_audio_equiv": row["loss_audio_equiv"],
                    "loss_own_coarse": row["loss_own_coarse"],
                    "loss_own_equiv": row["loss_own_equiv"],
                }
            )

    output_dir = HERE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    for filename, rows in (
        ("combined_method_metrics.csv", method_rows),
        ("combined_rescue_hurt.csv", transition_rows),
        ("combined_collapse_and_loss_curves.csv", collapse_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    for experiment, summary in summaries.items():
        new = summary["comparison"]["new_primary"]
        transition = new["rescue_hurt"]
        print(
            f"{experiment}: best AUD_OBJ epoch {summary['best_AUD_OBJ_epoch']} | "
            f"AUD {new['AUD_FINE']['cIoU']:.4f}/{new['AUD_FINE']['AUC']:.4f} | "
            f"OWN14 {new['OBJ_FINE']['cIoU']:.4f}/{new['OBJ_FINE']['AUC']:.4f} | "
            f"AUD_OBJ {new['AUD_OBJ']['cIoU']:.4f}/{new['AUD_OBJ']['AUC']:.4f} | "
            f"OGL {new['OGL']['cIoU']:.4f}/{new['OGL']['AUC']:.4f} | "
            f"R/H/N {transition['rescue']}/{transition['hurt']}/{transition['net']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
