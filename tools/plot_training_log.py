#!/usr/bin/env python3
"""Backfill epoch CSV and curves from a JSA text training log."""

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training_history import render_training_curves, write_history


METRIC_NAMES = {
    "AUD": "aud",
    "IMG_QUERY": "img_query",
    "IQR": "iqr",
    "OBJ_PRIOR": "obj_prior",
    "OGL": "ogl",
    "EXTRA_IQR_OGL": "extra_iqr_ogl",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument(
        "--output-dir",
        help="Defaults to the experiment directory containing logs/.",
    )
    return parser.parse_args()


def duration_seconds(value):
    parts = [int(part) for part in value.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Unsupported duration: {value}")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_log(path, weights):
    rows = {}
    current_epoch = None
    train_pattern = re.compile(
        r"^Train: \[(\d+)\].*"
        r"Info\s+[-+0-9.eE]+ \(\s*([-+0-9.eE]+)\).*"
        r"Con\s+[-+0-9.eE]+ \(\s*([-+0-9.eE]+)\).*"
        r"Div\s+[-+0-9.eE]+ \(\s*([-+0-9.eE]+)\).*"
        r"Att\s+[-+0-9.eE]+ \(\s*([-+0-9.eE]+)\)"
    )
    metric_pattern = re.compile(
        r"^(AUD|IMG_QUERY|IQR|OBJ_PRIOR|OGL|EXTRA_IQR_OGL)_"
        r"[^/]+/cIoU, auc(?:, best_cIoU, best_auc)?\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    )
    epoch_pattern = re.compile(r"^Epoch (\d+), Learning rate: ([-+0-9.eE]+)")
    time_pattern = re.compile(r"^Epoch (\d+)/\d+ finished in ([0-9:]+)")

    for line in Path(path).read_text(errors="replace").splitlines():
        match = train_pattern.match(line)
        if match:
            epoch = int(match.group(1)) + 1
            info, recon, div, attention = map(float, match.groups()[1:])
            row = rows.setdefault(epoch, {"epoch": epoch})
            row.update(
                {
                    "train_info_loss": info,
                    "train_recon_loss": recon,
                    "train_div_loss": div,
                    "train_attention_match_loss": attention,
                    "train_weighted_recon_loss": weights[0] * recon,
                    "train_weighted_div_loss": weights[1] * div,
                    "train_weighted_attention_match_loss": weights[2] * attention,
                    "train_total_loss": (
                        info + weights[0] * recon + weights[1] * div
                        + weights[2] * attention
                    ),
                }
            )
            continue

        match = epoch_pattern.match(line)
        if match:
            current_epoch = int(match.group(1))
            rows.setdefault(current_epoch, {"epoch": current_epoch})[
                "learning_rate"
            ] = float(match.group(2))
            continue

        match = metric_pattern.match(line)
        if match and current_epoch is not None:
            prefix = METRIC_NAMES[match.group(1)]
            row = rows.setdefault(current_epoch, {"epoch": current_epoch})
            row[f"{prefix}_ciou"] = float(match.group(2))
            row[f"{prefix}_auc"] = float(match.group(3))
            continue

        match = time_pattern.match(line)
        if match:
            epoch = int(match.group(1))
            rows.setdefault(epoch, {"epoch": epoch})["epoch_seconds"] = duration_seconds(
                match.group(2)
            )

    return [rows[epoch] for epoch in sorted(rows)]


def main():
    args = parse_args()
    log_path = Path(args.log_path).resolve()
    experiment_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else log_path.parent.parent
    )
    config_path = experiment_dir / "configs.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    weights = (
        float(config.get("lam1", 0.1)),
        float(config.get("lam2", 0.1)),
        float(config.get("lam3", 100.0)),
    )

    rows = parse_log(log_path, weights)
    if not rows:
        raise RuntimeError(f"No epoch records parsed from {log_path}")
    csv_path = experiment_dir / "epoch_metrics.csv"
    plot_path = experiment_dir / "training_curves.png"
    write_history(csv_path, rows)
    render_training_curves(csv_path, plot_path, title=experiment_dir.name)
    print(f"Parsed epochs: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(
        "Note: historical console losses were rounded to three decimals; "
        "future native CSV records retain full precision."
    )


if __name__ == "__main__":
    main()

