"""Publication-style result figures for Experiment 2.1R."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_fixed_slot_reliability_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "gray": "#8C8C8C",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def save_auroc(rows: list[dict], stem: Path) -> None:
    _style()
    labels = [row["feature"] for row in rows]
    values = [row["AUROC"] for row in rows]
    y = np.arange(len(labels))
    colors = [COLORS["blue"] if value >= 0.5 else COLORS["gray"] for value in values]
    fig, axis = plt.subplots(figsize=(6.75, 4.1), constrained_layout=True)
    bars = axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Fixed-slot Rescue-vs-Hurt AUROC")
    axis.set_title("Experiment 2.1R: Internal Reliability")
    axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    _save(fig, stem)


def save_seed_distributions(rows: list[dict], stem: Path) -> None:
    _style()
    features = (
        ("eval_seed_top20", "Evaluator-space top20"),
        ("raw_seed_top20", "Raw ownership top20"),
        ("js_divergence", "JS divergence"),
        ("centroid_distance", "Centroid distance"),
    )
    groups = ("Rescue", "Hurt", "Neutral")
    colors = (COLORS["green"], COLORS["red"], COLORS["gray"])
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.75), constrained_layout=True)
    for axis, (feature, title) in zip(axes, features):
        values = [[row[feature] for row in rows if row["outcome"] == group] for group in groups]
        boxes = axis.boxplot(values, tick_labels=groups, showfliers=False, patch_artist=True)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.72)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    _save(fig, stem)
