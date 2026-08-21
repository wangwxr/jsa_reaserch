"""Publication-quality training curves for Experiment 2.3."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_23_curves_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#8C8C8C",
}


def _read(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def render(history_path: Path, stem: Path, title: str, aud_ciou: float, aud_auc: float) -> None:
    rows = _read(history_path)
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
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
    epochs = [row["epoch"] for row in rows]
    value = lambda key: [row[key] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(8.1, 8.0), constrained_layout=True)

    for key, label, color in (
        ("loss_seed", "seed", COLORS["blue"]),
        ("loss_equiv", "equiv", COLORS["orange"]),
        ("loss_visual", "visual", COLORS["green"]),
        ("loss_mass", "mass", COLORS["red"]),
        ("loss_total", "total", COLORS["pink"]),
    ):
        axes[0, 0].plot(epochs, value(key), label=label, color=color)
    axes[0, 0].set_title("Raw losses")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(ncol=2)

    for key, label, color in (
        ("weighted_seed", "1.0 seed", COLORS["blue"]),
        ("weighted_equiv", "1.0 equiv", COLORS["orange"]),
        ("weighted_visual", "0.1 visual", COLORS["green"]),
        ("weighted_mass", "0.1 mass", COLORS["red"]),
    ):
        axes[0, 1].plot(epochs, value(key), label=label, color=color)
    axes[0, 1].set_title("Weighted contributions")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(ncol=2)

    axes[1, 0].plot(epochs, value("aud_spatial_ciou"), color=COLORS["red"], label="AUD_SPATIAL")
    axes[1, 0].plot(epochs, value("spatial_slot_ciou"), color=COLORS["green"], label="SPATIAL_SLOT0")
    axes[1, 0].axhline(aud_ciou, color=COLORS["gray"], linestyle="--", label="Frozen AUD")
    axes[1, 0].set_title("cIoU")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, value("aud_spatial_auc"), color=COLORS["red"], label="AUD_SPATIAL")
    axes[1, 1].plot(epochs, value("spatial_slot_auc"), color=COLORS["green"], label="SPATIAL_SLOT0")
    axes[1, 1].axhline(aud_auc, color=COLORS["gray"], linestyle="--", label="Frozen AUD")
    axes[1, 1].set_title("AUC")
    axes[1, 1].legend()

    axes[2, 0].plot(epochs, value("rescue"), color=COLORS["green"], label="Rescue")
    axes[2, 0].plot(epochs, value("hurt"), color=COLORS["red"], label="Hurt")
    axes[2, 0].plot(epochs, value("net_rescue"), color=COLORS["blue"], label="Net")
    axes[2, 0].axhline(0, color=COLORS["gray"], linewidth=0.8)
    axes[2, 0].set_title("Rescue / Hurt")
    axes[2, 0].legend()

    axes[2, 1].plot(epochs, value("slot0_mass"), color=COLORS["blue"], label="slot0 mass")
    axes[2, 1].plot(epochs, value("slot1_mass"), color=COLORS["orange"], label="slot1 mass")
    axes[2, 1].plot(epochs, value("ownership_entropy"), color=COLORS["pink"], label="entropy")
    axes[2, 1].set_title("Slot mass / entropy")
    axes[2, 1].set_ylim(0.0, 1.05)
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=11)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

