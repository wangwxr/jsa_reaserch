"""Experiment G training curves."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


def render_curves(
    history_path: Path,
    output_stem: Path,
    title: str,
    teacher_ciou: float,
    teacher_auc: float,
) -> None:
    cache = Path(tempfile.gettempdir()) / f"multigeom_l3_mpl_{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = np.asarray([float(row["epoch"]) for row in rows])

    def values(field: str) -> np.ndarray:
        return np.asarray([float(row[field]) for row in rows])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), constrained_layout=True)
    for field, label, color in (
        ("loss_total", "total", "#D55E00"),
        ("loss_coarse", "coarse KL", "#0072B2"),
        ("loss_equiv", "multi-geometry KL", "#009E73"),
    ):
        axes[0, 0].plot(epochs, values(field), label=label, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Spatial losses")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, values("aud_fine_ciou"), color="#D55E00", label="AUD_FINE")
    axes[0, 1].axhline(teacher_ciou, color="#777777", linestyle="--", label="AUD_L4")
    axes[0, 1].set_title("cIoU")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, values("aud_fine_auc"), color="#0072B2", label="AUD_FINE")
    axes[1, 0].axhline(teacher_auc, color="#777777", linestyle="--", label="AUD_L4")
    axes[1, 0].set_title("AUC")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, values("mean_valid_ratio"), label="valid ratio", color="#CC79A7")
    axes[1, 1].plot(epochs, values("actual_flip_ratio"), label="flip ratio", color="#E69F00")
    axes[1, 1].plot(epochs, values("mean_crop_scale"), label="crop area ratio", color="#009E73")
    axes[1, 1].set_title("Geometry statistics")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
