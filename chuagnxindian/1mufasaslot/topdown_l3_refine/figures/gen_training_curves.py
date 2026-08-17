#!/usr/bin/env python3
"""Publication-style stage-2 loss and localization curves."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


def render_training_curves(
    history_path: Path,
    output_stem: Path,
    title: str,
    aud_l4_ciou: float,
    aud_l4_auc: float,
) -> None:
    cache = Path(tempfile.gettempdir()) / f"topdown_l3_mpl_{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    epochs = np.asarray([float(row["epoch"]) for row in rows])

    def values(field: str) -> np.ndarray:
        return np.asarray([float(row[field]) for row in rows])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    loss_fields = [
        ("loss_refine", "Total"),
        ("loss_fine_match", "Fine match"),
        ("loss_coarse_aud", "Coarse audio"),
        ("loss_coarse_img", "Coarse image"),
    ]
    for index, (field, label) in enumerate(loss_fields):
        axes[0, 0].plot(epochs, values(field), color=colors[index], label=label)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Stage-2 refinement losses")
    axes[0, 0].set_ylabel("Loss (log)")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, values("aud_fine_ciou"), color=colors[1], label="AUD_FINE")
    axes[0, 1].axhline(aud_l4_ciou, color="#8C8C8C", linestyle="--", label="Frozen AUD_L4")
    axes[0, 1].set_title("AUD cIoU")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, values("aud_fine_auc"), color=colors[1], label="AUD_FINE")
    axes[0, 2].axhline(aud_l4_auc, color="#8C8C8C", linestyle="--", label="Frozen AUD_L4")
    axes[0, 2].set_title("AUD AUC")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, values("img_l4_ciou"), color="#8C8C8C", linestyle="--", label="IMG_L4 cIoU")
    axes[1, 0].plot(epochs, values("img_fine_ciou"), color=colors[0], label="IMG_FINE cIoU")
    axes[1, 0].plot(epochs, values("img_fine_auc"), color=colors[2], label="IMG_FINE AUC")
    axes[1, 0].set_title("Image-query localization")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, values("iqr_fine_ciou"), color=colors[3], label="IQR_FINE cIoU")
    axes[1, 1].plot(epochs, values("iqr_fine_auc"), color=colors[4], label="IQR_FINE AUC")
    axes[1, 1].set_title("Fine IQR")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    axes[1, 2].plot(
        epochs,
        values("aud_fine_ciou") - values("aud_l4_ciou"),
        color=colors[1],
        label="Δ cIoU",
    )
    axes[1, 2].plot(
        epochs,
        values("aud_fine_auc") - values("aud_l4_auc"),
        color=colors[0],
        label="Δ AUC",
    )
    axes[1, 2].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1, 2].set_title("AUD_FINE − AUD_L4")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
