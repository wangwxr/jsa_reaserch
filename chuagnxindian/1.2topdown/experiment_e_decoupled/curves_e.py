"""Curves including Experiment E map-similarity and fusion-gain diagnostics."""

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
    initial_metrics: dict,
) -> None:
    cache = Path(tempfile.gettempdir()) / f"experiment_e_mpl_{os.getuid()}"
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
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)

    for field, label in (
        ("info_loss", "InfoNCE"),
        ("recon_loss", "Reconstruction"),
        ("div_loss", "Divergence"),
        ("att_loss", "L4 attention"),
    ):
        axes[0, 0].plot(epochs, values(field), label=label)
    axes[0, 0].set_title("Original L3+L4 losses")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, values("loss_coarse_aud"), label="Coarse audio")
    axes[0, 1].plot(epochs, values("loss_coarse_img"), label="Coarse image")
    axes[0, 1].plot(epochs, values("refine_loss"), label="Fine total")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Decoupled fine losses (log)")
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, values("base_loss"), label="Base")
    axes[0, 2].plot(epochs, values("total_loss"), label="Total")
    axes[0, 2].set_title("Optimization objective")
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, values("aud_l4_ciou"), label="AUD_L4")
    axes[1, 0].plot(epochs, values("aud_fine_ciou"), label="AUD_FINE")
    axes[1, 0].axhline(
        initial_metrics["AUD_L4"]["cIoU"],
        color="gray",
        linestyle="--",
        label="Initial AUD_L4",
    )
    axes[1, 0].set_title("Audio cIoU")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, values("img_l4_ciou"), label="IMG_L4")
    axes[1, 1].plot(epochs, values("img_fine_ciou"), label="IMG_FINE")
    axes[1, 1].plot(epochs, values("iqr_fine_ciou"), label="IQR_FINE")
    axes[1, 1].set_title("Image/Fusion cIoU")
    axes[1, 1].legend()

    axes[1, 2].plot(
        epochs,
        values("aud_img_map_cosine"),
        label="AUD/IMG map cosine",
    )
    axes[1, 2].plot(
        epochs,
        values("fusion_gain_ciou"),
        label="IQR cIoU - AUD cIoU",
    )
    axes[1, 2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 2].set_title("Decoupling diagnostics")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
