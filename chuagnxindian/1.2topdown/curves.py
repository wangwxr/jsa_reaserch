"""Training curves for Experiment D."""

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
    cache = Path(tempfile.gettempdir()) / f"joint_topdown_mpl_{os.getuid()}"
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
        ("att_loss", "Attention"),
    ):
        axes[0, 0].plot(epochs, values(field), label=label)
    axes[0, 0].set_title("Original L3+L4 losses")
    axes[0, 0].legend()

    for field, label in (
        ("refine_loss", "Refine total"),
        ("loss_fine_match", "Fine match"),
        ("loss_coarse_aud", "Coarse audio"),
        ("loss_coarse_img", "Coarse image"),
    ):
        axes[0, 1].plot(epochs, values(field), label=label)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Refinement losses (log)")
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, values("base_loss"), label="Base")
    axes[0, 2].plot(epochs, values("total_loss"), label="Total")
    axes[0, 2].set_title("Optimization objective")
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, values("aud_l4_ciou"), label="AUD_L4")
    axes[1, 0].plot(epochs, values("aud_fine_ciou"), label="AUD_FINE")
    axes[1, 0].axhline(
        initial_metrics["AUD_L4"]["cIoU"], color="gray", linestyle="--",
        label="Initial AUD_L4",
    )
    axes[1, 0].set_title("AUD cIoU")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, values("aud_l4_auc"), label="AUD_L4")
    axes[1, 1].plot(epochs, values("aud_fine_auc"), label="AUD_FINE")
    axes[1, 1].plot(epochs, values("iqr_fine_auc"), label="IQR_FINE")
    axes[1, 1].axhline(
        initial_metrics["AUD_L4"]["AUC"], color="gray", linestyle="--",
        label="Initial AUD_L4",
    )
    axes[1, 1].set_title("AUD/IQR AUC")
    axes[1, 1].legend()

    axes[1, 2].plot(epochs, values("img_l4_ciou"), label="IMG_L4 cIoU")
    axes[1, 2].plot(epochs, values("img_fine_ciou"), label="IMG_FINE cIoU")
    axes[1, 2].plot(epochs, values("iqr_fine_ciou"), label="IQR_FINE cIoU")
    axes[1, 2].set_title("Image/Fusion cIoU")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
