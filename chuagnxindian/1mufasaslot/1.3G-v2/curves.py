"""Training curves for 1.3G-v2."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


def render_curves(history_path: Path, output_stem: Path, title: str) -> None:
    cache = Path(tempfile.gettempdir()) / f"1_3g_v2_mpl_{os.getuid()}"
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

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for field, label in (
        ("loss_aud_coarse", "AUD coarse"),
        ("loss_aud_equiv", "AUD equiv"),
        ("loss_img_coarse", "IMG coarse"),
        ("loss_total", "total"),
    ):
        axes[0, 0].plot(epochs, values(field), label=label)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Losses")
    axes[0, 0].legend()

    for field, label in (
        ("aud_fine_ciou", "AUD_FINE"),
        ("img_fine_ciou", "IMG_FINE"),
        ("iqr_fine_ciou", "IQR_FINE"),
    ):
        axes[0, 1].plot(epochs, values(field), label=label)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("cIoU")
    axes[0, 1].legend()

    for field, label in (
        ("aud_fine_auc", "AUD_FINE"),
        ("img_fine_auc", "IMG_FINE"),
        ("iqr_fine_auc", "IQR_FINE"),
    ):
        axes[1, 0].plot(epochs, values(field), label=label)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("AUC")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, values("grad_cosine"), label="grad cosine")
    axes[1, 1].plot(epochs, values("grad_norm_ratio"), label="IMG/AUD norm")
    axes[1, 1].plot(
        epochs,
        values("negative_grad_cosine_ratio"),
        label="negative ratio",
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Gradient diagnostics")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Epoch")
    fig.suptitle(title)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
