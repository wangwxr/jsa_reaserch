"""Fixed-selection qualitative panels for Experiment 2.5."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_25_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _overlay(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, pad=2.0)
    axis.axis("off")


def save_sample_panel(payload: dict, output_path: Path) -> None:
    """Save the required deterministic 10-view comparison."""
    _style()
    image = payload["image"]
    row = payload["row"]
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.8), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.03, hspace=0.03)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(
        np.ma.masked_where(payload["GT"] <= 0, payload["GT"]),
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        alpha=0.58,
    )
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    entries = (
        (axes[0, 2], "ORIGINAL_G_AUD", "Original G AUD", "IoU_ORIGINAL_G_AUD"),
        (axes[0, 3], "L3_POOLED", "Pooled L3 ownership", "IoU_SLOT_L3_POOLED"),
        (
            axes[0, 4],
            "L3_NATIVE_UPDATE",
            "Native-update L3",
            "IoU_SLOT_L3_NATIVE_UPDATE",
        ),
        (
            axes[1, 0],
            "L3_NATIVE_READOUT",
            "Native-readout L3",
            "IoU_SLOT_L3_NATIVE_READOUT",
        ),
        (axes[1, 1], "ORIGINAL_HR14", "Original HR14", "IoU_ORIGINAL_HR14"),
        (axes[1, 2], "OWN14_24", "2.4 OWN14", "IoU_OWN14_24"),
        (axes[1, 3], "CROSS_FUSION", "Cross fusion", "IoU_CROSS_FUSION"),
        (axes[1, 4], "OGL", "OGL reference", "IoU_OGL"),
    )
    for axis, key, title, iou_key in entries:
        _overlay(axis, image, payload[key], f"{title}\nIoU={float(row[iou_key]):.3f}")

    fig.suptitle(
        f"{payload['sample_id']} | fixed 2.2 selection: {payload['categories']}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
