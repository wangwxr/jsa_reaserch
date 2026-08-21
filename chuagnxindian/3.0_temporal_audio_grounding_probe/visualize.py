"""Qualitative temporal panels for Experiment 3.0."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_30_mpl")
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


def save_panel(payload: dict, output_path: Path) -> None:
    _style()
    image = payload["image"]
    row = payload["row"]
    fig, axes = plt.subplots(3, 4, figsize=(14.4, 10.0), constrained_layout=True)
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
        (axes[0, 2], "FULL_AUD", "FULL_AUD", "IoU_FULL_AUD"),
        (axes[0, 3], "CHUNK_1", "CHUNK_1", "IoU_CHUNK_1"),
        (axes[1, 0], "CHUNK_2", "CHUNK_2", "IoU_CHUNK_2"),
        (axes[1, 1], "CHUNK_3", "CHUNK_3", "IoU_CHUNK_3"),
        (axes[1, 2], "CHUNK_4", "CHUNK_4", "IoU_CHUNK_4"),
        (axes[1, 3], "TEMP_MEAN_4", "TEMP_MEAN_4", "IoU_TEMP_MEAN_4"),
        (axes[2, 0], "TEMP_GEO_4", "TEMP_GEO_4", "IoU_TEMP_GEO_4"),
        (
            axes[2, 1],
            "FULL_TEMP_GEO_4",
            "FULL_TEMP_GEO_4",
            "IoU_FULL_TEMP_GEO_4",
        ),
        (axes[2, 2], "OGL", "OGL", "IoU_OGL"),
        (axes[2, 3], "TEMP_STD", "TEMP_STD", None),
    )
    for axis, key, title, iou_key in entries:
        if iou_key is None:
            label = title
        else:
            label = f"{title}\nIoU={float(row[iou_key]):.3f}"
        _overlay(axis, image, payload[key], label)

    fig.suptitle(
        f"{payload['sample_id']} | {payload['categories']}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

