"""Deterministic qualitative panels for Experiment 3.2."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_32_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _overlay(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, fontsize=8, pad=2)
    axis.axis("off")


def save_panel(payload: dict, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
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
        (axes[0, 2], "ORIGINAL_AUD", "ORIGINAL_AUD", "IoU_ORIGINAL_AUD"),
        (axes[0, 3], "T4_CHUNK1", "T4 M1", "IoU_T4_CHUNK1"),
        (axes[1, 0], "T4_CHUNK2", "T4 M2", "IoU_T4_CHUNK2"),
        (axes[1, 1], "T4_CHUNK3", "T4 M3", "IoU_T4_CHUNK3"),
        (axes[1, 2], "T4_CHUNK4", "T4 M4", "IoU_T4_CHUNK4"),
        (axes[1, 3], "T4_RAW_MEAN", "T4 RAW MEAN", "IoU_T4_RAW_MEAN"),
        (axes[2, 0], "T4_RAW_GEO", "T4 RAW GEO", "IoU_T4_RAW_GEO"),
        (axes[2, 1], "T4_NORM_GEO", "T4 NORM GEO", "IoU_T4_NORM_GEO"),
        (axes[2, 2], "TEMP_STD", "T4 TEMP STD", None),
        (axes[2, 3], "OGL_REFERENCE", "OGL", "IoU_OGL_REFERENCE"),
    )
    for axis, key, title, iou_key in entries:
        label = title if iou_key is None else f"{title}\nIoU={float(row[iou_key]):.3f}"
        _overlay(axis, image, payload[key], label)

    fig.suptitle(f"{payload['sample_id']} | {payload['categories']}", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

