"""Deterministic qualitative panels for Experiment 4.0."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_40_mpl")
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
    plt.rcParams.update({"font.size": 8, "figure.dpi": 180, "savefig.dpi": 300})
    image = payload["image"]
    row = payload["row"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), constrained_layout=True)

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
        (axes[0, 2], "AUD", "AUD_FINE", "IoU_AUD"),
        (axes[0, 3], "IMG", "IMG_QUERY", "IoU_IMG"),
        (axes[0, 4], "IQR", "IQR", "IoU_IQR"),
        (axes[1, 0], "OBJ", "OBJ_PRIOR", "IoU_OBJ"),
        (axes[1, 1], "OGL", "OGL", "IoU_OGL"),
        (axes[1, 2], "AUD_IMG_DIFF", "|AUD-IMG|", None),
        (axes[1, 3], "AUD_OBJ_DIFF", "|AUD-OBJ|", None),
    )
    for axis, key, title, iou_key in entries:
        label = title if iou_key is None else f"{title}\nIoU={float(row[iou_key]):.3f}"
        _overlay(axis, image, payload[key], label)

    axes[1, 4].axis("off")
    axes[1, 4].text(
        0.02,
        0.98,
        payload["category"].replace("_", " "),
        va="top",
        ha="left",
        fontsize=10,
        wrap=True,
    )
    fig.suptitle(payload["sample_id"], fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

