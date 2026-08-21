"""Fixed qualitative comparison panels for Experiment 3.1."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_31_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def overlay(axis, image, heatmap, title):
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def save_panel(payload: dict, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "figure.dpi": 180,
            "savefig.dpi": 260,
        }
    )
    image = payload["image"]
    maps = payload["maps"]
    row = payload["row"]
    fig, axes = plt.subplots(3, 4, figsize=(14.2, 10.2), constrained_layout=True)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(
        np.ma.masked_where(payload["gt"] <= 0, payload["gt"]),
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        alpha=0.58,
    )
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    entries = (
        (axes[0, 2], "BASE_AUD", "Baseline AUD", "baseline_aud_iou"),
        (axes[0, 3], "NEW_AUD", "3.1 AUD", "new_aud_iou"),
        (axes[1, 0], "BASE_IMG", "Baseline IMG", "baseline_img_iou"),
        (axes[1, 1], "NEW_IMG", "3.1 IMG", "new_img_iou"),
        (axes[1, 2], "BASE_IQR", "Baseline IQR", "baseline_iqr_iou"),
        (axes[1, 3], "NEW_IQR", "3.1 IQR", "new_iqr_iou"),
        (axes[2, 0], "A3_QUERY_AUD", "A3 query diagnostic", "a3_query_iou"),
        (axes[2, 1], "OGL", "3.1 OGL", "new_ogl_iou"),
    )
    for axis, key, title, iou_key in entries:
        overlay(axis, image, maps[key], f"{title}\nIoU={row[iou_key]:.3f}")

    axes[2, 2].imshow(maps["AUD_ABS_DELTA"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[2, 2].set_title("|3.1 AUD - baseline AUD|")
    axes[2, 2].axis("off")
    axes[2, 3].imshow(maps["IQR_ABS_DELTA"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[2, 3].set_title("|3.1 IQR - baseline IQR|")
    axes[2, 3].axis("off")

    fig.suptitle(f"{payload['sample_id']} | {payload['category']}", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
