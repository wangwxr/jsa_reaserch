"""Deterministic qualitative panels for Experiment 4.1."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_41_mpl")
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
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.2), constrained_layout=True)

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
        (axes[0, 2], "AUD", f"AUD\nIoU={payload['iou_aud']:.3f}"),
        (axes[1, 0], "IMG", f"IMG\nIoU={payload['iou_img']:.3f}"),
        (axes[1, 1], "DISAGREEMENT", "|AUD-IMG|"),
        (axes[1, 2], "IQR", f"Fixed IQR\nIoU={payload['iou_iqr']:.3f}"),
        (axes[2, 0], "SELECTED", f"Selected\nIoU={payload['iou_selected']:.3f}"),
        (axes[2, 1], "OGL", f"OGL\nIoU={payload['iou_ogl']:.3f}"),
    )
    for axis, key, title in entries:
        _overlay(axis, image, payload[key], title)
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.98,
        f"{payload['category'].replace('_', ' ')}\n\n"
        f"Evidence: SEMANTIC_SLOT\n"
        f"E(AUD)={payload['e_aud']:.4f}\n"
        f"E(IMG)={payload['e_img']:.4f}\n"
        f"Delta={payload['delta']:+.4f}\n"
        f"Choice={'IMG' if payload['delta'] > 0 else 'AUD'}",
        va="top",
        ha="left",
        fontsize=9,
    )
    fig.suptitle(payload["sample_id"], fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

