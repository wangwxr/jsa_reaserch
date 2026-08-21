"""Deterministic qualitative panels for Experiment 5.1."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_51_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _overlay(axis, image: np.ndarray, value: np.ndarray, title: str, cmap: str = "turbo") -> None:
    axis.imshow(image)
    axis.imshow(np.ma.masked_where(value <= 0, value), cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, fontsize=7, pad=2)
    axis.axis("off")


def save_panel(payload: dict, output_path: Path) -> None:
    plt.rcParams.update({"font.size": 7, "figure.dpi": 160, "savefig.dpi": 260})
    image = payload["image"]
    fig, axes = plt.subplots(4, 4, figsize=(12.4, 12.4), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    _overlay(axes[0, 1], image, payload["GT"], "GT", "Reds")
    entries = (
        (0, 2, "STAGE1_AUD", "Stage1 AUD"),
        (0, 3, "STAGE1_IMG", "Stage1 IMG"),
        (1, 0, "P", "Agreement P"),
        (1, 1, "AUD", "AUD_FINE"),
        (1, 2, "PROP_F34", "PROP_F34"),
        (1, 3, "PROP_K34", "PROP_K34"),
        (2, 0, "IMG", "IMG_QUERY"),
        (2, 1, "OGL", "OGL"),
        (2, 2, "RAW_F34", "Raw similarity F34"),
        (2, 3, "RAW_K34", "Raw similarity K34"),
    )
    for row, column, key, title in entries:
        _overlay(axes[row, column], image, payload[key], title)
    for axis in axes[3]:
        axis.axis("off")
    axes[3, 0].text(
        0.0,
        1.0,
        f"Category: {payload['category']}\n"
        f"Group: {payload['group']}\n"
        f"Seed purity: {payload['seed_purity']:.4f}\n"
        f"Seed recall: {payload['seed_recall']:.4f}\n"
        f"Seed confidence: {payload['seed_confidence']:.4f}",
        va="top",
        ha="left",
        fontsize=9,
    )
    axes[3, 2].text(
        0.0,
        1.0,
        f"IoU AUD: {payload['iou_aud']:.4f}\n"
        f"IoU PROP_F34: {payload['iou_f34']:.4f}\n"
        f"IoU PROP_K34: {payload['iou_k34']:.4f}\n"
        f"IoU IMG: {payload['iou_img']:.4f}\n"
        f"IoU OGL: {payload['iou_ogl']:.4f}",
        va="top",
        ha="left",
        fontsize=9,
    )
    fig.suptitle(payload["sample_id"], fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
