"""Deterministic qualitative panels for Experiment 5.0."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_50_mpl")
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
        (1, 0, "A20", "A20"),
        (1, 1, "I20", "I20"),
        (1, 2, "P", "P = A20 & I20"),
        (1, 3, "NA", "AUD-extra candidate"),
        (2, 0, "NI", "IMG-extra"),
        (2, 1, "STAGE2_AUD", "Stage2 AUD_FINE"),
        (2, 2, "STAGE2_IMG", "Stage2 IMG"),
        (2, 3, "OGL", "OGL"),
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
        f"P FG purity: {payload['p_fg_purity']:.4f}\n"
        f"AUD-extra BG purity: {payload['na_bg_purity']:.4f}\n"
        f"P area: {payload['p_area']}\n"
        f"AUD-extra area: {payload['na_area']}",
        va="top",
        ha="left",
        fontsize=9,
    )
    axes[3, 2].text(
        0.0,
        1.0,
        "Stage1 seeds are the primary teacher.\n"
        "Stage2 maps are diagnostic only.\n"
        "GT/OGL are analysis-only.",
        va="top",
        ha="left",
        fontsize=9,
    )
    fig.suptitle(payload["sample_id"], fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
