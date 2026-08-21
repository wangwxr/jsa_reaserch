"""Deterministic qualitative panels for Experiment 4.2."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_42_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _image(axis, value: np.ndarray, title: str) -> None:
    axis.imshow(np.clip(value, 0.0, 1.0))
    axis.set_title(title, fontsize=7, pad=2)
    axis.axis("off")


def _overlay(axis, image: np.ndarray, value: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(value, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, fontsize=7, pad=2)
    axis.axis("off")


def save_panel(payload: dict, output_path: Path) -> None:
    plt.rcParams.update({"font.size": 7, "figure.dpi": 160, "savefig.dpi": 260})
    image = payload["image"]
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 12.2), constrained_layout=True)

    _image(axes[0, 0], image, "Original Image")
    _overlay(axes[0, 1], image, payload["GT"], "GT")
    _overlay(axes[0, 2], image, payload["AUD"], f"AUD IoU={payload['iou_aud']:.3f}")
    _overlay(axes[0, 3], image, payload["IMG"], f"IMG IoU={payload['iou_img']:.3f}")

    _overlay(axes[1, 0], image, payload["DISAGREEMENT"], "|AUD-IMG|")
    _overlay(axes[1, 1], image, payload["MASK_A20"], "MASK_A20")
    _overlay(axes[1, 2], image, payload["MASK_I20"], "MASK_I20")
    _overlay(axes[1, 3], image, payload["SELECTED"], f"Selected IoU={payload['iou_selected']:.3f}")

    _image(axes[2, 0], payload["KEEP_A_BLUR"], "KEEP_A blur")
    _image(axes[2, 1], payload["KEEP_I_BLUR"], "KEEP_I blur")
    _image(axes[2, 2], payload["REMOVE_A_BLUR"], "REMOVE_A blur")
    _image(axes[2, 3], payload["REMOVE_I_BLUR"], "REMOVE_I blur")

    _overlay(axes[3, 0], image, payload["OGL"], f"OGL IoU={payload['iou_ogl']:.3f}")
    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.0,
        1.0,
        f"S original  {payload['S_original']:+.4f}\n"
        f"S keep A    {payload['S_keep_A']:+.4f}\n"
        f"S keep I    {payload['S_keep_I']:+.4f}\n"
        f"S remove A  {payload['S_remove_A']:+.4f}\n"
        f"S remove I  {payload['S_remove_I']:+.4f}",
        va="top",
        ha="left",
        fontsize=8,
    )
    axes[3, 2].axis("off")
    axes[3, 2].text(
        0.0,
        1.0,
        f"CF(AUD)  {payload['CF_A']:+.4f}\n"
        f"CF(IMG)  {payload['CF_I']:+.4f}\n"
        f"Delta CF {payload['DELTA_CF']:+.4f}\n\n"
        f"True better: {payload['true_branch']}\n"
        f"Selected: {payload['selected_branch']}",
        va="top",
        ha="left",
        fontsize=8,
    )
    axes[3, 3].axis("off")
    axes[3, 3].text(
        0.0,
        1.0,
        payload["category"].replace("_", " "),
        va="top",
        ha="left",
        fontsize=9,
    )
    fig.suptitle(payload["sample_id"], fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
