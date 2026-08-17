#!/usr/bin/env python3
"""Save fixed, non-cherry-picked top-down refinement panels."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / f"topdown_l3_mpl_{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _normalize_map(heatmap: torch.Tensor) -> np.ndarray:
    resized = F.interpolate(
        heatmap[None], size=(224, 224), mode="bicubic", align_corners=False
    )[0, 0].detach().cpu().numpy()
    span = resized.max() - resized.min()
    return (resized - resized.min()) / span if span != 0 else resized


def save_panel(
    output_stem: Path,
    sample_name: str,
    image: torch.Tensor,
    gt_map: torch.Tensor,
    aud_l4: torch.Tensor,
    aud_fine: torch.Tensor,
    img_fine: torch.Tensor,
) -> None:
    plt = _setup_matplotlib()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    rgb = (image.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1)
    rgb = rgb.permute(1, 2, 0).numpy()
    gt = gt_map.detach().cpu().numpy()
    gt_overlay = rgb.copy()
    mask = gt > 0
    gt_overlay[mask] = 0.45 * gt_overlay[mask] + 0.55 * np.array([0.0, 0.62, 0.45])

    aud_l4_map = _normalize_map(aud_l4)
    aud_l4_up = F.interpolate(
        aud_l4[None], size=(14, 14), mode="bilinear", align_corners=False
    )[0]
    aud_l4_up_map = _normalize_map(aud_l4_up)
    aud_fine_map = _normalize_map(aud_fine)
    img_fine_map = _normalize_map(img_fine)

    def overlay(heatmap):
        colored = plt.get_cmap("magma")(heatmap)[..., :3]
        return np.clip(0.48 * rgb + 0.52 * colored, 0, 1)

    panels = [
        (rgb, "Image"),
        (gt_overlay, "GT"),
        (overlay(aud_l4_map), "AUD_L4 (7×7)"),
        (overlay(aud_l4_up_map), "AUD_L4 up (14×14)"),
        (overlay(aud_fine_map), "AUD_FINE (14×14)"),
        (overlay(img_fine_map), "IMG_FINE (14×14)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.7))
    for axis, (panel, title) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(sample_name, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
