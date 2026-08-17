"""Generate reproducible qualitative refinement panels (PNG and vector PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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


def _normalized_224(heatmap: torch.Tensor) -> np.ndarray:
    resized = F.interpolate(
        heatmap[None], size=(224, 224), mode="bicubic", align_corners=False
    )[0, 0].detach().cpu().numpy()
    span = resized.max() - resized.min()
    return (resized - resized.min()) / span if span != 0 else resized


def _overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    colored = plt.get_cmap("magma")(heatmap)[..., :3]
    return np.clip(0.48 * image + 0.52 * colored, 0.0, 1.0)


def save_qualitative_panel(
    output_stem: Path,
    sample_name: str,
    image: torch.Tensor,
    gt_map: torch.Tensor,
    aud_l4: torch.Tensor,
    a4_up: torch.Tensor,
    l3_affinity: torch.Tensor,
    native_refined: torch.Tensor,
    pooled_refined: torch.Tensor,
    aud_iou: float,
    refined_iou: float,
) -> None:
    """Save a fixed sample with all requested maps and the pooled control."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    rgb = (image.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1)
    rgb = rgb.permute(1, 2, 0).numpy()
    gt = gt_map.detach().cpu().numpy()

    gt_overlay = rgb.copy()
    gt_mask = gt > 0
    gt_overlay[gt_mask] = 0.45 * gt_overlay[gt_mask] + 0.55 * np.array(
        [0.0, 0.62, 0.45]
    )

    aud = _normalized_224(aud_l4)
    upsampled = _normalized_224(a4_up)
    affinity = _normalized_224(l3_affinity)
    refined = _normalized_224(native_refined)
    pooled = _normalized_224(pooled_refined)
    native_minus_pooled = refined - pooled

    panels = [
        (rgb, "Image"),
        (gt_overlay, "GT"),
        (_overlay(rgb, aud), f"AUD L4 (7×7)\nIoU={aud_iou:.3f}"),
        (_overlay(rgb, upsampled), "A4 up (14×14)"),
        (_overlay(rgb, affinity), "L3 affinity (14×14)"),
        (_overlay(rgb, refined), f"Native refined\nIoU={refined_iou:.3f}"),
        (_overlay(rgb, pooled), "Pooled refined (7×7)"),
        (native_minus_pooled, "Native − pooled"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.7))
    for axis, (panel, title) in zip(axes.flat, panels):
        if panel.ndim == 2 and title == "Native − pooled":
            axis.imshow(panel, cmap="coolwarm", vmin=-1, vmax=1)
        else:
            axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(sample_name, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
