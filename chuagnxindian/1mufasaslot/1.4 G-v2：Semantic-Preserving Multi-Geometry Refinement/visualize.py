"""Fixed audit visualizations for multi-geometry augmentation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch


def _setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / f"multigeom_l3_mpl_{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    restored = tensor.detach().cpu().float() * std + mean
    return restored.clamp(0, 1).permute(1, 2, 0).numpy()


def _map(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().squeeze().numpy()


def save_augmentation_panel(
    path: Path,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    fine_a: torch.Tensor,
    fine_b: torch.Tensor,
    fine_b_to_a: torch.Tensor,
    valid_mask: torch.Tensor,
    records: list[dict],
    max_samples: int = 4,
) -> None:
    plt = _setup_matplotlib()
    count = min(max_samples, image_a.shape[0], len(records))
    fig, axes = plt.subplots(count, 6, figsize=(15, 2.7 * count), squeeze=False)
    columns = ("View A", "View B", "Fine A", "Fine B", "Fine B → A", "VALID_MASK")
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=10, fontweight="bold")
    for row in range(count):
        axes[row, 0].imshow(_image(image_a[row]))
        axes[row, 1].imshow(_image(image_b[row]))
        axes[row, 2].imshow(_map(fine_a[row]), cmap="jet")
        axes[row, 3].imshow(_map(fine_b[row]), cmap="jet")
        axes[row, 4].imshow(_map(fine_b_to_a[row]), cmap="jet")
        axes[row, 5].imshow(_map(valid_mask[row]), cmap="gray", vmin=0, vmax=1)
        record = records[row]
        axes[row, 0].set_ylabel(
            f"top={record['crop_top']} left={record['crop_left']}\n"
            f"h={record['crop_height']} w={record['crop_width']} "
            f"flip={record['flipped']}",
            fontsize=8,
        )
        for column in range(6):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_synthetic_geometry_panel(
    path: Path,
    original: torch.Tensor,
    transformed: torch.Tensor,
    recovered: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.6))
    values = (
        (_map(original), "Original asymmetric map", "viridis"),
        (_map(transformed), "Crop+resize+flip", "viridis"),
        (_map(recovered), "Inverse warp", "viridis"),
        (_map(valid_mask), "VALID_MASK", "gray"),
        (_map((recovered - original).abs() * valid_mask), "Valid-region error", "magma"),
    )
    for axis, (value, title, cmap) in zip(axes, values):
        axis.imshow(value, cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
