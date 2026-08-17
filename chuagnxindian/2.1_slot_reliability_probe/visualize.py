"""Deterministic publication-style plots for Experiment 2.1."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_slot_reliability_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#8C8C8C",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )


def _save_both(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def _overlay(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title)
    axis.axis("off")


def save_sample_panel(payload: dict, output_path: Path) -> None:
    _style()
    image = payload["image"]
    row = payload["row"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.5), constrained_layout=True)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    axes[0, 1].imshow(
        np.ma.masked_where(payload["GT"] <= 0, payload["GT"]),
        cmap="Reds",
        alpha=0.58,
    )
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    entries = (
        (axes[0, 2], "AUD_FINE", "AUD_FINE", "IoU_AUD"),
        (axes[0, 3], "SLOT0", "Slot0 ownership", "IoU_SLOT0"),
        (axes[1, 0], "SLOT1", "Slot1 ownership", "IoU_SLOT1"),
        (axes[1, 1], "SLOT_SELECTED", "Audio-selected slot", "IoU_SLOT_SELECTED"),
        (axes[1, 2], "AUD_SELECTED", "AUD + selected", "IoU_AUD_SELECTED"),
        (axes[1, 3], "OGL", "OGL reference", "IoU_OGL"),
    )
    for axis, key, label, iou_key in entries:
        _overlay(axis, image, payload[key], f"{label}\nIoU={row[iou_key]:.3f}")

    diagnostics = (
        f"s0={row['sim_slot0']:.3f}, s1={row['sim_slot1']:.3f}, "
        f"selected={int(row['selected_slot'])}, margin={row['semantic_margin']:.3f}\n"
        f"Cown={row['ownership_confidence']:.3f}, "
        f"seed20={row['seed_containment_top20']:.3f}, "
        f"JS={row['js_divergence']:.3f}, extent={row['extent_ratio']:.3f}  |  "
        f"{payload['categories']}"
    )
    fig.suptitle(f"{payload['sample_id']}\n{diagnostics}", fontsize=10.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_selection_manifest(selected: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "categories", "selection_rule"],
        )
        writer.writeheader()
        writer.writerows(selected)


def save_auroc_figure(rows: list[dict], output_stem: Path) -> None:
    _style()
    labels = [row["feature"] for row in rows]
    values = [row["AUROC"] for row in rows]
    y = np.arange(len(labels))
    colors = [COLORS["blue"] if value >= 0.5 else COLORS["gray"] for value in values]
    fig, axis = plt.subplots(figsize=(6.75, 4.2), constrained_layout=True)
    bars = axis.barh(y, values, color=colors, height=0.62)
    axis.axvline(0.5, color=COLORS["vermillion"], linestyle="--", linewidth=1.1)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Rescue-vs-Hurt AUROC")
    axis.set_title("Internal Reliability Discriminability")
    axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    _save_both(fig, output_stem)


def save_outcome_figure(summary: dict, output_stem: Path) -> None:
    _style()
    labels = ["Fixed slot0", "Audio-selected", "OGL"]
    rescue = [
        summary["fixed_rescue"],
        summary["selected_rescue"],
        summary["ogl_rescue"],
    ]
    hurt = [
        summary["fixed_hurt"],
        summary["selected_hurt"],
        summary["ogl_hurt"],
    ]
    net = [r - h for r, h in zip(rescue, hurt)]
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(6.75, 3.1), constrained_layout=True)
    axis.bar(x - width, rescue, width, label="Rescue", color=COLORS["green"])
    axis.bar(x, hurt, width, label="Hurt", color=COLORS["vermillion"])
    axis.bar(x + width, net, width, label="Net", color=COLORS["blue"])
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Samples")
    axis.set_title("Completion Outcomes at IoU ≥ 0.5")
    axis.legend(ncol=3)
    _save_both(fig, output_stem)


def save_feature_boxplots(per_sample: list[dict], output_stem: Path) -> None:
    _style()
    features = (
        ("semantic_margin", "Semantic margin"),
        ("ownership_confidence", "Ownership confidence"),
        ("seed_containment_top20", "Seed containment top20"),
        ("centroid_distance", "Centroid distance"),
        ("js_divergence", "JS divergence"),
        ("extent_ratio", "Extent ratio"),
    )
    groups = ("Rescue", "Hurt", "Neutral")
    group_colors = (COLORS["green"], COLORS["vermillion"], COLORS["gray"])
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), constrained_layout=True)
    for axis, (feature, title) in zip(axes.ravel(), features):
        values = [
            [row[feature] for row in per_sample if row["outcome"] == group]
            for group in groups
        ]
        box = axis.boxplot(values, tick_labels=groups, showfliers=False, patch_artist=True)
        for patch, color in zip(box["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
    _save_both(fig, output_stem)

