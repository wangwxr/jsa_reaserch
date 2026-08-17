"""Deterministic publication-style diagnostics for the 2.0 slot probe."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_slot_probe_mpl")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "legend.frameon": False,
        }
    )


def _overlay(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title)
    axis.axis("off")


def save_sample_panel(payload: dict, output_path: Path) -> None:
    """Save the requested ten-view panel with one fixed 2x5 layout."""
    _style()
    image = payload["image"]
    gt = payload["GT"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), constrained_layout=True)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    axes[0, 1].imshow(np.ma.masked_where(gt <= 0, gt), cmap="Reds", alpha=0.58)
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")

    entries = [
        (axes[0, 2], "AUD_FINE"),
        (axes[0, 3], "IMG_QUERY"),
        (axes[0, 4], "SLOT_L3"),
        (axes[1, 0], "SLOT_L4"),
        (axes[1, 1], "AUD_SLOT_L3"),
        (axes[1, 2], "AUD_SLOT_L4"),
        (axes[1, 3], "OBJ_PRIOR"),
        (axes[1, 4], "OGL"),
    ]
    for axis, name in entries:
        iou = payload["ious"].get(name, float("nan"))
        _overlay(axis, image, payload[name], f"{name}\nIoU={iou:.3f}")

    categories = ", ".join(payload["categories"])
    fig.suptitle(
        f"{payload['sample_id']}  |  {categories}", fontsize=12, fontweight="bold"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_selection_manifest(selected: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "categories", "selection_rule"],
        )
        writer.writeheader()
        for payload in selected:
            writer.writerow(
                {
                    "sample_id": payload["sample_id"],
                    "categories": "|".join(payload["categories"]),
                    "selection_rule": payload["selection_rule"],
                }
            )


def save_metric_figure(summary_rows: list[dict], output_stem: Path) -> None:
    _style()
    official = [
        row
        for row in summary_rows
        if row["method"]
        in (
            "AUD_FINE",
            "SLOT_L3",
            "SLOT_L4",
            "AUD_SLOT_L3",
            "AUD_SLOT_L4",
            "OBJ_PRIOR",
            "OGL",
        )
    ]
    labels = [row["method"] for row in official]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 3.2), constrained_layout=True)
    ciou = axis.bar(
        x - width / 2,
        [row["cIoU"] for row in official],
        width,
        label="cIoU",
        color=OKABE_ITO[0],
    )
    auc = axis.bar(
        x + width / 2,
        [row["AUC"] for row in official],
        width,
        label="AUC",
        color=OKABE_ITO[1],
    )
    axis.bar_label(ciou, fmt="%.3f", fontsize=7, padding=2)
    axis.bar_label(auc, fmt="%.3f", fontsize=7, padding=2)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=24, ha="right")
    axis.legend(ncol=2)
    axis.set_title("Internal Slot Objectness Probe")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def save_rescue_figure(rows: list[dict], output_stem: Path) -> None:
    _style()
    labels = [row["method"] for row in rows if row["method"] != "OGL"]
    chosen = [row for row in rows if row["method"] != "OGL"]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(5.8, 3.0), constrained_layout=True)
    for offset, field, label, color in (
        (-width, "rescue_count", "Rescue", OKABE_ITO[2]),
        (0.0, "hurt_count", "Hurt", OKABE_ITO[3]),
        (width, "net_rescue", "Net", OKABE_ITO[0]),
    ):
        axis.bar(
            x + offset,
            [row[field] for row in chosen],
            width,
            label=label,
            color=color,
        )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Samples")
    axis.set_title("AUD failure rescue / success hurt")
    axis.legend(ncol=3)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
