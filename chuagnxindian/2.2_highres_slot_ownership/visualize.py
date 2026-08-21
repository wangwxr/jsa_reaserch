"""Publication-style figures and qualitative panels for Experiment 2.2."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_highres_slot_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
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


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def save_method_figure(rows: list[dict], stem: Path) -> None:
    _style()
    labels = [row["method"] for row in rows]
    ciou = [row["cIoU"] for row in rows]
    auc = [row["AUC"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
    axis.bar(x - width / 2, ciou, width, label="cIoU", color=COLORS["blue"])
    axis.bar(x + width / 2, auc, width, label="AUC", color=COLORS["orange"])
    axis.set_xticks(x, labels, rotation=23, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("High-Resolution Internal Slot Ownership")
    axis.legend(ncol=2)
    _save(fig, stem)


def save_reliability_comparison(rows: list[dict], stem: Path) -> None:
    _style()
    features = [row["feature"] for row in rows if row["candidate"] == "7x7"]
    lookup = {(row["candidate"], row["feature"]): row["AUROC"] for row in rows}
    y = np.arange(len(features))
    height = 0.36
    fig, axis = plt.subplots(figsize=(6.75, 4.1), constrained_layout=True)
    axis.barh(
        y - height / 2,
        [lookup[("7x7", feature)] for feature in features],
        height,
        label="Slot L4 7×7",
        color=COLORS["gray"],
    )
    axis.barh(
        y + height / 2,
        [lookup[("HR14", feature)] for feature in features],
        height,
        label="Slot HR 14×14",
        color=COLORS["blue"],
    )
    axis.axvline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.0)
    axis.set_yticks(y, features)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Candidate-specific Rescue-vs-Hurt AUROC")
    axis.legend(loc="lower right")
    _save(fig, stem)


def save_rescue_figure(rows: list[dict], stem: Path) -> None:
    _style()
    labels = [row["candidate"] for row in rows]
    rescue = [row["rescue"] for row in rows]
    hurt = [row["hurt"] for row in rows]
    net = [row["net_rescue"] for row in rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    axis.bar(x - width, rescue, width, label="Rescue", color=COLORS["green"])
    axis.bar(x, hurt, width, label="Hurt", color=COLORS["red"])
    axis.bar(x + width, net, width, label="Net", color=COLORS["blue"])
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Samples")
    axis.set_title("7×7 vs High-Resolution Candidate")
    axis.legend(ncol=3)
    _save(fig, stem)


def _overlay(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, fontsize=8.2, pad=2.0)
    axis.axis("off")


def save_sample_panel(payload: dict, output_path: Path) -> None:
    _style()
    image = payload["image"]
    row = payload["row"]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.0), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.08, wspace=0.04, hspace=0.04)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(np.ma.masked_where(payload["GT"] <= 0, payload["GT"]), cmap="Reds", alpha=0.58)
    axes[0, 1].set_title("GT")
    axes[0, 1].axis("off")
    entries = (
        (axes[0, 2], "AUD", "AUD_FINE", "IoU_AUD"),
        (axes[0, 3], "SLOT7", "SLOT_L4 7×7", "IoU_SLOT7"),
        (axes[1, 0], "SLOTHR", "SLOT_L4 HR14", "IoU_SLOTHR"),
        (axes[1, 1], "AUD7", "AUD + Slot7", "IoU_AUD7"),
        (axes[1, 2], "AUDHR", "AUD + SlotHR", "IoU_AUDHR"),
        (axes[1, 3], "OGL", "OGL reference", "IoU_OGL"),
    )
    for axis, key, title, iou_key in entries:
        _overlay(axis, image, payload[key], f"{title}\nIoU={row[iou_key]:.3f}")
    fig.suptitle(
        f"{payload['sample_id']}  |  {payload['categories']}\n"
        f"raw20: 7={row['raw_seed_top20_7']:.3f}, HR={row['raw_seed_top20_HR14']:.3f}; "
        f"JS: 7={row['js_divergence_7']:.3f}, HR={row['js_divergence_HR14']:.3f}",
        fontsize=10.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_manifest(selected: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "categories", "selection_rule"))
        writer.writeheader()
        writer.writerows(selected)
