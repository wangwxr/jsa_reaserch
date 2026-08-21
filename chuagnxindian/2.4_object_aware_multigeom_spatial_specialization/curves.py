"""Training and collapse-audit curves for Experiment 2.4."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_24_curves_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#777777",
}


def _read(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def render(history_path: Path, stem: Path, title: str) -> None:
    rows = _read(history_path)
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    epochs = [row["epoch"] for row in rows]
    value = lambda key: [row[key] for row in rows]
    fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.0), constrained_layout=True)

    for key, label, color in (
        ("loss_audio_coarse", "audio coarse", COLORS["blue"]),
        ("loss_audio_equiv", "audio equiv", COLORS["sky"]),
        ("loss_own_coarse", "own coarse", COLORS["orange"]),
        ("loss_own_equiv", "own equiv", COLORS["red"]),
        ("loss_total", "total", COLORS["pink"]),
    ):
        axes[0, 0].plot(epochs, value(key), label=label, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Four raw losses")
    axes[0, 0].legend(ncol=2)

    axes[0, 1].plot(epochs, value("own7_slot0_mass"), label="OWN7 slot0", color=COLORS["gray"])
    axes[0, 1].plot(epochs, value("own14_slot0_mass"), label="OWN14 slot0", color=COLORS["blue"])
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Slot0 mass")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, value("own7_entropy"), label="OWN7 entropy", color=COLORS["gray"])
    axes[1, 0].plot(epochs, value("own14_entropy"), label="OWN14 entropy", color=COLORS["pink"])
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Categorical ownership entropy")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, value("pooled_ownership_mae"), color=COLORS["orange"])
    axes[1, 1].set_title("MAE: AvgPool(OWN14) vs OWN7")

    for key, label, color in (
        ("aud_fine_ciou", "AUD_FINE", COLORS["blue"]),
        ("obj_fine_ciou", "OBJ_FINE", COLORS["green"]),
        ("aud_obj_ciou", "AUD_OBJ", COLORS["red"]),
        ("ogl_ciou", "OGL", COLORS["gray"]),
    ):
        axes[2, 0].plot(epochs, value(key), label=label, color=color)
    axes[2, 0].set_title("cIoU")
    axes[2, 0].legend(ncol=2)

    for key, label, color in (
        ("aud_fine_auc", "AUD_FINE", COLORS["blue"]),
        ("obj_fine_auc", "OBJ_FINE", COLORS["green"]),
        ("aud_obj_auc", "AUD_OBJ", COLORS["red"]),
        ("ogl_auc", "OGL", COLORS["gray"]),
    ):
        axes[2, 1].plot(epochs, value(key), label=label, color=color)
    axes[2, 1].set_title("AUC")
    axes[2, 1].legend(ncol=2)

    axes[3, 0].plot(epochs, value("rescue"), label="Rescue", color=COLORS["green"])
    axes[3, 0].plot(epochs, value("hurt"), label="Hurt", color=COLORS["red"])
    axes[3, 0].plot(epochs, value("net"), label="Net", color=COLORS["blue"])
    axes[3, 0].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[3, 0].set_title("Same-checkpoint Rescue / Hurt")
    axes[3, 0].legend()

    axes[3, 1].plot(epochs, value("fixed_ref_rescue"), label="Rescue", color=COLORS["green"])
    axes[3, 1].plot(epochs, value("fixed_ref_hurt"), label="Hurt", color=COLORS["red"])
    axes[3, 1].plot(epochs, value("fixed_ref_net"), label="Net", color=COLORS["blue"])
    axes[3, 1].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[3, 1].set_title("Original 1.3G fixed-reference transition")
    axes[3, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    fig.suptitle(title, fontsize=11)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
