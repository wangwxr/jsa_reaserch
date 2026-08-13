#!/usr/bin/env python3
"""Small, dependency-light helpers for epoch-level training history plots."""

import csv
import os
import tempfile
from pathlib import Path


HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "train_total_loss",
    "train_info_loss",
    "train_recon_loss",
    "train_div_loss",
    "train_attention_match_loss",
    "train_weighted_recon_loss",
    "train_weighted_div_loss",
    "train_weighted_attention_match_loss",
    "aud_ciou",
    "aud_auc",
    "img_query_ciou",
    "img_query_auc",
    "iqr_ciou",
    "iqr_auc",
    "obj_prior_ciou",
    "obj_prior_auc",
    "ogl_ciou",
    "ogl_auc",
    "extra_iqr_ogl_ciou",
    "extra_iqr_ogl_auc",
)


def _atomic_write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = handle.name
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})
    os.replace(temporary_path, path)


def write_history(path, rows):
    """Atomically write records ordered by one-based epoch number."""
    by_epoch = {int(row["epoch"]): dict(row) for row in rows}
    _atomic_write_rows(path, [by_epoch[epoch] for epoch in sorted(by_epoch)])


def update_history(path, record):
    """Insert or replace one epoch while preserving any existing records."""
    path = Path(path)
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if int(row["epoch"]) != int(record["epoch"])]
    rows.append(dict(record))
    write_history(path, rows)


def _read_numeric_history(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = {}
        for field in HISTORY_FIELDS:
            value = raw.get(field, "")
            row[field] = float(value) if value not in {"", None} else float("nan")
        rows.append(row)
    return sorted(rows, key=lambda row: row["epoch"])


def render_training_curves(history_path, output_path, title=None):
    """Render four compact epoch-level panels to an atomically replaced PNG."""
    rows = _read_numeric_history(history_path)
    if not rows:
        raise RuntimeError(f"No epoch records found in {history_path}")

    cache_dir = Path(tempfile.gettempdir()) / f"jsa_matplotlib_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = np.asarray([row["epoch"] for row in rows])

    def values(field):
        return np.asarray([row[field] for row in rows], dtype=float)

    def plot_available(axis, fields, logarithmic=False):
        plotted = False
        for field, label in fields:
            series = values(field)
            valid = np.isfinite(series)
            if logarithmic:
                valid &= series > 0
            if valid.any():
                axis.plot(epochs[valid], series[valid], linewidth=1.8, label=label)
                plotted = True
        if logarithmic and plotted:
            axis.set_yscale("log")
        if plotted:
            axis.legend(fontsize=8, ncol=2)
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=axis.transAxes)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Epoch")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plot_available(
        axes[0, 0],
        (
            ("train_total_loss", "Total"),
            ("train_info_loss", "InfoNCE"),
            ("train_weighted_recon_loss", "0.1 x Recon"),
            ("train_weighted_div_loss", "0.1 x Div"),
            ("train_weighted_attention_match_loss", "100 x Match"),
        ),
        logarithmic=True,
    )
    axes[0, 0].set_title("Weighted training losses (log scale)")
    axes[0, 0].set_ylabel("Loss")

    plot_available(
        axes[0, 1],
        (
            ("train_info_loss", "InfoNCE"),
            ("train_recon_loss", "Recon"),
            ("train_div_loss", "Divergence"),
            ("train_attention_match_loss", "Attention match"),
        ),
        logarithmic=True,
    )
    axes[0, 1].set_title("Raw training losses (log scale)")
    axes[0, 1].set_ylabel("Loss")

    plot_available(
        axes[1, 0],
        (
            ("aud_ciou", "AUD"),
            ("img_query_ciou", "IMG query"),
            ("iqr_ciou", "IQR"),
            ("ogl_ciou", "OGL"),
            ("extra_iqr_ogl_ciou", "Extra IQR+OGL"),
        ),
    )
    axes[1, 0].set_title("Validation cIoU (AP@0.5)")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0, 1)

    plot_available(
        axes[1, 1],
        (
            ("aud_auc", "AUD"),
            ("img_query_auc", "IMG query"),
            ("iqr_auc", "IQR"),
            ("ogl_auc", "OGL"),
            ("extra_iqr_ogl_auc", "Extra IQR+OGL"),
        ),
    )
    axes[1, 1].set_title("Validation AUC")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0, 1)

    iqr = values("iqr_ciou")
    valid_iqr = np.isfinite(iqr)
    if valid_iqr.any():
        valid_indices = np.flatnonzero(valid_iqr)
        best_index = valid_indices[np.argmax(iqr[valid_iqr])]
        best_epoch = int(epochs[best_index])
        best_score = iqr[best_index]
        for axis in axes.flat:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.45)
        axes[1, 0].scatter(
            [best_epoch], [best_score], color="black", marker="*", s=90,
            zorder=5, label=f"Best IQR: e{best_epoch}",
        )
        axes[1, 0].legend(fontsize=8, ncol=2)

    if title is None:
        title = Path(output_path).parent.name
    fig.suptitle(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = handle.name
    try:
        fig.savefig(temporary_path, format="png", dpi=150)
        os.replace(temporary_path, output_path)
    finally:
        plt.close(fig)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

