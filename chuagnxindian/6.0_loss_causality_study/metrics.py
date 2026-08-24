"""Evaluator-compatible metrics and causal spatial diagnostics for 6.0."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import rankdata
from sklearn import metrics as sklearn_metrics


EPS = 1e-8
METHODS = (
    "AUD",
    "IMG_QUERY",
    "IQR",
    "OBJ_PRIOR",
    "OGL",
    "EXTRA_IQR_OGL",
    "SHRINK_BASE",
    "EXPAND_BASE",
)


def minmax_normalize(array):
    # Preserve the evaluator's incoming float32 dtype.  Casting to float64
    # changes values by only ~1e-8, but that is enough to flip many pixels
    # lying exactly on the official threshold=0.6 boundary.
    array = np.asarray(array)
    lo = array.min()
    hi = array.max()
    if hi - lo > 0:
        return (array - lo) / (hi - lo)
    return array.copy()


def probability_normalize(array):
    array = np.asarray(array, dtype=np.float64)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= EPS:
        return np.full_like(array, 1.0 / array.size)
    return array / total


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > EPS else 0.0


def spearman(x, y):
    return pearson(rankdata(np.asarray(x).reshape(-1)), rankdata(np.asarray(y).reshape(-1)))


def js_divergence(x, y):
    p = probability_normalize(x).reshape(-1)
    q = probability_normalize(y).reshape(-1)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + EPS) / (m + EPS)))
    kl_qm = np.sum(q * np.log((q + EPS) / (m + EPS)))
    return float(0.5 * (kl_pm + kl_qm))


def top_overlap(x, y, fraction=0.1):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    count = max(1, int(math.ceil(fraction * x.size)))
    ix = set(np.argpartition(x, -count)[-count:].tolist())
    iy = set(np.argpartition(y, -count)[-count:].tolist())
    return len(ix.intersection(iy)) / count


def entropy_and_effective_area(array):
    p = probability_normalize(array).reshape(-1)
    entropy = float(-np.sum(p * np.log(p + EPS)))
    active_tokens = float(np.exp(entropy))
    return entropy, active_tokens, active_tokens / p.size


def binary_iou(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def localization_stats(prediction, gt, threshold=0.6):
    pred_mask = np.asarray(prediction) >= threshold
    gt_array = np.asarray(gt)
    # Official JSA cIoU keeps Flickr's 0.5 consensus pixels soft.
    official_intersection = np.sum(pred_mask * gt_array)
    official_false_positive = np.sum(pred_mask * (gt_array == 0))
    official_union = np.sum(gt_array) + official_false_positive
    iou = (
        float(official_intersection / official_union)
        if official_union
        else 0.0
    )

    # The requested precision/coverage/leakage audit instead uses the binary
    # union of all annotated boxes, hence gt > 0 here.
    gt_mask = gt_array > 0
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    false_positive = np.logical_and(pred_mask, ~gt_mask).sum()
    precision = float(intersection / pred_area) if pred_area else 0.0
    coverage = float(intersection / gt_area) if gt_area else 0.0
    leakage = float(false_positive / pred_area) if pred_area else 0.0
    return {
        "iou": iou,
        "precision": precision,
        "coverage": coverage,
        "outside_leakage": leakage,
        "predicted_area_ratio": float(pred_area / pred_mask.size),
        "binary": pred_mask,
    }


def summarize_iou(values):
    values = np.asarray(values, dtype=np.float64)
    success_curve = [float(np.mean(values >= 0.05 * i)) for i in range(21)]
    thresholds = [0.05 * i for i in range(21)]
    return {
        # The official JSA code prints AP@IoU=.5 under the cIoU label.
        "cIoU": float(np.mean(values >= 0.5)),
        "AUC": float(sklearn_metrics.auc(thresholds, success_curve)),
        "mean_IoU": float(values.mean()),
    }


def aggregate_scalar(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


class NormReducer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return tensor.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return tensor.unsqueeze(self.dim)


def build_object_prior(device):
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)), NormReducer(1), Unsqueeze(1)
    )
    model.requires_grad_(False)
    return model.to(device).eval()


def _expand_multiview_batch(image, spec, bboxes, names):
    if image.ndim == 3:
        image = image.unsqueeze(0)
        spec = spec.unsqueeze(0)
        bboxes = bboxes.unsqueeze(0)
    if image.ndim != 5:
        return image, spec, bboxes, list(names)
    batch, views, channels, height, width = image.shape
    image = image.reshape(batch * views, channels, height, width)
    spec = spec.reshape(batch * views, *spec.shape[2:])
    bboxes = bboxes.reshape(batch * views, *bboxes.shape[2:]).squeeze(1)
    expanded = []
    for name in names:
        expanded.extend([name] * views)
    return image, spec, bboxes, expanded


@torch.no_grad()
def evaluate_model(model, test_loader, object_model, device, alpha=0.6):
    """Run the unchanged evaluator plus native-7x7 redundancy diagnostics."""
    model.eval()
    object_model.eval()
    per_sample = []
    method_ious = defaultdict(list)
    nan_inf_count = 0

    for image, spec, bboxes, names, _labels in test_loader:
        image, spec, bboxes, names = _expand_multiview_batch(
            image, spec, bboxes, names
        )
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        img_native, aud_native = model(image, spec)
        obj_native = object_model(image)
        native_tensors = (img_native, aud_native, obj_native)
        nan_inf_count += sum(
            int((~torch.isfinite(tensor)).sum().item())
            for tensor in native_tensors
        )

        aud_224 = F.interpolate(
            aud_native, (224, 224), mode="bicubic", align_corners=False
        ).cpu().numpy()
        img_224 = F.interpolate(
            img_native, (224, 224), mode="bicubic", align_corners=False
        ).cpu().numpy()
        obj_224 = F.interpolate(
            obj_native, (224, 224), mode="bicubic", align_corners=False
        ).cpu().numpy()
        aud_7 = aud_native.cpu().numpy()
        img_7 = img_native.cpu().numpy()
        gt_batch = bboxes.cpu().numpy()

        for index, name in enumerate(names):
            aud = minmax_normalize(aud_224[index, 0])
            img = minmax_normalize(img_224[index, 0])
            obj = minmax_normalize(obj_224[index, 0])
            iqr = minmax_normalize(alpha * aud + (1.0 - alpha) * img)
            ogl = minmax_normalize(alpha * aud + (1.0 - alpha) * obj)
            extra = minmax_normalize(
                alpha * aud
                + (1.0 - alpha) * 0.5 * img
                + (1.0 - alpha) * 0.5 * obj
            )
            gt = gt_batch[index]

            aud_stats = localization_stats(aud, gt)
            img_stats = localization_stats(img, gt)
            shrink_binary = np.logical_and(
                aud_stats["binary"], img_stats["binary"]
            )
            expand = np.maximum(aud, img)
            maps = {
                "AUD": aud,
                "IMG_QUERY": img,
                "IQR": iqr,
                "OBJ_PRIOR": obj,
                "OGL": ogl,
                "EXTRA_IQR_OGL": extra,
                "SHRINK_BASE": shrink_binary.astype(np.float64),
                "EXPAND_BASE": expand,
            }

            row = {"sample_id": str(name)}
            for method, prediction in maps.items():
                stats = localization_stats(prediction, gt)
                method_ious[method].append(stats["iou"])
                prefix = method.lower()
                row[f"{prefix}_iou"] = stats["iou"]
                row[f"{prefix}_precision"] = stats["precision"]
                row[f"{prefix}_coverage"] = stats["coverage"]
                row[f"{prefix}_outside_leakage"] = stats["outside_leakage"]
                row[f"{prefix}_predicted_area_ratio"] = stats[
                    "predicted_area_ratio"
                ]

            aud_prob = probability_normalize(aud_7[index, 0])
            img_prob = probability_normalize(img_7[index, 0])
            row["aud_img_pearson"] = pearson(aud_prob, img_prob)
            row["aud_img_spearman"] = spearman(aud_prob, img_prob)
            row["aud_img_js"] = js_divergence(aud_prob, img_prob)
            row["aud_img_top10_overlap"] = top_overlap(aud_prob, img_prob)
            row["aud_img_threshold_mask_iou"] = binary_iou(
                aud_stats["binary"], img_stats["binary"]
            )
            aud_entropy, aud_active, aud_active_ratio = entropy_and_effective_area(
                aud_prob
            )
            img_entropy, img_active, img_active_ratio = entropy_and_effective_area(
                img_prob
            )
            row.update(
                {
                    "aud_attention_entropy": aud_entropy,
                    "aud_effective_active_area": aud_active,
                    "aud_effective_active_ratio": aud_active_ratio,
                    "img_attention_entropy": img_entropy,
                    "img_effective_active_area": img_active,
                    "img_effective_active_ratio": img_active_ratio,
                    "shrink_base_iou_gain_vs_aud": row["shrink_base_iou"]
                    - row["aud_iou"],
                    "expand_base_iou_gain_vs_aud": row["expand_base_iou"]
                    - row["aud_iou"],
                }
            )
            per_sample.append(row)

    metrics = {method: summarize_iou(method_ious[method]) for method in METHODS}
    diagnostics = {
        key: aggregate_scalar(per_sample, key)
        for key in (
            "aud_img_pearson",
            "aud_img_spearman",
            "aud_img_js",
            "aud_img_top10_overlap",
            "aud_img_threshold_mask_iou",
            "aud_attention_entropy",
            "aud_effective_active_area",
            "aud_effective_active_ratio",
            "img_attention_entropy",
            "img_effective_active_area",
            "img_effective_active_ratio",
            "aud_precision",
            "aud_coverage",
            "aud_outside_leakage",
            "aud_predicted_area_ratio",
            "shrink_base_iou_gain_vs_aud",
            "expand_base_iou_gain_vs_aud",
        )
    }
    diagnostics.update(
        {
            "IQR_minus_AUD_cIoU": metrics["IQR"]["cIoU"]
            - metrics["AUD"]["cIoU"],
            "IQR_minus_AUD_AUC": metrics["IQR"]["AUC"]
            - metrics["AUD"]["AUC"],
            "OGL_minus_AUD_cIoU": metrics["OGL"]["cIoU"]
            - metrics["AUD"]["cIoU"],
            "OGL_minus_AUD_AUC": metrics["OGL"]["AUC"]
            - metrics["AUD"]["AUC"],
            "nan_inf_count": nan_inf_count,
            "sample_count": len(per_sample),
            "shrink_expand_scope_note": (
                "SHRINK_BASE/EXPAND_BASE are base AUD-IMG branch-interaction "
                "diagnostics, not Experiment 5.2 PROP_F34/K34 oracles."
            ),
        }
    )
    return {"metrics": metrics, "diagnostics": diagnostics}, per_sample


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_training_curves(epoch_csv, png_path, pdf_path, title):
    import pandas as pd

    frame = pd.read_csv(epoch_csv)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    epoch = frame["epoch"]
    raw_losses = [
        "info_loss",
        "rec_img_loss",
        "rec_aud_loss",
        "div_img_loss",
        "div_aud_loss",
        "att_space_loss",
        "att_time_loss",
    ]
    for key in raw_losses:
        axes[0, 0].plot(epoch, frame[key], label=key.replace("_loss", ""))
    axes[0, 0].set_yscale("symlog", linthresh=1e-5)
    axes[0, 0].set_title("Raw loss terms")
    axes[0, 0].legend(fontsize=7, ncol=2)

    weighted = [f"weighted_{key}" for key in raw_losses[1:]]
    weighted.insert(0, "weighted_info_loss")
    for key in weighted:
        axes[0, 1].plot(epoch, frame[key], label=key.replace("weighted_", ""))
    axes[0, 1].plot(epoch, frame["total_loss"], color="black", label="total")
    axes[0, 1].set_yscale("symlog", linthresh=1e-5)
    axes[0, 1].set_title("Weighted loss contributions")
    axes[0, 1].legend(fontsize=7, ncol=2)

    for prefix, color in (("aud", "#0072B2"), ("img", "#D55E00"), ("iqr", "#009E73")):
        axes[1, 0].plot(epoch, frame[f"{prefix}_ciou"], color=color, label=f"{prefix.upper()} cIoU")
        axes[1, 0].plot(epoch, frame[f"{prefix}_auc"], color=color, linestyle="--", label=f"{prefix.upper()} AUC")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Localization metrics")
    axes[1, 0].legend(fontsize=7, ncol=2)

    axes[1, 1].plot(epoch, frame["aud_img_pearson"], label="AUD-IMG Pearson")
    axes[1, 1].plot(epoch, frame["aud_predicted_area_ratio"], label="AUD area")
    axes[1, 1].plot(epoch, frame["aud_precision"], label="precision")
    axes[1, 1].plot(epoch, frame["aud_coverage"], label="coverage")
    axes[1, 1].plot(epoch, frame["aud_outside_leakage"], label="outside leakage")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Spatial diagnostics @ threshold 0.6")
    axes[1, 1].legend(fontsize=7)

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25, linewidth=0.6)
    fig.suptitle(title)
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
