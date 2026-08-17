"""Unchanged JSA localization protocol generalized to 7x7 or 14x14 maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


@dataclass
class ProtocolAccumulator:
    sample_ious: list[float] = field(default_factory=list)
    sample_names: list[str] = field(default_factory=list)

    @staticmethod
    def resize_and_normalize(heatmap: torch.Tensor) -> np.ndarray:
        resized = F.interpolate(
            heatmap, size=(224, 224), mode="bicubic", align_corners=False
        )
        values = resized.detach().cpu().numpy()[:, 0]
        flat = values.reshape(values.shape[0], -1)
        minima = flat.min(axis=1)[:, None, None]
        maxima = flat.max(axis=1)[:, None, None]
        spans = maxima - minima
        normalized = values.copy()
        nonconstant = spans[:, 0, 0] != 0
        normalized[nonconstant] = (
            values[nonconstant] - minima[nonconstant]
        ) / spans[nonconstant]
        return normalized

    def update(
        self,
        heatmap: torch.Tensor,
        gt_maps: torch.Tensor | np.ndarray,
        names: Sequence[str],
    ) -> np.ndarray:
        prediction = self.resize_and_normalize(heatmap)
        ground_truth = (
            gt_maps.detach().cpu().numpy()
            if isinstance(gt_maps, torch.Tensor)
            else np.asarray(gt_maps)
        )
        inferred = (prediction >= 0.6).astype(np.float64)
        intersection = (inferred * ground_truth).sum(axis=(1, 2))
        denominator = ground_truth.sum(axis=(1, 2)) + (
            inferred * (ground_truth == 0)
        ).sum(axis=(1, 2))
        sample_ious = intersection / denominator
        self.sample_ious.extend(sample_ious.tolist())
        self.sample_names.extend(str(name) for name in names)
        return sample_ious

    def update_normalized(
        self,
        prediction: np.ndarray,
        gt_maps: torch.Tensor | np.ndarray,
        names: Sequence[str],
    ) -> np.ndarray:
        """Accumulate an already resized/min-max-normalized prediction."""
        ground_truth = (
            gt_maps.detach().cpu().numpy()
            if isinstance(gt_maps, torch.Tensor)
            else np.asarray(gt_maps)
        )
        inferred = (prediction >= 0.6).astype(np.float64)
        intersection = (inferred * ground_truth).sum(axis=(1, 2))
        denominator = ground_truth.sum(axis=(1, 2)) + (
            inferred * (ground_truth == 0)
        ).sum(axis=(1, 2))
        sample_ious = intersection / denominator
        self.sample_ious.extend(sample_ious.tolist())
        self.sample_names.extend(str(name) for name in names)
        return sample_ious

    def finalize(self) -> dict[str, float]:
        values = np.asarray(self.sample_ious, dtype=np.float64)
        if values.size == 0:
            raise RuntimeError("Cannot finalize an empty evaluator")
        curve = [np.mean(values >= 0.05 * index) for index in range(21)]
        thresholds = [0.05 * index for index in range(21)]
        return {
            # Preserve test_model.py's existing printed "cIoU" convention.
            "cIoU": float(np.mean(values >= 0.5)),
            "AUC": float(metrics.auc(thresholds, curve)),
            "mean_sample_cIoU": float(values.mean()),
            "num_samples": int(values.size),
        }


def evaluate_maps(
    output: dict[str, torch.Tensor],
    gt_maps: torch.Tensor,
    names: Sequence[str],
    accumulators: dict[str, ProtocolAccumulator],
) -> dict[str, np.ndarray]:
    batch_ious = {}
    for method in ("AUD_L4", "AUD_FINE", "IMG_L4", "IMG_FINE"):
        batch_ious[method] = accumulators[method].update(
            output[method], gt_maps, names
        )
    # Match test_model.py exactly: each component is first resized and min-max
    # normalized, then mixed with alpha=0.6, then normalized once more.
    aud = ProtocolAccumulator.resize_and_normalize(output["AUD_FINE"])
    img = ProtocolAccumulator.resize_and_normalize(output["IMG_FINE"])
    mixed = 0.6 * aud + 0.4 * img
    flat = mixed.reshape(mixed.shape[0], -1)
    minima = flat.min(axis=1)[:, None, None]
    maxima = flat.max(axis=1)[:, None, None]
    spans = maxima - minima
    normalized = mixed.copy()
    nonconstant = spans[:, 0, 0] != 0
    normalized[nonconstant] = (
        mixed[nonconstant] - minima[nonconstant]
    ) / spans[nonconstant]
    batch_ious["IQR_FINE"] = accumulators["IQR_FINE"].update_normalized(
        normalized, gt_maps, names
    )
    return batch_ious
