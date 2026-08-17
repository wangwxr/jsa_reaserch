"""Unchanged JSA cIoU/AUC protocol for Experiment G."""

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

    def update(
        self,
        heatmap: torch.Tensor,
        gt_maps: torch.Tensor | np.ndarray,
        _names: Sequence[str],
    ) -> None:
        resized = F.interpolate(
            heatmap, size=(224, 224), mode="bicubic", align_corners=False
        )
        values = resized.detach().cpu().numpy()[:, 0]
        flat = values.reshape(values.shape[0], -1)
        minima = flat.min(axis=1)[:, None, None]
        maxima = flat.max(axis=1)[:, None, None]
        spans = maxima - minima
        prediction = values.copy()
        nonconstant = spans[:, 0, 0] != 0
        prediction[nonconstant] = (
            values[nonconstant] - minima[nonconstant]
        ) / spans[nonconstant]

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
        self.sample_ious.extend((intersection / denominator).tolist())

    def finalize(self) -> dict[str, float]:
        values = np.asarray(self.sample_ious, dtype=np.float64)
        if values.size == 0:
            raise RuntimeError("Cannot finalize an empty evaluator")
        thresholds = [0.05 * index for index in range(21)]
        curve = [np.mean(values >= threshold) for threshold in thresholds]
        return {
            "cIoU": float(np.mean(values >= 0.5)),
            "AUC": float(metrics.auc(thresholds, curve)),
            "mean_sample_cIoU": float(values.mean()),
            "num_samples": int(values.size),
        }
