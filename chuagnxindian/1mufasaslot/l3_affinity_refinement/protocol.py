"""Local arbitrary-resolution wrapper for the unchanged JSA localization metric."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


@dataclass
class ProtocolAccumulator:
    """Accumulate the exact threshold=0.6 JSA sample IoUs in batches.

    The existing repository prints ``finalize_AP50()`` under the ``cIoU`` label.
    We intentionally preserve that behavior so old and new reported numbers are
    directly comparable. ``mean_sample_cIoU`` is additionally exposed for audit.
    """

    sample_ious: list[float] = field(default_factory=list)
    sample_names: list[str] = field(default_factory=list)

    @staticmethod
    def _resize_and_normalize(heatmap: torch.Tensor) -> np.ndarray:
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
        prediction = self._resize_and_normalize(heatmap)
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
        success_curve = [np.mean(values >= 0.05 * index) for index in range(21)]
        thresholds = [0.05 * index for index in range(21)]
        return {
            # This is the existing test_model.py output convention.
            "cIoU": float(np.mean(values >= 0.5)),
            "AUC": float(metrics.auc(thresholds, success_curve)),
            "mean_sample_cIoU": float(values.mean()),
            "num_samples": int(values.size),
        }


def metric_key(
    method: str,
    tau_aff: float | None,
    alpha: float | None,
) -> tuple[str, float | None, float | None]:
    return method, tau_aff, alpha


def format_names(names: Iterable[object]) -> list[str]:
    return [str(name) for name in names]
