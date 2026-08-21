"""Unchanged 2.2 localization protocol and deterministic 2.4 diagnostics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_24_eval_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from common import PROJECT_ROOT, flatten_eval_batch
from dataset import inverse_normalize
import test_model


REFERENCE_ROOT = (
    PROJECT_ROOT / "chuagnxindian" / "2.2_highres_slot_ownership" / "results"
)
METHODS = ("AUD_FINE", "OBJ_FINE", "AUD_OBJ", "OBJ_PRIOR", "OGL")


def object_prior_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def normalize_map(value: np.ndarray) -> np.ndarray:
    """Exact equivalent of the root evaluator's per-sample min-max normalization."""
    value = np.asarray(value)
    minimum = value.min()
    maximum = value.max()
    if maximum - minimum != 0:
        return (value - minimum) / (maximum - minimum)
    return value.copy()


def fuse_maps(first: np.ndarray, second: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    return normalize_map(alpha * first + (1.0 - alpha) * second)


def sample_iou(prediction: np.ndarray, gt_map: np.ndarray) -> float:
    inferred = prediction >= 0.6
    intersection = np.sum(inferred * gt_map)
    denominator = np.sum(gt_map) + np.sum(inferred * (gt_map == 0))
    return float(intersection / denominator)


def summarize_ious(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    thresholds = np.arange(21, dtype=np.float64) * 0.05
    curve = [float(np.mean(array >= threshold)) for threshold in thresholds]
    return {
        "cIoU": float(np.mean(array >= 0.5)),
        "AUC": float(sklearn_metrics.auc(thresholds, curve)),
        "mean_sample_cIoU": float(array.mean()),
        "num_samples": int(array.size),
    }


def resize_maps(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: F.interpolate(
            tensor, size=(224, 224), mode="bicubic", align_corners=False
        )
        .detach()
        .cpu()
        .numpy()[:, 0]
        for name, tensor in tensors.items()
    }


def outcome(reference_iou: float, candidate_iou: float) -> str:
    if reference_iou < 0.5 and candidate_iou >= 0.5:
        return "Rescue"
    if reference_iou >= 0.5 and candidate_iou < 0.5:
        return "Hurt"
    return "Neutral"


def load_reference_rows(experiment: str) -> dict[str, dict[str, str]]:
    path = REFERENCE_ROOT / experiment / "per_sample_metrics.csv"
    with path.open(encoding="utf-8") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"Empty Experiment 2.2 reference: {path}")
    return rows


def load_reference_summary(experiment: str) -> dict[str, Any]:
    payload = json.loads((REFERENCE_ROOT / "combined_summary.json").read_text(encoding="utf-8"))
    return payload["datasets"][experiment]


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device: torch.device,
    object_model: nn.Module,
    object_cache: dict[str, torch.Tensor],
    reference_rows: dict[str, dict[str, str]],
    description: str = "Evaluate 2.4",
) -> dict[str, Any]:
    model.eval()
    values = {method: [] for method in METHODS}
    rows: list[dict[str, Any]] = []

    for image, spec, bboxes, names, _labels in tqdm(
        loader, desc=description, dynamic_ncols=True
    ):
        image, spec, bboxes, names = flatten_eval_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = model(image, spec)

        if all(name in object_cache for name in names):
            object_prior = torch.stack([object_cache[name] for name in names]).to(
                device, non_blocking=True
            )
        else:
            object_prior = object_model(image)
            for index, name in enumerate(names):
                object_cache[name] = object_prior[index].detach().cpu()

        resized = resize_maps(
            {
                "AUD_FINE": output["AUD_FINE"],
                "OBJ_FINE": output["OBJ_FINE"],
                "OBJ_PRIOR": object_prior,
            }
        )
        ground_truth = bboxes.numpy()
        for index, sample_id in enumerate(names):
            if sample_id not in reference_rows:
                raise RuntimeError(f"Missing Experiment 2.2 reference sample: {sample_id}")
            maps = {
                key: normalize_map(value[index]) for key, value in resized.items()
            }
            maps["AUD_OBJ"] = fuse_maps(maps["AUD_FINE"], maps["OBJ_FINE"], 0.6)
            maps["OGL"] = fuse_maps(maps["AUD_FINE"], maps["OBJ_PRIOR"], 0.6)
            ious = {
                key: sample_iou(maps[key], ground_truth[index]) for key in METHODS
            }
            for key, value in ious.items():
                values[key].append(value)

            reference = reference_rows[sample_id]
            original_aud = float(reference["IoU_AUD"])
            old_hr14 = float(reference["IoU_SLOTHR"])
            old_fusion = float(reference["IoU_AUDHR"])
            rows.append(
                {
                    "sample_id": sample_id,
                    **{f"IoU_{key}": value for key, value in ious.items()},
                    "IoU_ORIGINAL_1_3G_AUD": original_aud,
                    "IoU_ORIGINAL_2_2_HR14": old_hr14,
                    "IoU_ORIGINAL_2_2_AUD_HR14": old_fusion,
                    "outcome_same_checkpoint": outcome(
                        ious["AUD_FINE"], ious["AUD_OBJ"]
                    ),
                    "outcome_original_1_3g_reference": outcome(
                        original_aud, ious["AUD_OBJ"]
                    ),
                    "outcome_original_2_2_hr14": reference["outcome_HR14"],
                }
            )

    if len(rows) != len(reference_rows):
        raise RuntimeError(
            f"Evaluation/reference sample count mismatch: {len(rows)} vs {len(reference_rows)}"
        )
    metrics = {method: summarize_ious(values[method]) for method in METHODS}
    same_rescue = sum(row["outcome_same_checkpoint"] == "Rescue" for row in rows)
    same_hurt = sum(row["outcome_same_checkpoint"] == "Hurt" for row in rows)
    fixed_rescue = sum(
        row["outcome_original_1_3g_reference"] == "Rescue" for row in rows
    )
    fixed_hurt = sum(
        row["outcome_original_1_3g_reference"] == "Hurt" for row in rows
    )
    oracle_values = [
        max(aud, fusion)
        for aud, fusion in zip(values["AUD_FINE"], values["AUD_OBJ"])
    ]
    return {
        "metrics": metrics,
        "rescue_hurt": {
            "same_checkpoint": {
                "rescue": same_rescue,
                "hurt": same_hurt,
                "net": same_rescue - same_hurt,
            },
            "original_1_3g_reference": {
                "rescue": fixed_rescue,
                "hurt": fixed_hurt,
                "net": fixed_rescue - fixed_hurt,
            },
        },
        "oracle": summarize_ious(oracle_values),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_detailed_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if key != "rows"}
    (output_dir / "detailed_results.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "per_sample_metrics.csv", result["rows"])
    write_csv(
        output_dir / "method_metrics.csv",
        [{"method": key, **value} for key, value in result["metrics"].items()],
    )


def select_qualitative(experiment: str, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    source = REFERENCE_ROOT / experiment / "qualitative" / "selection_manifest.csv"
    with source.open(encoding="utf-8") as handle:
        selected = [dict(row) for row in csv.DictReader(handle)]
    selected_ids = {row["sample_id"] for row in selected}
    for category in ("Rescue", "Hurt"):
        added = 0
        for row in rows:
            if row["outcome_same_checkpoint"] != category:
                continue
            sample_id = row["sample_id"]
            if sample_id in selected_ids:
                continue
            selected.append(
                {
                    "sample_id": sample_id,
                    "categories": f"NEW_{category.upper()}",
                    "selection_rule": "first-in-test-order same-checkpoint outcome",
                }
            )
            selected_ids.add(sample_id)
            added += 1
            if added == 2:
                break
    return selected


@torch.inference_mode()
def save_qualitative(
    model,
    original_model,
    loader,
    object_model: nn.Module,
    object_cache: dict[str, torch.Tensor],
    selected: list[dict[str, str]],
    detailed_rows: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_lookup = {row["sample_id"]: row for row in selected}
    metric_lookup = {row["sample_id"]: row for row in detailed_rows}
    written: set[str] = set()

    for image, spec, bboxes, names, _labels in loader:
        image, spec, bboxes, names = flatten_eval_batch(image, spec, bboxes, names)
        indices = [index for index, name in enumerate(names) if name in selected_lookup]
        if not indices:
            continue
        selected_image = image[indices].to(device, non_blocking=True).float()
        selected_spec = spec[indices].to(device, non_blocking=True).float()
        selected_names = [names[index] for index in indices]
        selected_gt = bboxes[indices].numpy()
        new_output = model(selected_image, selected_spec)
        original_output = original_model(selected_image, selected_spec)
        if all(name in object_cache for name in selected_names):
            object_prior = torch.stack(
                [object_cache[name] for name in selected_names]
            ).to(device, non_blocking=True)
        else:
            object_prior = object_model(selected_image)
        resized = resize_maps(
            {
                "ORIGINAL_AUD": original_output["AUD_FINE"],
                "ORIGINAL_HR14": original_output["OBJ_FINE"],
                "NEW_AUD": new_output["AUD_FINE"],
                "NEW_OWN14": new_output["OBJ_FINE"],
                "OWN7": new_output["OWN7_SLOT0"],
                "OBJ_PRIOR": object_prior,
            }
        )

        for local_index, sample_id in enumerate(selected_names):
            maps = {
                key: normalize_map(value[local_index]) for key, value in resized.items()
            }
            maps["NEW_AUD_OBJ"] = fuse_maps(maps["NEW_AUD"], maps["NEW_OWN14"], 0.6)
            maps["OGL"] = fuse_maps(maps["NEW_AUD"], maps["OBJ_PRIOR"], 0.6)
            rgb = inverse_normalize(selected_image[local_index].cpu()).permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0.0, 1.0)
            panels = (
                ("Image", rgb, None),
                ("GT", selected_gt[local_index], "gray"),
                ("original 1.3G AUD_FINE", maps["ORIGINAL_AUD"], "jet"),
                ("original 2.2 HR14 ownership", maps["ORIGINAL_HR14"], "jet"),
                ("NEW AUD_FINE", maps["NEW_AUD"], "jet"),
                ("NEW OWN14", maps["NEW_OWN14"], "jet"),
                ("NEW AUD+OWN14", maps["NEW_AUD_OBJ"], "jet"),
                ("Stage1 OWN7 upsample", maps["OWN7"], "jet"),
                ("OGL", maps["OGL"], "jet"),
            )
            fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.2), constrained_layout=True)
            for axis, (title, value, cmap) in zip(axes.flat, panels):
                axis.imshow(value, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            metrics = metric_lookup[sample_id]
            fig.suptitle(
                f"{sample_id} | {selected_lookup[sample_id]['categories']} | "
                f"NEW outcome={metrics['outcome_same_checkpoint']}",
                fontsize=10,
            )
            order = next(
                index for index, row in enumerate(selected, start=1) if row["sample_id"] == sample_id
            )
            fig.savefig(output_dir / f"{order:02d}_{sample_id}.png", dpi=220)
            plt.close(fig)
            written.add(sample_id)
        if len(written) == len(selected_lookup):
            break

    if written != set(selected_lookup):
        raise RuntimeError(f"Missing qualitative samples: {set(selected_lookup) - written}")
    write_csv(output_dir / "selection_manifest.csv", selected)
