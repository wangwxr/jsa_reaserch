"""Exact 2.0/2.2 evaluator and deterministic qualitative export for 2.3."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_23_eval_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import runtime


ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)


def _outcome(aud_iou: float, fusion_iou: float) -> str:
    if aud_iou < 0.5 and fusion_iou >= 0.5:
        return "Rescue"
    if aud_iou >= 0.5 and fusion_iou < 0.5:
        return "Hurt"
    return "Neutral"


def _oracle(aud: list[float], fusion: list[float]) -> dict[str, float]:
    return runtime.probe20.summarize_ious(
        [max(first, second) for first, second in zip(aud, fusion)]
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device: torch.device,
    object_model=None,
    include_alpha: bool = False,
) -> dict[str, Any]:
    model.eval()
    values: dict[str, list[float]] = {
        "AUD_FINE": [],
        "SPATIAL_SLOT0": [],
        "AUD_SPATIAL": [],
        "OLD_HR14": [],
        "AUD_OLD_HR14": [],
    }
    if object_model is not None:
        values.update({"OBJ_PRIOR": [], "OGL": []})
    alpha_values = {alpha: [] for alpha in ALPHAS} if include_alpha else {}
    rows: list[dict[str, Any]] = []

    for image, spec, bboxes, names, _labels in tqdm(
        loader, desc="Evaluate 2.3", dynamic_ncols=True
    ):
        image, spec, bboxes, names = runtime.probe20.flatten_eval_batch(
            image, spec, bboxes, names
        )
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = model(image, spec)
        tensors = {
            "AUD": output["AUD_FINE"],
            "NEW": output["SPATIAL_SLOT0"],
            "OLD": output["OLD_OWNERSHIP"][:, 0].reshape(-1, 1, 14, 14),
        }
        if object_model is not None:
            tensors["OBJ"] = object_model(image)
        resized = runtime.probe20.resize_maps(tensors)
        ground_truth = bboxes.numpy()

        for index, sample_id in enumerate(names):
            maps = {
                key: runtime.probe20.normalize_map(value[index])
                for key, value in resized.items()
            }
            aud_new = runtime.probe20.fuse_maps(maps["AUD"], maps["NEW"], 0.6)
            aud_old = runtime.probe20.fuse_maps(maps["AUD"], maps["OLD"], 0.6)
            sample_maps = {
                "AUD_FINE": maps["AUD"],
                "SPATIAL_SLOT0": maps["NEW"],
                "AUD_SPATIAL": aud_new,
                "OLD_HR14": maps["OLD"],
                "AUD_OLD_HR14": aud_old,
            }
            if object_model is not None:
                sample_maps["OBJ_PRIOR"] = maps["OBJ"]
                sample_maps["OGL"] = runtime.probe20.fuse_maps(
                    maps["AUD"], maps["OBJ"], 0.6
                )
            ious = {
                key: runtime.probe20.sample_iou(value, ground_truth[index])
                for key, value in sample_maps.items()
            }
            for key, value in ious.items():
                values[key].append(value)

            row = {"sample_id": str(sample_id), **{f"IoU_{k}": v for k, v in ious.items()}}
            row["outcome_old"] = _outcome(ious["AUD_FINE"], ious["AUD_OLD_HR14"])
            row["outcome_new"] = _outcome(ious["AUD_FINE"], ious["AUD_SPATIAL"])
            if include_alpha:
                for alpha in ALPHAS:
                    fused = runtime.probe20.fuse_maps(
                        maps["AUD"], maps["NEW"], alpha
                    )
                    iou = runtime.probe20.sample_iou(fused, ground_truth[index])
                    alpha_values[alpha].append(iou)
                    row[f"IoU_ALPHA_{alpha:.1f}"] = iou
            rows.append(row)

    metrics = {
        method: runtime.probe20.summarize_ious(method_values)
        for method, method_values in values.items()
    }
    rescue = sum(row["outcome_new"] == "Rescue" for row in rows)
    hurt = sum(row["outcome_new"] == "Hurt" for row in rows)
    old_rescue = sum(row["outcome_old"] == "Rescue" for row in rows)
    old_hurt = sum(row["outcome_old"] == "Hurt" for row in rows)
    result = {
        "metrics": metrics,
        "rescue_hurt": {
            "new": {"rescue": rescue, "hurt": hurt, "net": rescue - hurt},
            "old_hr14": {
                "rescue": old_rescue,
                "hurt": old_hurt,
                "net": old_rescue - old_hurt,
            },
        },
        "oracle": {
            "new": _oracle(values["AUD_FINE"], values["AUD_SPATIAL"]),
            "old_hr14": _oracle(values["AUD_FINE"], values["AUD_OLD_HR14"]),
        },
        "rows": rows,
    }
    if include_alpha:
        result["alpha_sweep"] = {
            f"{alpha:.1f}": runtime.probe20.summarize_ious(alpha_values[alpha])
            for alpha in ALPHAS
        }
    return result


def select_qualitative(rows: list[dict[str, Any]], count: int = 12) -> list[dict[str, str]]:
    buckets = {
        "OLD_HR_HURT_NEW_FIXED": [],
        "OLD_HR_RESCUE_NEW_PRESERVED": [],
        "NEW_RESCUE": [],
        "NEW_HURT": [],
        "BOTH_FAIL": [],
    }
    for row in rows:
        aud_ok = row["IoU_AUD_FINE"] >= 0.5
        old_ok = row["IoU_AUD_OLD_HR14"] >= 0.5
        new_ok = row["IoU_AUD_SPATIAL"] >= 0.5
        if aud_ok and not old_ok and new_ok:
            buckets["OLD_HR_HURT_NEW_FIXED"].append(row)
        if not aud_ok and old_ok and new_ok:
            buckets["OLD_HR_RESCUE_NEW_PRESERVED"].append(row)
        if not aud_ok and new_ok:
            buckets["NEW_RESCUE"].append(row)
        if aud_ok and not new_ok:
            buckets["NEW_HURT"].append(row)
        if not aud_ok and not old_ok and not new_ok:
            buckets["BOTH_FAIL"].append(row)

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    ordered = list(buckets)
    while len(selected) < count:
        added = False
        for category in ordered:
            while buckets[category] and buckets[category][0]["sample_id"] in used:
                buckets[category].pop(0)
            if buckets[category] and len(selected) < count:
                row = buckets[category].pop(0)
                used.add(row["sample_id"])
                categories = []
                aud_ok = row["IoU_AUD_FINE"] >= 0.5
                old_ok = row["IoU_AUD_OLD_HR14"] >= 0.5
                new_ok = row["IoU_AUD_SPATIAL"] >= 0.5
                if aud_ok and not old_ok and new_ok:
                    categories.append("OLD_HR_HURT_NEW_FIXED")
                if not aud_ok and old_ok and new_ok:
                    categories.append("OLD_HR_RESCUE_NEW_PRESERVED")
                if not aud_ok and new_ok:
                    categories.append("NEW_RESCUE")
                if aud_ok and not new_ok:
                    categories.append("NEW_HURT")
                if not aud_ok and not old_ok and not new_ok:
                    categories.append("BOTH_FAIL")
                selected.append(
                    {
                        "sample_id": row["sample_id"],
                        "categories": "|".join(categories),
                        "selection_rule": "first-in-test-order round-robin over predefined categories",
                    }
                )
                added = True
        if not added:
            break
    if len(selected) < count:
        for row in rows:
            if row["sample_id"] not in used:
                selected.append(
                    {
                        "sample_id": row["sample_id"],
                        "categories": "TEST_ORDER_FILL",
                        "selection_rule": "test-order fill after predefined categories",
                    }
                )
                used.add(row["sample_id"])
                if len(selected) == count:
                    break
    return selected


def _overlay(axis, image: np.ndarray, heatmap: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.58)
    axis.set_title(title, fontsize=8.2, pad=2.0)
    axis.axis("off")


def _save_panel(payload: dict[str, Any], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    image = payload["image"]
    row = payload["row"]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.0), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.08, wspace=0.04, hspace=0.04)
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
        (axes[0, 2], "AUD", "AUD_FINE", "IoU_AUD_FINE"),
        (axes[0, 3], "OLD", "OLD HR14 ownership", "IoU_OLD_HR14"),
        (axes[1, 0], "NEW", "NEW SPATIAL_SLOT0", "IoU_SPATIAL_SLOT0"),
        (axes[1, 1], "AUDOLD", "AUD + OLD HR14", "IoU_AUD_OLD_HR14"),
        (axes[1, 2], "AUDNEW", "AUD + NEW Spatial", "IoU_AUD_SPATIAL"),
        (axes[1, 3], "OGL", "OGL reference", "IoU_OGL"),
    )
    for axis, key, title, metric in entries:
        _overlay(axis, image, payload[key], f"{title}\nIoU={row[metric]:.3f}")
    fig.suptitle(
        f"{payload['sample_id']} | {payload['categories']}", fontsize=10.5
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


@torch.inference_mode()
def save_qualitative(
    model,
    loader,
    object_model,
    selected: list[dict[str, str]],
    rows: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
) -> None:
    selected_lookup = {row["sample_id"]: row for row in selected}
    metrics_lookup = {row["sample_id"]: row for row in rows}
    written: set[str] = set()
    model.eval()
    for image, spec, bboxes, names, _labels in tqdm(
        loader, desc="Qualitative 2.3", dynamic_ncols=True
    ):
        image, spec, bboxes, names = runtime.probe20.flatten_eval_batch(
            image, spec, bboxes, names
        )
        wanted = [index for index, name in enumerate(names) if str(name) in selected_lookup]
        if not wanted:
            continue
        image_gpu = image.to(device, non_blocking=True).float()
        spec_gpu = spec.to(device, non_blocking=True).float()
        output = model(image_gpu, spec_gpu)
        tensors = {
            "AUD": output["AUD_FINE"],
            "OLD": output["OLD_OWNERSHIP"][:, 0].reshape(-1, 1, 14, 14),
            "NEW": output["SPATIAL_SLOT0"],
            "OBJ": object_model(image_gpu),
        }
        resized = runtime.probe20.resize_maps(tensors)
        for index in wanted:
            sample_id = str(names[index])
            maps = {
                key: runtime.probe20.normalize_map(value[index])
                for key, value in resized.items()
            }
            payload = {
                "sample_id": sample_id,
                "categories": selected_lookup[sample_id]["categories"],
                "row": metrics_lookup[sample_id],
                "image": runtime.probe20.inverse_normalize(image[index]).permute(1, 2, 0).numpy(),
                "GT": bboxes[index].numpy(),
                "AUD": maps["AUD"],
                "OLD": maps["OLD"],
                "NEW": maps["NEW"],
                "AUDOLD": runtime.probe20.fuse_maps(maps["AUD"], maps["OLD"], 0.6),
                "AUDNEW": runtime.probe20.fuse_maps(maps["AUD"], maps["NEW"], 0.6),
                "OGL": runtime.probe20.fuse_maps(maps["AUD"], maps["OBJ"], 0.6),
            }
            rank = next(i for i, row in enumerate(selected, 1) if row["sample_id"] == sample_id)
            _save_panel(payload, output_dir / f"{rank:02d}_{sample_id}.png")
            written.add(sample_id)
        if len(written) == len(selected):
            break
    if written != set(selected_lookup):
        raise RuntimeError(f"Missing qualitative IDs: {set(selected_lookup) - written}")
    with (output_dir / "selection_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "categories", "selection_rule")
        )
        writer.writeheader()
        writer.writerows(selected)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
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
    serializable = {key: value for key, value in result.items() if key != "rows"}
    (output_dir / "detailed_results.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "per_sample_metrics.csv", result["rows"])
    write_csv(
        output_dir / "method_metrics.csv",
        [{"method": key, **value} for key, value in result["metrics"].items()],
    )
    if "alpha_sweep" in result:
        write_csv(
            output_dir / "alpha_sweep.csv",
            [
                {"alpha_aud": alpha, **metric}
                for alpha, metric in result["alpha_sweep"].items()
            ],
        )

