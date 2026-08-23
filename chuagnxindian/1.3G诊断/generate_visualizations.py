#!/usr/bin/env python3
"""Read-only qualitative diagnosis for the formal Experiment 1.3G checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
G_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot" / "1.3G-multigeom_equivariant_l3_refine"
for import_path in (G_ROOT, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from common import (  # noqa: E402
    EXPERIMENTS,
    build_model,
    load_base_config,
    setup_seed,
)
from dataset import get_test_dataset  # noqa: E402
import test_model  # noqa: E402


FORMAL = {
    "vggss": {
        "registry": "vggss_144k",
        "experiment": "1.3G-multigeom_equivariant_l3_refine_vggss_144k_best",
        "checkpoint": "vggss_best.pth",
    },
    "flickr": {
        "registry": "flickr_144k",
        "experiment": "1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5_best",
        "checkpoint": "flickr_best.pth",
    },
}

METHODS = (
    "AUD",
    "IMG_QUERY",
    "IQR",
    "OBJ_PRIOR",
    "OGL",
    "EXTRA_IQR_OGL",
)

DISPLAY_TITLES = {
    "AUD": "AUD\nQa → K34",
    "IMG_QUERY": "IMG branch\nQ4 → K4",
    "IQR": "IQR\n0.6 AUD + 0.4 IMG",
    "OBJ_PRIOR": "OGL branch\nOBJ_PRIOR",
    "OGL": "AUD + OGL branch\nOfficial OGL",
    "EXTRA_IQR_OGL": "AUD + IMG + OGL\nEXTRA_IQR_OGL",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


class DiagnosticReadout(nn.Module):
    """Expose the unchanged official IMG_L4 and Experiment G AUD_FINE maps."""

    def __init__(self, refinement: nn.Module):
        super().__init__()
        self.refinement = refinement

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_l4, _aud_l4 = self.refinement.teacher.forward_eval(image, audio)
        aud_fine = self.refinement(image, audio)["AUD_FINE"]
        return img_l4, aud_fine


def object_prior_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def normalize_map(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    minimum = float(value.min())
    maximum = float(value.max())
    if maximum > minimum:
        return (value - minimum) / (maximum - minimum)
    return value.copy()


def resize_and_normalize(heatmap: torch.Tensor) -> np.ndarray:
    resized = F.interpolate(
        heatmap, size=(224, 224), mode="bicubic", align_corners=False
    )
    values = resized.detach().cpu().numpy()[:, 0]
    return np.stack([normalize_map(value) for value in values])


def denormalize_images(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[None, :, None, None]
    rgb = (image * std + mean).clamp(0, 1)
    return rgb.permute(0, 2, 3, 1).detach().cpu().numpy()


def sample_iou(prediction: np.ndarray, gt: np.ndarray, threshold: float = 0.6) -> float:
    infer = prediction >= threshold
    numerator = np.sum(infer * gt)
    denominator = np.sum(gt) + np.sum(infer * (gt == 0))
    return float(numerator / denominator) if denominator > 0 else 0.0


def gt_overlay(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    result = image.copy()
    mask = gt > 0
    red = np.zeros_like(result)
    red[..., 0] = 1.0
    alpha = np.clip(gt[..., None], 0, 1) * 0.58
    result = result * (1.0 - alpha) + red * alpha
    # White boundary keeps overlapping/consensus Flickr annotations visible.
    boundary = mask ^ (
        F.max_pool2d(
            torch.from_numpy(mask.astype(np.float32))[None, None], 3, 1, 1
        )[0, 0].numpy()
        > 0
    )
    result[boundary] = 1.0
    return np.clip(result, 0, 1)


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    color = plt.get_cmap("magma")(np.clip(heatmap, 0, 1))[..., :3]
    return np.clip(0.42 * image + 0.58 * color, 0, 1)


def save_individual_maps(
    output_dir: Path,
    sample_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    maps: dict[str, np.ndarray],
) -> None:
    sample_dir = output_dir / "individual" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(sample_dir / "00_image.png", image)
    plt.imsave(sample_dir / "01_gt.png", gt_overlay(image, gt))
    for index, method in enumerate(METHODS, start=2):
        plt.imsave(
            sample_dir / f"{index:02d}_{method.lower()}.png",
            heatmap_overlay(image, maps[method]),
        )
    np.savez_compressed(sample_dir / "raw_maps.npz", GT=gt, **maps)


def save_sample_panel(
    output_path: Path,
    dataset_name: str,
    sample_id: str,
    label: str,
    image: np.ndarray,
    gt: np.ndarray,
    maps: dict[str, np.ndarray],
    ious: dict[str, float],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.1), constrained_layout=True)
    panels = [
        ("Image", image),
        ("GT localization", gt_overlay(image, gt)),
        *[
            (f"{DISPLAY_TITLES[method]}\nIoU@0.6={ious[method]:.3f}", heatmap_overlay(image, maps[method]))
            for method in METHODS
        ],
    ]
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title, fontweight="semibold")
        axis.axis("off")
    fig.suptitle(
        f"1.3G diagnosis · {dataset_name} · {sample_id} · label: {label}",
        fontsize=12,
        fontweight="bold",
    )
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="magma")
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=axes[:, 2:].ravel().tolist(),
        fraction=0.018,
        pad=0.015,
        label="Evaluator-normalized response",
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_contact_sheet(output_path: Path, rows: list[dict[str, Any]], dataset_name: str) -> None:
    columns = ("Image", "GT", *METHODS)
    fig, axes = plt.subplots(
        len(rows), len(columns), figsize=(17.2, 2.18 * len(rows)), squeeze=False
    )
    for row_index, row in enumerate(rows):
        panels = [
            row["image"],
            gt_overlay(row["image"], row["gt"]),
            *[heatmap_overlay(row["image"], row["maps"][method]) for method in METHODS],
        ]
        for column_index, (axis, panel) in enumerate(zip(axes[row_index], panels)):
            axis.imshow(panel)
            axis.axis("off")
            if row_index == 0:
                title = columns[column_index]
                if title in DISPLAY_TITLES:
                    title = DISPLAY_TITLES[title]
                axis.set_title(title, fontsize=9, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(
                    f"{row['sample_id']}\n{row['label']}", fontsize=7, rotation=0,
                    ha="right", va="center", labelpad=58,
                )
    fig.suptitle(
        f"Formal 1.3G qualitative diagnosis — {dataset_name}\n"
        "fixed evenly-spaced test samples; identical evaluator normalization",
        fontsize=13,
        fontweight="bold",
        y=0.998,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.968, bottom=0.01, wspace=0.025, hspace=0.10)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def fixed_indices(length: int, count: int) -> list[int]:
    count = min(count, length)
    return np.linspace(0, length - 1, num=count, dtype=int).tolist()


@torch.inference_mode()
def run_dataset(dataset_name: str, gpu: int, count: int) -> dict[str, Any]:
    formal = FORMAL[dataset_name]
    registry = EXPERIMENTS[formal["registry"]]
    checkpoint_path = PROJECT_ROOT / "checkpoints" / formal["experiment"] / formal["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_before = checkpoint_state(checkpoint_path)

    device = torch.device("cuda", gpu)
    torch.cuda.set_device(gpu)
    config = load_base_config(registry)
    config.gpu = gpu
    config.testset = registry["dataset"]
    config.workers = 0
    config.alpha = 0.6
    setup_seed(config.seed)

    refinement, base_checkpoint = build_model(config, registry, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "multi_geometry_equivariant_l3_refine":
        raise RuntimeError(f"Unexpected architecture: {checkpoint.get('architecture')}")
    refinement.student.proj3_spatial.load_state_dict(
        checkpoint["proj3_spatial_state_dict"], strict=True
    )
    refinement.student.adapter.load_state_dict(
        checkpoint["topdown_adapter_state_dict"], strict=True
    )
    refinement.requires_grad_(False).eval()
    readout = DiagnosticReadout(refinement).to(device).requires_grad_(False).eval()
    object_model = object_prior_model().to(device).requires_grad_(False).eval()

    dataset = get_test_dataset(config, registry["dataset"])
    indices = fixed_indices(len(dataset), count)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=min(10, len(indices)), shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    output_dir = HERE / "outputs" / dataset_name
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    subset_offset = 0
    nan_inf_count = 0

    for image, audio, gt, sample_ids, labels in loader:
        image = image.to(device, non_blocking=True).float()
        audio = audio.to(device, non_blocking=True).float()
        gt_np = gt.numpy().astype(np.float32)
        img_l4, aud_fine = readout(image, audio)
        obj_raw = object_model(image)

        aud = resize_and_normalize(aud_fine)
        img = resize_and_normalize(img_l4)
        obj = resize_and_normalize(obj_raw)
        iqr = np.stack([normalize_map(0.6 * a + 0.4 * i) for a, i in zip(aud, img)])
        ogl = np.stack([normalize_map(0.6 * a + 0.4 * o) for a, o in zip(aud, obj)])
        extra = np.stack(
            [normalize_map(0.6 * a + 0.2 * i + 0.2 * o) for a, i, o in zip(aud, img, obj)]
        )
        method_batches = {
            "AUD": aud,
            "IMG_QUERY": img,
            "IQR": iqr,
            "OBJ_PRIOR": obj,
            "OGL": ogl,
            "EXTRA_IQR_OGL": extra,
        }
        rgb = denormalize_images(image)
        nan_inf_count += sum(
            int((~np.isfinite(values)).sum()) for values in method_batches.values()
        )

        for local_index, sample_id in enumerate(sample_ids):
            maps = {method: method_batches[method][local_index] for method in METHODS}
            ious = {
                method: sample_iou(maps[method], gt_np[local_index]) for method in METHODS
            }
            dataset_index = indices[subset_offset + local_index]
            label = str(labels[local_index])
            sample_id = str(sample_id)
            save_sample_panel(
                panels_dir / f"{subset_offset + local_index:02d}_{sample_id}.png",
                dataset_name.upper(), sample_id, label, rgb[local_index],
                gt_np[local_index], maps, ious,
            )
            save_individual_maps(output_dir, sample_id, rgb[local_index], gt_np[local_index], maps)
            manifest_rows.append(
                {
                    "display_order": subset_offset + local_index,
                    "dataset_index": dataset_index,
                    "sample_id": sample_id,
                    "label": label,
                    **{f"IoU_{method}": ious[method] for method in METHODS},
                }
            )
            contact_rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "image": rgb[local_index],
                    "gt": gt_np[local_index],
                    "maps": maps,
                }
            )
        subset_offset += len(sample_ids)

    with (output_dir / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    save_contact_sheet(output_dir / f"1.3G_{dataset_name}_contact_sheet.png", contact_rows, dataset_name.upper())

    checkpoint_after = checkpoint_state(checkpoint_path)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("Checkpoint changed during read-only visualization")
    if nan_inf_count:
        raise RuntimeError(f"Detected {nan_inf_count} NaN/Inf heatmap values")

    result = {
        "dataset": dataset_name,
        "formal_checkpoint": checkpoint_before,
        "base_checkpoint": str(base_checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset_size": len(dataset),
        "fixed_dataset_indices": indices,
        "num_visualized_samples": len(manifest_rows),
        "alpha": 0.6,
        "threshold_for_displayed_iou": 0.6,
        "normalization": "official evaluator: bicubic to 224x224 then per-map min-max",
        "map_definitions": {
            "AUD": "1.3G AUD_FINE: Qa -> K34 (14x14)",
            "IMG_QUERY": "frozen visual Q4 -> K4 (7x7)",
            "IQR": "norm(0.6*AUD + 0.4*IMG_QUERY)",
            "OBJ_PRIOR": "ImageNet ResNet18 external OGL branch output",
            "OGL": "norm(0.6*AUD + 0.4*OBJ_PRIOR)",
            "EXTRA_IQR_OGL": "norm(0.6*AUD + 0.2*IMG_QUERY + 0.2*OBJ_PRIOR)",
        },
        "read_only_audit": {
            "optimizer_created": False,
            "backward_called": False,
            "all_refinement_parameters_require_grad_false": all(
                not parameter.requires_grad for parameter in refinement.parameters()
            ),
            "all_object_prior_parameters_require_grad_false": all(
                not parameter.requires_grad for parameter in object_model.parameters()
            ),
            "checkpoint_before_after_identical": checkpoint_before == checkpoint_after,
            "nan_inf_count": nan_inf_count,
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    refinement.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(FORMAL), default=["vggss", "flickr"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    results = [
        run_dataset(dataset, arguments.gpu, arguments.num_samples)
        for dataset in arguments.datasets
    ]
    (HERE / "outputs" / "summary.json").write_text(
        json.dumps({"experiments": results}, indent=2), encoding="utf-8"
    )
    for result in results:
        print(
            f"Saved {result['num_visualized_samples']} fixed {result['dataset']} samples: "
            f"{HERE / 'outputs' / result['dataset']}",
            flush=True,
        )
    print("Read-only 1.3G visualization diagnosis complete.", flush=True)


if __name__ == "__main__":
    main()
