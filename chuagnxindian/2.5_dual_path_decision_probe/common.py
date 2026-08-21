"""Shared loading, evaluation, and zero-training audit helpers for Experiment 2.5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_25_mpl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn import metrics as sklearn_metrics
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
ABLATION_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
G_ROOT = V11_ROOT / "1.3G-multigeom_equivariant_l3_refine"
NATIVE_UPDATE_ROOT = V11_ROOT / "1.1.1_14_14_L3"
D24_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.4_object_aware_multigeom_spatial_specialization"
R22_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.2_highres_slot_ownership" / "results"

for import_path in (PROJECT_ROOT, V11_ROOT, ABLATION_ROOT):
    sys.path.insert(0, str(import_path))

from dataset import get_test_dataset, inverse_normalize  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402
import test_model  # noqa: E402

sys.path.insert(0, str(G_ROOT))
from model import MultiGeometryEquivariantRefinement as GRefinement  # noqa: E402

sys.path.insert(0, str(NATIVE_UPDATE_ROOT))
from model_native_l3 import MUFASAJSA11NativeL3  # noqa: E402


EXPERIMENTS = {
    "vggss_10k": {
        "dataset": "vggss",
        "stage1": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
        "checkpoint": "vggss_best.pth",
        "native_update": "1.1.1_14_14_L3_vggss_10k",
        "g": "1.3G-multigeom_equivariant_l3_refine_vggss_10k",
        "batch_size": 256,
        "workers": 16,
        "formal_144k": False,
    },
    "flickr_10k": {
        "dataset": "flickr",
        "stage1": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
        "checkpoint": "flickr_best.pth",
        "native_update": "1.1.1_14_14_L3_flickr_10k",
        "g": "1.3G-multigeom_equivariant_l3_refine_flickr_10k_frame8_center5",
        "batch_size": 32,
        "workers": 12,
        "formal_144k": False,
    },
    "vggss_144k": {
        "dataset": "vggss",
        "stage1": "mufasa_ablation2_l3_l4_ablation_vggss_144k",
        "checkpoint": "vggss_best.pth",
        "native_update": "1.1.1_14_14_L3_vggss_144k",
        "g": "1.3G-multigeom_equivariant_l3_refine_vggss_144k",
        "d24": "2.4_object_aware_multigeom_spatial_specialization_vggss_144k",
        "reference_key": "vggss_144k",
        "batch_size": 256,
        "workers": 16,
        "formal_144k": True,
    },
    "flickr_144k": {
        "dataset": "flickr",
        "stage1": "mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5",
        "checkpoint": "flickr_best.pth",
        "native_update": "1.1.1_14_14_L3_flickr_144k",
        "g": "1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5",
        "d24": "2.4_object_aware_multigeom_spatial_specialization_flickr_144k_frame8_center5",
        "reference_key": "flickr_144k",
        "batch_size": 32,
        "workers": 12,
        "formal_144k": True,
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for role, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        output[role] = {
            "path": str(resolved),
            "sha256": sha256(resolved),
            "mtime_ns": resolved.stat().st_mtime_ns,
            "size": resolved.stat().st_size,
        }
    return output


def verify_snapshots(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    all_unchanged = True
    for role, reference in before.items():
        path = Path(reference["path"])
        after_hash = sha256(path)
        after_mtime = path.stat().st_mtime_ns
        unchanged = after_hash == reference["sha256"] and after_mtime == reference["mtime_ns"]
        rows[role] = {
            **reference,
            "sha256_after": after_hash,
            "mtime_ns_after": after_mtime,
            "unchanged": unchanged,
        }
        all_unchanged = all_unchanged and unchanged
    return {"files": rows, "all_unchanged": all_unchanged}


def load_config(experiment: str) -> argparse.Namespace:
    path = PROJECT_ROOT / "checkpoints" / experiment / "configs.json"
    return argparse.Namespace(**json.loads(path.read_text(encoding="utf-8")))


def load_stage1(registry: dict[str, Any], device: torch.device) -> MUFASAL3L4:
    config = load_config(registry["stage1"])
    checkpoint_path = PROJECT_ROOT / "checkpoints" / registry["stage1"] / registry["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    model = MUFASAL3L4(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    freeze(model)
    return model


def load_original_g(registry: dict[str, Any], device: torch.device) -> GRefinement:
    teacher = load_stage1(registry, device)
    model = GRefinement(teacher).to(device)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / registry["g"] / registry["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "multi_geometry_equivariant_l3_refine":
        raise RuntimeError(f"Unexpected original G architecture: {checkpoint.get('architecture')}")
    model.student.proj3_spatial.load_state_dict(checkpoint["proj3_spatial_state_dict"], strict=True)
    model.student.adapter.load_state_dict(checkpoint["topdown_adapter_state_dict"], strict=True)
    model.eval()
    freeze(model)
    return model


def load_24(registry: dict[str, Any], device: torch.device):
    module = load_module("experiment_25_d24_model", D24_ROOT / "model.py")
    teacher = load_stage1(registry, device)
    model = module.MultiGeometryEquivariantRefinement(teacher).to(device)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / registry["d24"] / registry["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "object_aware_multigeom_spatial_specialization":
        raise RuntimeError(f"Unexpected 2.4 architecture: {checkpoint.get('architecture')}")
    model.student.proj3_spatial.load_state_dict(checkpoint["proj3_spatial_state_dict"], strict=True)
    model.student.adapter.load_state_dict(checkpoint["topdown_adapter_state_dict"], strict=True)
    model.eval()
    freeze(model)
    return model


def load_native_update(registry: dict[str, Any], device: torch.device) -> MUFASAJSA11NativeL3:
    config = load_config(registry["native_update"])
    checkpoint_path = PROJECT_ROOT / "checkpoints" / registry["native_update"] / registry["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    model = MUFASAJSA11NativeL3(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    freeze(model)
    return model


def freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def object_prior_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    freeze(model)
    return model


def build_loader(config: argparse.Namespace, registry: dict[str, Any]) -> DataLoader:
    config.testset = registry["dataset"]
    dataset = get_test_dataset(config, registry["dataset"])
    return DataLoader(
        dataset,
        batch_size=registry["batch_size"],
        shuffle=False,
        num_workers=registry["workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=registry["workers"] > 0,
    )


def flatten_batch(image, spec, bboxes, names):
    if image.ndim == 3:
        image = image.unsqueeze(0)
        spec = spec.unsqueeze(0)
        bboxes = bboxes.unsqueeze(0)
    if image.ndim == 5:
        batch, clips, channels, height, width = image.shape
        image = image.reshape(batch * clips, channels, height, width)
        _, _, channels, frequency, time = spec.shape
        spec = spec.reshape(batch * clips, channels, frequency, time)
        _, _, channels, height, width = bboxes.shape
        bboxes = bboxes.reshape(batch * clips, channels, height, width).squeeze(1)
        names = [name for name in names for _ in range(clips)]
    return image, spec, bboxes, [str(name) for name in names]


def normalize_map(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    minimum = value.min()
    maximum = value.max()
    if maximum - minimum != 0:
        return (value - minimum) / (maximum - minimum)
    return value.copy()


def fuse_maps(first: np.ndarray, second: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    return normalize_map(alpha * first + (1.0 - alpha) * second)


def resize_maps(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: F.interpolate(tensor, (224, 224), mode="bicubic", align_corners=False)
        .detach()
        .cpu()
        .numpy()[:, 0]
        for name, tensor in tensors.items()
    }


def sample_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    inferred = prediction >= 0.6
    intersection = np.sum(inferred * ground_truth)
    denominator = np.sum(ground_truth) + np.sum(inferred * (ground_truth == 0))
    return float(intersection / denominator)


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    thresholds = np.arange(21, dtype=np.float64) * 0.05
    curve = [float(np.mean(array >= threshold)) for threshold in thresholds]
    return {
        "cIoU": float(np.mean(array >= 0.5)),
        "AUC": float(sklearn_metrics.auc(thresholds, curve)),
        "mean_sample_cIoU": float(array.mean()),
        "num_samples": int(array.size),
    }


def transition(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    reference_success = np.asarray(reference) >= 0.5
    candidate_success = np.asarray(candidate) >= 0.5
    rescue = (~reference_success) & candidate_success
    hurt = reference_success & (~candidate_success)
    oracle = np.maximum(np.asarray(reference), np.asarray(candidate)).tolist()
    return {
        "rescue": int(rescue.sum()),
        "hurt": int(hurt.sum()),
        "net": int(rescue.sum() - hurt.sum()),
        "oracle": summarize(oracle),
    }


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64, copy=False).ravel()
    second = second.astype(np.float64, copy=False).ravel()
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return math.nan
    return float(np.dot(first, second) / denominator)


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return safe_pearson(
        rankdata(first.ravel(), method="average"),
        rankdata(second.ravel(), method="average"),
    )


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.clip(first.astype(np.float64, copy=False).ravel(), 0.0, None)
    second = np.clip(second.astype(np.float64, copy=False).ravel(), 0.0, None)
    first = (first + 1e-12) / (first.sum() + 1e-12 * first.size)
    second = (second + 1e-12) / (second.sum() + 1e-12 * second.size)
    middle = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / middle))
        + 0.5 * np.sum(second * np.log(second / middle))
    )


def aggregate_distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "median": float(np.nanmedian(array)),
        "num_samples": int(array.size),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_paths(registry: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "stage1_l3_l4": PROJECT_ROOT / "checkpoints" / registry["stage1"] / registry["checkpoint"],
        "native_update": PROJECT_ROOT / "checkpoints" / registry["native_update"] / registry["checkpoint"],
    }
    if registry["formal_144k"]:
        paths.update(
            {
                "original_1_3g": PROJECT_ROOT / "checkpoints" / registry["g"] / registry["checkpoint"],
                "experiment_2_4": PROJECT_ROOT / "checkpoints" / registry["d24"] / registry["checkpoint"],
            }
        )
    return paths


def source_artifact_paths(registry: dict[str, Any]) -> dict[str, Path]:
    native_dir = PROJECT_ROOT / "checkpoints" / registry["native_update"]
    paths = {"native_update_per_sample": native_dir / "ownership_per_sample.csv"}
    if registry["formal_144k"]:
        reference = registry["reference_key"]
        paths.update(
            {
                "experiment_2_2_summary": R22_ROOT / reference / "summary.json",
                "experiment_2_2_manifest": R22_ROOT / reference / "qualitative" / "selection_manifest.csv",
                "experiment_2_4_summary": PROJECT_ROOT / "checkpoints" / registry["d24"] / "summary.json",
            }
        )
    return paths


__all__ = [
    "EXPERIMENTS",
    "HERE",
    "PROJECT_ROOT",
    "R22_ROOT",
    "aggregate_distribution",
    "build_loader",
    "checkpoint_paths",
    "flatten_batch",
    "fuse_maps",
    "inverse_normalize",
    "js_divergence",
    "load_24",
    "load_config",
    "load_native_update",
    "load_original_g",
    "load_stage1",
    "normalize_map",
    "object_prior_model",
    "read_csv",
    "resize_maps",
    "sample_iou",
    "sha256",
    "snapshot_files",
    "source_artifact_paths",
    "spearman",
    "safe_pearson",
    "summarize",
    "transition",
    "verify_snapshots",
    "write_csv",
    "write_json",
]
