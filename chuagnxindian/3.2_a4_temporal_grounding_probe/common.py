"""Shared zero-training helpers for Experiment 3.2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_32_mpl")

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
REFERENCE_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.2_highres_slot_ownership" / "results"
OBJECT_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

for import_path in (PROJECT_ROOT, V11_ROOT, ABLATION_ROOT):
    sys.path.insert(0, str(import_path))

from dataset import get_test_dataset, inverse_normalize  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402
import test_model  # noqa: E402

sys.path.insert(0, str(G_ROOT))
from model import MultiGeometryEquivariantRefinement as GRefinement  # noqa: E402


EXPERIMENTS = {
    "vggss_144k": {
        "dataset": "vggss",
        "stage1": "mufasa_ablation2_l3_l4_ablation_vggss_144k",
        "g": "1.3G-multigeom_equivariant_l3_refine_vggss_144k",
        "checkpoint": "vggss_best.pth",
        "reference_key": "vggss_144k",
        "batch_size": 256,
        "workers": 16,
    },
    "flickr_144k": {
        "dataset": "flickr",
        "stage1": "mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5",
        "g": "1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5",
        "checkpoint": "flickr_best.pth",
        "reference_key": "flickr_144k",
        "batch_size": 32,
        "workers": 12,
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
    files = {}
    all_unchanged = True
    for role, reference in before.items():
        path = Path(reference["path"])
        after_hash = sha256(path)
        after_mtime = path.stat().st_mtime_ns
        unchanged = after_hash == reference["sha256"] and after_mtime == reference["mtime_ns"]
        files[role] = {
            **reference,
            "sha256_after": after_hash,
            "mtime_ns_after": after_mtime,
            "unchanged": unchanged,
        }
        all_unchanged = all_unchanged and unchanged
    return {"files": files, "all_unchanged": all_unchanged}


def freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


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
        raise RuntimeError(f"Unexpected G architecture: {checkpoint.get('architecture')}")
    model.student.proj3_spatial.load_state_dict(checkpoint["proj3_spatial_state_dict"], strict=True)
    model.student.adapter.load_state_dict(checkpoint["topdown_adapter_state_dict"], strict=True)
    model.eval()
    freeze(model)
    return model


def object_prior_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    model.eval()
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
    value = np.asarray(value, dtype=np.float64)
    minimum = value.min()
    maximum = value.max()
    if maximum - minimum != 0:
        return (value - minimum) / (maximum - minimum)
    return value.copy()


def minmax_tensor(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(start_dim=2)
    minimum = flat.min(dim=-1, keepdim=True).values.view(value.shape[0], value.shape[1], 1, 1)
    maximum = flat.max(dim=-1, keepdim=True).values.view(value.shape[0], value.shape[1], 1, 1)
    return (value - minimum) / (maximum - minimum).clamp_min(1e-12)


def resize_tensor(value: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    return F.interpolate(value, (224, 224), mode=mode, align_corners=False)


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
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    reference_success = reference_array >= 0.5
    candidate_success = candidate_array >= 0.5
    rescue = (~reference_success) & candidate_success
    hurt = reference_success & (~candidate_success)
    return {
        "rescue": int(rescue.sum()),
        "hurt": int(hurt.sum()),
        "net": int(rescue.sum() - hurt.sum()),
        "oracle": summarize(np.maximum(reference_array, candidate_array).tolist()),
    }


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return math.nan
    return float(np.dot(first, second) / denominator)


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return safe_pearson(
        rankdata(np.asarray(first).ravel(), method="average"),
        rankdata(np.asarray(second).ravel(), method="average"),
    )


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.clip(np.asarray(first, dtype=np.float64).ravel(), 0.0, None)
    second = np.clip(np.asarray(second, dtype=np.float64).ravel(), 0.0, None)
    first = (first + 1e-12) / (first.sum() + 1e-12 * first.size)
    second = (second + 1e-12) / (second.sum() + 1e-12 * second.size)
    middle = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / middle))
        + 0.5 * np.sum(second * np.log(second / middle))
    )


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(array)) if array.size else math.nan,
        "median": float(np.nanmedian(array)) if array.size else math.nan,
        "std": float(np.nanstd(array)) if array.size else math.nan,
        "num_samples": int(np.isfinite(array).sum()),
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
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def checkpoint_paths(registry: dict[str, Any]) -> dict[str, Path]:
    return {
        "stage1_l3_l4": PROJECT_ROOT / "checkpoints" / registry["stage1"] / registry["checkpoint"],
        "original_1_3g": PROJECT_ROOT / "checkpoints" / registry["g"] / registry["checkpoint"],
        "evaluation_only_object_prior": OBJECT_CHECKPOINT,
    }


def reference_summary(registry: dict[str, Any]) -> dict[str, Any]:
    path = REFERENCE_ROOT / registry["reference_key"] / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def reference_metric(summary: dict[str, Any], method: str) -> dict[str, Any]:
    for row in summary["method_metrics"]:
        if row["method"] == method:
            return row
    raise KeyError(method)

