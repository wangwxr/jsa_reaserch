"""Experiment registry and unchanged data/checkpoint construction for G."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
V11_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
ABLATION_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
for import_path in (PROJECT_ROOT, V11_ROOT, ABLATION_ROOT):
    sys.path.insert(0, str(import_path))

from dataset import get_test_dataset, get_train_dataset  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402
from model import MultiGeometryEquivariantRefinement  # noqa: E402


EXPERIMENTS = {
    "vggss_10k": {
        "dataset": "vggss",
        "base_experiment": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
        "base_checkpoint": "vggss_best.pth",
        "default_experiment": "1.3G-multigeom_equivariant_l3_refine_vggss_10k",
        "eval_batch_size": 256,
        "workers": 16,
        "expected_aud": (0.4015, 0.4074),
    },
    "flickr_10k": {
        "dataset": "flickr",
        "base_experiment": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
        "base_checkpoint": "flickr_best.pth",
        "default_experiment": "1.3G-multigeom_equivariant_l3_refine_flickr_10k_frame8_center5",
        "eval_batch_size": 32,
        "workers": 12,
        "expected_aud": (0.7640, 0.5916),
    },
    "vggss_144k": {
        "dataset": "vggss",
        "base_experiment": "mufasa_ablation2_l3_l4_ablation_vggss_144k",
        "base_checkpoint": "vggss_best.pth",
        "default_experiment": "1.3G-multigeom_equivariant_l3_refine_vggss_144k",
        "eval_batch_size": 256,
        "workers": 16,
        "expected_aud": (0.4002, 0.4127),
    },
    "flickr_144k": {
        "dataset": "flickr",
        "base_experiment": "mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5",
        "base_checkpoint": "flickr_best.pth",
        "default_experiment": "1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5",
        "eval_batch_size": 32,
        "workers": 12,
        "expected_aud": (0.8040, 0.6228),
    },
}


def setup_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_base_config(registry: dict) -> argparse.Namespace:
    path = PROJECT_ROOT / "checkpoints" / registry["base_experiment"] / "configs.json"
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architecture") != "mufasa_ablation2_l3_l4_ablation":
        raise RuntimeError(f"Unexpected base architecture: {config.get('architecture')}")
    return argparse.Namespace(**config)


def base_checkpoint_path(registry: dict) -> Path:
    path = (
        PROJECT_ROOT
        / "checkpoints"
        / registry["base_experiment"]
        / registry["base_checkpoint"]
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def build_model(
    config: argparse.Namespace,
    registry: dict,
    device: torch.device,
    minimum_valid_ratio: float = 0.2,
) -> tuple[MultiGeometryEquivariantRefinement, Path]:
    checkpoint_path = base_checkpoint_path(registry)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key.replace("module.", ""): value for key, value in checkpoint["model"].items()
    }
    teacher = MUFASAL3L4(config)
    teacher.load_state_dict(state, strict=True)
    model = MultiGeometryEquivariantRefinement(
        teacher, minimum_valid_ratio=minimum_valid_ratio
    ).to(device)
    return model, checkpoint_path


def build_datasets(config: argparse.Namespace, registry: dict):
    train_dataset = get_train_dataset(
        config,
        hard_img=config.hard_img,
        hard_aud=config.hard_aud,
        rand_aud=config.rand_aud,
    )
    test_dataset = get_test_dataset(config, registry["dataset"])
    return train_dataset, test_dataset


def build_train_loader(dataset, config: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.workers > 0,
        prefetch_factor=2 if config.workers > 0 else None,
    )


def build_test_loader(dataset, config: argparse.Namespace, registry: dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=registry["eval_batch_size"],
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=config.workers > 0,
    )


def flatten_eval_batch(image, spec, bboxes, names):
    if image.ndim == 3:
        image = image.unsqueeze(0)
        spec = spec.unsqueeze(0)
        bboxes = bboxes.unsqueeze(0)
    if image.ndim == 5:
        batch_size, clips, channels, height, width = image.shape
        image = image.reshape(batch_size * clips, channels, height, width)
        _, _, channels, frequency, time = spec.shape
        spec = spec.reshape(batch_size * clips, channels, frequency, time)
        _, _, channels, height, width = bboxes.shape
        bboxes = bboxes.reshape(batch_size * clips, channels, height, width).squeeze(1)
        names = [name for name in names for _ in range(clips)]
    return image, spec, bboxes, [str(name) for name in names]


def parameter_counts(model: MultiGeometryEquivariantRefinement) -> dict[str, int]:
    teacher_total = sum(parameter.numel() for parameter in model.teacher.parameters())
    teacher_frozen = sum(
        parameter.numel()
        for parameter in model.teacher.parameters()
        if not parameter.requires_grad
    )
    proj3_total = sum(
        parameter.numel() for parameter in model.student.proj3_spatial.parameters()
    )
    adapter_total = sum(parameter.numel() for parameter in model.student.adapter.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_teacher_parameters": teacher_total,
        "frozen_teacher_parameters": teacher_frozen,
        "proj3_spatial_parameters": proj3_total,
        "topdown_adapter_parameters": adapter_total,
        "spatial_student_parameters": proj3_total + adapter_total,
        "trainable_parameters": trainable,
    }
