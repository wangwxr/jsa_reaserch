"""Data, checkpoint and parameter-state helpers for Experiment D."""

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
PROJECT_ROOT = HERE.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
ABLATION_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
FROZEN_TOPDOWN_ROOT = V11_ROOT / "topdown_l3_refine"
for import_path in (PROJECT_ROOT, V11_ROOT, ABLATION_ROOT, FROZEN_TOPDOWN_ROOT, HERE):
    sys.path.insert(0, str(import_path))

from dataset import get_test_dataset, get_train_dataset  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402
from model import JointL3L4TopDownModel  # noqa: E402


EXPERIMENTS = {
    "vggss_10k": {
        "dataset": "vggss",
        "base_experiment": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
        "base_checkpoint": "vggss_best.pth",
        "default_experiment": "joint_topdown_l3_refine_vggss_10k",
        "eval_batch_size": 256,
        "workers": 16,
        "expected_aud": (0.4015, 0.4074),
        "expected_img": (0.4064, 0.4094),
    },
    "flickr_10k": {
        "dataset": "flickr",
        "base_experiment": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
        "base_checkpoint": "flickr_best.pth",
        "default_experiment": "joint_topdown_l3_refine_flickr_10k_frame8_center5",
        "eval_batch_size": 32,
        "workers": 12,
        "expected_aud": (0.7640, 0.5916),
        "expected_img": (0.7520, 0.5890),
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
    config: argparse.Namespace, registry: dict, device: torch.device
) -> tuple[JointL3L4TopDownModel, Path]:
    checkpoint_path = base_checkpoint_path(registry)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model"].items()
    }
    base_model = MUFASAL3L4(config)
    # This snapshot is the authoritative freezing rule of the original model.
    original_trainability = {
        name: parameter.requires_grad
        for name, parameter in base_model.named_parameters()
    }
    base_model.load_state_dict(state, strict=True)
    model = JointL3L4TopDownModel(
        base_model, original_trainability=original_trainability
    ).to(device)
    return model, checkpoint_path


def build_datasets(config: argparse.Namespace, registry: dict):
    train_dataset = get_train_dataset(
        config,
        hard_img=config.hard_img,
        hard_aud=config.hard_aud,
        rand_aud=config.rand_aud,
    )
    return train_dataset, get_test_dataset(config, registry["dataset"])


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
        bboxes = bboxes.reshape(
            batch_size * clips, channels, height, width
        ).squeeze(1)
        names = [name for name in names for _ in range(clips)]
    return image, spec, bboxes, [str(name) for name in names]


def parameter_audit(model: JointL3L4TopDownModel) -> dict[str, object]:
    base_named = dict(model.base_model.named_parameters())
    base_total = sum(parameter.numel() for parameter in base_named.values())
    base_trainable = sum(
        parameter.numel() for parameter in base_named.values()
        if parameter.requires_grad
    )
    base_frozen = base_total - base_trainable
    head_total = sum(parameter.numel() for parameter in model.refinement_head.parameters())
    trainable_total = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_modules = sorted(
        {
            name.split(".", 1)[0]
            for name, parameter in model.base_model.named_parameters()
            if parameter.requires_grad
        }
    )
    frozen_names = [
        name
        for name, parameter in model.base_model.named_parameters()
        if not parameter.requires_grad
    ]
    return {
        "total_base_parameters": base_total,
        "trainable_base_parameters": base_trainable,
        "frozen_base_parameters": base_frozen,
        "refinement_head_parameters": head_total,
        "trainable_parameters": trainable_total,
        "optimizer_expected_parameters": base_trainable + head_total,
        "trainable_base_top_level_modules": trainable_modules,
        "frozen_base_parameter_names": frozen_names,
    }
