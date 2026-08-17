#!/usr/bin/env python3
"""Evaluate an Experiment G best checkpoint with the standard six JSA maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from common import (
    EXPERIMENTS,
    PROJECT_ROOT,
    build_model,
    build_test_loader,
    load_base_config,
    setup_seed,
)
from dataset import get_test_dataset
import test_model


class FullMetricModel(nn.Module):
    """Expose frozen L4 IMG_QUERY together with the learned 14x14 AUD_FINE."""

    def __init__(self, refinement: nn.Module):
        super().__init__()
        self.refinement = refinement

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The first output follows the unchanged L3+L4 Q4 -> K4 evaluation path.
        img_l4, _aud_l4 = self.refinement.teacher.forward_eval(image, audio)
        # The second output is Experiment G's primary Qa -> K34 result.
        aud_fine = self.refinement(image, audio)["AUD_FINE"]
        return img_l4, aud_fine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def object_prior_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    return model


def main() -> None:
    arguments = parse_args()
    registry = EXPERIMENTS[arguments.experiment]
    experiment_name = arguments.experiment_name or registry["default_experiment"]
    checkpoint_name = arguments.checkpoint or f"{registry['dataset']}_best.pth"
    experiment_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    checkpoint_path = experiment_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    config.alpha = 0.6
    config.model_dir = str(PROJECT_ROOT / "checkpoints")
    config.experiment_name = experiment_name
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
    refinement.eval()

    test_dataset = get_test_dataset(config, registry["dataset"])
    test_loader = build_test_loader(test_dataset, config, registry)
    model = FullMetricModel(refinement).to(device).eval()
    object_model = object_prior_model().to(device).eval()

    print(f"Experiment G checkpoint: {checkpoint_path}", flush=True)
    print(f"Base L3+L4 checkpoint: {base_checkpoint}", flush=True)
    print("AUD = AUD_FINE (Qa -> K34, 14x14)", flush=True)
    print("IMG_QUERY = frozen teacher IMG_L4 (Q4 -> K4, 7x7)", flush=True)
    print("alpha = 0.6; evaluator = unchanged root test_model.validate_img_aud", flush=True)

    values = test_model.validate_img_aud(
        test_loader,
        model,
        object_model,
        str(experiment_dir / "viz_full_metrics"),
        registry["dataset"],
        -1,
        config,
    )
    names = (
        "AUD",
        "IMG_QUERY",
        "IQR",
        "OBJ_PRIOR",
        "OGL",
        "EXTRA_IQR_OGL",
    )
    metrics = {
        name: {"cIoU": float(values[2 * index]), "AUC": float(values[2 * index + 1])}
        for index, name in enumerate(names)
    }
    result = {
        "architecture": "multi_geometry_equivariant_l3_refine",
        "experiment": experiment_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_checkpoint": str(base_checkpoint),
        "alpha": 0.6,
        "map_definitions": {
            "AUD": "AUD_FINE: frozen Qa -> learned K34, 14x14",
            "IMG_QUERY": "frozen L3+L4 teacher Q4 -> K4, 7x7",
            "IQR": "normalize(0.6 * AUD + 0.4 * IMG_QUERY)",
            "OBJ_PRIOR": "unchanged ImageNet ResNet18 object prior",
            "OGL": "normalize(0.6 * AUD + 0.4 * OBJ_PRIOR)",
            "EXTRA_IQR_OGL": "normalize(0.6 * AUD + 0.2 * IMG_QUERY + 0.2 * OBJ_PRIOR)",
        },
        "metrics": metrics,
    }
    output_path = experiment_dir / "best_full_six_metrics.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in names:
        print(
            f"{name}_{registry['dataset']}/cIoU, auc "
            f"{metrics[name]['cIoU']:.4f} {metrics[name]['AUC']:.4f}",
            flush=True,
        )
    print(f"Saved full metrics: {output_path}", flush=True)
    refinement.close()


if __name__ == "__main__":
    main()
