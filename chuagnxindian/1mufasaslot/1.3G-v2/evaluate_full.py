#!/usr/bin/env python3
"""Evaluate a 1.3G-v2 best checkpoint with old six maps plus fine visual maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    EXPERIMENTS,
    build_model,
    build_test_loader,
    load_base_config,
    setup_seed,
)
from dataset import get_test_dataset  # noqa: E402
from train import print_metrics, validate  # noqa: E402
import test_model  # noqa: E402


class FullMetricModel(nn.Module):
    """Expose old IMG_L4 and v2 AUD_FINE to the unchanged six-map evaluator."""

    def __init__(self, refinement: nn.Module):
        super().__init__()
        self.refinement = refinement

    def forward(self, image: torch.Tensor, audio: torch.Tensor):
        output = self.refinement(image, audio)
        return output["IMG_L4"], output["AUD_FINE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--checkpoint")
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
    if checkpoint.get("architecture") != "1.3G-v2_visual_semantic_preservation":
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
    extra_metrics, _ = validate(refinement, test_loader, device)
    print_metrics(extra_metrics, prefix="V2/")

    full_model = FullMetricModel(refinement).to(device).eval()
    object_model = object_prior_model().to(device).eval()
    values = test_model.validate_img_aud(
        test_loader,
        full_model,
        object_model,
        str(experiment_dir / "viz_full_metrics"),
        registry["dataset"],
        -1,
        config,
    )
    names = ("AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL")
    six_metrics = {
        name: {"cIoU": float(values[2 * index]), "AUC": float(values[2 * index + 1])}
        for index, name in enumerate(names)
    }
    # AUD and IMG_QUERY must be the same tensors as AUD_FINE and IMG_L4.
    reproduction = {
        "AUD_vs_AUD_FINE_cIoU": abs(
            six_metrics["AUD"]["cIoU"] - extra_metrics["AUD_FINE"]["cIoU"]
        ),
        "AUD_vs_AUD_FINE_AUC": abs(
            six_metrics["AUD"]["AUC"] - extra_metrics["AUD_FINE"]["AUC"]
        ),
        "IMG_QUERY_vs_IMG_L4_cIoU": abs(
            six_metrics["IMG_QUERY"]["cIoU"] - extra_metrics["IMG_L4"]["cIoU"]
        ),
        "IMG_QUERY_vs_IMG_L4_AUC": abs(
            six_metrics["IMG_QUERY"]["AUC"] - extra_metrics["IMG_L4"]["AUC"]
        ),
    }
    if max(reproduction.values()) > 1e-10:
        raise RuntimeError(f"Full evaluator map reproduction failed: {reproduction}")

    result = {
        "architecture": "1.3G-v2_visual_semantic_preservation",
        "experiment": experiment_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_checkpoint": str(base_checkpoint),
        "checkpoint_selection": "AUD_FINE_cIoU",
        "alpha": 0.6,
        "six_metrics": six_metrics,
        "fine_metrics": extra_metrics,
        "reproduction": reproduction,
        "map_definitions": {
            "AUD": "AUD_FINE: frozen Qa -> learned K34, 14x14",
            "IMG_QUERY": "IMG_L4: frozen Q4 -> frozen K4, 7x7",
            "IQR": "0.6 AUD_FINE + 0.4 IMG_L4 in unchanged evaluator",
            "IMG_FINE": "frozen Q4 -> learned K34, 14x14",
            "IQR_FINE": "native 14x14 0.6 AUD_FINE + 0.4 IMG_FINE",
        },
    }
    output_path = experiment_dir / "best_full_metrics.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in names:
        print(
            f"{name}_{registry['dataset']}/cIoU, auc "
            f"{six_metrics[name]['cIoU']:.4f} {six_metrics[name]['AUC']:.4f}",
            flush=True,
        )
    print(f"Saved: {output_path}", flush=True)
    refinement.close()


if __name__ == "__main__":
    main()
