#!/usr/bin/env python3
"""Evaluate a trained top-down L3 refinement head without any external prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    EXPERIMENTS,
    PROJECT_ROOT,
    build_model,
    build_test_dataset,
    build_test_loader,
    load_base_config,
    parameter_counts,
    setup_seed,
)
from train import metric_matches, print_metrics, validate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-qualitative", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    registry = EXPERIMENTS[arguments.experiment]
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    experiment_name = arguments.experiment_name or registry["default_experiment"]
    checkpoint_path = arguments.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            arguments.model_dir
            / experiment_name
            / f"{registry['dataset']}_best.pth"
        )
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal evaluation")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded_base = Path(checkpoint["base_checkpoint_path"]).resolve()
    if recorded_base != base_checkpoint:
        raise RuntimeError(
            "Refinement checkpoint was trained from a different base: "
            f"recorded={recorded_base}, expected={base_checkpoint}"
        )
    model.refinement_head.load_state_dict(
        checkpoint["refinement_head_state_dict"], strict=True
    )
    print(json.dumps(parameter_counts(model), indent=2), flush=True)
    print(f"Base checkpoint: {base_checkpoint}", flush=True)
    print(f"Refinement checkpoint: {checkpoint_path}", flush=True)

    test_dataset = build_test_dataset(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    output_dir = checkpoint_path.parent / "test_outputs"
    qualitative_dir = None if arguments.no_qualitative else output_dir / "qualitative"
    metrics = validate(
        model,
        test_loader,
        device,
        qualitative_dir=qualitative_dir,
    )
    if not metric_matches(metrics["AUD_L4"], registry["expected_aud"]):
        raise RuntimeError("Frozen AUD_L4 no longer reproduces the formal result")
    if not metric_matches(metrics["IMG_L4"], registry["expected_img"]):
        raise RuntimeError("Frozen IMG_L4 no longer reproduces the formal result")
    print_metrics(metrics, prefix="TEST/")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / f"{checkpoint_path.stem}_metrics.json"
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {metric_path}", flush=True)
    if qualitative_dir is not None:
        print(f"Qualitative panels: {qualitative_dir}", flush=True)
    model.close()


if __name__ == "__main__":
    main()
