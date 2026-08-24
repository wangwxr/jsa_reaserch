#!/usr/bin/env python3
"""Evaluate the unchanged formal L3+L4 10k checkpoint as FULL_REFERENCE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    RESULTS_ROOT,
    checkpoint_identity,
    load_checkpoint_model,
    namespace_from_json,
    reference_paths,
    require_cuda,
    setup_paths,
    setup_seed,
    write_json,
)

setup_paths()

from dataset import get_test_dataset  # noqa: E402
from metrics import build_object_prior, evaluate_model, write_rows  # noqa: E402
from model import LossFactorizedL3L4  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["vggss", "flickr"])
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--workers", type=int)
    cli = parser.parse_args()

    device = require_cuda(cli.gpu)
    torch.cuda.set_device(cli.gpu)
    setup_seed(12345)
    config_path, checkpoint_path = reference_paths(cli.dataset)
    workers = cli.workers if cli.workers is not None else (16 if cli.dataset == "vggss" else 12)
    args = namespace_from_json(config_path, gpu=cli.gpu, workers=workers, batch_size=256)
    before = checkpoint_identity(checkpoint_path)

    model = LossFactorizedL3L4(args)
    model, _ = load_checkpoint_model(model, checkpoint_path, device)
    model.requires_grad_(False).eval()
    test_dataset = get_test_dataset(args, args.testset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    object_model = build_object_prior(device)
    result, per_sample = evaluate_model(
        model, test_loader, object_model, device, alpha=args.alpha
    )
    after = checkpoint_identity(checkpoint_path)
    if before != after:
        raise RuntimeError("FULL_REFERENCE checkpoint identity changed")

    output = RESULTS_ROOT / "full_reference" / cli.dataset
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "per_sample_metrics.csv", per_sample)
    summary = {
        "configuration": "FULL_REFERENCE",
        "dataset": cli.dataset,
        "checkpoint": before,
        "checkpoint_after": after,
        "checkpoint_identical": before == after,
        "optimizer_created": False,
        "backward_called": False,
        "metrics": result["metrics"],
        "diagnostics": result["diagnostics"],
    }
    write_json(output / "summary.json", summary)
    print(f"FULL_REFERENCE {cli.dataset}")
    for method in ("AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL"):
        metric = result["metrics"][method]
        print(f"{method:20s} {metric['cIoU']:.4f} {metric['AUC']:.4f}")


if __name__ == "__main__":
    main()
