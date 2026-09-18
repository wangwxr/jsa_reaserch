#!/usr/bin/env python3
"""Pre-training CRAM gradient-scale audit; it never evaluates test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import common
import objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(common.DATASETS), required=True)
    parser.add_argument("--seed", choices=common.SEEDS, type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


GRADIENT_SCALE = float(2**16)


def gradient_norm(loss: torch.Tensor, model: torch.nn.Module) -> float:
    # Match the production GradScaler path; direct half-precision backward makes
    # the small attention-MSE gradients underflow before their norm is observed.
    (loss * GRADIENT_SCALE).backward()
    return sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in model.parameters() if parameter.grad is not None
    ) ** 0.5 / GRADIENT_SCALE


def build_train_loader(dataset, orders: np.ndarray, seed: int, workers: int):
    sampler = common.EpochOrderSampler(orders)
    sampler.set_epoch(0)
    return DataLoader(
        dataset, batch_size=common.BATCH_SIZE, sampler=sampler, num_workers=workers,
        pin_memory=True, drop_last=True, persistent_workers=False,
        prefetch_factor=2 if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


def negative_spec(spec: torch.Tensor, offsets: np.ndarray, device: torch.device) -> torch.Tensor:
    positions = torch.from_numpy(np.asarray(offsets, dtype=np.int64)).to(device)
    if torch.any(positions < 0) or torch.any(positions >= len(spec)):
        raise RuntimeError("Frozen negative offsets do not match this full training batch")
    return spec[positions]


def main() -> None:
    args = parse_args(); torch.cuda.set_device(args.gpu); device = torch.device(f"cuda:{args.gpu}")
    init_path, orders_path = common.initialization_path(args.dataset, args.seed), common.order_path(args.dataset, args.seed)
    negative_path = common.HERE / "configs" / f"{args.dataset}_seed{args.seed}_negative_offsets.npy"
    if not negative_path.is_file(): raise FileNotFoundError(negative_path)
    common.setup_seed(args.seed)
    model = common.build_model(args.dataset, device, init_path, trainable=True)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before = common.state_sha256(model.state_dict())
    orders = np.load(orders_path, mmap_mode="r"); negatives = np.load(negative_path, mmap_mode="r")
    loader = build_train_loader(common.train_dataset(args.dataset), orders, args.seed, common.DATASETS[args.dataset]["workers"])
    frame, spec, _boxes, _ids, _labels = next(iter(loader)); del loader
    frame, spec = frame.to(device).float(), spec.to(device).float()
    wrong = negative_spec(spec, negatives[0, :len(spec)], device)

    def run(which: str) -> float:
        common.setup_seed(args.seed + 8_700_001)
        model.zero_grad(set_to_none=True); model.train()
        with torch.amp.autocast("cuda"):
            output = objective.forward_components(model, frame, spec)
            if which == "info": loss = output["info_loss"]
            elif which == "match": loss = output["match_loss"]
            elif which == "cram":
                with torch.random.fork_rng(devices=[args.gpu]):
                    torch.manual_seed(args.seed + 87_000_001); torch.cuda.manual_seed_all(args.seed + 87_000_001)
                    loss = objective.cram_components(model, output, wrong)["cram_loss"]
            else: raise ValueError(which)
        return gradient_norm(loss, model)

    raw = {name: run(name) for name in ("info", "match", "cram")}
    model.load_state_dict(initial, strict=True)
    after = common.state_sha256(model.state_dict())
    weighted = {"info": raw["info"], "original_match": 100.0 * raw["match"], "cram_lambda_0_1": 0.1 * raw["cram"], "cram_lambda_0_5": 0.5 * raw["cram"], "cram_lambda_1_0": raw["cram"]}
    if weighted["original_match"] <= 0:
        raise RuntimeError("Original match gradient underflowed despite production-equivalent scaling")
    c3_allowed = weighted["cram_lambda_0_1"] < weighted["original_match"] / 100.0
    payload = {"dataset": args.dataset, "seed": args.seed, "batch_size": len(spec), "gradient_scale": GRADIENT_SCALE, "raw_gradient_l2": raw, "weighted_gradient_l2": weighted, "c3_allowed": c3_allowed, "c3_rule": "allow lambda=1.0 only when 0.1*||grad L_CRAM|| is below 1% of 100*||grad original L_match||", "initialization": common.file_snapshot(init_path), "negative_mapping": common.file_snapshot(negative_path), "model_state_restored": before == after, "optimizer_created": False, "test_data_used": False, "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md"), "git": common.git_metadata()}
    if not payload["model_state_restored"]: raise RuntimeError("Gradient audit changed model state")
    common.write_json(common.HERE / "gradient_audit" / f"{args.dataset}_seed{args.seed}.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
