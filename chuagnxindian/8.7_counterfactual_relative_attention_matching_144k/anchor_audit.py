#!/usr/bin/env python3
"""Measure the matched A2V-to-V2V anchor distance on one fixed training batch."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common
import objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(common.DATASETS), required=True)
    parser.add_argument("--seed", choices=common.SEEDS, type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    order = np.load(common.order_path(args.dataset, args.seed), mmap_mode="r")
    sampler = common.EpochOrderSampler(order)
    sampler.set_epoch(0)
    loader = DataLoader(
        common.train_dataset(args.dataset), batch_size=common.BATCH_SIZE,
        sampler=sampler, num_workers=common.DATASETS[args.dataset]["workers"],
        pin_memory=True, drop_last=True,
    )
    frame, spec, _boxes, _ids, _labels = next(iter(loader))
    frame, spec = frame.to(device).float(), spec.to(device).float()
    rows = []
    for group in common.GROUPS:
        checkpoint = common.checkpoint_path(args.dataset, group, args.seed, "best")
        common.setup_seed(args.seed)
        model = common.build_model(args.dataset, device, checkpoint, trainable=False)
        model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            output = objective.forward_components(model, frame, spec)
            target = output["v2v_prob"][:, 0]
            d_pos = F.mse_loss(output["a2v_prob"][:, 0], target, reduction="none").mean(1)
        rows.append({
            "dataset": args.dataset, "seed": args.seed, "group": group,
            "d_pos_mean": float(d_pos.mean()), "d_pos_median": float(d_pos.median()),
            "checkpoint": common.file_snapshot(checkpoint),
        })
    payload = {"fixed_train_epoch": 0, "batch_size": len(spec), "rows": rows,
               "test_data_used": False, "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md")}
    path = common.HERE / "gradient_audit" / f"{args.dataset}_anchor_retention_seed{args.seed}.json"
    common.write_json(path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
