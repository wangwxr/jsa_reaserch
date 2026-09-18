#!/usr/bin/env python3
"""Freeze the shared K=4 in-batch different-video negatives for 10k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common


K = 4


def video_id(sample_id: str, dataset_name: str) -> str:
    if dataset_name == "vggss":
        stem, suffix = sample_id.rsplit("_", 1)
        if suffix.isdigit():
            return stem
    return sample_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(common.DATASETS), required=True)
    args = parser.parse_args()
    seed = 12345
    path = common.HERE / "configs" / f"{args.dataset}_seed{seed}_negative_offsets.npy"
    audit_path = path.with_suffix(".json")
    if path.exists() or audit_path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    dataset = common.train_dataset(args.dataset)
    sample_ids = np.asarray([Path(value).stem for value in dataset.image_files], dtype=str)
    videos = np.asarray([video_id(value, args.dataset) for value in sample_ids], dtype=str)
    orders = np.load(common.order_path(args.dataset, seed), mmap_mode="r")
    if orders.shape != (common.EPOCHS, len(dataset)):
        raise RuntimeError(f"Unexpected order shape: {orders.shape}")
    offsets = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16,
                                        shape=(common.EPOCHS, len(dataset), K))
    offsets.fill(-1)
    rng = np.random.default_rng(seed + 87_000_001)
    full_batches = len(dataset) // common.BATCH_SIZE
    violations = 0
    for epoch in range(common.EPOCHS):
        order = np.asarray(orders[epoch])
        for batch in range(full_batches):
            start = batch * common.BATCH_SIZE
            indices = order[start:start + common.BATCH_SIZE]
            batch_videos = videos[indices]
            for local in range(common.BATCH_SIZE):
                eligible = np.flatnonzero(batch_videos != batch_videos[local])
                chosen = rng.choice(eligible, size=K, replace=False)
                offsets[epoch, start + local] = chosen.astype(np.int16)
                violations += int(np.any(batch_videos[chosen] == batch_videos[local]))
    del offsets
    sample_path = common.HERE / "configs" / f"{args.dataset}_seed{seed}_train_sample_ids.npy"
    np.save(sample_path, sample_ids)
    payload = {"dataset": args.dataset, "seed": seed, "negative_count": K,
               "sampler": "uniform_without_replacement_within_fixed_training_batch",
               "different_video_rule": "VGGSound strips trailing _frame; Flickr uses its sample/video ID",
               "offsets": common.file_snapshot(path), "orders": common.file_snapshot(common.order_path(args.dataset, seed)),
               "train_sample_ids": common.file_snapshot(sample_path), "epochs": common.EPOCHS,
               "full_batches_per_epoch": full_batches, "dropped_tail_per_epoch": len(dataset) % common.BATCH_SIZE,
               "same_video_violations": violations, "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md"),
               "git": common.git_metadata()}
    if violations:
        raise RuntimeError(f"Found {violations} same-video negative violations")
    common.write_json(audit_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
