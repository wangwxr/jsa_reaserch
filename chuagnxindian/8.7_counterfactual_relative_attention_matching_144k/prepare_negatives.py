#!/usr/bin/env python3
"""Freeze same-batch, different-video CRAM negative offsets before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import common


K = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(common.DATASETS), required=True)
    parser.add_argument("--seed", choices=common.SEEDS, type=int, required=True)
    return parser.parse_args()


def video_id(sample_id: str, dataset_name: str) -> str:
    if dataset_name == "vggss":
        stem, suffix = sample_id.rsplit("_", 1)
        if suffix.isdigit():
            return stem
    return sample_id


def main() -> None:
    args = parse_args()
    path = common.HERE / "configs" / f"{args.dataset}_seed{args.seed}_negative_offsets.npy"
    audit_path = path.with_suffix(".json")
    if path.exists() or audit_path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen negatives: {path}")
    dataset = common.train_dataset(args.dataset)
    sample_ids = np.asarray([Path(value).stem for value in dataset.image_files], dtype=str)
    videos = np.asarray([video_id(value, args.dataset) for value in sample_ids], dtype=str)
    orders = np.load(common.order_path(args.dataset, args.seed), mmap_mode="r")
    if orders.shape != (common.EPOCHS, len(dataset)):
        raise RuntimeError(f"Unexpected order shape: {orders.shape}")
    offsets = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.int16, shape=(common.EPOCHS, len(dataset), K)
    )
    offsets.fill(-1)
    rng = np.random.default_rng(args.seed + 87_000_001)
    full_batches = len(dataset) // common.BATCH_SIZE
    violations = 0
    for epoch in range(common.EPOCHS):
        order = np.asarray(orders[epoch])
        for batch in range(full_batches):
            start = batch * common.BATCH_SIZE
            stop = start + common.BATCH_SIZE
            batch_indices = order[start:stop]
            batch_videos = videos[batch_indices]
            for local in range(common.BATCH_SIZE):
                eligible = np.flatnonzero(batch_videos != batch_videos[local])
                if len(eligible) < K:
                    raise RuntimeError(f"Insufficient different-video negatives at epoch {epoch}, batch {batch}")
                chosen = rng.choice(eligible, size=K, replace=False)
                offsets[epoch, start + local] = chosen.astype(np.int16)
                violations += int(np.any(batch_videos[chosen] == batch_videos[local]))
    del offsets
    manifest = common.HERE / "configs" / f"{args.dataset}_seed{args.seed}_train_sample_ids.npy"
    np.save(manifest, sample_ids)
    audit = {
        "dataset": args.dataset,
        "seed": args.seed,
        "negative_count": K,
        "sampler": "uniform_without_replacement_within_fixed_training_batch",
        "different_video_rule": "VGGSound strips trailing _frame; Flickr uses its sample/video ID",
        "offsets": common.file_snapshot(path),
        "orders": common.file_snapshot(common.order_path(args.dataset, args.seed)),
        "train_sample_ids": common.file_snapshot(manifest),
        "epochs": common.EPOCHS,
        "full_batches_per_epoch": full_batches,
        "dropped_tail_per_epoch": len(dataset) % common.BATCH_SIZE,
        "same_video_violations": violations,
        "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md"),
        "git": common.git_metadata(),
    }
    if violations:
        raise RuntimeError(f"Found {violations} same-video negatives")
    common.write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
