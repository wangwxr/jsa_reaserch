#!/usr/bin/env python3
"""Create one shared deterministic 10k initialization and epoch orders."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

import common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(common.DATASETS), required=True)
    args = parser.parse_args()
    seed = 12345
    init = common.initialization_path(args.dataset, seed)
    orders_path = common.order_path(args.dataset, seed)
    audit_path = common.HERE / "configs" / f"{args.dataset}_seed{seed}_preparation.json"
    if any(path.exists() for path in (init, orders_path, audit_path)):
        raise RuntimeError(f"Refusing to overwrite prepared 10k assets for {args.dataset}")
    common.setup_seed(seed)
    dataset = common.train_dataset(args.dataset)
    if len(dataset) != 10_000:
        raise RuntimeError(f"Expected exactly 10,000 samples, got {len(dataset)}")
    model = common.build_model(args.dataset, torch.device("cpu"), trainable=True)
    state_hash = common.state_sha256(model.state_dict())
    init.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "dataset": args.dataset, "seed": seed,
                "state_sha256": state_hash,
                "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md")}, init)
    orders_path.parent.mkdir(parents=True, exist_ok=True)
    orders = np.lib.format.open_memmap(orders_path, mode="w+", dtype=np.int32,
                                       shape=(common.EPOCHS, len(dataset)))
    hashes = []
    for epoch in range(common.EPOCHS):
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + 1_000_003 * epoch)).numpy().astype(np.int32)
        orders[epoch] = order
        hashes.append(hashlib.sha256(order.tobytes()).hexdigest())
    del orders
    payload = {"dataset": args.dataset, "seed": seed, "samples": len(dataset),
               "epochs": common.EPOCHS, "initialization": common.file_snapshot(init),
               "initial_state_sha256": state_hash, "orders": common.file_snapshot(orders_path),
               "epoch_order_sha256": hashes, "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md"),
               "git": common.git_metadata()}
    common.write_json(audit_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
