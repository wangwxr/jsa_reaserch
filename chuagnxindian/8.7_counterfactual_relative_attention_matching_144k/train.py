#!/usr/bin/env python3
"""Stage-1 CRAM training; Stage 2 is deliberately absent from this entry point."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common
import objective


LAMBDA = {"C1": 0.1, "C2": 0.5, "C3": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(common.DATASETS), required=True)
    parser.add_argument("--group", choices=tuple(LAMBDA), required=True)
    parser.add_argument("--seed", choices=common.SEEDS, type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_map(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    low = flat.min(1).values[:, None, None, None]
    high = flat.max(1).values[:, None, None, None]
    return (values - low) / (high - low).clamp_min(1e-12)


def iou_values(score: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    prediction = score >= 0.6
    intersection = (prediction.float() * gt.float()).flatten(1).sum(1)
    false_positive = (prediction.float() * (gt == 0)).flatten(1).sum(1)
    return (intersection / (gt.flatten(1).sum(1) + false_positive).clamp_min(1)).cpu().numpy()


@torch.inference_mode()
def validate(model, dataset_name: str, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loader = DataLoader(common.test_dataset(dataset_name), batch_size=common.DATASETS[dataset_name]["eval_batch"], shuffle=False, num_workers=common.DATASETS[dataset_name]["workers"], pin_memory=True, drop_last=False, persistent_workers=False)
    results = {"AUD": [], "IQR": []}
    for frame, audio, gt, _ids, _labels in loader:
        image_map, audio_map = model(frame.to(device).float(), audio.to(device).float())
        image_map = normalize_map(F.interpolate(image_map, (224, 224), mode="bicubic", align_corners=False))
        audio_map = normalize_map(F.interpolate(audio_map, (224, 224), mode="bicubic", align_corners=False))
        results["AUD"].append(iou_values(audio_map[:, 0], torch.as_tensor(gt, device=device)))
        results["IQR"].append(iou_values(normalize_map(0.6 * audio_map + 0.4 * image_map)[:, 0], torch.as_tensor(gt, device=device)))
    model.train(was_training)
    thresholds = np.arange(21, dtype=float) * 0.05
    output = {}
    for name, parts in results.items():
        values = np.concatenate(parts)
        output[f"{name}_cIoU"] = float((values >= 0.5).mean())
        output[f"{name}_AUC"] = float(np.trapezoid([(values >= value).mean() for value in thresholds], thresholds))
    return output


def build_train_loader(dataset, orders: np.ndarray, epoch: int, seed: int, workers: int):
    sampler = common.EpochOrderSampler(orders)
    sampler.set_epoch(epoch)
    return DataLoader(dataset, batch_size=common.BATCH_SIZE, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True, persistent_workers=False, prefetch_factor=2 if workers else None, generator=torch.Generator().manual_seed(seed + 10_000_019 * epoch))


def gradient_l2(model: torch.nn.Module) -> float:
    return sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.grad is not None) ** 0.5


def negative_positions(offsets: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    positions = torch.from_numpy(np.asarray(offsets, dtype=np.int64)).to(device, non_blocking=True)
    if torch.any(positions < 0) or torch.any(positions >= batch_size):
        raise RuntimeError("Frozen negative offsets do not match this full training batch")
    return positions


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, args: argparse.Namespace, metrics: dict[str, float], initial_hash: str, negative_snapshot: dict[str, Any], include_training_state: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": "mufasa_ablation2_l3_l4_ablation",
        "experiment": "8.7_counterfactual_relative_attention_matching",
        "dataset": args.dataset, "group": args.group, "seed": args.seed, "epoch": epoch,
        "lambda_cram": LAMBDA[args.group], "negative_mapping": negative_snapshot,
        "model": model.state_dict(), "metrics": metrics,
        "initial_state_sha256": initial_hash,
        "model_state_sha256": common.state_sha256(model.state_dict()),
        "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md"),
    }
    if include_training_state:
        payload.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    run_dir = common.run_dir(args.dataset, args.group, args.seed)
    selected_path = common.checkpoint_path(args.dataset, args.group, args.seed, "best")
    final_path = common.checkpoint_path(args.dataset, args.group, args.seed, "final")
    latest_path = run_dir / "latest.pth"
    if (selected_path.exists() or final_path.exists()) and not args.resume:
        raise RuntimeError(f"Refusing to overwrite completed run {run_dir}")
    if args.resume and not latest_path.is_file():
        raise RuntimeError(f"Cannot resume without {latest_path}")
    init_path, orders_path = common.initialization_path(args.dataset, args.seed), common.order_path(args.dataset, args.seed)
    negative_path = common.HERE / "configs" / f"{args.dataset}_seed{args.seed}_negative_offsets.npy"
    negative_audit_path = negative_path.with_suffix(".json")
    if not all(path.is_file() for path in (init_path, orders_path, negative_path, negative_audit_path)):
        raise FileNotFoundError("Missing reused initialization/order or frozen CRAM negatives")
    common.setup_seed(args.seed)
    model = common.build_model(args.dataset, device, init_path, trainable=True)
    initial_hash = common.state_sha256(model.state_dict())
    if initial_hash != torch.load(init_path, map_location="cpu", weights_only=False)["state_sha256"]:
        raise RuntimeError("Canonical initialization hash mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=common.LR, weight_decay=common.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    orders = np.load(orders_path, mmap_mode="r")
    negatives = np.load(negative_path, mmap_mode="r")
    dataset = common.train_dataset(args.dataset)
    if orders.shape != (common.EPOCHS, len(dataset)) or negatives.shape != (common.EPOCHS, len(dataset), 4):
        raise RuntimeError("Frozen order/negative shape mismatch")
    start_epoch, best_iqr, validation, rows = 0, -math.inf, {}, []
    curve_path = getattr(common, "curve_root", lambda: common.HERE)() / "training_curves" / args.dataset / args.group / f"seed{args.seed}.csv"
    if args.resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True); optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_iqr = int(checkpoint["epoch"]), float(checkpoint["metrics"]["best_iqr"])
        validation = {k: float(v) for k, v in checkpoint["metrics"].items() if k != "best_iqr"}
        if curve_path.is_file():
            with curve_path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    negative_snapshot = common.file_snapshot(negative_path)
    started = time.time()
    for epoch in range(start_epoch, common.EPOCHS):
        epoch_start = time.time(); common.setup_seed(args.seed + 100_003 * epoch)
        loader = build_train_loader(dataset, orders, epoch, args.seed, common.DATASETS[args.dataset]["workers"])
        totals: dict[str, float] = {}; count = 0; grad_samples = []
        for batch_index, (frame, spec, _boxes, ids, _labels) in enumerate(loader):
            start = batch_index * common.BATCH_SIZE
            expected = [Path(dataset.image_files[int(index)]).stem for index in orders[epoch, start:start + len(ids)]]
            if [str(value) for value in ids] != expected:
                raise RuntimeError(f"Training sample order mismatch at epoch={epoch}, batch={batch_index}")
            frame, spec = frame.to(device, non_blocking=True).float(), spec.to(device, non_blocking=True).float()
            positions = negative_positions(negatives[epoch, start:start + len(spec)], len(spec), device)
            wrong = spec[positions]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                output = objective.forward_components(model, frame, spec)
                # Isolate CRAM's mask RNG so the original forward sequence remains C0-comparable.
                with torch.random.fork_rng(devices=[args.gpu]):
                    torch.manual_seed(args.seed + 87_000_001 * (epoch + 1) + batch_index)
                    torch.cuda.manual_seed_all(args.seed + 87_000_001 * (epoch + 1) + batch_index)
                    cram = objective.cram_components(model, output, wrong, source_spec=spec, negative_indices=positions)
                loss = objective.total_loss(output, cram["cram_loss"], LAMBDA[args.group])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            scaler.scale(loss).backward()
            if batch_index % 50 == 0:
                scaler.unscale_(optimizer); grad_samples.append(gradient_l2(model))
            scaler.step(optimizer); scaler.update()
            batch_size = len(spec); count += batch_size
            values = {"info_loss": output["info_loss"], "recon_loss": output["recon_loss"], "div_loss": output["div_loss"], "a2v_loss": output["a2v_loss"], "v2a_loss": output["v2a_loss"], "original_match_loss": output["match_loss"], "d_pos": cram["d_pos"], "d_neg": cram["d_neg"], "g_match": cram["g_match"], "cram_loss": cram["cram_loss"], "weighted_cram_loss": LAMBDA[args.group] * cram["cram_loss"], "total_loss": loss}
            for name, value in values.items(): totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            if batch_index % 50 == 0 or batch_index + 1 == len(loader):
                eta = (time.time() - epoch_start) / (batch_index + 1) * (len(loader) - batch_index - 1)
                print(f"{args.dataset} {args.group} seed{args.seed} epoch {epoch+1}/{common.EPOCHS} batch {batch_index+1}/{len(loader)} total={float(loss):.4f} d_pos={float(cram['d_pos']):.6f} d_neg={float(cram['d_neg']):.6f} gap={float(cram['g_match']):.6f} ETA={timedelta(seconds=int(eta))}", flush=True)
        del loader
        validation = validate(model, args.dataset, device)
        record: dict[str, Any] = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - epoch_start, "gradient_l2_mean": float(np.mean(grad_samples)), "parameter_l2": common.parameter_l2(model), **{name: value / count for name, value in totals.items()}, **validation}
        rows.append(record); print(json.dumps(record), flush=True)
        with curve_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record)); writer.writeheader(); writer.writerows(rows)
        if validation["IQR_cIoU"] > best_iqr:
            best_iqr = validation["IQR_cIoU"]; save_checkpoint(selected_path, model, optimizer, scaler, epoch + 1, args, validation, initial_hash, negative_snapshot)
        save_checkpoint(latest_path, model, optimizer, scaler, epoch + 1, args, {**validation, "best_iqr": best_iqr}, initial_hash, negative_snapshot, include_training_state=True)
    save_checkpoint(final_path, model, optimizer, scaler, common.EPOCHS, args, validation, initial_hash, negative_snapshot)
    payload = {"dataset": args.dataset, "group": args.group, "seed": args.seed, "lambda_cram": LAMBDA[args.group], "epochs": common.EPOCHS, "batch_size": common.BATCH_SIZE, "loss_weights": {"info": 1.0, "recon": 0.1, "div": 0.1, "match": 100.0, "cram": LAMBDA[args.group]}, "initialization": common.file_snapshot(init_path), "orders": common.file_snapshot(orders_path), "negative_mapping": negative_snapshot, "selected_checkpoint": common.file_snapshot(selected_path), "final_checkpoint": common.file_snapshot(final_path), "protocol_sha256": common.sha256(common.HERE / "PROTOCOL.md"), "git": common.git_metadata(), "elapsed_seconds": time.time() - started}
    common.write_json(run_dir / "config.json", payload); print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    train(parse_args())
