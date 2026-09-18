#!/usr/bin/env python3
"""Independent Stage-2+CRAM trainer; default is a no-write gradient sanity check."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
EXP87 = HERE.parent
ROOT = EXP87.parents[1]
RECIPE = ROOT / "chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine"
EXP86 = ROOT / "chuagnxindian/8.6_loss_to_decision_causal_ablation"
sys.path.insert(0, str(RECIPE))

import common as original_common
from geometry import sample_random_resized_crop
from protocol import ProtocolAccumulator


LAMBDAS = {"lambda025k": 25_000.0, "lambda050k": 50_000.0, "lambda100k": 100_000.0}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the local additive wrapper under an explicit module name only after the
# original recipe has imported its own `model` module through `common`.
Stage2WithCRAM = load_module("stage2_with_cram_model", HERE / "model.py").Stage2WithCRAM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def c3_teacher_path(dataset: str) -> Path:
    return EXP87 / f"checkpoints/C3/{dataset}/seed12345/selected_best.pth"


def build_model(dataset: str, device: torch.device):
    registry = copy.deepcopy(original_common.EXPERIMENTS[f"{dataset}_144k"])
    teacher_path = c3_teacher_path(dataset)
    registry["base_checkpoint"] = str(teacher_path)
    config = original_common.load_base_config(registry)
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    teacher = original_common.MUFASAL3L4(config)
    teacher.load_state_dict(state, strict=True)
    model = Stage2WithCRAM(teacher).to(device)
    return model, config, registry, teacher_path


def video_id(name: str, dataset: str) -> str:
    return name[:11] if dataset == "vggss" else name


def wrong_audio_batch(audio: torch.Tensor, names, dataset: str, epoch: int, batch: int) -> tuple[torch.Tensor, np.ndarray]:
    """K=4 deterministic, uniform, other-video batch negatives without labels."""
    rng = np.random.default_rng(12345 + 1_000_003 * epoch + 10_007 * batch)
    names = [str(name) for name in names]
    indices = np.empty((len(names), 4), dtype=np.int64)
    for anchor, name in enumerate(names):
        candidates = [index for index, other in enumerate(names) if video_id(name, dataset) != video_id(other, dataset)]
        if len(candidates) < 4:
            raise RuntimeError("Batch does not contain four different-video negatives")
        indices[anchor] = rng.choice(candidates, size=4, replace=False)
    positions = torch.from_numpy(indices).to(audio.device)
    return audio[positions], indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--variant", choices=tuple(LAMBDAS), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--run-train", action="store_true")
    return parser.parse_args()


def gradient_l2(parameters) -> float:
    return sum(float(p.grad.detach().float().square().sum()) for p in parameters if p.grad is not None) ** .5


def validate(model, loader, device: torch.device) -> dict:
    model.eval(); accum = ProtocolAccumulator()
    with torch.inference_mode():
        for image, audio, gt, names, _labels in loader:
            image, audio, gt, names = original_common.flatten_eval_batch(image, audio, gt, names)
            output = model(image.to(device).float(), audio.to(device).float())
            accum.update(output["AUD_FINE"], gt, names)
    return accum.finalize()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu); device = torch.device(f"cuda:{args.gpu}")
    original_common.setup_seed(12345)
    model, config, registry, teacher_path = build_model(args.dataset, device)
    audit_common = load_module("stage2_cram_eval_cache", EXP86 / "common.py")
    train_dataset = original_common.get_train_dataset(config, hard_img=config.hard_img, hard_aud=config.hard_aud, rand_aud=config.rand_aud)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    image, audio, _gt, names, _labels = next(iter(loader))
    image, audio = image.to(device).float(), audio.to(device).float()
    geometry = sample_random_resized_crop(len(image), 224, 224, device, scale=(.6, 1.0), ratio=(.9, 1.1), flip_probability=.5)
    wrong, indices = wrong_audio_batch(audio, names, args.dataset, epoch=0, batch=0)
    model.train(); model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        output = model.forward_two_views(image, audio, geometry)
        original = model.spatial_losses(output, lambda_equiv=1.0)
        cram = model.stage2_cram_components(output, wrong)
        total = original["loss_total"] + LAMBDAS[args.variant] * cram["cram_loss"]
    key_gradient = torch.autograd.grad(
        cram["cram_loss"], output["FINE_KEYS_A"], retain_graph=True, allow_unused=True
    )[0]
    # Match the production mixed-precision path. At the zero-initialized
    # adapter, direct fp16 backward can underflow although the scaled training
    # gradient is valid.
    sanity_scaler = torch.amp.GradScaler("cuda")
    sanity_scaler.scale(total).backward()
    total_gradient = gradient_l2(model.student.parameters())
    # Separate scaled gradient norms document whether a Stage-1-derived lambda
    # is on a plausible Stage-2 scale before any epoch is allowed to start.
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        original_output = model.forward_two_views(image, audio, geometry)
        original_only = model.spatial_losses(original_output, lambda_equiv=1.0)["loss_total"]
    sanity_scaler.scale(original_only).backward()
    original_gradient = gradient_l2(model.student.parameters())
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        cram_output = model.forward_two_views(image, audio, geometry)
        cram_only = model.stage2_cram_components(cram_output, wrong)["cram_loss"]
    sanity_scaler.scale(cram_only).backward()
    cram_gradient = gradient_l2(model.student.parameters())
    # A large read-only probe exposes the raw derivative scale even if AMP's
    # normal initial scale rounds it to zero at the fp16 F34 boundary.
    model.zero_grad(set_to_none=True)
    probe_scale = 1e8
    with torch.amp.autocast("cuda"):
        probe_output = model.forward_two_views(image, audio, geometry)
        probe_cram = model.stage2_cram_components(probe_output, wrong)["cram_loss"]
    sanity_scaler.scale(probe_scale * probe_cram).backward()
    cram_probe_gradient = gradient_l2(model.student.parameters())
    checks = {
        "dataset": args.dataset, "variant": args.variant, "lambda_stage2_cram": LAMBDAS[args.variant],
        "teacher_checkpoint": str(teacher_path.resolve()), "teacher_sha256": sha256(teacher_path),
        "batch_size": len(image), "wrong_audio_k": 4,
        "different_video_negatives": bool(all(video_id(str(names[i]), args.dataset) != video_id(str(names[j]), args.dataset) for i in range(len(names)) for j in indices[i])),
        "teacher_has_gradient": any(p.grad is not None for p in model.teacher.parameters()),
        "student_gradient_l2_scaled": total_gradient,
        "fine_keys_requires_grad": bool(output["FINE_KEYS_A"].requires_grad),
        "cram_loss_requires_grad": bool(cram["cram_loss"].requires_grad),
        "cram_fine_keys_gradient_l2": float(key_gradient.float().square().sum().sqrt()) if key_gradient is not None else 0.0,
        "original_gradient_l2_scaled": original_gradient,
        "cram_gradient_l2_scaled": cram_gradient,
        "cram_gradient_l2_scaled_probe_x1e8": cram_probe_gradient,
        "estimated_cram_to_original_gradient_ratio": cram_probe_gradient / max(probe_scale * original_gradient, 1e-30),
        "estimated_weighted_cram_to_original_gradient_ratio": LAMBDAS[args.variant] * cram_probe_gradient / max(probe_scale * original_gradient, 1e-30),
        "weighted_cram_gradient_l2_scaled": LAMBDAS[args.variant] * cram_gradient,
        "weighted_cram_to_original_gradient_ratio": LAMBDAS[args.variant] * cram_gradient / max(original_gradient, 1e-30),
        "loss_coarse": float(original["loss_coarse"]), "loss_equiv": float(original["loss_equiv"]),
        "original_loss": float(original["loss_total"]), "d_pos": float(cram["d_pos"]),
        "d_neg": float(cram["d_neg"]), "g_match": float(cram["g_match"]),
        "cram_loss": float(cram["cram_loss"]), "total_loss": float(total),
        "finite": bool(torch.isfinite(total)), "optimizer_created": False,
    }
    if not checks["different_video_negatives"] or checks["teacher_has_gradient"] or checks["student_gradient_l2_scaled"] <= 0 or checks["estimated_weighted_cram_to_original_gradient_ratio"] <= 0 or not checks["finite"]:
        raise RuntimeError(f"Stage-2+CRAM sanity failed: {checks}")
    output_dir = HERE / "sanity" / args.dataset / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sanity.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if args.sanity_only:
        model.close()
        return
    if not args.run_train:
        print("Sanity passed. Add --run-train to begin this independent variant.")
        model.close()
        return

    run_dir = HERE / "checkpoints" / args.variant / args.dataset / "seed12345"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    eval_dataset = audit_common.test_dataset(args.dataset)
    eval_loader = DataLoader(
        eval_dataset, batch_size=registry["eval_batch_size"], shuffle=False,
        num_workers=registry["workers"], pin_memory=True, drop_last=False,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=5e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    manifest = {
        "experiment": f"8.7_stage2_cram_{int(LAMBDAS[args.variant])}", "dataset": args.dataset,
        "variant": args.variant, "lambda_stage2_cram": LAMBDAS[args.variant],
        "seed": 12345, "epochs": args.epochs, "batch_size": config.batch_size,
        "teacher_checkpoint": str(teacher_path.resolve()), "teacher_sha256": sha256(teacher_path),
        "original_stage2_loss": "L_coarse + 1.0*L_equiv", "cram": "Stage-1 direct relative attention matching on K34",
        "negative_sampling": {"K": 4, "source": "current original Stage-2 batch", "different_video": True,
                              "seed_formula": "12345 + 1000003*epoch + 10007*batch"},
        "optimizer": "AdamW", "lr": 5e-5, "weight_decay": .01,
        "stage2_architecture_modified": False, "stage2_original_losses_preserved": True,
        "source_sha256": {path.name: sha256(path) for path in (HERE / "train.py", HERE / "model.py", HERE / "PROTOCOL.md")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sanity": checks,
    }
    (run_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    history_path = run_dir / "epoch_metrics.csv"
    history, best = [], -math.inf
    # The loader, geometry, optimizer, and original spatial loss path are retained
    # from original Stage-2. Only the deterministic K=4 wrong-audio branch is added.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.workers, pin_memory=True, drop_last=True,
                              persistent_workers=config.workers > 0,
                              prefetch_factor=2 if config.workers > 0 else None)
    for epoch in range(args.epochs):
        model.train(); totals = {key: 0.0 for key in ("loss_coarse", "loss_equiv", "original_loss", "cram_loss", "weighted_cram_loss", "d_pos", "d_neg", "g_match", "total_loss")}; samples = 0
        epoch_start = time.time()
        epoch_anchor_ids, epoch_wrong_ids = [], []
        grad_norm_sum, grad_norm_min, grad_norm_max, grad_norm_count = 0.0, math.inf, 0.0, 0
        for batch_index, (image, audio, _gt, names, _labels) in enumerate(train_loader):
            image, audio = image.to(device, non_blocking=True).float(), audio.to(device, non_blocking=True).float()
            geometry = sample_random_resized_crop(len(image), image.shape[-2], image.shape[-1], device,
                                                   scale=(.6, 1.0), ratio=(.9, 1.1), flip_probability=.5)
            wrong, _indices = wrong_audio_batch(audio, names, args.dataset, epoch, batch_index)
            epoch_anchor_ids.extend(str(name) for name in names)
            epoch_wrong_ids.extend([[str(names[index]) for index in row] for row in _indices])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                output = model.forward_two_views(image, audio, geometry)
                original = model.spatial_losses(output, lambda_equiv=1.0)
                cram = model.stage2_cram_components(output, wrong)
                total = original["loss_total"] + LAMBDAS[args.variant] * cram["cram_loss"]
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_index}")
            scaler.scale(total).backward()
            if any(parameter.grad is not None for parameter in model.teacher.parameters()):
                raise RuntimeError("Frozen teacher received gradient")
            scaler.unscale_(optimizer)
            grad_norm = gradient_l2(model.student.parameters())
            if not math.isfinite(grad_norm) or grad_norm > 1e3:
                raise RuntimeError(f"Gradient explosion at epoch={epoch} batch={batch_index}: {grad_norm}")
            grad_norm_sum += grad_norm
            grad_norm_min = min(grad_norm_min, grad_norm)
            grad_norm_max = max(grad_norm_max, grad_norm)
            grad_norm_count += 1
            scaler.step(optimizer); scaler.update()
            batch_size = len(image); samples += batch_size
            values = {"loss_coarse": original["loss_coarse"], "loss_equiv": original["loss_equiv"],
                      "original_loss": original["loss_total"], "cram_loss": cram["cram_loss"],
                      "weighted_cram_loss": LAMBDAS[args.variant] * cram["cram_loss"],
                      "d_pos": cram["d_pos"], "d_neg": cram["d_neg"], "g_match": cram["g_match"], "total_loss": total}
            for key, value in values.items(): totals[key] += float(value.detach()) * batch_size
        metrics = validate(model, eval_loader, device)
        record = {"epoch": epoch + 1, "epoch_seconds": time.time() - epoch_start,
                  "lambda_stage2_cram": LAMBDAS[args.variant], **{key: value / samples for key, value in totals.items()},
                  "student_gradient_l2_mean": grad_norm_sum / max(grad_norm_count, 1),
                  "student_gradient_l2_min": grad_norm_min,
                  "student_gradient_l2_max": grad_norm_max,
                  "aud_fine_ciou": metrics["cIoU"], "aud_fine_auc": metrics["AUC"]}
        history.append(record)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record)); writer.writeheader(); writer.writerows(history)
        np.savez_compressed(
            run_dir / f"negative_audio_mapping_epoch{epoch + 1:03d}.npz",
            anchor_ids=np.asarray(epoch_anchor_ids), wrong_ids=np.asarray(epoch_wrong_ids),
        )
        state = {"architecture": "original_1.3G_stage2_plus_additive_cram", "epoch": epoch + 1,
                 "lambda_stage2_cram": LAMBDAS[args.variant], "metrics": metrics,
                 "proj3_spatial_state_dict": model.student.proj3_spatial.state_dict(),
                 "topdown_adapter_state_dict": model.student.adapter.state_dict(),
                 "teacher_checkpoint": str(teacher_path.resolve()), "teacher_sha256": sha256(teacher_path),
                 "protocol_sha256": sha256(HERE / "PROTOCOL.md")}
        torch.save(state, run_dir / "latest.pth")
        if metrics["cIoU"] > best:
            best = metrics["cIoU"]
            torch.save({**state, "selection_metric": "AUD_FINE_cIoU", "selection_score": best},
                       run_dir / f"{args.dataset}_best.pth")
        print(json.dumps(record), flush=True)
    torch.save(state, run_dir / "final.pth")
    model.close()


if __name__ == "__main__":
    main()
