#!/usr/bin/env python3
"""Experiment F: frozen semantic teacher plus equivariant L3 refinement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
mpl_cache = Path("/tmp") / f"equivariant_l3_mpl_{os.getuid()}"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    EXPERIMENTS,
    PROJECT_ROOT,
    build_datasets,
    build_model,
    build_test_loader,
    build_train_loader,
    flatten_eval_batch,
    load_base_config,
    parameter_counts,
    setup_seed,
)
from curves import render_curves  # noqa: E402
from protocol import ProtocolAccumulator  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "flip_fraction",
    "loss_equiv",
    "loss_coarse_a",
    "loss_coarse_b",
    "loss_coarse",
    "loss_spatial",
    "aud_l4_ciou",
    "aud_l4_auc",
    "aud_fine_ciou",
    "aud_fine_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lambda-equiv", type=float, default=1.0)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--init-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def validate(model, test_loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    accumulators = {
        "AUD_L4": ProtocolAccumulator(),
        "AUD_FINE": ProtocolAccumulator(),
    }
    for image, spec, bboxes, names, _labels in tqdm(
        test_loader, desc="Validate", dynamic_ncols=True
    ):
        image, spec, bboxes, names = flatten_eval_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = model(image, spec)
        for method, accumulator in accumulators.items():
            accumulator.update(output[method], bboxes, names)
    return {method: accumulator.finalize() for method, accumulator in accumulators.items()}


def print_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    for method in ("AUD_L4", "AUD_FINE"):
        value = metrics[method]
        print(
            f"{prefix}{method}/cIoU,AUC {value['cIoU']:.4f} {value['AUC']:.4f}",
            flush=True,
        )


def metric_matches(observed: dict[str, float], expected: tuple[float, float]) -> bool:
    return (
        f"{observed['cIoU']:.4f}" == f"{expected[0]:.4f}"
        and f"{observed['AUC']:.4f}" == f"{expected[1]:.4f}"
    )


def gradient_l1(parameters) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in parameters
        if parameter.grad is not None
    )


def run_sanity_checks(
    model,
    train_dataset,
    test_loader,
    config: argparse.Namespace,
    registry: dict,
    device: torch.device,
    lambda_equiv: float,
) -> dict[str, Any]:
    print("Running Experiment F pre-training sanity checks...", flush=True)
    baseline_metrics = validate(model, test_loader, device)
    print_metrics(baseline_metrics, prefix="SANITY/")
    if not metric_matches(baseline_metrics["AUD_L4"], registry["expected_aud"]):
        raise RuntimeError("Frozen teacher AUD_L4 does not reproduce the formal result")

    sanity_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, 8),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    frame, spec, _bboxes, _names, _labels = next(iter(sanity_loader))
    frame = frame.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    flip_mask = torch.arange(frame.shape[0], device=device) % 2 == 0

    model.train()
    model.zero_grad(set_to_none=True)
    output = model.forward_two_views(frame, spec, flip_mask)
    losses = model.spatial_losses(output, lambda_equiv=lambda_equiv)
    losses["loss_spatial"].backward()

    teacher_has_grad = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    student_grad = gradient_l1(model.student.parameters())
    adapter_last_grad = gradient_l1(model.student.adapter.layers[-1].parameters())
    proj3_grad = gradient_l1(model.student.proj3_spatial.parameters())
    copy_error = max(
        float((student_value.detach() - teacher_value.detach()).abs().max())
        for student_value, teacher_value in zip(
            model.student.proj3_spatial.parameters(),
            model.teacher.imgnet.proj3.parameters(),
        )
    )
    zero_init_error = float((output["F34_A"] - output["F4_UP_A"]).abs().max())
    f4_token_error = float(output["f4_token_error"])

    probability_errors = {}
    for method in (
        "AUD_L4_A",
        "AUD_L4_B_ALIGNED",
        "AUD_FINE_A",
        "AUD_FINE_B_ALIGNED",
    ):
        sums = output[method].sum(dim=(-2, -1))
        probability_errors[method] = float((sums - 1.0).abs().max())
    pooled = model.sum_pool_2x2(output["AUD_FINE_A"])
    pooled_error = float((pooled.sum(dim=(-2, -1)) - 1.0).abs().max())

    checks = {
        "teacher_aud_l4_reproduced": True,
        "teacher_has_any_gradient": teacher_has_grad,
        "student_gradient_l1": student_grad,
        "adapter_last_conv_gradient_l1": adapter_last_grad,
        "proj3_spatial_initial_gradient_l1": proj3_grad,
        "proj3_spatial_copy_max_abs_error": copy_error,
        "zero_init_f34_minus_up_f4_max_abs": zero_init_error,
        "f4_hook_minus_formal_tokens_max_abs": f4_token_error,
        "probability_sum_max_errors": probability_errors,
        "sum_pool_probability_sum_max_error": pooled_error,
        "note": (
            "The copied proj3 has zero gradient on the very first backward because "
            "the adapter's last convolution is zero-initialized; gradients reach it "
            "after that convolution receives its first update."
        ),
    }
    if teacher_has_grad:
        raise RuntimeError("Frozen teacher received a gradient")
    if student_grad <= 0 or adapter_last_grad <= 0:
        raise RuntimeError("Spatial student did not receive a nonzero gradient")
    if copy_error != 0:
        raise RuntimeError(f"proj3_spatial was not copied exactly: {copy_error}")
    if zero_init_error > 1e-7:
        raise RuntimeError(f"Adapter zero-init check failed: {zero_init_error}")
    if f4_token_error > 1e-7:
        raise RuntimeError(f"F4 source mismatch: {f4_token_error}")
    if max(probability_errors.values()) > 1e-5 or pooled_error > 1e-5:
        raise RuntimeError("A spatial probability distribution is not normalized")
    if not torch.isfinite(losses["loss_spatial"]):
        raise RuntimeError("Spatial loss is not finite")

    model.zero_grad(set_to_none=True)
    print(json.dumps(checks, indent=2), flush=True)
    print("All Experiment F sanity checks passed.", flush=True)
    return {"baseline_metrics": baseline_metrics, "checks": checks}


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    lambda_equiv: float,
    flip_probability: float,
) -> dict[str, float]:
    model.train()
    loss_fields = (
        "loss_equiv",
        "loss_coarse_a",
        "loss_coarse_b",
        "loss_coarse",
        "loss_spatial",
    )
    totals = {field: 0.0 for field in loss_fields}
    sample_count = 0
    flipped_count = 0
    batch_average = 0.0
    epoch_start = time.time()

    for batch_index, (frame, spec, _bboxes, _names, _labels) in enumerate(train_loader):
        batch_start = time.time()
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        flip_mask = torch.rand(frame.shape[0], device=device) < flip_probability

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            output = model.forward_two_views(frame, spec, flip_mask)
            losses = model.spatial_losses(output, lambda_equiv=lambda_equiv)
        scaler.scale(losses["loss_spatial"]).backward()
        if any(parameter.grad is not None for parameter in model.teacher.parameters()):
            raise RuntimeError("Frozen teacher received a gradient during training")
        scaler.step(optimizer)
        scaler.update()

        batch_size = frame.shape[0]
        sample_count += batch_size
        flipped_count += int(flip_mask.sum())
        for field in loss_fields:
            totals[field] += float(losses[field].detach()) * batch_size

        completed = batch_index + 1
        batch_seconds = time.time() - batch_start
        batch_average += (batch_seconds - batch_average) / completed
        if batch_index % 10 == 0 or completed == len(train_loader):
            remaining_batches = (
                len(train_loader) - completed
                + (total_epochs - epoch - 1) * len(train_loader)
            )
            eta = timedelta(seconds=int(batch_average * remaining_batches))
            print(
                f"Train [{epoch + 1}/{total_epochs}] [{completed}/{len(train_loader)}] "
                f"coarse={float(losses['loss_coarse']):.8f} "
                f"equiv={float(losses['loss_equiv']):.8f} "
                f"total={float(losses['loss_spatial']):.8f} "
                f"flip={float(flip_mask.float().mean()):.3f} ETA={eta}",
                flush=True,
            )

    result = {field: value / sample_count for field, value in totals.items()}
    result["flip_fraction"] = flipped_count / sample_count
    result["epoch_seconds"] = time.time() - epoch_start
    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, Any],
    base_checkpoint: Path,
    selection_metric: str | None = None,
) -> None:
    checkpoint = {
        "architecture": "frozen_semantic_teacher_equivariant_l3_refine",
        "base_checkpoint_path": str(base_checkpoint),
        "proj3_spatial_state_dict": model.student.proj3_spatial.state_dict(),
        "topdown_adapter_state_dict": model.student.adapter.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if selection_metric is not None:
        checkpoint["selection_metric"] = selection_metric
        checkpoint["selection_score"] = metrics["AUD_FINE"]["cIoU"]
    torch.save(checkpoint, path)


def append_history(path: Path, record: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    arguments = parse_args()
    if not 0 <= arguments.flip_probability <= 1:
        raise ValueError("--flip-probability must be in [0, 1]")
    registry = EXPERIMENTS[arguments.experiment]
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    epochs = config.epochs if arguments.epochs is None else arguments.epochs
    experiment_name = arguments.experiment_name or registry["default_experiment"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    counts = parameter_counts(model)
    print(json.dumps(counts, indent=2), flush=True)
    if counts["frozen_teacher_parameters"] != counts["total_teacher_parameters"]:
        raise RuntimeError("Not every teacher parameter is frozen")
    if counts["trainable_parameters"] != counts["spatial_student_parameters"]:
        raise RuntimeError("Trainable parameters are not exactly the spatial student")

    train_dataset, test_dataset = build_datasets(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    sanity = run_sanity_checks(
        model,
        train_dataset,
        test_loader,
        config,
        registry,
        device,
        arguments.lambda_equiv,
    )
    if arguments.sanity_only:
        model.close()
        print("Sanity-only mode: no optimizer step and no checkpoint written.")
        return

    model_dir = arguments.model_dir / experiment_name
    protected = ("latest.pth", "final.pth", f"{registry['dataset']}_best.pth")
    if any((model_dir / filename).exists() for filename in protected):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "architecture": "frozen_semantic_teacher_equivariant_l3_refine",
        "experiment_key": arguments.experiment,
        "experiment_name": experiment_name,
        "base_experiment": registry["base_experiment"],
        "base_checkpoint_path": str(base_checkpoint),
        "epochs": epochs,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "optimizer": "AdamW",
        "init_lr": arguments.init_lr,
        "weight_decay": arguments.weight_decay,
        "scheduler": False,
        "lambda_equiv": arguments.lambda_equiv,
        "flip_probability": arguments.flip_probability,
        "checkpoint_selection": "AUD_FINE_cIoU",
        "seed": config.seed,
        "train_data_path": config.train_data_path,
        "train_manifest_path": config.train_manifest_path,
        "test_data_path": config.test_data_path,
        "test_manifest_path": config.test_manifest_path,
        "test_gt_path": config.test_gt_path,
        "parameter_counts": counts,
        "sanity_checks": sanity["checks"],
        "forbidden_inputs": ["GT localization", "OGL", "OBJ_PRIOR"],
    }
    (model_dir / "configs.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (model_dir / "sanity_checks.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )

    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=arguments.init_lr,
        weight_decay=arguments.weight_decay,
    )
    optimized = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if optimized != counts["spatial_student_parameters"]:
        raise RuntimeError("Optimizer is not restricted to proj3_spatial + adapter")
    scaler = torch.amp.GradScaler("cuda")
    setup_seed(config.seed)
    train_loader = build_train_loader(train_dataset, config)

    history_path = model_dir / "epoch_metrics.csv"
    best_score = -math.inf
    run_start = time.time()
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            epochs,
            arguments.lambda_equiv,
            arguments.flip_probability,
        )
        print(
            f"Epoch {epoch + 1}/{epochs} losses: "
            f"coarse={train_metrics['loss_coarse']:.8f} "
            f"equiv={train_metrics['loss_equiv']:.8f} "
            f"total={train_metrics['loss_spatial']:.8f} "
            f"flip={train_metrics['flip_fraction']:.4f}",
            flush=True,
        )
        validation = validate(model, test_loader, device)
        print_metrics(validation, prefix=f"Epoch{epoch + 1}/")
        save_checkpoint(
            model_dir / "latest.pth",
            model,
            optimizer,
            epoch + 1,
            validation,
            base_checkpoint,
        )
        if validation["AUD_FINE"]["cIoU"] > best_score:
            best_score = validation["AUD_FINE"]["cIoU"]
            save_checkpoint(
                model_dir / f"{registry['dataset']}_best.pth",
                model,
                optimizer,
                epoch + 1,
                validation,
                base_checkpoint,
                selection_metric="AUD_FINE_cIoU",
            )
            print(
                f"Best student saved at epoch {epoch + 1}: "
                f"AUD_FINE cIoU={best_score:.4f}",
                flush=True,
            )

        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "aud_l4_ciou": validation["AUD_L4"]["cIoU"],
            "aud_l4_auc": validation["AUD_L4"]["AUC"],
            "aud_fine_ciou": validation["AUD_FINE"]["cIoU"],
            "aud_fine_auc": validation["AUD_FINE"]["AUC"],
        }
        append_history(history_path, record)
        render_curves(
            history_path,
            model_dir / "training_curves",
            experiment_name,
            sanity["baseline_metrics"]["AUD_L4"]["cIoU"],
            sanity["baseline_metrics"]["AUD_L4"]["AUC"],
        )
        elapsed = time.time() - run_start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{epochs} complete; overall ETA "
            f"{timedelta(seconds=int(remaining))}; estimated finish "
            f"{(datetime.now() + timedelta(seconds=remaining)):%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    final_metrics = validate(model, test_loader, device)
    save_checkpoint(
        model_dir / "final.pth",
        model,
        optimizer,
        epochs,
        final_metrics,
        base_checkpoint,
    )

    best_checkpoint = torch.load(
        model_dir / f"{registry['dataset']}_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    model.student.proj3_spatial.load_state_dict(
        best_checkpoint["proj3_spatial_state_dict"], strict=True
    )
    model.student.adapter.load_state_dict(
        best_checkpoint["topdown_adapter_state_dict"], strict=True
    )
    best_metrics = validate(model, test_loader, device)
    print_metrics(best_metrics, prefix="BEST/")
    (model_dir / "best_test_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )
    print(
        f"Total Experiment F training time: "
        f"{timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
