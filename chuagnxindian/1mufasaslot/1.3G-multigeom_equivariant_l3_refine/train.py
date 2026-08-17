#!/usr/bin/env python3
"""Experiment G: multi-geometry equivariant fine spatial refinement."""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
mpl_cache = Path("/tmp") / f"multigeom_l3_mpl_{os.getuid()}"
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
from geometry import (  # noqa: E402
    explicit_geometry,
    forward_grid,
    geometry_records,
    sample_random_resized_crop,
    warp_view_b_to_a,
)
from protocol import ProtocolAccumulator  # noqa: E402
from visualize import save_augmentation_panel, save_synthetic_geometry_panel  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "loss_coarse",
    "loss_equiv",
    "loss_total",
    "mean_valid_ratio",
    "skipped_small_overlap_samples",
    "actual_flip_ratio",
    "mean_crop_scale",
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
    parser.add_argument("--crop-scale-min", type=float, default=0.6)
    parser.add_argument("--crop-scale-max", type=float, default=1.0)
    parser.add_argument("--crop-ratio-min", type=float, default=0.9)
    parser.add_argument("--crop-ratio-max", type=float, default=1.1)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.2)
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


def run_geometry_sanity(output_dir: Path, device: torch.device) -> dict[str, float]:
    axis_y, axis_x = torch.meshgrid(
        torch.arange(14, device=device, dtype=torch.float32),
        torch.arange(14, device=device, dtype=torch.float32),
        indexing="ij",
    )
    asymmetric = (1.0 + 0.031 * axis_x + 0.047 * axis_y)[None, None]

    identity = explicit_geometry([0], [0], [224], [224], [False], 224, 224, device)
    identity_b = F.grid_sample(
        asymmetric, forward_grid(identity, 14, 14), align_corners=False
    )
    identity_back, identity_mask = warp_view_b_to_a(identity_b, identity)
    identity_error = float(((identity_back - asymmetric).abs() * identity_mask).max())

    flip = explicit_geometry([0], [0], [224], [224], [True], 224, 224, device)
    flip_b = F.grid_sample(asymmetric, forward_grid(flip, 14, 14), align_corners=False)
    old_flip_error = float((flip_b - torch.flip(asymmetric, dims=[-1])).abs().max())
    flip_back, flip_mask = warp_view_b_to_a(flip_b, flip)
    flip_recovery_error = float(((flip_back - asymmetric).abs() * flip_mask).max())

    crop_flip = explicit_geometry(
        [23], [17], [181], [193], [True], 224, 224, device
    )
    crop_b = F.grid_sample(
        asymmetric, forward_grid(crop_flip, 14, 14), align_corners=False
    )
    crop_back, crop_mask = warp_view_b_to_a(crop_b, crop_flip)
    crop_recovery_error = float(((crop_back - asymmetric).abs() * crop_mask).max())
    crop_valid_ratio = float(crop_mask.mean())
    save_synthetic_geometry_panel(
        output_dir / "synthetic_geometry_sanity.png",
        asymmetric[0],
        crop_b[0],
        crop_back[0],
        crop_mask[0],
    )

    checks = {
        "identity_inverse_max_abs_error": identity_error,
        "horizontal_flip_vs_experiment_f_max_abs_error": old_flip_error,
        "horizontal_flip_inverse_max_abs_error": flip_recovery_error,
        "crop_resize_flip_valid_recovery_max_abs_error": crop_recovery_error,
        "crop_resize_flip_valid_ratio": crop_valid_ratio,
    }
    if max(identity_error, old_flip_error, flip_recovery_error, crop_recovery_error) > 1e-5:
        raise RuntimeError(f"Synthetic geometry sanity failed: {checks}")
    return checks


def run_model_sanity(
    model,
    train_dataset,
    test_loader,
    config: argparse.Namespace,
    registry: dict,
    device: torch.device,
    arguments: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    print("Running Experiment G geometry and model sanity checks...", flush=True)
    geometry_checks = run_geometry_sanity(output_dir, device)
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
    frame, spec, _bboxes, names, _labels = next(iter(sanity_loader))
    frame = frame.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    geometry = sample_random_resized_crop(
        frame.shape[0],
        frame.shape[-2],
        frame.shape[-1],
        device,
        scale=(arguments.crop_scale_min, arguments.crop_scale_max),
        ratio=(arguments.crop_ratio_min, arguments.crop_ratio_max),
        flip_probability=arguments.flip_probability,
    )

    initial_student_state = copy.deepcopy(model.student.state_dict())
    model.train()
    model.zero_grad(set_to_none=True)
    output = model.forward_two_views(frame, spec, geometry)
    losses = model.spatial_losses(output, lambda_equiv=arguments.lambda_equiv)
    losses["loss_total"].backward()

    teacher_has_grad = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    initial_adapter_grad = gradient_l1(model.student.adapter.parameters())
    initial_proj3_grad = gradient_l1(model.student.proj3_spatial.parameters())
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
    for method in ("AUD_L4_A", "AUD_L4_B", "AUD_FINE_A", "AUD_FINE_B"):
        sums = output[method].sum(dim=(-2, -1))
        probability_errors[method] = float((sums - 1.0).abs().max())
    pooled_error = float(
        (
            model.sum_pool_2x2(output["AUD_FINE_A"]).sum(dim=(-2, -1)) - 1.0
        ).abs().max()
    )

    records = geometry_records(geometry)
    for index, record in enumerate(records):
        record["sample_id"] = str(names[index])
    (output_dir / "augmentation_geometry_samples.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    save_augmentation_panel(
        output_dir / "training_augmentation_audit.png",
        frame,
        output["VIEW_B"],
        output["AUD_FINE_A"],
        output["AUD_FINE_B"],
        output["AUD_FINE_B_TO_A"],
        output["VALID_MASK_14"],
        records,
    )

    # Zero initialization intentionally blocks proj3 on the first backward.
    # Take one temporary audit step, verify second-step gradient flow, then
    # restore the exact initial student weights before any formal training.
    audit_optimizer = torch.optim.AdamW(
        model.student.parameters(), lr=arguments.init_lr, weight_decay=arguments.weight_decay
    )
    audit_optimizer.step()
    audit_optimizer.zero_grad(set_to_none=True)
    second_output = model.forward_two_views(frame, spec, geometry)
    second_losses = model.spatial_losses(
        second_output, lambda_equiv=arguments.lambda_equiv
    )
    second_losses["loss_total"].backward()
    second_proj3_grad = gradient_l1(model.student.proj3_spatial.parameters())
    second_adapter_grad = gradient_l1(model.student.adapter.parameters())
    second_teacher_has_grad = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    model.student.load_state_dict(initial_student_state, strict=True)
    model.zero_grad(set_to_none=True)

    checks = {
        **geometry_checks,
        "teacher_aud_l4_reproduced": True,
        "teacher_has_any_gradient": teacher_has_grad or second_teacher_has_grad,
        "initial_adapter_gradient_l1": initial_adapter_grad,
        "initial_proj3_spatial_gradient_l1": initial_proj3_grad,
        "second_step_adapter_gradient_l1": second_adapter_grad,
        "second_step_proj3_spatial_gradient_l1": second_proj3_grad,
        "proj3_spatial_copy_max_abs_error": copy_error,
        "zero_init_f34_minus_up_f4_max_abs": zero_init_error,
        "f4_hook_minus_formal_tokens_max_abs": f4_token_error,
        "probability_sum_max_errors": probability_errors,
        "sum_pool_probability_sum_max_error": pooled_error,
        "mean_valid_ratio": float(losses["mean_valid_ratio"]),
        "skipped_small_overlap_samples": int(losses["skipped_small_overlap_samples"]),
        "loss_coarse": float(losses["loss_coarse"]),
        "loss_equiv": float(losses["loss_equiv"]),
        "loss_total": float(losses["loss_total"]),
        "student_restored_after_temporary_gradient_audit": True,
    }
    if checks["teacher_has_any_gradient"]:
        raise RuntimeError("Frozen teacher received a gradient")
    if initial_adapter_grad <= 0 or second_adapter_grad <= 0 or second_proj3_grad <= 0:
        raise RuntimeError("Student gradient audit failed")
    if copy_error != 0 or zero_init_error > 1e-7 or f4_token_error > 1e-7:
        raise RuntimeError("Initialization or feature-source audit failed")
    if max(probability_errors.values()) > 1e-5 or pooled_error > 1e-5:
        raise RuntimeError("A probability map is not normalized")
    if not all(torch.isfinite(losses[key]) for key in ("loss_coarse", "loss_equiv", "loss_total")):
        raise RuntimeError("A spatial loss is NaN or Inf")

    print(json.dumps(checks, indent=2), flush=True)
    print("All Experiment G sanity checks passed.", flush=True)
    return {"baseline_metrics": baseline_metrics, "checks": checks}


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    arguments: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    loss_fields = ("loss_coarse", "loss_equiv", "loss_total")
    totals = {field: 0.0 for field in loss_fields}
    valid_ratio_total = 0.0
    crop_scale_total = 0.0
    flipped_count = 0
    skipped_count = 0
    sample_count = 0
    batch_average = 0.0
    epoch_start = time.time()

    for batch_index, (frame, spec, _bboxes, _names, _labels) in enumerate(train_loader):
        batch_start = time.time()
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        geometry = sample_random_resized_crop(
            frame.shape[0],
            frame.shape[-2],
            frame.shape[-1],
            device,
            scale=(arguments.crop_scale_min, arguments.crop_scale_max),
            ratio=(arguments.crop_ratio_min, arguments.crop_ratio_max),
            flip_probability=arguments.flip_probability,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            output = model.forward_two_views(frame, spec, geometry)
            losses = model.spatial_losses(output, lambda_equiv=arguments.lambda_equiv)
        scaler.scale(losses["loss_total"]).backward()
        if any(parameter.grad is not None for parameter in model.teacher.parameters()):
            raise RuntimeError("Frozen teacher received a gradient during training")
        scaler.step(optimizer)
        scaler.update()

        batch_size = frame.shape[0]
        sample_count += batch_size
        flipped_count += int(geometry["flipped"].sum())
        skipped_count += int(losses["skipped_small_overlap_samples"])
        valid_ratio_total += float(losses["mean_valid_ratio"]) * batch_size
        crop_scale_total += float(output["mean_crop_scale"]) * batch_size
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
                f"total={float(losses['loss_total']):.8f} "
                f"valid={float(losses['mean_valid_ratio']):.3f} "
                f"skip={int(losses['skipped_small_overlap_samples'])} "
                f"flip={float(output['actual_flip_ratio']):.3f} "
                f"scale={float(output['mean_crop_scale']):.3f} ETA={eta}",
                flush=True,
            )

    result = {field: value / sample_count for field, value in totals.items()}
    result.update(
        {
            "mean_valid_ratio": valid_ratio_total / sample_count,
            "skipped_small_overlap_samples": skipped_count,
            "actual_flip_ratio": flipped_count / sample_count,
            "mean_crop_scale": crop_scale_total / sample_count,
            "epoch_seconds": time.time() - epoch_start,
        }
    )
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
        "architecture": "multi_geometry_equivariant_l3_refine",
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
        raise ValueError("--flip-probability must be in [0,1]")
    if not 0 < arguments.minimum_valid_ratio <= 1:
        raise ValueError("--minimum-valid-ratio must be in (0,1]")
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
    model, base_checkpoint = build_model(
        config, registry, device, minimum_valid_ratio=arguments.minimum_valid_ratio
    )
    counts = parameter_counts(model)
    print(json.dumps(counts, indent=2), flush=True)
    if counts["frozen_teacher_parameters"] != counts["total_teacher_parameters"]:
        raise RuntimeError("Not every teacher parameter is frozen")
    if counts["trainable_parameters"] != counts["spatial_student_parameters"]:
        raise RuntimeError("Trainable parameters are not exactly the spatial student")

    model_dir = arguments.model_dir / experiment_name
    protected = ("latest.pth", "final.pth", f"{registry['dataset']}_best.pth")
    if not arguments.sanity_only and any(
        (model_dir / filename).exists() for filename in protected
    ):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    audit_dir = (
        arguments.model_dir / "1.3G-multigeom_equivariant_l3_refine_sanity" / arguments.experiment
        if arguments.sanity_only
        else model_dir
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = build_datasets(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    sanity = run_model_sanity(
        model,
        train_dataset,
        test_loader,
        config,
        registry,
        device,
        arguments,
        audit_dir,
    )
    if arguments.sanity_only:
        model.close()
        print("Sanity-only mode: temporary audit update was restored; no training checkpoint written.")
        return

    run_config = {
        "architecture": "multi_geometry_equivariant_l3_refine",
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
        "random_resized_crop_scale": [arguments.crop_scale_min, arguments.crop_scale_max],
        "random_resized_crop_ratio": [arguments.crop_ratio_min, arguments.crop_ratio_max],
        "flip_probability": arguments.flip_probability,
        "minimum_valid_ratio": arguments.minimum_valid_ratio,
        "checkpoint_selection": "AUD_FINE_cIoU",
        "test_time_augmentation": False,
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
            arguments,
        )
        print(
            f"Epoch {epoch + 1}/{epochs} losses: "
            f"coarse={train_metrics['loss_coarse']:.8f} "
            f"equiv={train_metrics['loss_equiv']:.8f} "
            f"total={train_metrics['loss_total']:.8f} "
            f"valid={train_metrics['mean_valid_ratio']:.4f} "
            f"skipped={train_metrics['skipped_small_overlap_samples']} "
            f"flip={train_metrics['actual_flip_ratio']:.4f} "
            f"scale={train_metrics['mean_crop_scale']:.4f}",
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
                f"Best student saved at epoch {epoch + 1}: AUD_FINE cIoU={best_score:.4f}",
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
        f"Total Experiment G training time: {timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
