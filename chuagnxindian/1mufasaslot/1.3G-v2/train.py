#!/usr/bin/env python3
"""Train 1.3G-v2 with one added frozen-Q4 visual coarse KL constraint."""

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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
G_ROOT = HERE.parent / "1.3G-multigeom_equivariant_l3_refine"
sys.path.insert(0, str(HERE))
if str(G_ROOT) not in sys.path:
    sys.path.append(str(G_ROOT))
mpl_cache = Path("/tmp") / f"1_3g_v2_mpl_{os.getuid()}"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

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
from geometry import sample_random_resized_crop  # noqa: E402
from model import BaseMultiGeometryRefinement  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402
from protocol import ProtocolAccumulator  # noqa: E402


EVAL_METHODS = ("AUD_L4", "AUD_FINE", "IMG_L4", "IMG_FINE", "IQR_FINE")
DIAGNOSTIC_GRAD_SCALE = float(2**16)
HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "loss_aud_coarse",
    "loss_aud_equiv",
    "loss_img_coarse",
    "loss_total",
    "mean_valid_ratio",
    "skipped_small_overlap_samples",
    "actual_flip_ratio",
    "mean_crop_scale",
    "grad_norm_aud",
    "grad_norm_img",
    "grad_norm_ratio",
    "grad_cosine",
    "negative_grad_cosine_ratio",
    "gradient_diagnostic_batches",
    "aud_l4_ciou",
    "aud_l4_auc",
    "aud_fine_ciou",
    "aud_fine_auc",
    "img_l4_ciou",
    "img_l4_auc",
    "img_fine_ciou",
    "img_fine_auc",
    "iqr_fine_ciou",
    "iqr_fine_auc",
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
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=20)
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def validate(
    model,
    test_loader,
    device: torch.device,
    collect_samples: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    accumulators = {method: ProtocolAccumulator() for method in EVAL_METHODS}
    sample_rows: list[dict[str, Any]] = []
    for image, spec, bboxes, names, _labels in tqdm(
        test_loader, desc="Validate", dynamic_ncols=True
    ):
        image, spec, bboxes, names = flatten_eval_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = model(image, spec)
        maps = {
            "AUD_L4": output["AUD_L4"],
            "AUD_FINE": output["AUD_FINE"],
            "IMG_L4": output["IMG_L4"],
            "IMG_FINE": output["IMG_FINE"],
            # Native 14x14 mixture; ProtocolAccumulator applies the unchanged
            # resize and min-max normalization used by 5.3.
            "IQR_FINE": 0.6 * output["AUD_FINE"] + 0.4 * output["IMG_FINE"],
        }
        batch_values: dict[str, list[float]] = {}
        for method, accumulator in accumulators.items():
            before = len(accumulator.sample_ious)
            accumulator.update(maps[method], bboxes, names)
            batch_values[method] = accumulator.sample_ious[before:]
        if collect_samples:
            for index, name in enumerate(names):
                sample_rows.append(
                    {
                        "sample_index": len(sample_rows),
                        "sample_id": str(name),
                        **{
                            f"IoU_{method}": float(batch_values[method][index])
                            for method in EVAL_METHODS
                        },
                    }
                )
    return (
        {method: accumulator.finalize() for method, accumulator in accumulators.items()},
        sample_rows,
    )


def print_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    for method in EVAL_METHODS:
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


def gradients_for(loss: torch.Tensor, parameters: list[torch.nn.Parameter]):
    # Formal backward uses GradScaler.  The diagnostic must likewise avoid
    # FP16 underflow, but it must not touch parameter.grad or GradScaler state.
    scaled = torch.autograd.grad(
        loss * DIAGNOSTIC_GRAD_SCALE,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    return tuple(
        None if gradient is None else gradient / DIAGNOSTIC_GRAD_SCALE
        for gradient in scaled
    )


def gradient_pair_statistics(
    aud_grads: tuple[torch.Tensor | None, ...],
    img_grads: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    device = next(
        gradient.device
        for gradient in (*aud_grads, *img_grads)
        if gradient is not None
    )
    dot = torch.zeros((), device=device, dtype=torch.float64)
    aud_sq = torch.zeros_like(dot)
    img_sq = torch.zeros_like(dot)
    for aud_grad, img_grad in zip(aud_grads, img_grads):
        if aud_grad is not None:
            aud = aud_grad.detach().double()
            aud_sq += torch.sum(aud * aud)
        if img_grad is not None:
            img = img_grad.detach().double()
            img_sq += torch.sum(img * img)
        if aud_grad is not None and img_grad is not None:
            dot += torch.sum(aud_grad.detach().double() * img_grad.detach().double())
    aud_norm = torch.sqrt(aud_sq)
    img_norm = torch.sqrt(img_sq)
    cosine = dot / (aud_norm * img_norm + 1e-12)
    return {
        "grad_norm_aud": float(aud_norm),
        "grad_norm_img": float(img_norm),
        "grad_norm_ratio": float(img_norm / (aud_norm + 1e-12)),
        "grad_cosine": float(cosine),
    }


def build_original_g_model(
    config: argparse.Namespace,
    base_checkpoint: Path,
    device: torch.device,
    minimum_valid_ratio: float,
    student_state: dict[str, Any],
):
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    teacher = MUFASAL3L4(config)
    teacher.load_state_dict(
        {key.replace("module.", ""): value for key, value in checkpoint["model"].items()},
        strict=True,
    )
    base_model = BaseMultiGeometryRefinement(
        teacher, minimum_valid_ratio=minimum_valid_ratio
    ).to(device)
    base_model.student.load_state_dict(student_state, strict=True)
    base_model.train()
    return base_model


def run_sanity(
    model,
    train_dataset,
    test_loader,
    config: argparse.Namespace,
    registry: dict,
    base_checkpoint: Path,
    device: torch.device,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    print("Running 1.3G-v2 regression and gradient sanity...", flush=True)
    baseline_metrics, _ = validate(model, test_loader, device)
    print_metrics(baseline_metrics, prefix="SANITY/")
    if not metric_matches(baseline_metrics["AUD_L4"], registry["expected_aud"]):
        raise RuntimeError("Frozen teacher AUD_L4 did not reproduce Stage-1")

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
    base_model = build_original_g_model(
        config,
        base_checkpoint,
        device,
        arguments.minimum_valid_ratio,
        initial_student_state,
    )
    model.train()
    with torch.no_grad():
        base_output = base_model.forward_two_views(frame, spec, geometry)
        v2_output = model.forward_two_views(frame, spec, geometry)
        base_losses = base_model.spatial_losses(
            base_output, lambda_equiv=arguments.lambda_equiv
        )
        v2_lambda0 = model.spatial_losses(
            v2_output, lambda_equiv=arguments.lambda_equiv, lambda_img=0.0
        )
        base_l4_branch = base_model.teacher.slot_attn.visual_branches[-1]
        base_fine_tokens = base_output["F34_A"].flatten(2).transpose(1, 2)
        base_k34 = base_l4_branch.img_to_k(
            base_l4_branch.img_norm_input(base_fine_tokens)
        )
        regression_errors = {
            "AUD_L4": float((base_output["AUD_L4_A"] - v2_output["AUD_L4_A"]).abs().max()),
            "AUD_FINE": float((base_output["AUD_FINE_A"] - v2_output["AUD_FINE_A"]).abs().max()),
            "F34": float((base_output["F34_A"] - v2_output["F34_A"]).abs().max()),
            "K34": float((base_k34 - v2_output["K34_A"]).abs().max()),
            "L_aud_coarse": abs(float(base_losses["loss_coarse"] - v2_lambda0["loss_aud_coarse"])),
            "L_aud_equiv": abs(float(base_losses["loss_equiv"] - v2_lambda0["loss_aud_equiv"])),
            "L_total_lambda_img_0": abs(float(base_losses["loss_total"] - v2_lambda0["loss_total"])),
        }
    base_model.close()
    del base_model
    torch.cuda.empty_cache()
    if max(regression_errors.values()) > 1e-7:
        raise RuntimeError(f"lambda_img=0 regression failed: {regression_errors}")

    model.zero_grad(set_to_none=True)
    output = model.forward_two_views(frame, spec, geometry)
    losses = model.spatial_losses(
        output, lambda_equiv=arguments.lambda_equiv, lambda_img=1.0
    )
    parameters = [parameter for parameter in model.student.parameters() if parameter.requires_grad]
    aud_grads = gradients_for(
        losses["loss_aud_coarse"] + losses["loss_aud_equiv"], parameters
    )
    img_grads = gradients_for(losses["loss_img_coarse"], parameters)
    pair_stats = gradient_pair_statistics(aud_grads, img_grads)
    diagnostic_changed_grad = any(parameter.grad is not None for parameter in parameters)
    k34_grad = torch.autograd.grad(
        losses["loss_total"], output["K34_A"], retain_graph=True
    )[0]
    losses["loss_total"].backward()

    first_teacher_grad = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    first_adapter_grad = gradient_l1(model.student.adapter.parameters())
    first_proj3_grad = gradient_l1(model.student.proj3_spatial.parameters())
    img_grad_norm = pair_stats["grad_norm_img"]
    copy_error = max(
        float((student.detach() - teacher.detach()).abs().max())
        for student, teacher in zip(
            model.student.proj3_spatial.parameters(),
            model.teacher.imgnet.proj3.parameters(),
        )
    )
    zero_init_error = float((output["F34_A"] - output["F4_UP_A"]).abs().max())

    audit_optimizer = torch.optim.AdamW(
        model.student.parameters(), lr=arguments.init_lr, weight_decay=arguments.weight_decay
    )
    audit_optimizer.step()
    audit_optimizer.zero_grad(set_to_none=True)
    second_output = model.forward_two_views(frame, spec, geometry)
    second_losses = model.spatial_losses(
        second_output, lambda_equiv=arguments.lambda_equiv, lambda_img=1.0
    )
    second_losses["loss_total"].backward()
    second_proj3_grad = gradient_l1(model.student.proj3_spatial.parameters())
    second_adapter_grad = gradient_l1(model.student.adapter.parameters())
    second_teacher_grad = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    model.student.load_state_dict(initial_student_state, strict=True)
    model.zero_grad(set_to_none=True)

    probability_errors = {
        key: float((output[key].sum(dim=(-2, -1)) - 1.0).abs().max())
        for key in (
            "AUD_L4_A",
            "AUD_FINE_A",
            "IMG_L4_A",
            "IMG_FINE_A",
            "AUD_L4_B",
            "AUD_FINE_B",
            "IMG_L4_B",
            "IMG_FINE_B",
        )
    }
    checks = {
        "baseline_metrics": baseline_metrics,
        "lambda_img_zero_regression_max_abs_errors": regression_errors,
        "teacher_has_any_gradient": first_teacher_grad or second_teacher_grad,
        "QA_requires_grad": bool(output["QA"].requires_grad),
        "Q4_requires_grad": bool(output["Q4_A"].requires_grad),
        "K4_requires_grad": bool(output["K4_A"].requires_grad),
        "K34_requires_grad": bool(output["K34_A"].requires_grad),
        "K34_gradient_l2": float(k34_grad.norm()),
        "L_img_coarse_gradient_norm": img_grad_norm,
        "gradient_diagnostic_changed_parameter_grad": diagnostic_changed_grad,
        "initial_adapter_gradient_l1": first_adapter_grad,
        "initial_proj3_spatial_gradient_l1": first_proj3_grad,
        "second_step_adapter_gradient_l1": second_adapter_grad,
        "second_step_proj3_spatial_gradient_l1": second_proj3_grad,
        "proj3_spatial_copy_max_abs_error": copy_error,
        "zero_init_f34_minus_up_f4_max_abs": zero_init_error,
        "probability_sum_max_errors": probability_errors,
        "initial_gradient_conflict": pair_stats,
        "loss_aud_coarse": float(losses["loss_aud_coarse"]),
        "loss_aud_equiv": float(losses["loss_aud_equiv"]),
        "loss_img_coarse": float(losses["loss_img_coarse"]),
        "loss_total": float(losses["loss_total"]),
        "student_restored_after_temporary_audit_step": True,
    }
    if checks["teacher_has_any_gradient"]:
        raise RuntimeError("Frozen teacher received a gradient")
    if checks["QA_requires_grad"] or checks["Q4_requires_grad"] or checks["K4_requires_grad"]:
        raise RuntimeError("Frozen Qa/Q4/K4 unexpectedly require gradients")
    if not checks["K34_requires_grad"] or checks["K34_gradient_l2"] <= 0:
        raise RuntimeError("K34 does not carry student gradients")
    if img_grad_norm <= 0 or first_adapter_grad <= 0:
        raise RuntimeError("L_img_coarse has no student gradient")
    if second_proj3_grad <= 0 or second_adapter_grad <= 0:
        raise RuntimeError("Second-step student gradient audit failed")
    if diagnostic_changed_grad:
        raise RuntimeError("torch.autograd.grad changed formal .grad")
    if copy_error != 0 or zero_init_error > 1e-7:
        raise RuntimeError("Student initialization changed from G")
    if max(probability_errors.values()) > 1e-5:
        raise RuntimeError("A probability map is not normalized")
    if not all(
        torch.isfinite(losses[key])
        for key in ("loss_aud_coarse", "loss_aud_equiv", "loss_img_coarse", "loss_total")
    ):
        raise RuntimeError("A loss is NaN or Inf")
    print(json.dumps(checks, indent=2), flush=True)
    print("All 1.3G-v2 regression and gradient checks passed.", flush=True)
    return checks


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
    loss_fields = ("loss_aud_coarse", "loss_aud_equiv", "loss_img_coarse", "loss_total")
    totals = {field: 0.0 for field in loss_fields}
    valid_ratio_total = 0.0
    crop_scale_total = 0.0
    flipped_count = 0
    skipped_count = 0
    sample_count = 0
    batch_average = 0.0
    diagnostic_totals = {
        "grad_norm_aud": 0.0,
        "grad_norm_img": 0.0,
        "grad_norm_ratio": 0.0,
        "grad_cosine": 0.0,
    }
    diagnostic_count = 0
    negative_count = 0
    epoch_start = time.time()
    parameters = [parameter for parameter in model.student.parameters() if parameter.requires_grad]

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
            losses = model.spatial_losses(
                output, lambda_equiv=arguments.lambda_equiv, lambda_img=1.0
            )

        if batch_index % arguments.gradient_diagnostic_interval == 0:
            aud_grads = gradients_for(
                losses["loss_aud_coarse"] + losses["loss_aud_equiv"], parameters
            )
            img_grads = gradients_for(losses["loss_img_coarse"], parameters)
            diagnostic = gradient_pair_statistics(aud_grads, img_grads)
            if any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("Gradient diagnostic changed formal .grad")
            for key in diagnostic_totals:
                diagnostic_totals[key] += diagnostic[key]
            diagnostic_count += 1
            negative_count += int(diagnostic["grad_cosine"] < 0)

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
            remaining_batches = len(train_loader) - completed + (
                total_epochs - epoch - 1
            ) * len(train_loader)
            eta = timedelta(seconds=int(batch_average * remaining_batches))
            print(
                f"Train [{epoch + 1}/{total_epochs}] [{completed}/{len(train_loader)}] "
                f"aud_coarse={float(losses['loss_aud_coarse']):.8f} "
                f"aud_equiv={float(losses['loss_aud_equiv']):.8f} "
                f"img_coarse={float(losses['loss_img_coarse']):.8f} "
                f"total={float(losses['loss_total']):.8f} "
                f"valid={float(losses['mean_valid_ratio']):.3f} ETA={eta}",
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
            **{
                key: value / diagnostic_count
                for key, value in diagnostic_totals.items()
            },
            "negative_grad_cosine_ratio": negative_count / diagnostic_count,
            "gradient_diagnostic_batches": diagnostic_count,
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
        "architecture": "1.3G-v2_visual_semantic_preservation",
        "base_checkpoint_path": str(base_checkpoint),
        "proj3_spatial_state_dict": model.student.proj3_spatial.state_dict(),
        "topdown_adapter_state_dict": model.student.adapter.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "loss_definition": "L_aud_coarse + L_aud_equiv + L_img_coarse",
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
    if arguments.gradient_diagnostic_interval <= 0:
        raise ValueError("--gradient-diagnostic-interval must be positive")
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
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print(json.dumps(counts, indent=2), flush=True)
    print(json.dumps({"trainable_parameter_names": trainable_names}, indent=2), flush=True)
    if counts["frozen_teacher_parameters"] != counts["total_teacher_parameters"]:
        raise RuntimeError("Not every teacher parameter is frozen")
    if counts["trainable_parameters"] != counts["spatial_student_parameters"]:
        raise RuntimeError("Trainable parameters are not exactly the spatial student")
    if counts["new_trainable_parameters_relative_to_G"] != 0:
        raise RuntimeError("1.3G-v2 introduced a trainable parameter")
    if any(not name.startswith("student.") for name in trainable_names):
        raise RuntimeError("A non-student parameter is trainable")

    model_dir = arguments.model_dir / experiment_name
    protected = ("latest.pth", "final.pth", f"{registry['dataset']}_best.pth")
    if not arguments.sanity_only and any((model_dir / name).exists() for name in protected):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    audit_dir = (
        arguments.model_dir / "1.3G-v2_sanity" / arguments.experiment
        if arguments.sanity_only
        else model_dir
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = build_datasets(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    sanity = run_sanity(
        model,
        train_dataset,
        test_loader,
        config,
        registry,
        base_checkpoint,
        device,
        arguments,
    )
    (audit_dir / "sanity_checks.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    if arguments.sanity_only:
        model.close()
        print("Sanity-only mode complete; no training checkpoint written.", flush=True)
        return

    run_config = {
        "architecture": "1.3G-v2_visual_semantic_preservation",
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
        "lambda_aud_coarse": 1.0,
        "lambda_aud_equiv": arguments.lambda_equiv,
        "lambda_img_coarse": 1.0,
        "gradient_diagnostic_interval": arguments.gradient_diagnostic_interval,
        "random_resized_crop_scale": [arguments.crop_scale_min, arguments.crop_scale_max],
        "random_resized_crop_ratio": [arguments.crop_ratio_min, arguments.crop_ratio_max],
        "flip_probability": arguments.flip_probability,
        "minimum_valid_ratio": arguments.minimum_valid_ratio,
        "checkpoint_selection": "AUD_FINE_cIoU",
        "seed": config.seed,
        "train_data_path": config.train_data_path,
        "train_manifest_path": config.train_manifest_path,
        "test_data_path": config.test_data_path,
        "test_manifest_path": config.test_manifest_path,
        "test_gt_path": config.test_gt_path,
        "parameter_counts": counts,
        "trainable_parameter_names": trainable_names,
        "sanity_checks": sanity,
        "forbidden_inputs": ["GT localization", "OGL", "OBJ_PRIOR", "Q_VALUE"],
    }
    (model_dir / "configs.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
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
            model, train_loader, optimizer, scaler, device, epoch, epochs, arguments
        )
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"aud_coarse={train_metrics['loss_aud_coarse']:.8f} "
            f"aud_equiv={train_metrics['loss_aud_equiv']:.8f} "
            f"img_coarse={train_metrics['loss_img_coarse']:.8f} "
            f"total={train_metrics['loss_total']:.8f} "
            f"grad_ratio={train_metrics['grad_norm_ratio']:.4f} "
            f"grad_cos={train_metrics['grad_cosine']:.4f} "
            f"negative={train_metrics['negative_grad_cosine_ratio']:.4f}",
            flush=True,
        )
        validation, _ = validate(model, test_loader, device)
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
                f"Best saved at epoch {epoch + 1}: AUD_FINE cIoU={best_score:.4f}",
                flush=True,
            )

        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **{
                f"{method.lower()}_{metric.lower()}": validation[method][metric]
                for method in EVAL_METHODS
                for metric in ("cIoU", "AUC")
            },
        }
        # Preserve the explicit historical field spellings.
        record["aud_l4_ciou"] = validation["AUD_L4"]["cIoU"]
        record["aud_l4_auc"] = validation["AUD_L4"]["AUC"]
        record["aud_fine_ciou"] = validation["AUD_FINE"]["cIoU"]
        record["aud_fine_auc"] = validation["AUD_FINE"]["AUC"]
        record["img_l4_ciou"] = validation["IMG_L4"]["cIoU"]
        record["img_l4_auc"] = validation["IMG_L4"]["AUC"]
        record["img_fine_ciou"] = validation["IMG_FINE"]["cIoU"]
        record["img_fine_auc"] = validation["IMG_FINE"]["AUC"]
        record["iqr_fine_ciou"] = validation["IQR_FINE"]["cIoU"]
        record["iqr_fine_auc"] = validation["IQR_FINE"]["AUC"]
        record = {field: record[field] for field in HISTORY_FIELDS}
        append_history(history_path, record)
        render_curves(history_path, model_dir / "training_curves", experiment_name)
        elapsed = time.time() - run_start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{epochs} complete; ETA "
            f"{timedelta(seconds=int(remaining))}; finish "
            f"{(datetime.now() + timedelta(seconds=remaining)):%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    final_metrics, _ = validate(model, test_loader, device)
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
    best_metrics, sample_rows = validate(
        model, test_loader, device, collect_samples=True
    )
    print_metrics(best_metrics, prefix="BEST/")
    (model_dir / "best_test_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )
    write_csv(model_dir / "best_per_sample_iou.csv", sample_rows)
    print(
        f"Total 1.3G-v2 training time: {timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
