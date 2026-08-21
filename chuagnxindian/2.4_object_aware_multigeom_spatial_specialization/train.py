#!/usr/bin/env python3
"""Experiment 2.4: object-aware multi-geometry spatial specialization."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_24_train_mpl")

from common import (
    EXPERIMENTS,
    PROJECT_ROOT,
    build_datasets,
    build_model,
    build_test_loader,
    build_train_loader,
    load_base_config,
    parameter_counts,
    setup_seed,
)
from curves import render
from evaluation import (
    evaluate,
    load_reference_rows,
    load_reference_summary,
    object_prior_model,
    save_detailed_result,
    save_qualitative,
    select_qualitative,
)
from geometry import sample_random_resized_crop


HERE = Path(__file__).resolve().parent
LAMBDA_AUDIO_COARSE = 1.0
LAMBDA_AUDIO_EQUIV = 1.0
LAMBDA_OWN_COARSE = 1.0
LAMBDA_OWN_EQUIV = 1.0

HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "loss_audio_coarse",
    "loss_audio_equiv",
    "loss_own_coarse",
    "loss_own_equiv",
    "loss_total",
    "own7_slot0_mass",
    "own14_slot0_mass",
    "own7_entropy",
    "own14_entropy",
    "pooled_ownership_mae",
    "mean_valid_ratio",
    "skipped_small_overlap_samples",
    "actual_flip_ratio",
    "mean_crop_scale",
    "aud_fine_ciou",
    "aud_fine_auc",
    "obj_fine_ciou",
    "obj_fine_auc",
    "aud_obj_ciou",
    "aud_obj_auc",
    "obj_prior_ciou",
    "obj_prior_auc",
    "ogl_ciou",
    "ogl_auc",
    "rescue",
    "hurt",
    "net",
    "fixed_ref_rescue",
    "fixed_ref_hurt",
    "fixed_ref_net",
    "oracle_ciou",
    "oracle_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-output-root", type=Path, default=HERE / "smoke_results")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_audit(model) -> dict[str, Any]:
    counts = parameter_counts(model)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    invalid = [name for name in trainable_names if not name.startswith("student.")]
    audit = {
        **counts,
        "trainable_parameter_names": trainable_names,
        "invalid_trainable_parameter_names": invalid,
    }
    expected_names = {
        "student.proj3_spatial.weight",
        "student.proj3_spatial.bias",
        "student.adapter.layers.0.weight",
        "student.adapter.layers.0.bias",
        "student.adapter.layers.2.weight",
        "student.adapter.layers.2.bias",
    }
    if set(trainable_names) != expected_names:
        raise RuntimeError(f"Unexpected trainable parameters: {audit}")
    if counts != {
        "total_teacher_parameters": 36417282,
        "frozen_teacher_parameters": 36417282,
        "proj3_spatial_parameters": 131584,
        "topdown_adapter_parameters": 1311488,
        "spatial_student_parameters": 1443072,
        "trainable_parameters": 1443072,
    }:
        raise RuntimeError(f"Formal 1.3G parameter count mismatch: {audit}")
    return audit


def _component_gradient(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> tuple[torch.Tensor, dict[str, float]]:
    parameters = [parameter for _name, parameter in named_parameters]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    flat_parts = []
    proj_squared = torch.zeros((), device=loss.device, dtype=torch.float32)
    adapter_squared = torch.zeros((), device=loss.device, dtype=torch.float32)
    for (name, parameter), gradient in zip(named_parameters, gradients):
        value = (
            torch.zeros_like(parameter, dtype=torch.float32)
            if gradient is None
            else gradient.detach().float()
        )
        flat_parts.append(value.reshape(-1))
        if name.startswith("student.proj3_spatial."):
            proj_squared = proj_squared + value.square().sum()
        elif name.startswith("student.adapter."):
            adapter_squared = adapter_squared + value.square().sum()
        else:
            raise RuntimeError(f"Unexpected trainable gradient target: {name}")
    vector = torch.cat(flat_parts)
    return vector, {
        "proj3_spatial": float(proj_squared.sqrt()),
        "adapter": float(adapter_squared.sqrt()),
        "total": float(vector.norm()),
    }


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.norm() * second.norm()
    if float(denominator) == 0.0:
        return math.nan
    return float(torch.dot(first, second) / denominator)


def gradient_diagnostics(
    losses: dict[str, torch.Tensor], model
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    loss_keys = (
        "loss_audio_coarse",
        "loss_audio_equiv",
        "loss_own_coarse",
        "loss_own_equiv",
    )
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, dict[str, float]] = {}
    for key in loss_keys:
        vectors[key], norms[key] = _component_gradient(losses[key], named_parameters)
    total_audio = vectors["loss_audio_coarse"] + vectors["loss_audio_equiv"]
    total_ownership = vectors["loss_own_coarse"] + vectors["loss_own_equiv"]
    cosines = {
        "audio_coarse_vs_own_coarse": _cosine(
            vectors["loss_audio_coarse"], vectors["loss_own_coarse"]
        ),
        "audio_equiv_vs_own_equiv": _cosine(
            vectors["loss_audio_equiv"], vectors["loss_own_equiv"]
        ),
        "total_audio_vs_total_ownership": _cosine(total_audio, total_ownership),
    }
    return norms, cosines


def _state_max_error(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return max(float((before[key] - after[key]).abs().max()) for key in before)


def smoke_audit(
    model,
    train_dataset,
    device: torch.device,
    base_checkpoint: Path,
    parameter_report: dict[str, Any],
) -> dict[str, Any]:
    print("Running Experiment 2.4 first-batch smoke and gradient audit...", flush=True)
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    frame, spec, _bboxes, names, _labels = next(iter(loader))
    frame = frame.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    geometry = sample_random_resized_crop(
        frame.shape[0],
        frame.shape[-2],
        frame.shape[-1],
        device,
        scale=(0.6, 1.0),
        ratio=(0.9, 1.1),
        flip_probability=0.5,
    )

    initial_state = copy.deepcopy(model.student.state_dict())
    model.train()
    model.zero_grad(set_to_none=True)
    output = model.forward_two_views(frame, spec, geometry)
    losses = model.all_losses(
        output,
        lambda_audio_equiv=LAMBDA_AUDIO_EQUIV,
        lambda_own_coarse=LAMBDA_OWN_COARSE,
        lambda_own_equiv=LAMBDA_OWN_EQUIV,
    )
    audio_only = model.spatial_losses(output, lambda_equiv=LAMBDA_AUDIO_EQUIV)
    zero_ownership = model.all_losses(
        output,
        lambda_audio_equiv=LAMBDA_AUDIO_EQUIV,
        lambda_own_coarse=0.0,
        lambda_own_equiv=0.0,
    )
    degeneration_error = float(
        (audio_only["loss_total"] - zero_ownership["loss_total"]).abs()
    )
    initial_norms, initial_cosines = gradient_diagnostics(losses, model)
    losses["loss_total"].backward()
    teacher_has_gradient = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    state_after_backward_error = _state_max_error(initial_state, model.student.state_dict())

    copy_error = max(
        float((student.detach() - teacher.detach()).abs().max())
        for student, teacher in zip(
            model.student.proj3_spatial.parameters(),
            model.teacher.imgnet.proj3.parameters(),
        )
    )
    zero_init_error = float((output["F34_A"] - output["F4_UP_A"]).abs().max())
    slot_sum_errors = {
        key: float((output[key].sum(dim=1) - 1.0).abs().max())
        for key in ("OWN7_A", "OWN7_B", "OWN14_A", "OWN14_B")
    }
    pooled = model.average_pool_ownership(output["OWN14_A"])

    audit_optimizer = torch.optim.AdamW(
        model.student.parameters(), lr=5e-5, weight_decay=0.01
    )
    audit_optimizer.step()
    audit_optimizer.zero_grad(set_to_none=True)
    second_output = model.forward_two_views(frame, spec, geometry)
    second_losses = model.all_losses(second_output)
    second_norms, _second_cosines = gradient_diagnostics(second_losses, model)
    model.student.load_state_dict(initial_state, strict=True)
    model.zero_grad(set_to_none=True)
    restore_error = _state_max_error(initial_state, model.student.state_dict())

    raw_losses = {
        key: float(losses[key].detach())
        for key in (
            "loss_audio_coarse",
            "loss_audio_equiv",
            "loss_own_coarse",
            "loss_own_equiv",
            "loss_total",
        )
    }
    finite_values = [*raw_losses.values(), degeneration_error, copy_error, zero_init_error]
    for report in (initial_norms, second_norms):
        for group in report.values():
            finite_values.extend(group.values())
    finite_values.extend(initial_cosines.values())
    no_nan_or_inf = all(math.isfinite(value) for value in finite_values)
    checks = {
        "sample_ids": [str(name) for name in names],
        "base_stage1_checkpoint": str(base_checkpoint),
        "loaded_trained_1_3g_checkpoint": False,
        "parameter_audit": parameter_report,
        "F34_shape": list(output["F34_A"].shape),
        "K34_shape": list(output["K34_A"].shape),
        "Q4_shape": list(output["Q4_A"].shape),
        "OWN7_shape": list(output["OWN7_A"].shape),
        "OWN14_shape": list(output["OWN14_A"].shape),
        "avg_pool_OWN14_shape": list(pooled.shape),
        "Q4_A_requires_grad": output["Q4_A"].requires_grad,
        "Q4_B_requires_grad": output["Q4_B"].requires_grad,
        "Q4_A_B_share_storage": output["Q4_A"].data_ptr() == output["Q4_B"].data_ptr(),
        "OWN7_requires_grad": output["OWN7_A"].requires_grad,
        "OWN14_requires_grad": output["OWN14_A"].requires_grad,
        "OWN7_slot_sum_max_errors": {
            key: value for key, value in slot_sum_errors.items() if key.startswith("OWN7")
        },
        "OWN14_slot_sum_max_errors": {
            key: value for key, value in slot_sum_errors.items() if key.startswith("OWN14")
        },
        "raw_losses": raw_losses,
        "initial_gradient_norms": initial_norms,
        "gradient_cosines": initial_cosines,
        "post_temporary_step_gradient_norms": second_norms,
        "OWN7_slot0_mass": float(losses["own7_slot0_mass"]),
        "OWN14_slot0_mass": float(losses["own14_slot0_mass"]),
        "OWN7_entropy": float(losses["own7_entropy"]),
        "OWN14_entropy": float(losses["own14_entropy"]),
        "pooled_ownership_mae": float(losses["pooled_ownership_mae"]),
        "audio_only_degeneration_max_abs_error": degeneration_error,
        "proj3_spatial_copy_max_abs_error": copy_error,
        "zero_init_F34_minus_F4_up_max_abs_error": zero_init_error,
        "f4_token_error": float(output["f4_token_error"]),
        "teacher_has_gradient": teacher_has_gradient,
        "backward_changed_parameters_max_error": state_after_backward_error,
        "student_restore_max_error": restore_error,
        "no_nan_or_inf": no_nan_or_inf,
    }
    expected_shapes = {
        "F34_shape": [frame.shape[0], 512, 14, 14],
        "K34_shape": [frame.shape[0], 196, 512],
        "Q4_shape": [frame.shape[0], 2, 512],
        "OWN7_shape": [frame.shape[0], 2, 7, 7],
        "OWN14_shape": [frame.shape[0], 2, 14, 14],
        "avg_pool_OWN14_shape": [frame.shape[0], 2, 7, 7],
    }
    if any(checks[key] != value for key, value in expected_shapes.items()):
        raise RuntimeError(f"Smoke shape audit failed: {checks}")
    if checks["Q4_A_requires_grad"] or checks["Q4_B_requires_grad"] or checks["OWN7_requires_grad"]:
        raise RuntimeError("Frozen Stage1 ownership tensors require gradients")
    if checks["Q4_A_B_share_storage"] or not checks["OWN14_requires_grad"]:
        raise RuntimeError("View-specific Q4 or trainable OWN14 audit failed")
    if max(slot_sum_errors.values()) > 1e-6:
        raise RuntimeError(f"Ownership slot normalization failed: {checks}")
    if teacher_has_gradient or state_after_backward_error != 0.0 or restore_error != 0.0:
        raise RuntimeError(f"Freeze/restore audit failed: {checks}")
    if degeneration_error != 0.0 or copy_error != 0.0 or zero_init_error > 1e-7:
        raise RuntimeError(f"G initialization/degeneration audit failed: {checks}")
    if not no_nan_or_inf:
        raise RuntimeError(f"NaN/Inf in smoke audit: {checks}")
    for key, report in initial_norms.items():
        if report["adapter"] <= 0.0 or report["total"] <= 0.0:
            raise RuntimeError(f"Initial {key} did not reach the zero-init adapter: {checks}")
    for key, report in second_norms.items():
        if report["proj3_spatial"] <= 0.0 or report["adapter"] <= 0.0:
            raise RuntimeError(f"Post-step {key} did not reach both G modules: {checks}")

    print(json.dumps(checks, indent=2), flush=True)
    print("Experiment 2.4 smoke audit passed; student initialization restored.", flush=True)
    return checks


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train()
    metric_keys = (
        "loss_audio_coarse",
        "loss_audio_equiv",
        "loss_own_coarse",
        "loss_own_equiv",
        "loss_total",
        "own7_slot0_mass",
        "own14_slot0_mass",
        "own7_entropy",
        "own14_entropy",
        "pooled_ownership_mae",
        "mean_valid_ratio",
    )
    totals = {key: 0.0 for key in metric_keys}
    skipped = 0
    flipped = 0
    crop_scale = 0.0
    samples = 0
    batch_average = 0.0
    epoch_start = time.time()

    for batch_index, (frame, spec, _bboxes, _names, _labels) in enumerate(loader):
        batch_start = time.time()
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        geometry = sample_random_resized_crop(
            frame.shape[0],
            frame.shape[-2],
            frame.shape[-1],
            device,
            scale=(0.6, 1.0),
            ratio=(0.9, 1.1),
            flip_probability=0.5,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            output = model.forward_two_views(frame, spec, geometry)
            losses = model.all_losses(output)
        scaler.scale(losses["loss_total"]).backward()
        if any(parameter.grad is not None for parameter in model.teacher.parameters()):
            raise RuntimeError("Frozen Stage1 teacher received a training gradient")
        scaler.step(optimizer)
        scaler.update()

        batch_size = frame.shape[0]
        samples += batch_size
        skipped += int(losses["skipped_small_overlap_samples"])
        flipped += int(geometry["flipped"].sum())
        crop_scale += float(output["mean_crop_scale"]) * batch_size
        for key in metric_keys:
            totals[key] += float(losses[key].detach()) * batch_size

        completed = batch_index + 1
        batch_seconds = time.time() - batch_start
        batch_average += (batch_seconds - batch_average) / completed
        if batch_index % 10 == 0 or completed == len(loader):
            remaining_batches = len(loader) - completed + (epochs - epoch - 1) * len(loader)
            eta = timedelta(seconds=int(batch_average * remaining_batches))
            print(
                f"Train [{epoch + 1}/{epochs}] [{completed}/{len(loader)}] "
                f"aud_c={float(losses['loss_audio_coarse']):.7f} "
                f"aud_e={float(losses['loss_audio_equiv']):.7f} "
                f"own_c={float(losses['loss_own_coarse']):.7f} "
                f"own_e={float(losses['loss_own_equiv']):.7f} "
                f"m7={float(losses['own7_slot0_mass']):.3f} "
                f"m14={float(losses['own14_slot0_mass']):.3f} "
                f"H14={float(losses['own14_entropy']):.3f} ETA={eta}",
                flush=True,
            )

    result = {key: value / samples for key, value in totals.items()}
    result.update(
        {
            "skipped_small_overlap_samples": skipped,
            "actual_flip_ratio": flipped / samples,
            "mean_crop_scale": crop_scale / samples,
            "epoch_seconds": time.time() - epoch_start,
        }
    )
    return result


def append_history(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compact_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in validation.items() if key != "rows"}


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    validation: dict[str, Any],
    base_checkpoint: Path,
    selection_metric: str | None = None,
) -> None:
    payload = {
        "architecture": "object_aware_multigeom_spatial_specialization",
        "base_stage1_checkpoint": str(base_checkpoint),
        "proj3_spatial_state_dict": model.student.proj3_spatial.state_dict(),
        "topdown_adapter_state_dict": model.student.adapter.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        **compact_validation(validation),
    }
    if selection_metric is not None:
        method = "AUD_OBJ" if selection_metric == "AUD_OBJ_cIoU" else "AUD_FINE"
        payload["selection_metric"] = selection_metric
        payload["selection_score"] = validation["metrics"][method]["cIoU"]
        payload["selection_auc_diagnostic"] = validation["metrics"][method]["AUC"]
    torch.save(payload, path)


def load_student(path: Path, model) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "object_aware_multigeom_spatial_specialization":
        raise RuntimeError(f"Unexpected 2.4 checkpoint architecture: {payload.get('architecture')}")
    model.student.proj3_spatial.load_state_dict(
        payload["proj3_spatial_state_dict"], strict=True
    )
    model.student.adapter.load_state_dict(
        payload["topdown_adapter_state_dict"], strict=True
    )
    return payload


def record_tie(path: Path, saver: str, epoch: int, metric: dict[str, float], incumbent: dict[str, Any]) -> None:
    record = {
        "saver": saver,
        "tie_epoch": epoch,
        "cIoU": metric["cIoU"],
        "AUC": metric["AUC"],
        "incumbent_epoch": incumbent["epoch"],
        "incumbent_AUC": incumbent["AUC"],
        "saver_rule": "strict cIoU improvement only; AUC is diagnostic",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _metric_lookup(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in reference["method_metrics"]}


def _hr14_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in reference["rescue_hurt_oracle"] if row["candidate"] == "HR14")


def _assert_original_reproduction(
    observed: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    expected_methods = _metric_lookup(reference)
    mapping = {
        "AUD_FINE": "AUD_FINE",
        "OBJ_FINE": "SLOT_L4_HR14",
        "AUD_OBJ": "AUD_SLOT_L4_HR14",
        "OBJ_PRIOR": "OBJ_PRIOR",
        "OGL": "OGL",
    }
    errors: dict[str, dict[str, float]] = {}
    for observed_name, expected_name in mapping.items():
        errors[observed_name] = {}
        for metric in ("cIoU", "AUC"):
            error = abs(
                float(observed["metrics"][observed_name][metric])
                - float(expected_methods[expected_name][metric])
            )
            errors[observed_name][metric] = error
            if error > 1e-12:
                raise RuntimeError(f"Original 1.3G/2.2 reproduction failed: {errors}")
    return {"passed": True, "absolute_errors": errors}


def main() -> None:
    arguments = parse_args()
    if arguments.epochs != 50 and not arguments.smoke_only:
        raise ValueError("Formal Experiment 2.4 is fixed to 50 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)

    registry = EXPERIMENTS[arguments.experiment]
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    if registry["original_g_experiment"] in str(base_checkpoint):
        raise RuntimeError("Experiment 2.4 must initialize from Stage1, not trained 1.3G")
    base_hash_before = sha256(base_checkpoint)
    base_mtime_before = base_checkpoint.stat().st_mtime_ns
    parameters = parameter_audit(model)
    print(json.dumps(parameters, indent=2), flush=True)

    train_dataset, test_dataset = build_datasets(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    smoke = smoke_audit(model, train_dataset, device, base_checkpoint, parameters)
    if arguments.smoke_only:
        output_dir = arguments.smoke_output_root / arguments.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": arguments.experiment,
            "parameter_audit": parameters,
            "smoke": smoke,
            "base_stage1_checkpoint": str(base_checkpoint),
            "base_stage1_checkpoint_unchanged": (
                base_hash_before == sha256(base_checkpoint)
                and base_mtime_before == base_checkpoint.stat().st_mtime_ns
            ),
        }
        (output_dir / "smoke_audit.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        model.close()
        print("Smoke-only audit complete; no formal training checkpoint was written.", flush=True)
        return

    experiment_name = arguments.experiment_name or registry["default_experiment"]
    model_dir = arguments.model_dir / experiment_name
    protected = (
        "latest.pth",
        "final.pth",
        f"{registry['dataset']}_best.pth",
        f"{registry['dataset']}_best_aud_fine.pth",
    )
    if any((model_dir / name).exists() for name in protected):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = load_reference_rows(arguments.experiment)
    reference_summary = load_reference_summary(arguments.experiment)
    run_config = {
        "architecture": "object_aware_multigeom_spatial_specialization",
        "experiment_key": arguments.experiment,
        "experiment_name": experiment_name,
        "training_stages": 2,
        "stage1_checkpoint": str(base_checkpoint),
        "trained_1_3g_used_for_initialization": False,
        "epochs": 50,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "optimizer": "AdamW",
        "init_lr": 5e-5,
        "weight_decay": 0.01,
        "scheduler": False,
        "geometry": {
            "random_resized_crop_scale": [0.6, 1.0],
            "random_resized_crop_ratio": [0.9, 1.1],
            "horizontal_flip_probability": 0.5,
            "minimum_valid_ratio": 0.2,
        },
        "loss_weights": {
            "lambda_audio_coarse": LAMBDA_AUDIO_COARSE,
            "lambda_audio_equiv": LAMBDA_AUDIO_EQUIV,
            "lambda_own_coarse": LAMBDA_OWN_COARSE,
            "lambda_own_equiv": LAMBDA_OWN_EQUIV,
        },
        "primary_checkpoint_selection": "AUD_OBJ_cIoU_strict_greater",
        "diagnostic_checkpoint_selection": "AUD_FINE_cIoU_strict_greater",
        "fusion": "normalize(0.6 * normalize(AUD_FINE) + 0.4 * normalize(OBJ_FINE))",
        "evaluator": "2.2 resize/min-max/threshold=0.6/cIoU/AUC",
        "parameter_audit": parameters,
        "smoke_audit": smoke,
        "forbidden_losses": [
            "seed",
            "mass",
            "entropy",
            "balance",
            "reconstruction",
            "reliability",
            "segmentation",
            "OGL",
        ],
    }
    (model_dir / "configs.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (model_dir / "smoke_audit.json").write_text(
        json.dumps(smoke, indent=2), encoding="utf-8"
    )

    optimizer = torch.optim.AdamW(model.student.parameters(), lr=5e-5, weight_decay=0.01)
    optimized = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if optimized != 1443072:
        raise RuntimeError("Optimizer is not restricted to proj3_spatial + Adapter")
    scaler = torch.amp.GradScaler("cuda")
    setup_seed(config.seed)
    train_loader = build_train_loader(train_dataset, config)
    object_model = object_prior_model().to(device).eval()
    object_cache: dict[str, torch.Tensor] = {}

    history_path = model_dir / "epoch_metrics.csv"
    tie_path = model_dir / "checkpoint_ties.jsonl"
    best_primary = {"score": -math.inf, "epoch": None, "AUC": math.nan}
    best_audio = {"score": -math.inf, "epoch": None, "AUC": math.nan}
    run_start = time.time()
    last_validation = None

    for epoch in range(50):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, 50
        )
        validation = evaluate(
            model,
            test_loader,
            device,
            object_model,
            object_cache,
            reference_rows,
            description=f"Epoch {epoch + 1} validation",
        )
        last_validation = validation
        metrics = validation["metrics"]
        transitions = validation["rescue_hurt"]["same_checkpoint"]
        fixed_transitions = validation["rescue_hurt"]["original_1_3g_reference"]
        print(
            f"Epoch {epoch + 1}: AUD {metrics['AUD_FINE']['cIoU']:.4f}/{metrics['AUD_FINE']['AUC']:.4f} | "
            f"OWN14 {metrics['OBJ_FINE']['cIoU']:.4f}/{metrics['OBJ_FINE']['AUC']:.4f} | "
            f"AUD_OBJ {metrics['AUD_OBJ']['cIoU']:.4f}/{metrics['AUD_OBJ']['AUC']:.4f} | "
            f"OGL {metrics['OGL']['cIoU']:.4f}/{metrics['OGL']['AUC']:.4f} | "
            f"R/H/N {transitions['rescue']}/{transitions['hurt']}/{transitions['net']}",
            flush=True,
        )
        save_checkpoint(
            model_dir / "latest.pth",
            model,
            optimizer,
            epoch + 1,
            validation,
            base_checkpoint,
        )

        primary_metric = metrics["AUD_OBJ"]
        if primary_metric["cIoU"] > best_primary["score"]:
            best_primary = {
                "score": primary_metric["cIoU"],
                "epoch": epoch + 1,
                "AUC": primary_metric["AUC"],
            }
            save_checkpoint(
                model_dir / f"{registry['dataset']}_best.pth",
                model,
                optimizer,
                epoch + 1,
                validation,
                base_checkpoint,
                selection_metric="AUD_OBJ_cIoU",
            )
            print(f"New primary best epoch {epoch + 1}: AUD_OBJ cIoU={primary_metric['cIoU']:.4f}", flush=True)
        elif primary_metric["cIoU"] == best_primary["score"]:
            record_tie(tie_path, "AUD_OBJ", epoch + 1, primary_metric, best_primary)

        audio_metric = metrics["AUD_FINE"]
        if audio_metric["cIoU"] > best_audio["score"]:
            best_audio = {
                "score": audio_metric["cIoU"],
                "epoch": epoch + 1,
                "AUC": audio_metric["AUC"],
            }
            save_checkpoint(
                model_dir / f"{registry['dataset']}_best_aud_fine.pth",
                model,
                optimizer,
                epoch + 1,
                validation,
                base_checkpoint,
                selection_metric="AUD_FINE_cIoU",
            )
        elif audio_metric["cIoU"] == best_audio["score"]:
            record_tie(tie_path, "AUD_FINE", epoch + 1, audio_metric, best_audio)

        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "aud_fine_ciou": metrics["AUD_FINE"]["cIoU"],
            "aud_fine_auc": metrics["AUD_FINE"]["AUC"],
            "obj_fine_ciou": metrics["OBJ_FINE"]["cIoU"],
            "obj_fine_auc": metrics["OBJ_FINE"]["AUC"],
            "aud_obj_ciou": metrics["AUD_OBJ"]["cIoU"],
            "aud_obj_auc": metrics["AUD_OBJ"]["AUC"],
            "obj_prior_ciou": metrics["OBJ_PRIOR"]["cIoU"],
            "obj_prior_auc": metrics["OBJ_PRIOR"]["AUC"],
            "ogl_ciou": metrics["OGL"]["cIoU"],
            "ogl_auc": metrics["OGL"]["AUC"],
            "rescue": transitions["rescue"],
            "hurt": transitions["hurt"],
            "net": transitions["net"],
            "fixed_ref_rescue": fixed_transitions["rescue"],
            "fixed_ref_hurt": fixed_transitions["hurt"],
            "fixed_ref_net": fixed_transitions["net"],
            "oracle_ciou": validation["oracle"]["cIoU"],
            "oracle_auc": validation["oracle"]["AUC"],
        }
        append_history(history_path, row)
        render(history_path, model_dir / "training_curves", experiment_name)
        elapsed = time.time() - run_start
        remaining = elapsed / (epoch + 1) * (49 - epoch)
        print(
            f"Epoch {epoch + 1}/50 complete; ETA {timedelta(seconds=int(remaining))}; "
            f"estimated finish {(datetime.now() + timedelta(seconds=remaining)):%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    assert last_validation is not None
    save_checkpoint(
        model_dir / "final.pth",
        model,
        optimizer,
        50,
        last_validation,
        base_checkpoint,
    )

    primary_path = model_dir / f"{registry['dataset']}_best.pth"
    primary_payload = load_student(primary_path, model)
    primary_result = evaluate(
        model,
        test_loader,
        device,
        object_model,
        object_cache,
        reference_rows,
        description="Primary best AUD_OBJ",
    )
    result_dir = model_dir / "results"
    save_detailed_result(primary_result, result_dir)

    audio_path = model_dir / f"{registry['dataset']}_best_aud_fine.pth"
    audio_payload = load_student(audio_path, model)
    audio_result = evaluate(
        model,
        test_loader,
        device,
        object_model,
        object_cache,
        reference_rows,
        description="Diagnostic best AUD_FINE",
    )
    (result_dir / "best_aud_fine_diagnostic.json").write_text(
        json.dumps(compact_validation(audio_result), indent=2), encoding="utf-8"
    )
    load_student(primary_path, model)

    original_checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / registry["original_g_experiment"]
        / f"{registry['dataset']}_best.pth"
    )
    original_hash_before = sha256(original_checkpoint)
    original_mtime_before = original_checkpoint.stat().st_mtime_ns
    original_model, original_base_checkpoint = build_model(config, registry, device)
    original_payload = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
    if original_payload.get("architecture") != "multi_geometry_equivariant_l3_refine":
        raise RuntimeError("Unexpected original 1.3G checkpoint architecture")
    original_model.student.proj3_spatial.load_state_dict(
        original_payload["proj3_spatial_state_dict"], strict=True
    )
    original_model.student.adapter.load_state_dict(
        original_payload["topdown_adapter_state_dict"], strict=True
    )
    for parameter in original_model.parameters():
        parameter.requires_grad = False
    original_model.eval()
    original_result = evaluate(
        original_model,
        test_loader,
        device,
        object_model,
        object_cache,
        reference_rows,
        description="Original 1.3G and 2.2 reproduction",
    )
    reproduction = _assert_original_reproduction(original_result, reference_summary)
    (result_dir / "original_1_3g_2_2_reproduction.json").write_text(
        json.dumps(
            {**reproduction, **compact_validation(original_result)}, indent=2
        ),
        encoding="utf-8",
    )

    selected = select_qualitative(arguments.experiment, primary_result["rows"])
    save_qualitative(
        model,
        original_model,
        test_loader,
        object_model,
        object_cache,
        selected,
        primary_result["rows"],
        device,
        result_dir / "qualitative",
    )

    reference_methods = _metric_lookup(reference_summary)
    hr14_reference = _hr14_reference(reference_summary)
    new_metrics = primary_result["metrics"]
    new_transition = primary_result["rescue_hurt"]["same_checkpoint"]
    comparison = {
        "original_1_3g_AUD": reference_methods["AUD_FINE"],
        "original_2_2_HR14_standalone": reference_methods["SLOT_L4_HR14"],
        "original_2_2_HR14_fusion": reference_methods["AUD_SLOT_L4_HR14"],
        "original_2_2_HR14_rescue_hurt_oracle": hr14_reference,
        "new_primary": {
            "AUD_FINE": new_metrics["AUD_FINE"],
            "OBJ_FINE": new_metrics["OBJ_FINE"],
            "AUD_OBJ": new_metrics["AUD_OBJ"],
            "OBJ_PRIOR": new_metrics["OBJ_PRIOR"],
            "OGL": new_metrics["OGL"],
            "rescue_hurt": new_transition,
            "oracle": primary_result["oracle"],
        },
        "delta_vs_original_1_3g_AUD": {
            metric: new_metrics["AUD_FINE"][metric] - reference_methods["AUD_FINE"][metric]
            for metric in ("cIoU", "AUC")
        },
        "delta_fusion_vs_2_2_HR14": {
            metric: new_metrics["AUD_OBJ"][metric] - reference_methods["AUD_SLOT_L4_HR14"][metric]
            for metric in ("cIoU", "AUC")
        },
    }
    stage1_unchanged = (
        base_hash_before == sha256(base_checkpoint)
        and base_mtime_before == base_checkpoint.stat().st_mtime_ns
    )
    original_unchanged = (
        original_hash_before == sha256(original_checkpoint)
        and original_mtime_before == original_checkpoint.stat().st_mtime_ns
    )
    if not stage1_unchanged or not original_unchanged or original_base_checkpoint != base_checkpoint:
        raise RuntimeError("A source checkpoint changed or Stage1 source diverged")
    summary = {
        "experiment": "2.4 Object-Aware Multi-Geometry Spatial Specialization",
        "dataset": arguments.experiment,
        "best_AUD_OBJ_epoch": int(primary_payload["epoch"]),
        "best_AUD_FINE_epoch": int(audio_payload["epoch"]),
        "parameter_audit": parameters,
        "smoke_audit": smoke,
        "comparison": comparison,
        "collapse_audit_csv": str(history_path),
        "qualitative_ids": selected,
        "stage1_checkpoint": str(base_checkpoint),
        "stage1_checkpoint_unchanged": stage1_unchanged,
        "original_1_3g_checkpoint": str(original_checkpoint),
        "original_1_3g_loaded_only_after_training": True,
        "original_1_3g_checkpoint_unchanged": original_unchanged,
        "training_seconds": time.time() - run_start,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Total Experiment 2.4 time: {timedelta(seconds=int(time.time() - run_start))}", flush=True)
    original_model.close()
    model.close()


if __name__ == "__main__":
    main()
