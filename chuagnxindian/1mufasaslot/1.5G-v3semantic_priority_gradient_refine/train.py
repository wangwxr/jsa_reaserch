#!/usr/bin/env python3
"""Experiment G-v3: semantic-priority gradient projection."""

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
    sample_semantic_preserving_crop,
    transform_view_a_map_to_b,
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
    "mean_crop_area_ratio",
    "mean_crop_teacher_mass",
    "min_crop_teacher_mass",
    "mean_crop_attempts",
    "fallback_identity_count",
    "grad_norm_coarse",
    "grad_norm_equiv_before",
    "grad_norm_equiv_after",
    "grad_norm_final",
    "projection_rate",
    "mean_projection_alpha",
    "mean_cosine_before",
    "mean_cosine_after",
    "max_abs_conflict_cosine_after",
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lambda-equiv", type=float, default=1.0)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--crop-scale-min", type=float, default=0.6)
    parser.add_argument("--crop-scale-max", type=float, default=1.0)
    parser.add_argument("--crop-ratio-min", type=float, default=0.9)
    parser.add_argument("--crop-ratio-max", type=float, default=1.1)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.2)
    parser.add_argument("--min-teacher-mass", type=float, default=0.60)
    parser.add_argument("--max-crop-attempts", type=int, default=10)
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=20)
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


def loss_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def semantic_priority_projection(
    coarse_gradients: tuple[torch.Tensor | None, ...],
    equiv_gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
    epsilon: float = 1e-20,
) -> tuple[list[torch.Tensor], list[bool], dict[str, float | bool]]:
    """Project only the global equivariance gradient away from coarse."""
    coarse_values: list[torch.Tensor] = []
    equiv_values: list[torch.Tensor] = []
    active: list[bool] = []
    device = parameters[0].device
    dot = torch.zeros((), device=device, dtype=torch.float32)
    norm_coarse_sq = torch.zeros_like(dot)
    norm_equiv_sq = torch.zeros_like(dot)

    for coarse, equiv, parameter in zip(
        coarse_gradients, equiv_gradients, parameters
    ):
        coarse_value = (
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if coarse is None
            else coarse.float()
        )
        equiv_value = (
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if equiv is None
            else equiv.float()
        )
        coarse_values.append(coarse_value)
        equiv_values.append(equiv_value)
        active.append(coarse is not None or equiv is not None)
        dot = dot + (coarse_value * equiv_value).sum()
        norm_coarse_sq = norm_coarse_sq + coarse_value.square().sum()
        norm_equiv_sq = norm_equiv_sq + equiv_value.square().sum()

    projection_applied = bool(dot < 0)
    alpha = (
        dot / (norm_coarse_sq + epsilon)
        if projection_applied
        else torch.zeros_like(dot)
    )
    projected_equiv = [
        equiv - alpha * coarse if projection_applied else equiv
        for coarse, equiv in zip(coarse_values, equiv_values)
    ]
    final_gradients = [
        coarse + equiv
        for coarse, equiv in zip(coarse_values, projected_equiv)
    ]

    norm_coarse = norm_coarse_sq.sqrt()
    norm_equiv_before = norm_equiv_sq.sqrt()
    dot_after = torch.zeros_like(dot)
    norm_equiv_after_sq = torch.zeros_like(dot)
    norm_final_sq = torch.zeros_like(dot)
    for coarse, equiv_after, final in zip(
        coarse_values, projected_equiv, final_gradients
    ):
        dot_after = dot_after + (coarse * equiv_after).sum()
        norm_equiv_after_sq = norm_equiv_after_sq + equiv_after.square().sum()
        norm_final_sq = norm_final_sq + final.square().sum()
    norm_equiv_after = norm_equiv_after_sq.sqrt()
    norm_final = norm_final_sq.sqrt()
    cosine_before = dot / (norm_coarse * norm_equiv_before + epsilon)
    cosine_after = dot_after / (norm_coarse * norm_equiv_after + epsilon)

    stats: dict[str, float | bool] = {
        "grad_cosine_before": float(cosine_before.detach()),
        "grad_cosine_after": float(cosine_after.detach()),
        "grad_norm_coarse": float(norm_coarse.detach()),
        "grad_norm_equiv_before": float(norm_equiv_before.detach()),
        "grad_norm_equiv_after": float(norm_equiv_after.detach()),
        "grad_norm_final": float(norm_final.detach()),
        "projection_applied": projection_applied,
        "projection_alpha": float(alpha.detach()),
        "dot_before": float(dot.detach()),
        "dot_after": float(dot_after.detach()),
    }
    if projection_applied and abs(stats["grad_cosine_after"]) > 1e-4:
        raise RuntimeError(
            "Semantic-priority projection failed orthogonality: "
            f"cos_after={stats['grad_cosine_after']}"
        )
    return final_gradients, active, stats


def projected_loss_gradients(
    loss_coarse: torch.Tensor,
    loss_equiv: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    loss_scale: float,
    losses_already_scaled: bool = False,
) -> tuple[list[torch.Tensor], list[bool], dict[str, float | bool]]:
    """Compute both losses in one consistent scaled space, then unscale."""
    coarse_objective = (
        loss_coarse if losses_already_scaled else loss_coarse * loss_scale
    )
    equiv_objective = (
        loss_equiv if losses_already_scaled else loss_equiv * loss_scale
    )
    coarse_scaled = loss_gradients(
        coarse_objective, parameters, retain_graph=True
    )
    equiv_scaled = loss_gradients(
        equiv_objective, parameters, retain_graph=False
    )
    coarse = tuple(
        None if gradient is None else gradient.float() / loss_scale
        for gradient in coarse_scaled
    )
    equiv = tuple(
        None if gradient is None else gradient.float() / loss_scale
        for gradient in equiv_scaled
    )
    return semantic_priority_projection(coarse, equiv, parameters)


def assign_scaled_gradients(
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor],
    active: list[bool],
    loss_scale: float,
) -> None:
    """Write scaled projected grads so GradScaler.step unscales exactly once."""
    for parameter, gradient, is_active in zip(parameters, gradients, active):
        parameter.grad = (
            (gradient * loss_scale).to(dtype=parameter.dtype)
            if is_active
            else None
        )


def run_projection_sanity(device: torch.device) -> dict[str, float | bool]:
    """Deterministically audit global conflict, non-conflict, and None paths."""
    parameters = [
        torch.nn.Parameter(torch.zeros(2, device=device)),
        torch.nn.Parameter(torch.zeros(1, device=device)),
    ]
    coarse = (
        torch.tensor([1.0, 0.0], device=device),
        torch.tensor([2.0], device=device),
    )
    equiv_conflict = (
        torch.tensor([-1.0, 1.0], device=device),
        torch.tensor([-1.0], device=device),
    )
    projected, active, conflict = semantic_priority_projection(
        coarse, equiv_conflict, parameters
    )
    expected_alpha = -3.0 / 5.0
    expected_projected_equiv = (
        equiv_conflict[0] - expected_alpha * coarse[0],
        equiv_conflict[1] - expected_alpha * coarse[1],
    )
    recovered_equiv = tuple(
        final - coarse_value
        for final, coarse_value in zip(projected, coarse)
    )
    conflict_equiv_error = max(
        float((observed - expected).abs().max())
        for observed, expected in zip(recovered_equiv, expected_projected_equiv)
    )

    equiv_nonconflict = (
        torch.tensor([1.0, 1.0], device=device),
        torch.tensor([1.0], device=device),
    )
    nonconflict_final, _, nonconflict = semantic_priority_projection(
        coarse, equiv_nonconflict, parameters
    )
    nonconflict_change = max(
        float((final - (coarse_value + equiv_value)).abs().max())
        for final, coarse_value, equiv_value in zip(
            nonconflict_final, coarse, equiv_nonconflict
        )
    )

    none_final, none_active, none_stats = semantic_priority_projection(
        (None, coarse[1]), (equiv_nonconflict[0], None), parameters
    )
    none_error = max(
        float((none_final[0] - equiv_nonconflict[0]).abs().max()),
        float((none_final[1] - coarse[1]).abs().max()),
    )
    checks: dict[str, float | bool] = {
        "synthetic_conflict_projection_applied": bool(conflict["projection_applied"]),
        "synthetic_conflict_alpha_error": abs(
            float(conflict["projection_alpha"]) - expected_alpha
        ),
        "synthetic_conflict_equiv_max_error": conflict_equiv_error,
        "synthetic_conflict_cosine_after": float(conflict["grad_cosine_after"]),
        "synthetic_nonconflict_projection_applied": bool(
            nonconflict["projection_applied"]
        ),
        "synthetic_nonconflict_final_max_change": nonconflict_change,
        "synthetic_none_gradient_max_error": none_error,
        "synthetic_none_active_all": all(none_active),
        "synthetic_none_projection_applied": bool(none_stats["projection_applied"]),
        "synthetic_conflict_active_all": all(active),
    }
    if (
        not checks["synthetic_conflict_projection_applied"]
        or checks["synthetic_conflict_alpha_error"] > 1e-6
        or conflict_equiv_error > 1e-6
        or abs(checks["synthetic_conflict_cosine_after"]) > 1e-6
        or checks["synthetic_nonconflict_projection_applied"]
        or nonconflict_change != 0
        or none_error != 0
        or not checks["synthetic_conflict_active_all"]
        or not checks["synthetic_none_active_all"]
    ):
        raise RuntimeError(f"Synthetic gradient projection sanity failed: {checks}")
    return checks


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

    probability = asymmetric / asymmetric.sum(dim=(-2, -1), keepdim=True)
    identity_target = transform_view_a_map_to_b(probability, identity, (14, 14))
    identity_target_error = float((identity_target - probability).abs().max())
    flip_target = transform_view_a_map_to_b(probability, flip, (14, 14))
    double_flip_target = transform_view_a_map_to_b(flip_target, flip, (14, 14))
    double_flip_error = float((double_flip_target - probability).abs().max())
    transformed_probability_sum_error = float(
        (transform_view_a_map_to_b(probability, crop_flip, (14, 14)).sum() - 1.0).abs()
    )
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
        "identity_teacher_target_max_abs_error": identity_target_error,
        "double_flip_teacher_target_max_abs_error": double_flip_error,
        "transformed_teacher_probability_sum_error": transformed_probability_sum_error,
    }
    if max(
        identity_error,
        old_flip_error,
        flip_recovery_error,
        crop_recovery_error,
        identity_target_error,
        double_flip_error,
        transformed_probability_sum_error,
    ) > 1e-5:
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
    print("Running Experiment G-v3 geometry and model sanity checks...", flush=True)
    geometry_checks = run_geometry_sanity(output_dir, device)
    projection_checks = run_projection_sanity(device)
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

    initial_student_state = copy.deepcopy(model.student.state_dict())
    model.train()
    model.zero_grad(set_to_none=True)
    output = model.forward_two_views(
        frame,
        spec,
        crop_scale_range=(arguments.crop_scale_min, arguments.crop_scale_max),
        crop_ratio_range=(arguments.crop_ratio_min, arguments.crop_ratio_max),
        flip_probability=arguments.flip_probability,
        min_teacher_mass=arguments.min_teacher_mass,
        max_crop_attempts=arguments.max_crop_attempts,
    )
    geometry = output["geometry"]
    losses = model.spatial_losses(output, lambda_equiv=arguments.lambda_equiv)

    trainable_parameters = [
        parameter for parameter in model.student.parameters() if parameter.requires_grad
    ]
    grad_was_clear_before = all(
        parameter.grad is None for parameter in trainable_parameters
    )
    projected_gradients, active_gradients, projection_stats = projected_loss_gradients(
        losses["loss_coarse"],
        losses["loss_equiv"],
        trainable_parameters,
        loss_scale=65536.0,
    )
    grad_clear_after_projection = all(
        parameter.grad is None for parameter in trainable_parameters
    )
    assign_scaled_gradients(
        trainable_parameters,
        projected_gradients,
        active_gradients,
        loss_scale=1.0,
    )

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
    for method in ("AUD_L4_A", "TEACHER_B_TARGET", "AUD_FINE_A", "AUD_FINE_B"):
        sums = output[method].sum(dim=(-2, -1))
        probability_errors[method] = float((sums - 1.0).abs().max())
    pooled_error = max(
        float(
            (
                model.sum_pool_2x2(output[method]).sum(dim=(-2, -1)) - 1.0
            ).abs().max()
        )
        for method in ("AUD_FINE_A", "AUD_FINE_B")
    )

    recomputed_teacher_b = transform_view_a_map_to_b(
        output["AUD_L4_A"], geometry, output_size=(7, 7)
    )
    teacher_b_transform_error = float(
        (recomputed_teacher_b - output["TEACHER_B_TARGET"]).abs().max()
    )
    view_b_independent_teacher_absent = (
        "AUD_L4_B" not in output and "AUD_L4_B_TO_A" not in output
    )
    full_image_mass_error = float(
        (geometry["full_image_teacher_mass"] - 1.0).abs().max()
    )
    selected_masses = geometry["teacher_mass"]
    nonfallback = ~geometry["fallback_identity"]
    accepted_min_mass = (
        float(selected_masses[nonfallback].min()) if nonfallback.any() else 1.0
    )
    all_selected_meet_mass = bool(
        (selected_masses >= arguments.min_teacher_mass - 1e-6).all()
    )

    uniform_teacher = torch.full(
        (2, 1, 7, 7), 1.0 / 49.0, device=device
    )
    forced_fallback = sample_semantic_preserving_crop(
        uniform_teacher,
        image_height=frame.shape[-2],
        image_width=frame.shape[-1],
        scale=(0.6, 0.6),
        ratio=(1.0, 1.0),
        flip_probability=0.5,
        min_teacher_mass=0.99,
        max_crop_attempts=2,
    )
    fallback_logic_ok = bool(
        forced_fallback["fallback_identity"].all()
        and (forced_fallback["crop_top"] == 0).all()
        and (forced_fallback["crop_left"] == 0).all()
        and (forced_fallback["crop_height"] == frame.shape[-2]).all()
        and (forced_fallback["crop_width"] == frame.shape[-1]).all()
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
    second_output = model.forward_two_views(frame, spec, geometry=geometry)
    second_losses = model.spatial_losses(
        second_output, lambda_equiv=arguments.lambda_equiv
    )
    second_projected, second_active, second_projection_stats = projected_loss_gradients(
        second_losses["loss_coarse"],
        second_losses["loss_equiv"],
        trainable_parameters,
        loss_scale=65536.0,
    )
    assign_scaled_gradients(
        trainable_parameters,
        second_projected,
        second_active,
        loss_scale=1.0,
    )
    second_proj3_grad = gradient_l1(model.student.proj3_spatial.parameters())
    second_adapter_grad = gradient_l1(model.student.adapter.parameters())
    second_teacher_has_grad = any(
        parameter.grad is not None for parameter in model.teacher.parameters()
    )
    model.student.load_state_dict(initial_student_state, strict=True)
    model.zero_grad(set_to_none=True)

    checks = {
        **geometry_checks,
        **projection_checks,
        "teacher_aud_l4_reproduced": True,
        "teacher_has_any_gradient": teacher_has_grad or second_teacher_has_grad,
        "view_b_independent_teacher_target_absent": view_b_independent_teacher_absent,
        "teacher_b_target_from_transform_a_max_abs_error": teacher_b_transform_error,
        "full_image_crop_teacher_mass_max_error": full_image_mass_error,
        "accepted_crop_min_teacher_mass": accepted_min_mass,
        "all_selected_crops_meet_teacher_mass": all_selected_meet_mass,
        "forced_fallback_identity_ok": fallback_logic_ok,
        "initial_adapter_gradient_l1": initial_adapter_grad,
        "initial_proj3_spatial_gradient_l1": initial_proj3_grad,
        "second_step_adapter_gradient_l1": second_adapter_grad,
        "second_step_proj3_spatial_gradient_l1": second_proj3_grad,
        "proj3_spatial_copy_max_abs_error": copy_error,
        "zero_init_f34_minus_up_f4_max_abs": zero_init_error,
        "f4_hook_minus_formal_tokens_max_abs": f4_token_error,
        "probability_sum_max_errors": probability_errors,
        "sum_pool_probability_sum_max_error": pooled_error,
        "projection_grad_clear_before": grad_was_clear_before,
        "projection_grad_clear_after_autograd": grad_clear_after_projection,
        "real_projection_stats": projection_stats,
        "second_step_projection_stats": second_projection_stats,
        "loss_total_equals_component_sum_error": float(
            (
                losses["loss_total"]
                - losses["loss_coarse"]
                - losses["loss_equiv"]
            ).abs()
        ),
        "mean_valid_ratio": float(losses["mean_valid_ratio"]),
        "skipped_small_overlap_samples": int(losses["skipped_small_overlap_samples"]),
        "loss_coarse": float(losses["loss_coarse"]),
        "loss_equiv": float(losses["loss_equiv"]),
        "loss_total": float(losses["loss_total"]),
        "student_restored_after_temporary_gradient_audit": True,
    }
    if checks["teacher_has_any_gradient"]:
        raise RuntimeError("Frozen teacher received a gradient")
    if not view_b_independent_teacher_absent or teacher_b_transform_error > 1e-7:
        raise RuntimeError("View B coarse target is not exclusively Transform(T_A)")
    if (
        full_image_mass_error > 1e-6
        or not all_selected_meet_mass
        or not fallback_logic_ok
    ):
        raise RuntimeError("Semantic-preserving crop sanity failed")
    if initial_adapter_grad <= 0 or second_adapter_grad <= 0 or second_proj3_grad <= 0:
        raise RuntimeError("Student gradient audit failed")
    if copy_error != 0 or zero_init_error > 1e-7 or f4_token_error > 1e-7:
        raise RuntimeError("Initialization or feature-source audit failed")
    if max(probability_errors.values()) > 1e-5 or pooled_error > 1e-5:
        raise RuntimeError("A probability map is not normalized")
    if not grad_was_clear_before or not grad_clear_after_projection:
        raise RuntimeError("autograd.grad unexpectedly wrote parameter .grad")
    if checks["loss_total_equals_component_sum_error"] > 1e-8:
        raise RuntimeError("Logged loss_total is not loss_coarse + loss_equiv")
    if not all(torch.isfinite(losses[key]) for key in ("loss_coarse", "loss_equiv", "loss_total")):
        raise RuntimeError("A spatial loss is NaN or Inf")

    print(json.dumps(checks, indent=2), flush=True)
    print("All Experiment G-v3 sanity checks passed.", flush=True)
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
    projection_audit_path: Path | None = None,
) -> dict[str, float]:
    model.train()
    loss_fields = ("loss_coarse", "loss_equiv", "loss_total")
    totals = {field: 0.0 for field in loss_fields}
    projection_fields = (
        "grad_norm_coarse",
        "grad_norm_equiv_before",
        "grad_norm_equiv_after",
        "grad_norm_final",
        "grad_cosine_before",
        "grad_cosine_after",
        "projection_alpha",
    )
    projection_totals = {field: 0.0 for field in projection_fields}
    projection_count = 0
    projection_applied_count = 0
    max_abs_conflict_cosine_after = 0.0
    last_projection: dict[str, float | bool] = {
        **{field: math.nan for field in projection_fields},
        "projection_applied": False,
    }
    valid_ratio_total = 0.0
    crop_area_total = 0.0
    crop_teacher_mass_total = 0.0
    crop_teacher_mass_min = math.inf
    crop_attempts_total = 0.0
    fallback_count = 0
    flipped_count = 0
    skipped_count = 0
    sample_count = 0
    batch_average = 0.0
    epoch_start = time.time()
    trainable_parameters = [
        parameter
        for parameter in model.student.parameters()
        if parameter.requires_grad
    ]

    for batch_index, (frame, spec, _bboxes, _names, _labels) in enumerate(train_loader):
        batch_start = time.time()
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            output = model.forward_two_views(
                frame,
                spec,
                crop_scale_range=(arguments.crop_scale_min, arguments.crop_scale_max),
                crop_ratio_range=(arguments.crop_ratio_min, arguments.crop_ratio_max),
                flip_probability=arguments.flip_probability,
                min_teacher_mass=arguments.min_teacher_mass,
                max_crop_attempts=arguments.max_crop_attempts,
            )
            losses = model.spatial_losses(output, lambda_equiv=arguments.lambda_equiv)

        # Both component gradients are produced at the same AMP scale, brought
        # back to one unscaled FP32 space, combined once, and only then written
        # to .grad. loss_total is deliberately never backward-ed in G-v3.
        if any(parameter.grad is not None for parameter in trainable_parameters):
            raise RuntimeError("Projected step started with non-empty .grad")
        current_scale = float(scaler.get_scale())
        coarse_scaled = scaler.scale(losses["loss_coarse"])
        equiv_scaled = scaler.scale(losses["loss_equiv"])
        final_gradients, active_gradients, last_projection = projected_loss_gradients(
            coarse_scaled,
            equiv_scaled,
            trainable_parameters,
            loss_scale=current_scale,
            losses_already_scaled=True,
        )
        if any(parameter.grad is not None for parameter in trainable_parameters):
            raise RuntimeError("autograd.grad unexpectedly modified parameter .grad")
        assign_scaled_gradients(
            trainable_parameters,
            final_gradients,
            active_gradients,
            loss_scale=current_scale,
        )
        if any(parameter.grad is not None for parameter in model.teacher.parameters()):
            raise RuntimeError("Frozen teacher received a gradient during training")
        scaler.step(optimizer)
        scaler.update()

        projection_count += 1
        projection_applied = bool(last_projection["projection_applied"])
        projection_applied_count += int(projection_applied)
        for field in projection_fields:
            projection_totals[field] += float(last_projection[field])
        if projection_applied:
            max_abs_conflict_cosine_after = max(
                max_abs_conflict_cosine_after,
                abs(float(last_projection["grad_cosine_after"])),
            )
        if batch_index % arguments.gradient_diagnostic_interval == 0:
            audit_record = {
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "num_batches": len(train_loader),
                **last_projection,
            }
            audit_line = json.dumps(audit_record, sort_keys=True)
            print(f"ProjectionAudit {audit_line}", flush=True)
            if projection_audit_path is not None:
                with projection_audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(audit_line + "\n")

        batch_size = frame.shape[0]
        geometry = output["geometry"]
        sample_count += batch_size
        flipped_count += int(geometry["flipped"].sum())
        fallback_count += int(geometry["fallback_identity"].sum())
        skipped_count += int(losses["skipped_small_overlap_samples"])
        valid_ratio_total += float(losses["mean_valid_ratio"]) * batch_size
        crop_area_total += float(output["mean_crop_scale"]) * batch_size
        crop_teacher_mass_total += float(geometry["teacher_mass"].sum())
        crop_teacher_mass_min = min(
            crop_teacher_mass_min, float(geometry["teacher_mass"].min())
        )
        crop_attempts_total += float(geometry["crop_attempts"].sum())
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
                f"area={float(output['mean_crop_scale']):.3f} "
                f"mass={float(geometry['teacher_mass'].mean()):.3f} "
                f"attempts={float(geometry['crop_attempts'].mean()):.2f} "
                f"fallback={int(geometry['fallback_identity'].sum())} "
                f"projected={int(bool(last_projection['projection_applied']))} "
                f"cos={float(last_projection['grad_cosine_before']):.4f}->"
                f"{float(last_projection['grad_cosine_after']):.4e} "
                f"alpha={float(last_projection['projection_alpha']):.4e} ETA={eta}",
                flush=True,
            )

    if projection_count != len(train_loader):
        raise RuntimeError("Gradient projection was not executed for every batch")
    result = {field: value / sample_count for field, value in totals.items()}
    result.update(
        {
            "mean_valid_ratio": valid_ratio_total / sample_count,
            "skipped_small_overlap_samples": skipped_count,
            "actual_flip_ratio": flipped_count / sample_count,
            "mean_crop_area_ratio": crop_area_total / sample_count,
            "mean_crop_teacher_mass": crop_teacher_mass_total / sample_count,
            "min_crop_teacher_mass": crop_teacher_mass_min,
            "mean_crop_attempts": crop_attempts_total / sample_count,
            "fallback_identity_count": fallback_count,
            "grad_norm_coarse": projection_totals["grad_norm_coarse"]
            / projection_count,
            "grad_norm_equiv_before": projection_totals[
                "grad_norm_equiv_before"
            ]
            / projection_count,
            "grad_norm_equiv_after": projection_totals[
                "grad_norm_equiv_after"
            ]
            / projection_count,
            "grad_norm_final": projection_totals["grad_norm_final"]
            / projection_count,
            "projection_rate": projection_applied_count / projection_count,
            "mean_projection_alpha": projection_totals["projection_alpha"]
            / projection_count,
            "mean_cosine_before": projection_totals["grad_cosine_before"]
            / projection_count,
            "mean_cosine_after": projection_totals["grad_cosine_after"]
            / projection_count,
            "max_abs_conflict_cosine_after": max_abs_conflict_cosine_after,
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
        "architecture": "semantic_priority_gradient_refine_g_v3",
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
    if not 0 <= arguments.min_teacher_mass <= 1:
        raise ValueError("--min-teacher-mass must be in [0,1]")
    if arguments.max_crop_attempts < 1:
        raise ValueError("--max-crop-attempts must be positive")
    if arguments.gradient_diagnostic_interval < 1:
        raise ValueError("--gradient-diagnostic-interval must be positive")
    if arguments.lambda_equiv != 1.0:
        raise ValueError("G-v3 fixes lambda_equiv=1.0")
    if arguments.init_lr != 5e-5 or arguments.weight_decay != 0.01:
        raise ValueError("G-v3 fixes AdamW lr=5e-5 and weight_decay=0.01")
    if arguments.epochs != 10:
        raise ValueError("The first G-v3 experiment is fixed to 10 epochs")
    if (
        arguments.crop_scale_min != 0.6
        or arguments.crop_scale_max != 1.0
        or arguments.crop_ratio_min != 0.9
        or arguments.crop_ratio_max != 1.1
        or arguments.flip_probability != 0.5
        or arguments.min_teacher_mass != 0.60
        or arguments.max_crop_attempts != 10
        or arguments.minimum_valid_ratio != 0.2
    ):
        raise ValueError("G-v3 crop, flip, teacher-mass, and valid-ratio settings are fixed")
    registry = EXPERIMENTS[arguments.experiment]
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    epochs = arguments.epochs
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
        arguments.model_dir
        / "1.5G-v3semantic_priority_gradient_refine_sanity"
        / arguments.experiment
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
        "architecture": "semantic_priority_gradient_refine_g_v3",
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
        "lambda_coarse": 1.0,
        "lambda_equiv": arguments.lambda_equiv,
        "random_resized_crop_scale": [arguments.crop_scale_min, arguments.crop_scale_max],
        "random_resized_crop_ratio": [arguments.crop_ratio_min, arguments.crop_ratio_max],
        "flip_probability": arguments.flip_probability,
        "min_teacher_mass": arguments.min_teacher_mass,
        "max_crop_attempts": arguments.max_crop_attempts,
        "single_semantic_anchor": "AUD_L4_A; View B target = Transform_A_to_B(AUD_L4_A)",
        "minimum_valid_ratio": arguments.minimum_valid_ratio,
        "gradient_diagnostic_interval": arguments.gradient_diagnostic_interval,
        "gradient_combination": "semantic-priority global projection",
        "projection_priority": "coarse semantic gradient is never modified",
        "projection_scope": "one global dot/norm/alpha across all trainable parameters",
        "loss_total_usage": "logging only; loss_total.backward() is forbidden",
        "amp_projection_space": "unscaled FP32 gradients",
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
    projection_audit_path = model_dir / "gradient_projection_audit.jsonl"
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
            projection_audit_path=projection_audit_path,
        )
        print(
            f"Epoch {epoch + 1}/{epochs} losses: "
            f"coarse={train_metrics['loss_coarse']:.8f} "
            f"equiv={train_metrics['loss_equiv']:.8f} "
            f"total={train_metrics['loss_total']:.8f} "
            f"valid={train_metrics['mean_valid_ratio']:.4f} "
            f"skipped={train_metrics['skipped_small_overlap_samples']} "
            f"flip={train_metrics['actual_flip_ratio']:.4f} "
            f"area={train_metrics['mean_crop_area_ratio']:.4f} "
            f"mass_mean/min={train_metrics['mean_crop_teacher_mass']:.4f}/"
            f"{train_metrics['min_crop_teacher_mass']:.4f} "
            f"attempts={train_metrics['mean_crop_attempts']:.3f} "
            f"fallback={train_metrics['fallback_identity_count']} "
            f"grad_norm(c/e/e_proj/final)="
            f"{train_metrics['grad_norm_coarse']:.6f}/"
            f"{train_metrics['grad_norm_equiv_before']:.6f}/"
            f"{train_metrics['grad_norm_equiv_after']:.6f}/"
            f"{train_metrics['grad_norm_final']:.6f} "
            f"projection_rate={train_metrics['projection_rate']:.4f} "
            f"alpha={train_metrics['mean_projection_alpha']:.6e} "
            f"cos_before/after={train_metrics['mean_cosine_before']:.4f}/"
            f"{train_metrics['mean_cosine_after']:.4e} "
            f"max_abs_conflict_cos_after="
            f"{train_metrics['max_abs_conflict_cosine_after']:.4e}",
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
        f"Total Experiment G-v3 training time: {timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
