#!/usr/bin/env python3
"""Experiment 2.3: semantic-spatial decoupled Slot learning."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch

import runtime


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def _load_local_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(HERE / filename)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SemanticSpatialDecoupledModel = _load_local_module(
    "decoupled_model_23", "model.py"
).SemanticSpatialDecoupledModel
render = _load_local_module("decoupled_curves_23", "curves.py").render
evaluation = _load_local_module("decoupled_evaluation_23", "evaluation.py")
evaluate = evaluation.evaluate
save_detailed_result = evaluation.save_detailed_result
save_qualitative = evaluation.save_qualitative
select_qualitative = evaluation.select_qualitative

HISTORY_FIELDS = (
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "loss_seed",
    "loss_equiv",
    "loss_visual",
    "loss_mass",
    "weighted_seed",
    "weighted_equiv",
    "weighted_visual",
    "weighted_mass",
    "loss_total",
    "grad_norm_seed",
    "grad_norm_equiv",
    "grad_norm_visual",
    "grad_norm_mass",
    "gradient_diagnostic_batches",
    "slot0_mass",
    "slot1_mass",
    "ownership_entropy",
    "valid_overlap_ratio",
    "aud_ciou",
    "aud_auc",
    "spatial_slot_ciou",
    "spatial_slot_auc",
    "aud_spatial_ciou",
    "aud_spatial_auc",
    "rescue",
    "hurt",
    "net_rescue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=runtime.EXPERIMENTS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--init-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-seed", type=float, default=1.0)
    parser.add_argument("--lambda-equiv", type=float, default=1.0)
    parser.add_argument("--lambda-visual", type=float, default=0.1)
    parser.add_argument("--lambda-mass", type=float, default=0.1)
    parser.add_argument("--gradient-interval", type=int, default=50)
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
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    bad_names = [name for name in trainable_names if not name.startswith("spatial_slot.")]
    total = sum(parameter.numel() for parameter in model.parameters())
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    branch = sum(parameter.numel() for parameter in model.spatial_slot.parameters())
    audit = {
        "total_parameters": total,
        "frozen_parameters": frozen,
        "trainable_parameters": trainable,
        "spatial_slot_parameters": branch,
        "trainable_parameter_names": trainable_names,
        "invalid_trainable_names": bad_names,
    }
    if bad_names or trainable != branch or not trainable_names:
        raise RuntimeError(f"Parameter freeze audit failed: {audit}")
    return audit


def _global_gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(squared.sqrt())


def component_gradient_norms(
    losses: dict[str, torch.Tensor], parameters: list[torch.nn.Parameter]
) -> dict[str, float]:
    return {
        name: _global_gradient_norm(losses[f"loss_{name}"], parameters)
        for name in ("seed", "equiv", "visual", "mass")
    }


def _loss_kwargs(arguments: argparse.Namespace) -> dict[str, float]:
    return {
        "lambda_seed": arguments.lambda_seed,
        "lambda_equiv": arguments.lambda_equiv,
        "lambda_visual": arguments.lambda_visual,
        "lambda_mass": arguments.lambda_mass,
    }


def smoke_audit(
    model,
    train_dataset,
    test_loader,
    config,
    experiment: str,
    device: torch.device,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    print("Running 2.3 forward/backward/freeze smoke audit...", flush=True)
    loader = runtime.build_smoke_loader(train_dataset, batch_size=4)
    frame, spec, _bboxes, names, _labels = next(iter(loader))
    frame = frame.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    geometry = runtime.geometry.sample_random_resized_crop(
        frame.shape[0],
        frame.shape[-2],
        frame.shape[-1],
        device,
        scale=(0.6, 1.0),
        ratio=(0.9, 1.1),
        flip_probability=0.5,
    )
    before = copy.deepcopy(model.spatial_slot.state_dict())
    model.train()
    model.zero_grad(set_to_none=True)
    output = model.forward_two_views(frame, spec, geometry)
    losses = model.losses(output, **_loss_kwargs(arguments))
    parameters = list(model.spatial_slot.parameters())
    component_norms = component_gradient_norms(losses, parameters)
    losses["loss_total"].backward()

    branch_grad = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    frozen_has_gradient = any(
        parameter.grad is not None for parameter in model.refinement.parameters()
    )
    ownership_error = float(
        (output["OWNERSHIP_A"].sum(dim=1) - 1.0).abs().max()
    )
    old_ownership_error = float(
        (output["OLD_OWNERSHIP_A"].sum(dim=1) - 1.0).abs().max()
    )
    with torch.no_grad():
        official_aud = model.refinement(frame, spec)["AUD_FINE"]
    aud_error = float((official_aud - output["AUD_FINE_A"]).abs().max())
    state_error = max(
        float((before[key] - model.spatial_slot.state_dict()[key]).abs().max())
        for key in before
    )
    finite = all(
        torch.isfinite(losses[key])
        for key in ("loss_seed", "loss_equiv", "loss_visual", "loss_mass", "loss_total")
    )
    checks = {
        "sample_ids": [str(name) for name in names],
        "F34_shape": list(output["F34_A"].shape),
        "semantic_initial_slots_shape": list(output["SEMANTIC_INITIAL_A"].shape),
        "ownership_shape": list(output["OWNERSHIP_A"].shape),
        "F34_requires_grad": output["F34_A"].requires_grad,
        "semantic_initial_slots_requires_grad": output["SEMANTIC_INITIAL_A"].requires_grad,
        "AUD_FINE_requires_grad": output["AUD_FINE_A"].requires_grad,
        "ownership_slot_sum_max_error": ownership_error,
        "old_ownership_slot_sum_max_error": old_ownership_error,
        "local_vs_official_AUD_FINE_max_error": aud_error,
        "f4_token_error": float(output["f4_token_error"]),
        "frozen_model_has_gradient": frozen_has_gradient,
        "spatial_slot_gradient_l1": branch_grad,
        "component_gradient_norms": component_norms,
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "no_nan_or_inf": bool(finite),
        "slot0_is_fixed_target": True,
        "hungarian_matching": False,
        "smoke_backward_changed_parameters_max_error": state_error,
    }
    expected_shapes = {
        "F34_shape": [frame.shape[0], 512, 14, 14],
        "semantic_initial_slots_shape": [frame.shape[0], 2, 512],
        "ownership_shape": [frame.shape[0], 2, 196],
    }
    if any(checks[key] != value for key, value in expected_shapes.items()):
        raise RuntimeError(f"Smoke shape audit failed: {checks}")
    if checks["F34_requires_grad"] or checks["semantic_initial_slots_requires_grad"] or checks["AUD_FINE_requires_grad"]:
        raise RuntimeError("A frozen semantic tensor unexpectedly requires gradients")
    if frozen_has_gradient or branch_grad <= 0 or min(component_norms.values()) <= 0:
        raise RuntimeError(f"Gradient audit failed: {checks}")
    if max(ownership_error, old_ownership_error, aud_error, float(output["f4_token_error"]), state_error) > 1e-6:
        raise RuntimeError(f"Tensor audit failed: {checks}")
    if not finite:
        raise RuntimeError("NaN/Inf in smoke loss")
    model.zero_grad(set_to_none=True)

    print("Running full frozen AUD baseline reproduction...", flush=True)
    baseline = evaluate(model, test_loader, device)
    aud_metric = baseline["metrics"]["AUD_FINE"]
    if not runtime.metric_matches(aud_metric, runtime.EXPECTED_AUD[experiment]):
        raise RuntimeError(
            f"AUD baseline mismatch: {aud_metric} vs {runtime.EXPECTED_AUD[experiment]}"
        )
    checks["AUD_baseline"] = aud_metric
    checks["AUD_baseline_reproduced"] = True
    print(json.dumps(checks, indent=2), flush=True)
    return checks


def append_history(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
        "architecture": "semantic_spatial_decoupled_slot",
        "base_1_3g_checkpoint": str(base_checkpoint),
        "spatial_slot_state_dict": model.spatial_slot.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": validation["metrics"],
        "rescue_hurt": validation["rescue_hurt"],
        "oracle": validation["oracle"],
    }
    if selection_metric is not None:
        payload["selection_metric"] = selection_metric
        payload["selection_score"] = validation["metrics"]["AUD_SPATIAL"]["cIoU"]
    torch.save(payload, path)


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    arguments: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    metric_keys = (
        "loss_seed",
        "loss_equiv",
        "loss_visual",
        "loss_mass",
        "weighted_seed",
        "weighted_equiv",
        "weighted_visual",
        "weighted_mass",
        "loss_total",
        "slot0_mass",
        "slot1_mass",
        "ownership_entropy",
        "valid_overlap_ratio",
    )
    totals = {key: 0.0 for key in metric_keys}
    grad_totals = {key: 0.0 for key in ("seed", "equiv", "visual", "mass")}
    diagnostic_batches = 0
    samples = 0
    batch_average = 0.0
    epoch_start = time.time()
    parameters = list(model.spatial_slot.parameters())

    for batch_index, (frame, spec, _bboxes, _names, _labels) in enumerate(loader):
        batch_start = time.time()
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        geometry = runtime.geometry.sample_random_resized_crop(
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
            losses = model.losses(output, **_loss_kwargs(arguments))

        if batch_index % arguments.gradient_interval == 0:
            norms = component_gradient_norms(losses, parameters)
            for key, value in norms.items():
                grad_totals[key] += value
            diagnostic_batches += 1
        scaler.scale(losses["loss_total"]).backward()
        if any(parameter.grad is not None for parameter in model.refinement.parameters()):
            raise RuntimeError("Frozen 1.3G received a training gradient")
        scaler.step(optimizer)
        scaler.update()

        batch_size = frame.shape[0]
        samples += batch_size
        for key in metric_keys:
            totals[key] += float(losses[key].detach()) * batch_size
        completed = batch_index + 1
        batch_seconds = time.time() - batch_start
        batch_average += (batch_seconds - batch_average) / completed
        if batch_index % 10 == 0 or completed == len(loader):
            remaining = (
                len(loader) - completed
                + (arguments.epochs - epoch - 1) * len(loader)
            )
            eta = timedelta(seconds=int(batch_average * remaining))
            print(
                f"Train [{epoch + 1}/{arguments.epochs}] [{completed}/{len(loader)}] "
                f"seed={float(losses['loss_seed']):.6f} "
                f"equiv={float(losses['loss_equiv']):.6f} "
                f"visual={float(losses['loss_visual']):.6f} "
                f"mass={float(losses['loss_mass']):.6f} "
                f"total={float(losses['loss_total']):.6f} "
                f"m0={float(losses['slot0_mass']):.3f} "
                f"H={float(losses['ownership_entropy']):.3f} ETA={eta}",
                flush=True,
            )

    result = {key: value / samples for key, value in totals.items()}
    result.update(
        {
            f"grad_norm_{key}": grad_totals[key] / diagnostic_batches
            for key in grad_totals
        }
    )
    result["gradient_diagnostic_batches"] = diagnostic_batches
    result["epoch_seconds"] = time.time() - epoch_start
    return result


def _print_validation(epoch: int, validation: dict[str, Any]) -> None:
    metrics = validation["metrics"]
    counts = validation["rescue_hurt"]["new"]
    print(
        f"Epoch {epoch} AUD {metrics['AUD_FINE']['cIoU']:.4f}/{metrics['AUD_FINE']['AUC']:.4f} | "
        f"SPATIAL_SLOT0 {metrics['SPATIAL_SLOT0']['cIoU']:.4f}/{metrics['SPATIAL_SLOT0']['AUC']:.4f} | "
        f"AUD_SPATIAL {metrics['AUD_SPATIAL']['cIoU']:.4f}/{metrics['AUD_SPATIAL']['AUC']:.4f} | "
        f"Rescue/Hurt/Net {counts['rescue']}/{counts['hurt']}/{counts['net']}",
        flush=True,
    )


def main() -> None:
    arguments = parse_args()
    if arguments.epochs != 50 and not arguments.smoke_only:
        raise ValueError("Formal Experiment 2.3 is fixed to 50 epochs")
    if (
        arguments.init_lr != 5e-5
        or arguments.weight_decay != 0.01
        or arguments.lambda_seed != 1.0
        or arguments.lambda_equiv != 1.0
        or arguments.lambda_visual != 0.1
        or arguments.lambda_mass != 0.1
    ):
        raise ValueError("First-version 2.3 hyperparameters are fixed by protocol")

    loaded = runtime.load_formal_1_3g(arguments.experiment, arguments.gpu)
    (
        registry,
        _formal_name,
        _formal_dir,
        formal_checkpoint,
        _formal_payload,
        config,
        refinement,
        _base_checkpoint,
        test_loader,
        device,
    ) = loaded
    runtime.setup_seed(config.seed)
    model = SemanticSpatialDecoupledModel(refinement, runtime.geometry, iters=5).to(device)
    counts = parameter_audit(model)
    print(json.dumps(counts, indent=2), flush=True)
    train_dataset = runtime.build_train_dataset(config)

    formal_hash_before = sha256(formal_checkpoint)
    formal_mtime_before = formal_checkpoint.stat().st_mtime_ns
    smoke = smoke_audit(
        model,
        train_dataset,
        test_loader,
        config,
        arguments.experiment,
        device,
        arguments,
    )
    if arguments.smoke_only:
        output_dir = arguments.smoke_output_root / arguments.experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": arguments.experiment,
            "parameter_audit": counts,
            "smoke": smoke,
            "formal_checkpoint": str(formal_checkpoint),
            "formal_checkpoint_unchanged": (
                formal_hash_before == sha256(formal_checkpoint)
                and formal_mtime_before == formal_checkpoint.stat().st_mtime_ns
            ),
        }
        (output_dir / "smoke_audit.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        model.close()
        print("Smoke-only audit passed; no optimizer or checkpoint was created.", flush=True)
        return

    experiment_name = arguments.experiment_name or runtime.DEFAULT_NAMES[arguments.experiment]
    model_dir = arguments.model_dir / experiment_name
    protected = ("latest.pth", "final.pth", f"{registry['dataset']}_best.pth")
    if any((model_dir / name).exists() for name in protected):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "architecture": "semantic_spatial_decoupled_slot",
        "experiment": arguments.experiment,
        "experiment_name": experiment_name,
        "formal_1_3g_checkpoint": str(formal_checkpoint),
        "epochs": arguments.epochs,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "optimizer": "AdamW",
        "init_lr": arguments.init_lr,
        "weight_decay": arguments.weight_decay,
        "scheduler": False,
        "slot_iterations": 5,
        "lambda_seed": arguments.lambda_seed,
        "lambda_equiv": arguments.lambda_equiv,
        "lambda_visual": arguments.lambda_visual,
        "lambda_mass": arguments.lambda_mass,
        "seed_top_fraction": 0.10,
        "crop_scale": [0.6, 1.0],
        "crop_ratio": [0.9, 1.1],
        "flip_probability": 0.5,
        "checkpoint_selection": "AUD_SPATIAL_cIoU_alpha_0.6",
        "parameter_audit": counts,
        "smoke_audit": smoke,
        "forbidden_training_inputs": ["GT", "OGL", "OBJ_PRIOR"],
    }
    (model_dir / "configs.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (model_dir / "smoke_audit.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(
        model.spatial_slot.parameters(),
        lr=arguments.init_lr,
        weight_decay=arguments.weight_decay,
    )
    optimized = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if optimized != counts["spatial_slot_parameters"]:
        raise RuntimeError("Optimizer contains parameters outside spatial_slot")
    scaler = torch.amp.GradScaler("cuda")
    runtime.setup_seed(config.seed)
    train_loader = runtime.build_train_loader(train_dataset, config)
    history_path = model_dir / "epoch_metrics.csv"
    best_score = -math.inf
    start = time.time()

    for epoch in range(arguments.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, arguments
        )
        validation = evaluate(model, test_loader, device)
        _print_validation(epoch + 1, validation)
        save_checkpoint(
            model_dir / "latest.pth",
            model,
            optimizer,
            epoch + 1,
            validation,
            formal_checkpoint,
        )
        score = validation["metrics"]["AUD_SPATIAL"]["cIoU"]
        if score > best_score:
            best_score = score
            save_checkpoint(
                model_dir / f"{registry['dataset']}_best.pth",
                model,
                optimizer,
                epoch + 1,
                validation,
                formal_checkpoint,
                selection_metric="AUD_SPATIAL_cIoU_alpha_0.6",
            )
            print(f"New best epoch {epoch + 1}: AUD_SPATIAL cIoU={score:.4f}", flush=True)

        counts_epoch = validation["rescue_hurt"]["new"]
        metrics_epoch = validation["metrics"]
        history_row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "aud_ciou": metrics_epoch["AUD_FINE"]["cIoU"],
            "aud_auc": metrics_epoch["AUD_FINE"]["AUC"],
            "spatial_slot_ciou": metrics_epoch["SPATIAL_SLOT0"]["cIoU"],
            "spatial_slot_auc": metrics_epoch["SPATIAL_SLOT0"]["AUC"],
            "aud_spatial_ciou": metrics_epoch["AUD_SPATIAL"]["cIoU"],
            "aud_spatial_auc": metrics_epoch["AUD_SPATIAL"]["AUC"],
            "rescue": counts_epoch["rescue"],
            "hurt": counts_epoch["hurt"],
            "net_rescue": counts_epoch["net"],
        }
        append_history(history_path, history_row)
        render(
            history_path,
            model_dir / "training_curves",
            experiment_name,
            smoke["AUD_baseline"]["cIoU"],
            smoke["AUD_baseline"]["AUC"],
        )
        elapsed = time.time() - start
        remaining = elapsed / (epoch + 1) * (arguments.epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/50 complete in {timedelta(seconds=int(train_metrics['epoch_seconds']))}; "
            f"overall ETA {timedelta(seconds=int(remaining))}; finish "
            f"{(datetime.now() + timedelta(seconds=remaining)):%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    final_validation = evaluate(model, test_loader, device)
    save_checkpoint(
        model_dir / "final.pth",
        model,
        optimizer,
        arguments.epochs,
        final_validation,
        formal_checkpoint,
    )
    best_path = model_dir / f"{registry['dataset']}_best.pth"
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.spatial_slot.load_state_dict(best_payload["spatial_slot_state_dict"], strict=True)
    object_model = runtime.probe20.object_prior_model().to(device).eval()
    detailed = evaluate(
        model, test_loader, device, object_model=object_model, include_alpha=True
    )
    if not runtime.metric_matches(detailed["metrics"]["AUD_FINE"], runtime.EXPECTED_AUD[arguments.experiment]):
        raise RuntimeError("Best-checkpoint frozen AUD no longer matches 1.3G")
    result_dir = model_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_detailed_result(detailed, result_dir)
    selected = select_qualitative(detailed["rows"], count=12)
    save_qualitative(
        model,
        test_loader,
        object_model,
        selected,
        detailed["rows"],
        device,
        result_dir / "qualitative",
    )

    formal_unchanged = (
        formal_hash_before == sha256(formal_checkpoint)
        and formal_mtime_before == formal_checkpoint.stat().st_mtime_ns
    )
    summary = {
        "experiment": "2.3 Semantic-Spatial Decoupled Slot Learning",
        "dataset": arguments.experiment,
        "best_epoch": int(best_payload["epoch"]),
        "parameter_audit": counts,
        "smoke_audit": smoke,
        "best_results": {key: value for key, value in detailed.items() if key != "rows"},
        "formal_1_3g_checkpoint": str(formal_checkpoint),
        "formal_1_3g_checkpoint_unchanged": formal_unchanged,
        "training_seconds": time.time() - start,
        "qualitative_ids": selected,
    }
    if not formal_unchanged:
        raise RuntimeError("Formal 1.3G checkpoint changed during Experiment 2.3")
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Total Experiment 2.3 time: {timedelta(seconds=int(time.time() - start))}", flush=True)
    model.close()


if __name__ == "__main__":
    main()
