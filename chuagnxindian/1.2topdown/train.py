#!/usr/bin/env python3
"""Experiment D: joint fine-tuning of L3+L4 JSA and top-down refinement."""

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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
mpl_cache = Path("/tmp") / f"joint_topdown_mpl_{os.getuid()}"
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
    parameter_audit,
    setup_seed,
)
from curves import render_training_curves  # noqa: E402
from protocol import ProtocolAccumulator, evaluate_maps  # noqa: E402


MAP_METHODS = ("AUD_L4", "AUD_FINE", "IMG_L4", "IMG_FINE", "IQR_FINE")
LOSS_FIELDS = (
    "info_loss",
    "recon_loss",
    "div_loss",
    "att_loss",
    "loss_fine_match",
    "loss_coarse_aud",
    "loss_coarse_img",
    "refine_loss",
    "base_loss",
    "total_loss",
)
HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "epoch_seconds",
    *LOSS_FIELDS,
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
    parser.add_argument("--lambda-f", type=float, default=1.0)
    parser.add_argument("--init-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--sanity-only", action="store_true")
    # reciprocal_k=20 requires the diagnostic batch to contain at least 20
    # examples; 32 is still much cheaper than the formal batch size 256.
    parser.add_argument("--sanity-batch-size", type=int, default=32)
    return parser.parse_args()


def new_accumulators() -> dict[str, ProtocolAccumulator]:
    return {method: ProtocolAccumulator() for method in MAP_METHODS}


@torch.inference_mode()
def validate(model, test_loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    accumulators = new_accumulators()
    for image, spec, bboxes, names, _labels in tqdm(
        test_loader, desc="Validate", dynamic_ncols=True
    ):
        image, spec, bboxes, names = flatten_eval_batch(
            image, spec, bboxes, names
        )
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        evaluate_maps(model(image, spec), bboxes, names, accumulators)
    return {
        method: accumulator.finalize()
        for method, accumulator in accumulators.items()
    }


def print_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    for method in MAP_METHODS:
        value = metrics[method]
        print(
            f"{prefix}{method}/cIoU,AUC "
            f"{value['cIoU']:.4f} {value['AUC']:.4f}",
            flush=True,
        )


def metric_matches(observed: dict[str, float], expected: tuple[float, float]) -> bool:
    return (
        f"{observed['cIoU']:.4f}" == f"{expected[0]:.4f}"
        and f"{observed['AUC']:.4f}" == f"{expected[1]:.4f}"
    )


def compose_losses(
    output: dict[str, torch.Tensor],
    config: argparse.Namespace,
    epoch: int,
    lambda_f: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_base_loss = (
        output["info_loss"]
        + config.lam1 * output["recon_loss"]
        + config.lam2 * output["div_loss"]
        + config.lam3 * output["att_loss"]
    )
    # Preserve the original training rule. Current 10k configs use warmup=-1,
    # so the complete four-loss expression is active from epoch zero.
    base_loss = output["info_loss"] if epoch < config.warmup else full_base_loss
    total_loss = base_loss + lambda_f * output["refine_loss"]
    return base_loss, total_loss


def optimizer_parameter_count(optimizer) -> int:
    return sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def make_optimizer(model, init_lr: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters, lr=init_lr, weight_decay=weight_decay
    )


def run_sanity_checks(
    model,
    train_dataset,
    test_loader,
    config: argparse.Namespace,
    registry: dict,
    device: torch.device,
    lambda_f: float,
    sanity_batch_size: int,
) -> dict[str, Any]:
    print("Running Experiment D baseline and gradient audit...", flush=True)
    baseline_metrics = validate(model, test_loader, device)
    print_metrics(baseline_metrics, prefix="SANITY/")
    if not metric_matches(baseline_metrics["AUD_L4"], registry["expected_aud"]):
        raise RuntimeError("AUD_L4 does not reproduce the L3+L4 checkpoint")
    if not metric_matches(baseline_metrics["IMG_L4"], registry["expected_img"]):
        raise RuntimeError("IMG_L4 does not reproduce the L3+L4 checkpoint")

    loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, sanity_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    frame, spec, _bboxes, _names, _labels = next(iter(loader))
    frame = frame.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(frame, spec)
    base_loss, total_loss = compose_losses(
        output, config, epoch=0, lambda_f=lambda_f
    )
    total_loss.backward()

    trainable_base_grad_l1 = 0.0
    trainable_base_grad_tensors = 0
    frozen_base_grad_names = []
    for name, parameter in model.base_model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            trainable_base_grad_tensors += 1
            trainable_base_grad_l1 += float(parameter.grad.detach().abs().sum())
        if not parameter.requires_grad and parameter.grad is not None:
            frozen_base_grad_names.append(name)
    head_grad_l1 = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.refinement_head.parameters()
        if parameter.grad is not None
    )

    zero_init_error = float((output["F34"] - output["F4_UP"]).abs().max())
    source_token_error = float(output["source_token_max_abs"])
    probability_errors = {
        method: float(
            (output[method].sum(dim=(-2, -1)) - 1.0).abs().max()
        )
        for method in ("AUD_FINE", "IMG_FINE", "AUD_L4", "IMG_L4")
    }
    pooled_sum_error = float(
        (
            model.sum_pool_2x2(output["AUD_FINE"]).sum(dim=(-2, -1))
            - 1.0
        ).abs().max()
    )
    audit = {
        "aud_l4_reproduced": True,
        "img_l4_reproduced": True,
        "trainable_base_gradient_tensor_count": trainable_base_grad_tensors,
        "trainable_base_gradient_l1": trainable_base_grad_l1,
        "refinement_head_gradient_l1": head_grad_l1,
        "frozen_base_parameters_with_gradient": frozen_base_grad_names,
        "f34_minus_up_f4_max_abs": zero_init_error,
        "f4_hook_minus_formal_tokens_max_abs": source_token_error,
        "probability_sum_max_errors": probability_errors,
        "sum_pool_probability_sum_max_error": pooled_sum_error,
        "base_loss": float(base_loss.detach()),
        "refine_loss": float(output["refine_loss"].detach()),
        "total_loss": float(total_loss.detach()),
    }
    if trainable_base_grad_tensors == 0 or trainable_base_grad_l1 <= 0:
        raise RuntimeError("Trainable base model did not receive a nonzero gradient")
    if head_grad_l1 <= 0:
        raise RuntimeError("Refinement head did not receive a nonzero gradient")
    if frozen_base_grad_names:
        raise RuntimeError(
            f"Originally frozen parameters received gradients: {frozen_base_grad_names}"
        )
    if zero_init_error > 1e-7 or source_token_error > 1e-7:
        raise RuntimeError(f"Feature/zero-init check failed: {audit}")
    if max(probability_errors.values()) > 1e-5 or pooled_sum_error > 1e-5:
        raise RuntimeError(f"Probability normalization check failed: {audit}")
    print(json.dumps(audit, indent=2), flush=True)
    print("Experiment D sanity and gradient audit passed.", flush=True)
    model.zero_grad(set_to_none=True)
    return {"baseline_metrics": baseline_metrics, "gradient_audit": audit}


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device,
    config: argparse.Namespace,
    epoch: int,
    total_epochs: int,
    lambda_f: float,
) -> dict[str, float]:
    model.train()
    totals = {field: 0.0 for field in LOSS_FIELDS}
    sample_count = 0
    batch_average = 0.0
    epoch_start = time.time()
    for batch_index, (frame, spec, _bboxes, _names, _labels) in enumerate(train_loader):
        batch_start = time.time()
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            output = model(frame, spec)
            base_loss, total_loss = compose_losses(
                output, config, epoch=epoch, lambda_f=lambda_f
            )
        scaler.scale(total_loss).backward()
        if batch_index == 0:
            if not any(
                parameter.grad is not None
                for parameter in model.base_model.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError("Trainable base model has no gradient")
            frozen_grad = [
                name
                for name, parameter in model.base_model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if frozen_grad:
                raise RuntimeError(f"Frozen base parameters have gradients: {frozen_grad}")
        scaler.step(optimizer)
        scaler.update()

        batch_size = frame.shape[0]
        sample_count += batch_size
        batch_losses = {
            **{field: output[field] for field in LOSS_FIELDS if field in output},
            "base_loss": base_loss,
            "total_loss": total_loss,
        }
        for field in LOSS_FIELDS:
            totals[field] += float(batch_losses[field].detach()) * batch_size

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
                f"Train [{epoch + 1}/{total_epochs}] "
                f"[{completed}/{len(train_loader)}] "
                f"Info={float(output['info_loss']):.4f} "
                f"Recon={float(output['recon_loss']):.4f} "
                f"Div={float(output['div_loss']):.4f} "
                f"Att={float(output['att_loss']):.8f} "
                f"Refine={float(output['refine_loss']):.8e} "
                f"Total={float(total_loss):.4f} ETA={eta}",
                flush=True,
            )
    result = {field: value / sample_count for field, value in totals.items()}
    result["epoch_seconds"] = time.time() - epoch_start
    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, Any],
    base_checkpoint_path: Path,
    selection_metric: str | None = None,
) -> None:
    checkpoint = {
        "base_checkpoint_path": str(base_checkpoint_path),
        "model_state_dict": model.state_dict(),
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
    registry = EXPERIMENTS[arguments.experiment]
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    epochs = config.epochs if arguments.epochs is None else arguments.epochs
    init_lr = config.init_lr if arguments.init_lr is None else arguments.init_lr
    weight_decay = (
        config.weight_decay
        if arguments.weight_decay is None
        else arguments.weight_decay
    )
    experiment_name = arguments.experiment_name or registry["default_experiment"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Experiment D")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    counts = parameter_audit(model)
    print("Parameter/freeze audit:", flush=True)
    print(json.dumps(counts, indent=2), flush=True)
    if counts["trainable_parameters"] != counts["optimizer_expected_parameters"]:
        raise RuntimeError("Trainable parameter accounting mismatch")

    train_dataset, test_dataset = build_datasets(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    sanity = run_sanity_checks(
        model,
        train_dataset,
        test_loader,
        config,
        registry,
        device,
        arguments.lambda_f,
        arguments.sanity_batch_size,
    )
    if arguments.sanity_only:
        model.close()
        print("Sanity-only mode: no optimizer step or training checkpoint.")
        return

    # The gradient audit runs in train mode and updates BN buffers. Rebuild from
    # the source checkpoint so formal training starts from a pristine state.
    model.close()
    del model
    torch.cuda.empty_cache()
    setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    rebuilt_counts = parameter_audit(model)
    if rebuilt_counts != counts:
        raise RuntimeError("Parameter audit changed after pristine rebuild")

    model_dir = arguments.model_dir / experiment_name
    if any(
        (model_dir / filename).exists()
        for filename in ("latest.pth", "final.pth", f"{registry['dataset']}_best.pth")
    ):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "architecture": "experiment_d_joint_topdown_l3_refine",
        "experiment_key": arguments.experiment,
        "experiment_name": experiment_name,
        "base_experiment": registry["base_experiment"],
        "base_checkpoint_path": str(base_checkpoint),
        "epochs": epochs,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "optimizer": "AdamW",
        "init_lr": init_lr,
        "weight_decay": weight_decay,
        "scheduler": False,
        "warmup": config.warmup,
        "lam1": config.lam1,
        "lam2": config.lam2,
        "lam3": config.lam3,
        "lambda_f": arguments.lambda_f,
        "lambda_match": 1.0,
        "lambda_coarse": 1.0,
        "checkpoint_selection": "AUD_FINE_cIoU",
        "seed": config.seed,
        "parameter_audit": counts,
        "sanity_checks": sanity["gradient_audit"],
        "uses_ogl": False,
        "uses_obj_prior": False,
        "uses_gt_training_loss": False,
    }
    (model_dir / "configs.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (model_dir / "sanity_checks.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )

    optimizer = make_optimizer(model, init_lr, weight_decay)
    if optimizer_parameter_count(optimizer) != counts["optimizer_expected_parameters"]:
        raise RuntimeError("Optimizer parameter set does not match the freeze audit")
    scaler = torch.amp.GradScaler("cuda")
    setup_seed(config.seed)
    train_loader = build_train_loader(train_dataset, config)

    history_path = model_dir / "epoch_metrics.csv"
    curve_stem = model_dir / "training_curves"
    best_score = -math.inf
    run_start = time.time()
    for epoch in range(epochs):
        epoch_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            epoch,
            epochs,
            arguments.lambda_f,
        )
        print(
            f"Epoch {epoch + 1}/{epochs} mean losses: "
            + " ".join(
                f"{field}={epoch_metrics[field]:.8g}" for field in LOSS_FIELDS
            ),
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
                f"Best joint model saved at epoch {epoch + 1}: "
                f"AUD_FINE cIoU={best_score:.4f}",
                flush=True,
            )

        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_metrics["epoch_seconds"],
            **{field: epoch_metrics[field] for field in LOSS_FIELDS},
            "aud_l4_ciou": validation["AUD_L4"]["cIoU"],
            "aud_l4_auc": validation["AUD_L4"]["AUC"],
            "aud_fine_ciou": validation["AUD_FINE"]["cIoU"],
            "aud_fine_auc": validation["AUD_FINE"]["AUC"],
            "img_l4_ciou": validation["IMG_L4"]["cIoU"],
            "img_l4_auc": validation["IMG_L4"]["AUC"],
            "img_fine_ciou": validation["IMG_FINE"]["cIoU"],
            "img_fine_auc": validation["IMG_FINE"]["AUC"],
            "iqr_fine_ciou": validation["IQR_FINE"]["cIoU"],
            "iqr_fine_auc": validation["IQR_FINE"]["AUC"],
        }
        append_history(history_path, record)
        render_training_curves(
            history_path,
            curve_stem,
            experiment_name,
            sanity["baseline_metrics"],
        )
        elapsed = time.time() - run_start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{epochs} complete; "
            f"overall ETA {timedelta(seconds=int(remaining))}; "
            f"estimated finish "
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
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    best_metrics = validate(model, test_loader, device)
    print_metrics(best_metrics, prefix="BEST/")
    (model_dir / "best_test_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )
    print(
        f"Total Experiment D training time: "
        f"{timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
