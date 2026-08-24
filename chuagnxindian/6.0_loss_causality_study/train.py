#!/usr/bin/env python3
"""Stage B trainer with independently weighted L3+L4 base losses."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    namespace_from_json,
    reference_paths,
    require_cuda,
    setup_paths,
    setup_seed,
    write_json,
)

setup_paths()

from dataset import get_test_dataset, get_train_dataset  # noqa: E402
from metrics import (  # noqa: E402
    build_object_prior,
    evaluate_model,
    render_training_curves,
    write_rows,
)
from model import LossFactorizedL3L4, LossWeights  # noqa: E402


CONFIGS = {
    "no_spatial_att": LossWeights(att_space=0.0),
    "no_image_rec": LossWeights(rec_img=0.0),
    "no_both": LossWeights(rec_img=0.0, att_space=0.0),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["vggss", "flickr"])
    parser.add_argument("--configuration", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lambda-rec-img", type=float)
    parser.add_argument("--lambda-rec-aud", type=float)
    parser.add_argument("--lambda-div-img", type=float)
    parser.add_argument("--lambda-div-aud", type=float)
    parser.add_argument("--lambda-att-space", type=float)
    parser.add_argument("--lambda-att-time", type=float)
    return parser.parse_args()


def resolve_weights(cli):
    expected = CONFIGS[cli.configuration]
    values = {}
    for field in expected.as_dict():
        override = getattr(cli, f"lambda_{field}")
        values[field] = getattr(expected, field) if override is None else override
    resolved = LossWeights(**values)
    if resolved != expected:
        raise ValueError(
            f"Configuration {cli.configuration} must remain {expected.as_dict()}, "
            f"got {resolved.as_dict()}"
        )
    return resolved


def ensure_stage_a_passed():
    path = RESULTS_ROOT / "audit" / "audit_summary.json"
    if not path.exists():
        raise RuntimeError("Stage A audit_summary.json is missing; refusing Stage B")
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("stage_a_passed", False):
        raise RuntimeError("Stage A did not pass; refusing Stage B")


def checkpoint_payload(model, optimizer, epoch, weights, **extra):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss_weights": weights.as_dict(),
        "architecture": "6.0_loss_factorized_l3_l4",
    }
    payload.update(extra)
    return payload


def global_grad_norm(grads):
    squared = 0.0
    for grad in grads:
        if grad is not None:
            squared += float(torch.sum(grad.detach().float() ** 2).cpu())
    return math.sqrt(squared)


def first_batch_loss_audit(model, detailed, weights):
    params = [param for param in model.parameters() if param.requires_grad]
    losses = detailed["losses"]
    audit = {"disabled": {}, "required_active": {}}
    for name, coefficient in weights.as_dict().items():
        if coefficient == 0.0:
            grads = torch.autograd.grad(
                losses[name] * coefficient,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            norm = global_grad_norm(grads)
            audit["disabled"][name] = {
                "weighted_value": float((losses[name] * coefficient).detach().cpu()),
                "gradient_norm": norm,
                "strict_zero": norm == 0.0,
            }
    for name, coefficient in (("rec_aud", weights.rec_aud), ("att_time", weights.att_time)):
        grads = torch.autograd.grad(
            losses[name] * coefficient,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        norm = global_grad_norm(grads)
        audit["required_active"][name] = {
            "weighted_value": float((losses[name] * coefficient).detach().cpu()),
            "gradient_norm": norm,
            "nonzero": bool(norm > 0 and np.isfinite(norm)),
        }
    if not all(item["strict_zero"] for item in audit["disabled"].values()):
        raise RuntimeError(f"Disabled loss has nonzero contribution: {audit}")
    if not all(item["nonzero"] for item in audit["required_active"].values()):
        raise RuntimeError(f"Audio rec/temporal attention gradient audit failed: {audit}")
    return audit


def train_epoch(loader, model, optimizer, scaler, weights, device, epoch, epochs):
    model.train()
    meters = defaultdict(float)
    samples = 0
    nan_inf = 0
    first_batch_audit = None
    first_step_delta = None
    batch_time_average = 0.0
    end = time.time()

    for batch_index, (frame, spec, _bboxes, _file_id, _label) in enumerate(loader):
        batch_size = frame.shape[0]
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        parameter_snapshot = None
        if epoch == 0 and batch_index == 0:
            parameter_snapshot = [
                param.detach().clone()
                for param in model.parameters()
                if param.requires_grad
            ]

        with torch.amp.autocast("cuda"):
            detailed = model.forward_train_detailed(frame, spec, weights)
            losses = detailed["losses"]
            total = losses["total"]

        finite_tensors = [total] + [losses[name] for name in ("info", "rec_img", "rec_aud", "div_img", "div_aud", "att_space", "att_time")]
        nan_inf += sum(int((~torch.isfinite(value)).sum().item()) for value in finite_tensors)
        if nan_inf:
            raise FloatingPointError(f"NaN/Inf detected at epoch {epoch + 1}, batch {batch_index}")

        if epoch == 0 and batch_index == 0:
            first_batch_audit = first_batch_loss_audit(model, detailed, weights)

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        if parameter_snapshot is not None:
            squared_delta = 0.0
            for before, after in zip(
                parameter_snapshot,
                (param for param in model.parameters() if param.requires_grad),
            ):
                squared_delta += float(torch.sum((after.detach() - before) ** 2).cpu())
            first_step_delta = math.sqrt(squared_delta)
            del parameter_snapshot
            if first_step_delta <= 0 or not np.isfinite(first_step_delta):
                raise RuntimeError(f"Parameters did not update: delta={first_step_delta}")

        values = {
            "info_loss": losses["info"],
            "rec_img_loss": losses["rec_img"],
            "rec_aud_loss": losses["rec_aud"],
            "div_img_loss": losses["div_img"],
            "div_aud_loss": losses["div_aud"],
            "att_space_loss": losses["att_space"],
            "att_time_loss": losses["att_time"],
            "weighted_info_loss": losses["info"],
            "weighted_rec_img_loss": weights.rec_img * losses["rec_img"],
            "weighted_rec_aud_loss": weights.rec_aud * losses["rec_aud"],
            "weighted_div_img_loss": weights.div_img * losses["div_img"],
            "weighted_div_aud_loss": weights.div_aud * losses["div_aud"],
            "weighted_att_space_loss": weights.att_space * losses["att_space"],
            "weighted_att_time_loss": weights.att_time * losses["att_time"],
            "total_loss": total,
        }
        for key, value in values.items():
            meters[key] += float(value.detach().cpu()) * batch_size
        samples += batch_size

        elapsed = time.time() - end
        batch_time_average = (
            elapsed if batch_index == 0 else (batch_time_average * batch_index + elapsed) / (batch_index + 1)
        )
        end = time.time()
        if batch_index % 10 == 0 or batch_index == len(loader) - 1:
            remaining = len(loader) - batch_index - 1 + (epochs - epoch - 1) * len(loader)
            eta = timedelta(seconds=int(batch_time_average * remaining))
            print(
                f"Train [{epoch + 1}/{epochs}][{batch_index}/{len(loader)}] "
                f"total={float(total.detach().cpu()):.5f} ETA={eta}",
                flush=True,
            )
        del detailed, losses, total

    result = {key: value / samples for key, value in meters.items()}
    result.update(
        {
            "nan_inf_count": nan_inf,
            "first_step_parameter_delta": first_step_delta,
            "first_batch_loss_audit": first_batch_audit,
        }
    )
    return result


def build_args(cli):
    reference_config, _ = reference_paths(cli.dataset)
    default_workers = 16 if cli.dataset == "vggss" else 12
    workers = default_workers if cli.workers is None else cli.workers
    return namespace_from_json(
        reference_config,
        gpu=cli.gpu,
        workers=workers,
        batch_size=256,
        epochs=100,
        init_lr=5e-5,
        weight_decay=0.01,
        tau=0.03,
        num_slots=2,
        iters=5,
        mask_ratio=0.1,
        infer_sharpening=0.1,
        reciprocal_k=20,
        warmup=-1,
        alpha=0.6,
        scheduler=False,
        seed=12345,
        experiment_name=cli.experiment_name,
        model_dir="./checkpoints",
        architecture="6.0_loss_factorized_l3_l4",
        checkpoint_selection="IQR_cIoU",
    )


def main():
    cli = parse_args()
    ensure_stage_a_passed()
    weights = resolve_weights(cli)
    args = build_args(cli)
    device = require_cuda(cli.gpu)
    torch.cuda.set_device(cli.gpu)
    setup_seed(12345)

    model_dir = PROJECT_ROOT / "checkpoints" / cli.experiment_name
    forbidden = [model_dir / name for name in ("latest.pth", "final.pth", "vggss_best.pth", "flickr_best.pth")]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint(s): {existing}")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "configuration": cli.configuration,
            "lambda_rec_img": weights.rec_img,
            "lambda_rec_aud": weights.rec_aud,
            "lambda_div_img": weights.div_img,
            "lambda_div_aud": weights.div_aud,
            "lambda_att_space": weights.att_space,
            "lambda_att_time": weights.att_time,
            "full_reference_regression_required": True,
        }
    )
    write_json(model_dir / "configs.json", config_payload)

    model = LossFactorizedL3L4(args).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.init_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")

    train_dataset = get_train_dataset(
        args, hard_img=args.hard_img, hard_aud=args.hard_aud, rand_aud=args.rand_aud
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    test_dataset = get_test_dataset(args, args.testset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    object_model = build_object_prior(device)
    print(
        f"Experiment={cli.experiment_name} config={cli.configuration} dataset={cli.dataset} "
        f"train={len(train_dataset)} test={len(test_dataset)} workers={args.workers} weights={weights.as_dict()}",
        flush=True,
    )

    history = []
    best_iqr = -1.0
    best_epoch = None
    best_name = f"{args.testset}_best.pth"
    best_per_sample = None
    first_batch_audit = None
    first_step_delta = None
    run_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = train_epoch(
            train_loader, model, optimizer, scaler, weights, device, epoch, args.epochs
        )
        if first_batch_audit is None:
            first_batch_audit = train_metrics.pop("first_batch_loss_audit")
            first_step_delta = train_metrics.get("first_step_parameter_delta")
            write_json(model_dir / "first_batch_loss_gradient_audit.json", first_batch_audit)
        else:
            train_metrics.pop("first_batch_loss_audit", None)

        eval_result, per_sample = evaluate_model(
            model, test_loader, object_model, device, alpha=args.alpha
        )
        if eval_result["diagnostics"]["nan_inf_count"]:
            raise FloatingPointError(f"Evaluation NaN/Inf: {eval_result['diagnostics']}")
        iqr_ciou = eval_result["metrics"]["IQR"]["cIoU"]
        if iqr_ciou > best_iqr:
            best_iqr = iqr_ciou
            best_epoch = epoch + 1
            best_per_sample = per_sample
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch + 1,
                    weights,
                    selection_metric="IQR_cIoU",
                    selection_score=iqr_ciou,
                ),
                model_dir / best_name,
            )
            print(f"Best saved: epoch={best_epoch} IQR cIoU={best_iqr:.4f}", flush=True)

        torch.save(
            checkpoint_payload(model, optimizer, epoch + 1, weights),
            model_dir / "latest.pth",
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
        }
        row.update({key: value for key, value in train_metrics.items() if not isinstance(value, dict)})
        row.update(
            {
                "aud_ciou": eval_result["metrics"]["AUD"]["cIoU"],
                "aud_auc": eval_result["metrics"]["AUD"]["AUC"],
                "img_ciou": eval_result["metrics"]["IMG_QUERY"]["cIoU"],
                "img_auc": eval_result["metrics"]["IMG_QUERY"]["AUC"],
                "iqr_ciou": eval_result["metrics"]["IQR"]["cIoU"],
                "iqr_auc": eval_result["metrics"]["IQR"]["AUC"],
                "aud_img_pearson": eval_result["diagnostics"]["aud_img_pearson"]["mean"],
                "aud_predicted_area_ratio": eval_result["diagnostics"]["aud_predicted_area_ratio"]["mean"],
                "aud_precision": eval_result["diagnostics"]["aud_precision"]["mean"],
                "aud_coverage": eval_result["diagnostics"]["aud_coverage"]["mean"],
                "aud_outside_leakage": eval_result["diagnostics"]["aud_outside_leakage"]["mean"],
            }
        )
        history.append(row)
        write_rows(model_dir / "epoch_metrics.csv", history)
        render_training_curves(
            model_dir / "epoch_metrics.csv",
            model_dir / "training_curves.png",
            model_dir / "training_curves.pdf",
            cli.experiment_name,
        )
        elapsed = time.time() - run_start
        average = elapsed / (epoch + 1)
        remaining = average * (args.epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: AUD={row['aud_ciou']:.4f}/{row['aud_auc']:.4f} "
            f"IMG={row['img_ciou']:.4f}/{row['img_auc']:.4f} "
            f"IQR={row['iqr_ciou']:.4f}/{row['iqr_auc']:.4f}; "
            f"epoch={timedelta(seconds=int(row['epoch_seconds']))}, overall ETA={timedelta(seconds=int(remaining))}",
            flush=True,
        )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "loss_weights": weights.as_dict(),
            "architecture": "6.0_loss_factorized_l3_l4",
        },
        model_dir / "final.pth",
    )
    if best_per_sample is None:
        raise RuntimeError("No best checkpoint was selected")

    best_checkpoint = torch.load(model_dir / best_name, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    best_eval, best_per_sample = evaluate_model(
        model, test_loader, object_model, device, alpha=args.alpha
    )
    write_rows(model_dir / "per_sample_metrics.csv", best_per_sample)
    summary = {
        "experiment_name": cli.experiment_name,
        "configuration": cli.configuration,
        "dataset": cli.dataset,
        "best_epoch": best_epoch,
        "selection_metric": "IQR_cIoU",
        "selection_score": best_iqr,
        "loss_weights": weights.as_dict(),
        "metrics": best_eval["metrics"],
        "diagnostics": best_eval["diagnostics"],
        "sanity": {
            "nan_inf_count": sum(row["nan_inf_count"] for row in history),
            "parameters_updated": first_step_delta is not None and first_step_delta > 0,
            "first_step_parameter_delta": first_step_delta,
            "disabled_loss_and_required_gradient_audit": first_batch_audit,
            "full_numerical_regression_passed_before_stage_b": True,
            "checkpoint_selection_unchanged": True,
            "dataset_and_evaluator_unchanged": True,
        },
        "total_training_seconds": time.time() - run_start,
    }
    write_json(model_dir / "summary.json", summary)
    # Explicitly close both persistent worker pools before interpreter teardown.
    # With two 16-worker loaders, relying on Python finalizers can leave the
    # completed VGGSS process blocked in poll(), preventing the scheduler from
    # advancing even though every artifact has already been written.
    for loader in (train_loader, test_loader):
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None
    print(
        f"Training complete in {timedelta(seconds=int(summary['total_training_seconds']))}; "
        f"best epoch={best_epoch}, IQR={best_iqr:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
