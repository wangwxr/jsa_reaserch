#!/usr/bin/env python3
"""Experiment 6.1 trainer: only spatial-attention weight varies over time."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    EXPERIMENT_ROOT,
    PROJECT_ROOT,
    namespace_from_json,
    reference_paths,
    require_cuda,
    setup_paths,
    setup_seed,
    write_json,
)

setup_paths()

from dataset import get_test_dataset, get_train_dataset  # noqa: E402
from metrics import build_object_prior, evaluate_model, render_training_curves, write_rows  # noqa: E402
from model import LossFactorizedL3L4, LossWeights, compose_total  # noqa: E402
from schedule import SCHEDULES, lambda_att_space, schedule_table  # noqa: E402


AUDIT_EPOCHS = {1, 5, 10, 20, 21, 25, 30, 35, 40, 41, 50, 60, 80, 100}
EXPERIMENT_NAMES = {
    ("early_high_late_low", "vggss"): "6.1_early_high_late_low_vggss_10k",
    ("early_high_late_low", "flickr"): "6.1_early_high_late_low_flickr_10k_frame8_center5",
    ("early_low_late_high", "vggss"): "6.1_early_low_late_high_vggss_10k",
    ("early_low_late_high", "flickr"): "6.1_early_low_late_high_flickr_10k_frame8_center5",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("vggss", "flickr"))
    parser.add_argument("--schedule", required=True, choices=SCHEDULES)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def build_args(cli):
    expected_name = EXPERIMENT_NAMES[(cli.schedule, cli.dataset)]
    if cli.experiment_name != expected_name:
        raise ValueError(f"Exact experiment name required: {expected_name}")
    reference_config, _ = reference_paths(cli.dataset)
    workers = (16 if cli.dataset == "vggss" else 12) if cli.workers is None else cli.workers
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
        architecture="6.1_spatial_att_temporal_schedule",
        checkpoint_selection="IQR_cIoU",
    )


def weights_for_epoch(schedule_name, epoch_one_based):
    return LossWeights(att_space=lambda_att_space(schedule_name, epoch_one_based))


def checkpoint_payload(model, optimizer, epoch, schedule_name, weights, **extra):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "schedule": schedule_name,
        "loss_weights": weights.as_dict(),
        "architecture": "6.1_spatial_att_temporal_schedule",
    }
    payload.update(extra)
    return payload


def global_grad_norm(grads):
    return math.sqrt(
        sum(float(torch.sum(grad.detach().float() ** 2).cpu()) for grad in grads if grad is not None)
    )


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def make_fixed_audit_batch(dataset, batch_size=32):
    state = capture_rng_state()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    frame, spec, *_ = next(iter(loader))
    restore_rng_state(state)
    return frame.contiguous(), spec.contiguous()


def parameter_groups(model):
    named = dict(model.named_parameters())
    predicates = {
        "audio_backbone": lambda n: n.startswith("audnet."),
        "audio_slot_branch": lambda n: n.startswith("slot_attn.audio_branch."),
        "audio_query_projection": lambda n: n.startswith("slot_attn.audio_branch.aud_to_q."),
        "visual_backbone": lambda n: n.startswith("imgnet."),
        "visual_l4_key_projection": lambda n: n.startswith("slot_attn.visual_branches.1.img_to_k."),
        "shared_slots": lambda n: n == "slot_attn.slots",
        "total_trainable_parameters": lambda n: named[n].requires_grad,
    }
    groups = {}
    for group, predicate in predicates.items():
        params = [param for name, param in named.items() if param.requires_grad and predicate(name)]
        if not params:
            raise RuntimeError(f"Empty gradient parameter group: {group}")
        groups[group] = params
    return groups


def subset_grads(all_params, grads, subset):
    wanted = {id(param) for param in subset}
    return [grad for param, grad in zip(all_params, grads) if id(param) in wanted]


def preflight_regression(model, fixed_batch, device, output_path):
    rng = capture_rng_state()
    versions = {name: param._version for name, param in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    was_training = model.training
    model.eval()
    frame, spec = (tensor.to(device).float() for tensor in fixed_batch)
    with torch.enable_grad():
        detailed = model.forward_train_detailed(frame, spec, LossWeights())
        losses = detailed["losses"]
        old_total = (
            losses["info"]
            + 0.1 * (losses["rec_img"] + losses["rec_aud"])
            + 0.1 * (losses["div_img"] + losses["div_aud"])
            + 100.0 * (losses["att_space"] + losses["att_time"])
        )
        full_error = float(torch.abs(losses["total"] - old_total).detach().cpu())
        zero_weights = LossWeights(att_space=0.0)
        zero_total = compose_total(losses, zero_weights)
        params = [param for param in model.parameters() if param.requires_grad]
        zero_grads = torch.autograd.grad(
            losses["att_space"] * 0.0, params, retain_graph=True, allow_unused=True
        )
        zero_total_space_grad = torch.autograd.grad(
            zero_total, losses["att_space"], retain_graph=False
        )[0]
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    restore_rng_state(rng)
    buffer_unchanged = all(torch.equal(buffers[n], b) for n, b in model.named_buffers())
    parameter_unchanged = all(versions[n] == p._version for n, p in model.named_parameters())
    result = {
        "passed": bool(
            full_error <= 1e-7
            and global_grad_norm(zero_grads) == 0.0
            and float(zero_total_space_grad) == 0.0
            and buffer_unchanged
            and parameter_unchanged
        ),
        "full_total_max_abs_error": full_error,
        "lambda_zero_weighted_gradient_norm": global_grad_norm(zero_grads),
        "lambda_zero_total_derivative_wrt_att_space": float(zero_total_space_grad),
        "parameters_unchanged": parameter_unchanged,
        "buffers_unchanged": buffer_unchanged,
        "fixed_weights": LossWeights().as_dict(),
        "forward_called_once_only; schedule_not_an_input": True,
    }
    write_json(output_path, result)
    if not result["passed"]:
        raise RuntimeError(f"Preflight loss regression failed: {result}")
    del detailed, losses, frame, spec
    return result


def gradient_trajectory_audit(model, fixed_batch, weights, device, epoch):
    rng = capture_rng_state()
    versions = {name: param._version for name, param in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    frame, spec = (tensor.to(device).float() for tensor in fixed_batch)
    params = [param for param in model.parameters() if param.requires_grad]
    groups = parameter_groups(model)
    with torch.enable_grad():
        detailed = model.forward_train_detailed(frame, spec, weights)
        weighted_space = weights.att_space * detailed["losses"]["att_space"]
        total = detailed["losses"]["total"]
        space_grads = torch.autograd.grad(weighted_space, params, retain_graph=True, allow_unused=True)
        total_grads = torch.autograd.grad(total, params, retain_graph=False, allow_unused=True)

    row = {
        "epoch": epoch,
        "current_lambda_att_space": weights.att_space,
        "weighted_att_space_loss": float(weighted_space.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    for name, subset in groups.items():
        space_norm = global_grad_norm(subset_grads(params, space_grads, subset))
        total_norm = global_grad_norm(subset_grads(params, total_grads, subset))
        row[f"{name}_grad_norm_weighted_att_space"] = space_norm
        row[f"{name}_grad_norm_total"] = total_norm
        row[f"{name}_grad_norm_ratio"] = space_norm / (total_norm + 1e-12)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    restore_rng_state(rng)
    row["parameters_unchanged"] = all(versions[n] == p._version for n, p in model.named_parameters())
    row["buffers_unchanged"] = all(torch.equal(buffers[n], b) for n, b in model.named_buffers())
    row["parameters_with_grad_after"] = sum(p.grad is not None for p in model.parameters())
    row["rng_restored"] = True
    if not row["parameters_unchanged"] or not row["buffers_unchanged"] or row["parameters_with_grad_after"]:
        raise RuntimeError(f"Gradient audit was invasive: {row}")
    return row


def train_epoch(loader, model, optimizer, scaler, weights, device, epoch_index, epochs):
    model.train()
    meters = defaultdict(float)
    sample_count = 0
    nan_inf = 0
    first_step_delta = None
    skipped_initial_steps = 0
    average_batch_time = 0.0
    end = time.time()
    for batch_index, (frame, spec, _boxes, _ids, _labels) in enumerate(loader):
        batch_size = frame.shape[0]
        frame = frame.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        snapshot = None
        if epoch_index == 0 and first_step_delta is None:
            snapshot = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        with torch.amp.autocast("cuda"):
            detailed = model.forward_train_detailed(frame, spec, weights)
            losses = detailed["losses"]
            total = losses["total"]
        finite = [total] + [losses[key] for key in ("info", "rec_img", "rec_aud", "div_img", "div_aud", "att_space", "att_time")]
        nan_inf += sum(int((~torch.isfinite(value)).sum()) for value in finite)
        if nan_inf:
            raise FloatingPointError(f"NaN/Inf epoch={epoch_index + 1} batch={batch_index}")
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        if snapshot is not None:
            candidate_delta = math.sqrt(sum(
                float(torch.sum((after.detach() - before) ** 2).cpu())
                for before, after in zip(snapshot, (p for p in model.parameters() if p.requires_grad))
            ))
            if not np.isfinite(candidate_delta):
                raise RuntimeError(f"Non-finite parameter delta: {candidate_delta}")
            if candidate_delta > 0:
                first_step_delta = candidate_delta
            else:
                # GradScaler may legitimately skip an overflowing initial AMP
                # step.  This is an audit event, not an algorithmic failure;
                # require a later step in the same epoch to update parameters.
                skipped_initial_steps += 1
        values = {
            "info_loss": losses["info"],
            "rec_img_loss": losses["rec_img"], "rec_aud_loss": losses["rec_aud"],
            "div_img_loss": losses["div_img"], "div_aud_loss": losses["div_aud"],
            "att_space_loss": losses["att_space"], "att_time_loss": losses["att_time"],
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
        sample_count += batch_size
        elapsed = time.time() - end
        average_batch_time = elapsed if batch_index == 0 else (average_batch_time * batch_index + elapsed) / (batch_index + 1)
        end = time.time()
        if batch_index % 10 == 0 or batch_index == len(loader) - 1:
            remaining = len(loader) - batch_index - 1 + (epochs - epoch_index - 1) * len(loader)
            print(
                f"Train [{epoch_index + 1}/{epochs}][{batch_index}/{len(loader)}] "
                f"lambda_space={weights.att_space:.1f} total={float(total.detach().cpu()):.5f} "
                f"ETA={timedelta(seconds=int(average_batch_time * remaining))}", flush=True,
            )
    result = {key: value / sample_count for key, value in meters.items()}
    if epoch_index == 0 and first_step_delta is None:
        raise RuntimeError("No parameter update occurred anywhere in the first epoch")
    result.update({
        "nan_inf_count": nan_inf,
        "first_step_parameter_delta": first_step_delta,
        "skipped_initial_amp_steps": skipped_initial_steps,
    })
    return result


def epoch_row(epoch, weights, train_metrics, evaluation, optimizer, seconds):
    metrics, diagnostics = evaluation["metrics"], evaluation["diagnostics"]
    row = {
        "epoch": epoch,
        "current_lambda_att_space": weights.att_space,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_seconds": seconds,
    }
    row.update(train_metrics)
    row.update({
        "aud_ciou": metrics["AUD"]["cIoU"], "aud_auc": metrics["AUD"]["AUC"],
        "img_ciou": metrics["IMG_QUERY"]["cIoU"], "img_auc": metrics["IMG_QUERY"]["AUC"],
        "iqr_ciou": metrics["IQR"]["cIoU"], "iqr_auc": metrics["IQR"]["AUC"],
        "aud_img_pearson_mean": diagnostics["aud_img_pearson"]["mean"],
        "aud_img_pearson_median": diagnostics["aud_img_pearson"]["median"],
        "aud_img_spearman_mean": diagnostics["aud_img_spearman"]["mean"],
        "aud_img_js": diagnostics["aud_img_js"]["mean"],
        "aud_predicted_area_ratio": diagnostics["aud_predicted_area_ratio"]["mean"],
        "aud_precision": diagnostics["aud_precision"]["mean"],
        "aud_coverage": diagnostics["aud_coverage"]["mean"],
        "aud_outside_leakage": diagnostics["aud_outside_leakage"]["mean"],
        "fusion_synergy_ciou": diagnostics["fusion_synergy_ciou"],
        "fusion_synergy_auc": diagnostics["fusion_synergy_auc"],
        "sample_synergy_mean": diagnostics["sample_synergy"]["mean"],
        "sample_synergy_median": diagnostics["sample_synergy"]["median"],
        "sample_synergy_positive_fraction": diagnostics["sample_synergy_positive_fraction"],
        "sample_synergy_ge_001_fraction": diagnostics["sample_synergy_ge_001_fraction"],
        "iqr_beats_aud_and_img_fraction": diagnostics["iqr_beats_aud_and_img_fraction"],
    })
    return row


def close_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def main():
    cli = parse_args()
    schedule_test = EXPERIMENT_ROOT / "results" / "preflight" / "schedule_tests.json"
    if not schedule_test.exists() or not json.loads(schedule_test.read_text())["passed"]:
        raise RuntimeError("test_schedule.py has not passed; refusing formal training")
    args = build_args(cli)
    device = require_cuda(cli.gpu)
    torch.cuda.set_device(cli.gpu)
    setup_seed(12345)

    model_dir = PROJECT_ROOT / "checkpoints" / cli.experiment_name
    if model_dir.exists() and any(model_dir.glob("*.pth")):
        raise FileExistsError(f"Refusing to overwrite existing checkpoints in {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "schedule": cli.schedule,
        "schedule_table": schedule_table(cli.schedule),
        "lambda_rec_img": 0.1, "lambda_rec_aud": 0.1,
        "lambda_div_img": 0.1, "lambda_div_aud": 0.1,
        "lambda_att_time": 100.0,
        "audit_epochs": sorted(AUDIT_EPOCHS),
        "gradient_audit_batch_size": 32,
    })
    write_json(model_dir / "configs.json", config)

    model = LossFactorizedL3L4(args).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.init_lr, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    train_dataset = get_train_dataset(args, hard_img=args.hard_img, hard_aud=args.hard_aud, rand_aud=args.rand_aud)
    fixed_batch = make_fixed_audit_batch(train_dataset)
    preflight = preflight_regression(model, fixed_batch, device, model_dir / "preflight_regression.json")
    # The fixed-batch capture and regression must not perturb formal training order/RNG.
    setup_seed(12345)
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    test_dataset = get_test_dataset(args, args.testset)
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False, persistent_workers=args.workers > 0,
    )
    object_model = build_object_prior(device)
    print(
        f"Experiment={cli.experiment_name} schedule={cli.schedule} dataset={cli.dataset} "
        f"train={len(train_dataset)} test={len(test_dataset)} workers={args.workers}", flush=True,
    )

    history, gradient_history = [], []
    best_iqr, best_epoch = -1.0, None
    best_name = f"{args.testset}_best.pth"
    first_step_delta = None
    final_eval = final_per_sample = None
    run_start = time.time()
    try:
        for epoch_index in range(100):
            epoch = epoch_index + 1
            weights = weights_for_epoch(cli.schedule, epoch)
            epoch_start = time.time()
            trained = train_epoch(train_loader, model, optimizer, scaler, weights, device, epoch_index, 100)
            if first_step_delta is None:
                first_step_delta = trained["first_step_parameter_delta"]
            evaluation, per_sample = evaluate_model(model, test_loader, object_model, device, alpha=0.6)
            if evaluation["diagnostics"]["nan_inf_count"]:
                raise FloatingPointError(f"Evaluation NaN/Inf at epoch {epoch}")
            row = epoch_row(epoch, weights, trained, evaluation, optimizer, time.time() - epoch_start)
            history.append(row)
            if row["iqr_ciou"] > best_iqr:
                best_iqr, best_epoch = row["iqr_ciou"], epoch
                torch.save(
                    checkpoint_payload(model, optimizer, epoch, cli.schedule, weights,
                                       selection_metric="IQR_cIoU", selection_score=best_iqr),
                    model_dir / best_name,
                )
                print(f"Best saved: epoch={epoch} IQR cIoU={best_iqr:.4f}", flush=True)
            torch.save(checkpoint_payload(model, optimizer, epoch, cli.schedule, weights), model_dir / "latest.pth")

            if epoch in AUDIT_EPOCHS:
                gradient_history.append(gradient_trajectory_audit(model, fixed_batch, weights, device, epoch))
                write_rows(model_dir / "gradient_trajectory.csv", gradient_history)
            write_rows(model_dir / "epoch_metrics.csv", history)
            render_training_curves(
                model_dir / "epoch_metrics.csv", model_dir / "gradient_trajectory.csv",
                model_dir / "training_curves.png", model_dir / "training_curves.pdf", cli.experiment_name,
            )
            if epoch == 100:
                final_eval, final_per_sample = evaluation, per_sample
            elapsed = time.time() - run_start
            remaining = elapsed / epoch * (100 - epoch)
            print(
                f"Epoch {epoch}/100 lambda={weights.att_space:.1f}: "
                f"AUD={row['aud_ciou']:.4f}/{row['aud_auc']:.4f} "
                f"IMG={row['img_ciou']:.4f}/{row['img_auc']:.4f} "
                f"IQR={row['iqr_ciou']:.4f}/{row['iqr_auc']:.4f} "
                f"synergy={row['fusion_synergy_ciou']:+.4f}; overall ETA={timedelta(seconds=int(remaining))}",
                flush=True,
            )

        torch.save({
            "model": model.state_dict(), "epoch": 100, "schedule": cli.schedule,
            "loss_weights": weights_for_epoch(cli.schedule, 100).as_dict(),
            "architecture": "6.1_spatial_att_temporal_schedule",
        }, model_dir / "final.pth")
        if best_epoch is None:
            raise RuntimeError("No best checkpoint selected")
        checkpoint = torch.load(model_dir / best_name, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        best_eval, best_per_sample = evaluate_model(model, test_loader, object_model, device, alpha=0.6)
        write_rows(model_dir / "per_sample_metrics.csv", best_per_sample)
        write_rows(model_dir / "final_per_sample_metrics.csv", final_per_sample)
        diagnostic_bests = {
            metric: {"value": max(row[metric] for row in history), "epoch": max(history, key=lambda row: row[metric])["epoch"]}
            for metric in ("aud_ciou", "img_ciou", "iqr_ciou")
        }
        summary = {
            "experiment_name": cli.experiment_name, "configuration": cli.schedule,
            "dataset": cli.dataset, "best_epoch": best_epoch,
            "selection_metric": "IQR_cIoU", "selection_score": best_iqr,
            "metrics": best_eval["metrics"], "diagnostics": best_eval["diagnostics"],
            "final_epoch": {"metrics": final_eval["metrics"], "diagnostics": final_eval["diagnostics"]},
            "training_diagnostic_bests": diagnostic_bests,
            "preflight": preflight,
            "gradient_trajectory_audited_epochs": sorted(AUDIT_EPOCHS),
            "sanity": {
                "nan_inf_count": sum(row["nan_inf_count"] for row in history),
                "parameters_updated": bool(first_step_delta and first_step_delta > 0),
                "first_step_parameter_delta": first_step_delta,
                "schedule_only_changes_att_space_multiplier": True,
                "checkpoint_selection_unchanged": True,
                "dataset_and_evaluator_unchanged": True,
            },
            "total_training_seconds": time.time() - run_start,
        }
        write_json(model_dir / "summary.json", summary)
        print(
            f"Training complete in {timedelta(seconds=int(summary['total_training_seconds']))}; "
            f"best epoch={best_epoch}, IQR={best_iqr:.4f}", flush=True,
        )
    finally:
        close_loader(train_loader)
        close_loader(test_loader)


if __name__ == "__main__":
    main()
