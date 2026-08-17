#!/usr/bin/env python3
"""Stage-2 training for frozen L3+L4 top-down native-L3 refinement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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
mpl_cache = Path("/tmp") / f"topdown_l3_mpl_{os.getuid()}"
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
from figures.gen_qualitative import save_panel  # noqa: E402
from figures.gen_training_curves import render_training_curves  # noqa: E402
from protocol import ProtocolAccumulator, evaluate_maps  # noqa: E402


HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "epoch_seconds",
    "loss_fine_match",
    "loss_coarse_aud",
    "loss_coarse_img",
    "loss_refine",
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
    parser.add_argument("--lambda-match", type=float, default=1.0)
    parser.add_argument("--lambda-coarse", type=float, default=1.0)
    parser.add_argument("--init-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def new_accumulators() -> dict[str, ProtocolAccumulator]:
    return {
        method: ProtocolAccumulator()
        for method in ("AUD_L4", "AUD_FINE", "IMG_L4", "IMG_FINE", "IQR_FINE")
    }


@torch.inference_mode()
def validate(
    model,
    test_loader,
    device: torch.device,
    qualitative_dir: Path | None = None,
    num_qualitative: int = 10,
) -> dict[str, Any]:
    model.eval()
    accumulators = new_accumulators()
    selected_indices = set()
    if qualitative_dir is not None:
        selected_indices = set(
            np.linspace(0, len(test_loader.dataset) - 1, num_qualitative, dtype=int).tolist()
        )
    qualitative_rows = []
    global_offset = 0

    for image, spec, bboxes, names, _labels in tqdm(
        test_loader, desc="Validate", dynamic_ncols=True
    ):
        image, spec, bboxes, names = flatten_eval_batch(
            image, spec, bboxes, names
        )
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = model(image, spec)
        batch_ious = evaluate_maps(output, bboxes, names, accumulators)

        if selected_indices:
            for local_index, name in enumerate(names):
                dataset_index = global_offset + local_index
                if dataset_index not in selected_indices:
                    continue
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                save_panel(
                    qualitative_dir / f"{dataset_index:05d}_{safe_name}",
                    name,
                    image[local_index],
                    bboxes[local_index],
                    output["AUD_L4"][local_index],
                    output["AUD_FINE"][local_index],
                    output["IMG_FINE"][local_index],
                )
                qualitative_rows.append(
                    {
                        "dataset_index": dataset_index,
                        "sample_id": name,
                        "selection_rule": "10 evenly spaced indices in fixed test-loader order",
                        "AUD_L4_sample_IoU": batch_ious["AUD_L4"][local_index],
                        "AUD_FINE_sample_IoU": batch_ious["AUD_FINE"][local_index],
                        "delta_sample_IoU": batch_ious["AUD_FINE"][local_index]
                        - batch_ious["AUD_L4"][local_index],
                    }
                )
        global_offset += len(names)

    metrics = {method: evaluator.finalize() for method, evaluator in accumulators.items()}
    if qualitative_rows:
        path = qualitative_dir.parent / "qualitative_sample_ids.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qualitative_rows[0]))
            writer.writeheader()
            writer.writerows(qualitative_rows)
    return metrics


def print_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    for method in ("AUD_L4", "AUD_FINE", "IMG_L4", "IMG_FINE", "IQR_FINE"):
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


def run_sanity_checks(
    model,
    train_dataset,
    test_loader,
    config: argparse.Namespace,
    registry: dict,
    device: torch.device,
    lambda_match: float,
    lambda_coarse: float,
) -> dict[str, Any]:
    print("Running mandatory pre-training sanity checks...", flush=True)
    baseline_metrics = validate(model, test_loader, device)
    print_metrics(baseline_metrics, prefix="SANITY/")
    if not metric_matches(baseline_metrics["AUD_L4"], registry["expected_aud"]):
        raise RuntimeError("Check 1 failed: AUD_L4 does not reproduce formal result")
    if not metric_matches(baseline_metrics["IMG_L4"], registry["expected_img"]):
        raise RuntimeError("Check 1 failed: IMG_L4 does not reproduce formal result")

    sanity_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    frame, spec, _bboxes, _names, _labels = next(iter(sanity_loader))
    frame = frame.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(frame, spec)
    losses = model.refinement_losses(
        output, lambda_match=lambda_match, lambda_coarse=lambda_coarse
    )
    losses["loss_refine"].backward()

    base_has_grad = any(
        parameter.grad is not None for parameter in model.base_model.parameters()
    )
    head_grad_norm = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.refinement_head.parameters()
        if parameter.grad is not None
    )
    zero_init_error = float((output["F34"] - output["F4_UP"]).abs().max())
    source_token_error = float(output["source_token_max_abs"])

    probability_errors = {}
    for method in ("AUD_FINE", "IMG_FINE", "AUD_L4", "IMG_L4"):
        sums = output[method].sum(dim=(-2, -1))
        probability_errors[method] = float((sums - 1.0).abs().max())
    pooled = model.sum_pool_2x2(output["AUD_FINE"])
    pooled_sum_error = float(
        (pooled.sum(dim=(-2, -1)) - 1.0).abs().max()
    )

    checks = {
        "check1_aud_l4_reproduced": True,
        "check1_img_l4_reproduced": True,
        "check2_base_has_any_grad": base_has_grad,
        "check3_head_gradient_l1": head_grad_norm,
        "check4_f34_minus_up_f4_max_abs": zero_init_error,
        "check4_f4_hook_minus_formal_tokens_max_abs": source_token_error,
        "check5_probability_sum_max_errors": probability_errors,
        "check6_sum_pool_probability_sum_max_error": pooled_sum_error,
    }
    if base_has_grad:
        raise RuntimeError("Check 2 failed: a frozen base parameter received gradient")
    if not head_grad_norm > 0:
        raise RuntimeError("Check 3 failed: refinement head gradient is zero")
    if zero_init_error > 1e-7:
        raise RuntimeError(f"Check 4 failed: zero-init error={zero_init_error}")
    if source_token_error > 1e-7:
        raise RuntimeError(f"F4 source mismatch: max_abs={source_token_error}")
    if max(probability_errors.values()) > 1e-5:
        raise RuntimeError(f"Check 5 failed: {probability_errors}")
    if pooled_sum_error > 1e-5:
        raise RuntimeError(f"Check 6 failed: pooled sum error={pooled_sum_error}")
    model.zero_grad(set_to_none=True)
    print(json.dumps(checks, indent=2), flush=True)
    print("All mandatory sanity checks passed.", flush=True)
    return {"baseline_metrics": baseline_metrics, "checks": checks}


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    total_epochs: int,
    lambda_match: float,
    lambda_coarse: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss_fine_match": 0.0,
        "loss_coarse_aud": 0.0,
        "loss_coarse_img": 0.0,
        "loss_refine": 0.0,
    }
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
            losses = model.refinement_losses(
                output,
                lambda_match=lambda_match,
                lambda_coarse=lambda_coarse,
            )
        scaler.scale(losses["loss_refine"]).backward()
        if any(parameter.grad is not None for parameter in model.base_model.parameters()):
            raise RuntimeError("Frozen base received a gradient during training")
        scaler.step(optimizer)
        scaler.update()

        batch_size = frame.shape[0]
        sample_count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        batch_seconds = time.time() - batch_start
        completed = batch_index + 1
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
                f"match={float(losses['loss_fine_match']):.6f} "
                f"coarse_aud={float(losses['loss_coarse_aud']):.6f} "
                f"coarse_img={float(losses['loss_coarse_img']):.6f} "
                f"total={float(losses['loss_refine']):.6f} ETA={eta}",
                flush=True,
            )
    result = {key: value / sample_count for key, value in totals.items()}
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
        "refinement_head_state_dict": model.refinement_head.state_dict(),
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
    # Match the current optimized JSA launchers; this changes throughput only.
    config.workers = registry["workers"]
    epochs = config.epochs if arguments.epochs is None else arguments.epochs
    experiment_name = arguments.experiment_name or registry["default_experiment"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal stage-2 training")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    counts = parameter_counts(model)
    print(json.dumps(counts, indent=2), flush=True)
    if counts["frozen_base_parameters"] != counts["total_base_parameters"]:
        raise RuntimeError("Not every base parameter is frozen")
    if counts["trainable_parameters"] != counts["refinement_head_parameters"]:
        raise RuntimeError("Trainable parameter count does not equal head count")

    train_dataset, test_dataset = build_datasets(config, registry)
    test_loader = build_test_loader(test_dataset, config, registry)
    sanity = run_sanity_checks(
        model,
        train_dataset,
        test_loader,
        config,
        registry,
        device,
        arguments.lambda_match,
        arguments.lambda_coarse,
    )
    if arguments.sanity_only:
        model.close()
        print("Sanity-only mode: no optimizer step and no checkpoint written.")
        return

    model_dir = arguments.model_dir / experiment_name
    if any(
        (model_dir / filename).exists()
        for filename in ("latest.pth", "final.pth", f"{registry['dataset']}_best.pth")
    ):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "architecture": "topdown_l3_refine_stage2",
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
        "lambda_match": arguments.lambda_match,
        "lambda_coarse": arguments.lambda_coarse,
        "checkpoint_selection": "AUD_FINE_cIoU",
        "seed": config.seed,
        "train_data_path": config.train_data_path,
        "train_manifest_path": config.train_manifest_path,
        "test_data_path": config.test_data_path,
        "test_manifest_path": config.test_manifest_path,
        "test_gt_path": config.test_gt_path,
        "parameter_counts": counts,
        "sanity_checks": sanity["checks"],
    }
    (model_dir / "configs.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (model_dir / "sanity_checks.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )

    # The optimizer is intentionally restricted to the new head.
    optimizer = torch.optim.AdamW(
        model.refinement_head.parameters(),
        lr=arguments.init_lr,
        weight_decay=arguments.weight_decay,
    )
    optimized_parameters = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if optimized_parameters != counts["refinement_head_parameters"]:
        raise RuntimeError("Optimizer contains parameters outside the refinement head")
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
            epoch,
            epochs,
            arguments.lambda_match,
            arguments.lambda_coarse,
        )
        print(
            f"Epoch {epoch + 1}/{epochs} mean losses: "
            f"loss_fine_match={epoch_metrics['loss_fine_match']:.8f} "
            f"loss_coarse_aud={epoch_metrics['loss_coarse_aud']:.8f} "
            f"loss_coarse_img={epoch_metrics['loss_coarse_img']:.8f} "
            f"loss_refine={epoch_metrics['loss_refine']:.8f}",
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
                f"Best refinement head saved at epoch {epoch + 1}: "
                f"AUD_FINE cIoU={best_score:.4f}",
                flush=True,
            )

        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_metrics["epoch_seconds"],
            **{key: epoch_metrics[key] for key in (
                "loss_fine_match", "loss_coarse_aud", "loss_coarse_img", "loss_refine"
            )},
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
            sanity["baseline_metrics"]["AUD_L4"]["cIoU"],
            sanity["baseline_metrics"]["AUD_L4"]["AUC"],
        )
        elapsed = time.time() - run_start
        average_epoch = elapsed / (epoch + 1)
        remaining = average_epoch * (epochs - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{epochs} complete; "
            f"overall ETA {timedelta(seconds=int(remaining))}; "
            f"estimated finish {(datetime.now() + timedelta(seconds=remaining)):%Y-%m-%d %H:%M:%S}",
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
    model.refinement_head.load_state_dict(
        best_checkpoint["refinement_head_state_dict"], strict=True
    )
    best_metrics = validate(
        model,
        test_loader,
        device,
        qualitative_dir=model_dir / "qualitative",
    )
    print_metrics(best_metrics, prefix="BEST/")
    (model_dir / "best_test_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )
    print(
        f"Total stage-2 training time: "
        f"{timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
