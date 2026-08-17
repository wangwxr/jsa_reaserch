#!/usr/bin/env python3
"""Experiment E: joint training with decoupled fine spatial anchors."""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
EXPERIMENT_D_DIR = HERE.parent
PROJECT_ROOT = HERE.parents[2]
mpl_cache = Path("/tmp") / f"experiment_e_mpl_{os.getuid()}"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

# Load Experiment D as a library under a unique name. This preserves D files
# byte-for-byte while reusing its optimizer, loss composition, epoch loop,
# sanity audit, checkpoint and unchanged evaluator implementation.
spec = importlib.util.spec_from_file_location(
    "experiment_d_train_lib", EXPERIMENT_D_DIR / "train.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load Experiment D training library")
dtrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtrain)
dcommon = sys.modules["common"]

sys.path.insert(0, str(HERE))
from curves_e import render_training_curves  # noqa: E402
from model_e import DecoupledFineSpatialModel  # noqa: E402


HISTORY_FIELDS = [
    *dtrain.HISTORY_FIELDS,
    "aud_img_map_cosine",
    "fusion_gain_ciou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True, choices=("vggss_10k", "flickr_10k")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment-name")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--sanity-batch-size", type=int, default=32)
    return parser.parse_args()


def build_model(config, registry, device):
    checkpoint_path = dcommon.base_checkpoint_path(registry)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model"].items()
    }
    base_model = dcommon.MUFASAL3L4(config)
    original_trainability = {
        name: parameter.requires_grad
        for name, parameter in base_model.named_parameters()
    }
    base_model.load_state_dict(state, strict=True)
    model = DecoupledFineSpatialModel(
        base_model, original_trainability=original_trainability
    ).to(device)
    return model, checkpoint_path


@torch.inference_mode()
def validate(model, test_loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    accumulators = dtrain.new_accumulators()
    cosine_sum = 0.0
    sample_count = 0
    for image, spec, bboxes, names, _labels in tqdm(
        test_loader, desc="Validate E", dynamic_ncols=True
    ):
        image, spec, bboxes, names = dcommon.flatten_eval_batch(
            image, spec, bboxes, names
        )
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = model(image, spec)
        dtrain.evaluate_maps(output, bboxes, names, accumulators)

        aud_flat = output["AUD_FINE"].flatten(start_dim=1)
        img_flat = output["IMG_FINE"].flatten(start_dim=1)
        cosine = F.cosine_similarity(aud_flat, img_flat, dim=1)
        cosine_sum += float(cosine.sum())
        sample_count += cosine.numel()

    metrics = {
        method: evaluator.finalize()
        for method, evaluator in accumulators.items()
    }
    metrics["diagnostics"] = {
        "aud_img_map_cosine": cosine_sum / sample_count,
        "fusion_gain_ciou": (
            metrics["IQR_FINE"]["cIoU"]
            - metrics["AUD_FINE"]["cIoU"]
        ),
    }
    return metrics


def print_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    dtrain.print_metrics(metrics, prefix=prefix)
    diagnostics = metrics["diagnostics"]
    print(
        f"{prefix}DIAG/AUD_IMG_MAP_COSINE "
        f"{diagnostics['aud_img_map_cosine']:.6f}",
        flush=True,
    )
    print(
        f"{prefix}DIAG/IQR_MINUS_AUD_cIoU "
        f"{diagnostics['fusion_gain_ciou']:+.6f}",
        flush=True,
    )


def append_history(path: Path, record: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    arguments = parse_args()
    registry = dcommon.EXPERIMENTS[arguments.experiment]
    config = dcommon.load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    epochs = config.epochs if arguments.epochs is None else arguments.epochs
    experiment_name = arguments.experiment_name or (
        "experiment_e_decoupled_vggss_10k"
        if arguments.experiment == "vggss_10k"
        else "experiment_e_decoupled_flickr_10k_frame8_center5"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Experiment E")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    dcommon.setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    counts = dcommon.parameter_audit(model)
    print("Experiment E parameter/freeze audit:", flush=True)
    print(json.dumps(counts, indent=2), flush=True)
    if counts["trainable_parameters"] != counts["optimizer_expected_parameters"]:
        raise RuntimeError("Trainable parameter accounting mismatch")

    train_dataset, test_dataset = dcommon.build_datasets(config, registry)
    test_loader = dcommon.build_test_loader(test_dataset, config, registry)
    sanity = dtrain.run_sanity_checks(
        model,
        train_dataset,
        test_loader,
        config,
        registry,
        device,
        lambda_f=1.0,
        sanity_batch_size=arguments.sanity_batch_size,
    )
    # An architectural assertion in addition to the gradient audit: the
    # removed loss is exactly zero, not merely assigned a small weight.
    if model.refinement_losses(
        {
            "AUD_FINE": torch.full((1, 1, 14, 14), 1 / 196, device=device),
            "IMG_FINE": torch.full((1, 1, 14, 14), 1 / 196, device=device),
            "AUD_L4": torch.full((1, 1, 7, 7), 1 / 49, device=device),
            "IMG_L4": torch.full((1, 1, 7, 7), 1 / 49, device=device),
        }
    )["loss_fine_match"].item() != 0.0:
        raise RuntimeError("Experiment E fine matching loss is not exactly zero")
    print("Experiment E loss audit: loss_fine_match is exactly zero.")
    if arguments.sanity_only:
        model.close()
        print("Sanity-only mode: no optimizer step or checkpoint written.")
        return

    # Rebuild after train-mode gradient audit so formal training starts from
    # the pristine L3+L4 checkpoint and zero-initialized adapter.
    model.close()
    del model
    torch.cuda.empty_cache()
    dcommon.setup_seed(config.seed)
    model, base_checkpoint = build_model(config, registry, device)
    if dcommon.parameter_audit(model) != counts:
        raise RuntimeError("Parameter audit changed after pristine rebuild")

    model_dir = arguments.model_dir / experiment_name
    if any(
        (model_dir / filename).exists()
        for filename in (
            "latest.pth",
            "final.pth",
            f"{registry['dataset']}_best.pth",
        )
    ):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "architecture": "experiment_e_decoupled_fine_spatial_learning",
        "parent_experiment": "experiment_d_joint_topdown_l3_refine",
        "experiment_key": arguments.experiment,
        "experiment_name": experiment_name,
        "base_experiment": registry["base_experiment"],
        "base_checkpoint_path": str(base_checkpoint),
        "epochs": epochs,
        "batch_size": config.batch_size,
        "workers": config.workers,
        "optimizer": "AdamW",
        "init_lr": config.init_lr,
        "weight_decay": config.weight_decay,
        "scheduler": False,
        "warmup": config.warmup,
        "lam1": config.lam1,
        "lam2": config.lam2,
        "lam3": config.lam3,
        "lambda_f": 1.0,
        "fine_pixel_matching": False,
        "loss_fine_match": 0.0,
        "lambda_coarse": 1.0,
        "checkpoint_selection": "AUD_FINE_cIoU",
        "diagnostics": ["aud_img_map_cosine", "fusion_gain_ciou"],
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

    optimizer = dtrain.make_optimizer(
        model, init_lr=config.init_lr, weight_decay=config.weight_decay
    )
    if (
        dtrain.optimizer_parameter_count(optimizer)
        != counts["optimizer_expected_parameters"]
    ):
        raise RuntimeError("Optimizer parameter set differs from Experiment D")
    scaler = torch.amp.GradScaler("cuda")
    dcommon.setup_seed(config.seed)
    train_loader = dcommon.build_train_loader(train_dataset, config)

    history_path = model_dir / "epoch_metrics.csv"
    curve_stem = model_dir / "training_curves"
    best_score = -math.inf
    run_start = time.time()
    for epoch in range(epochs):
        epoch_metrics = dtrain.train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            epoch,
            epochs,
            lambda_f=1.0,
        )
        if epoch_metrics["loss_fine_match"] != 0.0:
            raise RuntimeError("Fine matching loss became nonzero")
        print(
            f"Epoch {epoch + 1}/{epochs} mean losses: "
            + " ".join(
                f"{field}={epoch_metrics[field]:.8g}"
                for field in dtrain.LOSS_FIELDS
            ),
            flush=True,
        )
        validation = validate(model, test_loader, device)
        print_metrics(validation, prefix=f"Epoch{epoch + 1}/")
        dtrain.save_checkpoint(
            model_dir / "latest.pth",
            model,
            optimizer,
            epoch + 1,
            validation,
            base_checkpoint,
        )
        if validation["AUD_FINE"]["cIoU"] > best_score:
            best_score = validation["AUD_FINE"]["cIoU"]
            dtrain.save_checkpoint(
                model_dir / f"{registry['dataset']}_best.pth",
                model,
                optimizer,
                epoch + 1,
                validation,
                base_checkpoint,
                selection_metric="AUD_FINE_cIoU",
            )
            print(
                f"Best Experiment E model saved at epoch {epoch + 1}: "
                f"AUD_FINE cIoU={best_score:.4f}",
                flush=True,
            )

        diagnostics = validation["diagnostics"]
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_metrics["epoch_seconds"],
            **{
                field: epoch_metrics[field]
                for field in dtrain.LOSS_FIELDS
            },
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
            "aud_img_map_cosine": diagnostics["aud_img_map_cosine"],
            "fusion_gain_ciou": diagnostics["fusion_gain_ciou"],
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
    dtrain.save_checkpoint(
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
        f"Total Experiment E training time: "
        f"{timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )
    model.close()


if __name__ == "__main__":
    main()
