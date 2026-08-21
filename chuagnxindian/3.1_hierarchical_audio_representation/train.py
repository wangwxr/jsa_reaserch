#!/usr/bin/env python3
"""Formal Stage1 trainer for Experiment 3.1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
BASELINE_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
for path in (PROJECT_ROOT, V11_ROOT, BASELINE_ROOT, HERE):
    sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402
import wandb  # noqa: E402

import test_model  # noqa: E402
import train_slot  # noqa: E402
import utils  # noqa: E402
from dataset import get_test_dataset, get_train_dataset  # noqa: E402
from model import HierarchicalAudioStage1  # noqa: E402
from training_history import render_training_curves  # noqa: E402

import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=sorted(common.SETTINGS))
    parser.add_argument("--gpu", required=True, type=int)
    return parser.parse_args()


def build_object_prior(gpu: int) -> nn.Module:
    model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        train_slot.NormReducer(dim=1),
        train_slot.Unsqueeze(1),
    )
    return model.cuda(gpu).eval()


def mean_diagnostics(sums: dict[str, float], samples: int) -> dict[str, float]:
    return {key: value / max(samples, 1) for key, value in sums.items()}


def train_epoch(loader, model, optimizer, scaler, epoch, args):
    model.train()
    meters = {
        "info": train_slot.AverageMeter("Info", ":.3f"),
        "recon": train_slot.AverageMeter("Recon", ":.3f"),
        "div": train_slot.AverageMeter("Div", ":.3f"),
        "att": train_slot.AverageMeter("Att", ":.3f"),
        "total": train_slot.AverageMeter("Total", ":.3f"),
        "batch": train_slot.AverageMeter("Time", ":6.3f"),
        "data": train_slot.AverageMeter("Data", ":6.3f"),
    }
    diagnostic_sums: dict[str, float] = {}
    diagnostic_samples = 0
    end = time.time()

    for index, (frame, spec, _bboxes, _file_id, _label) in enumerate(loader):
        batch_size = frame.shape[0]
        meters["data"].update(time.time() - end)
        frame = frame.cuda(args.gpu, non_blocking=True)
        spec = spec.cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            detailed = model.forward_train_detailed(frame.float(), spec.float())
            info_loss, recon_loss, div_loss, att_loss = detailed["losses"]
            if epoch < args.warmup:
                total_loss = info_loss
            else:
                total_loss = (
                    info_loss
                    + args.lam1 * recon_loss
                    + args.lam2 * div_loss
                    + args.lam3 * att_loss
                )
        if not all(
            torch.isfinite(value).all().item()
            for value in (info_loss, recon_loss, div_loss, att_loss, total_loss)
        ):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={index}")

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        for name, value in (
            ("info", info_loss),
            ("recon", recon_loss),
            ("div", div_loss),
            ("att", att_loss),
            ("total", total_loss),
        ):
            meters[name].update(value.item(), batch_size)
        diagnostic_names = list(detailed["diagnostics"])
        diagnostic_values = torch.stack(
            [detailed["diagnostics"][name] for name in diagnostic_names]
        ).detach().float().cpu().tolist()
        for name, value in zip(diagnostic_names, diagnostic_values):
            diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + (
                value * batch_size
            )
        diagnostic_samples += batch_size

        meters["batch"].update(time.time() - end)
        end = time.time()
        if index % 10 == 0 or index == len(loader) - 1:
            remaining = len(loader) - index - 1
            eta = timedelta(seconds=int(meters["batch"].avg * remaining))
            print(
                f"Train [{epoch + 1}/{args.epochs}] [{index + 1}/{len(loader)}] "
                f"Info {meters['info'].val:.4f} ({meters['info'].avg:.4f}) "
                f"Recon {meters['recon'].val:.4f} ({meters['recon'].avg:.4f}) "
                f"Div {meters['div'].val:.4f} ({meters['div'].avg:.4f}) "
                f"Att {meters['att'].val:.6f} ({meters['att'].avg:.6f}) "
                f"ETA {eta}",
                flush=True,
            )

    metrics = {
        "train_total_loss": meters["total"].avg,
        "train_info_loss": meters["info"].avg,
        "train_recon_loss": meters["recon"].avg,
        "train_div_loss": meters["div"].avg,
        "train_attention_match_loss": meters["att"].avg,
        "train_weighted_recon_loss": args.lam1 * meters["recon"].avg,
        "train_weighted_div_loss": args.lam2 * meters["div"].avg,
        "train_weighted_attention_match_loss": args.lam3 * meters["att"].avg,
    }
    metrics.update(mean_diagnostics(diagnostic_sums, diagnostic_samples))
    return metrics


@torch.no_grad()
def evaluate(loader, model, object_prior, args):
    values = test_model.validate_img_aud(
        loader,
        model,
        object_prior,
        str(Path(args.model_dir) / args.experiment_name / "viz"),
        args.testset,
        0,
        args,
    )
    names = (
        "aud_ciou",
        "aud_auc",
        "img_query_ciou",
        "img_query_auc",
        "iqr_ciou",
        "iqr_auc",
        "obj_prior_ciou",
        "obj_prior_auc",
        "ogl_ciou",
        "ogl_auc",
        "extra_iqr_ogl_ciou",
        "extra_iqr_ogl_auc",
    )
    return dict(zip(names, values))


def run(setting: str, gpu: int) -> None:
    args = common.load_baseline_config(setting, gpu=gpu)
    output_dir = common.experiment_dir(setting)
    if any((output_dir / name).exists() for name in ("latest.pth", "final.pth", args.testset + "_best.pth")):
        raise RuntimeError(f"Refusing to overwrite existing experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    train_slot.setup_seed(args.seed)
    torch.cuda.set_device(gpu)
    common.write_json(output_dir / "configs.json", vars(args))
    model = HierarchicalAudioStage1(args).cuda(gpu)

    if args.optimizer == "adam":
        optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)
    elif args.optimizer == "sgd":
        optimizer, scheduler = utils.build_optimizer_and_scheduler_sgd(model, args)
    else:
        raise ValueError(args.optimizer)
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = get_train_dataset(
        args,
        hard_img=args.hard_img,
        hard_aud=args.hard_aud,
        rand_aud=args.rand_aud,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    test_dataset = get_test_dataset(args, args.testset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    object_prior = build_object_prior(gpu)

    print(json.dumps(vars(args), indent=2), flush=True)
    print(
        f"train samples={len(train_dataset)}, batches={len(train_loader)}, "
        f"test samples={len(test_dataset)}",
        flush=True,
    )
    print("eval_audio_query_source=A4", flush=True)

    wandb_run = wandb.init(
        project=f"SSL_JSA_{args.trainset}",
        config=vars(args),
        name=args.experiment_name,
        anonymous="allow",
        mode="disabled",
    )
    history: list[dict] = []
    best_iqr = float("-inf")
    run_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = train_epoch(
            train_loader, model, optimizer, scaler, epoch, args
        )
        if args.scheduler:
            scheduler.step()
        eval_metrics = evaluate(test_loader, model, object_prior, args)

        if eval_metrics["iqr_ciou"] > best_iqr:
            best_iqr = eval_metrics["iqr_ciou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "selection_metric": "IQR_cIoU",
                    "selection_score": best_iqr,
                    "eval_audio_query_source": "A4",
                },
                output_dir / f"{args.testset}_best.pth",
            )
            print(
                f"Best model saved: epoch={epoch + 1}, IQR cIoU={best_iqr:.4f}",
                flush=True,
            )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            output_dir / "latest.pth",
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
            **train_metrics,
            **eval_metrics,
        }
        history.append(row)
        common.write_csv(output_dir / "epoch_metrics.csv", history)
        render_training_curves(
            output_dir / "epoch_metrics.csv",
            output_dir / "training_curves.png",
            title=args.experiment_name,
        )
        elapsed = time.time() - run_start
        remaining = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
        print(json.dumps(row, indent=2), flush=True)
        print(
            f"Epoch {epoch + 1}/{args.epochs} complete; "
            f"overall ETA {timedelta(seconds=int(remaining))}; "
            f"finish {datetime.now() + timedelta(seconds=remaining):%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs},
        output_dir / "final.pth",
    )
    wandb_run.finish()
    print(
        f"Training complete in {timedelta(seconds=int(time.time() - run_start))}",
        flush=True,
    )


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.setting, arguments.gpu)
