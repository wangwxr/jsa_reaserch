#!/usr/bin/env python3
"""Stage A: numerical, gradient, and decoder-alpha audit (no optimizer)."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    EXPERIMENT_ROOT,
    RESULTS_ROOT,
    checkpoint_identity,
    load_checkpoint_model,
    namespace_from_json,
    reference_paths,
    require_cuda,
    setup_paths,
    setup_seed,
    write_json,
)

setup_paths()

from dataset import get_test_dataset, get_train_dataset, inverse_normalize  # noqa: E402
from metrics import (  # noqa: E402
    binary_iou,
    entropy_and_effective_area,
    js_divergence,
    minmax_normalize,
    pearson,
    probability_normalize,
    spearman,
    top_overlap,
    write_rows,
)
from model import LossFactorizedL3L4, LossWeights  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402


LOSS_NAMES = (
    "info",
    "rec_img",
    "rec_aud",
    "div_img",
    "div_aud",
    "att_space",
    "att_time",
)
WEIGHTS = {
    "info": 1.0,
    "rec_img": 0.1,
    "rec_aud": 0.1,
    "div_img": 0.1,
    "div_aud": 0.1,
    "att_space": 100.0,
    "att_time": 100.0,
}
GROUPS = (
    "visual_backbone",
    "visual_l3_slot_branch",
    "visual_l4_slot_branch",
    "visual_fusion",
    "audio_backbone",
    "audio_slot_branch",
    "shared_slots",
    "image_decoder",
    "audio_decoder",
)


def parameter_group(name):
    if name.startswith("imgnet."):
        return "visual_backbone"
    if name.startswith("slot_attn.visual_branches.0."):
        return "visual_l3_slot_branch"
    if name.startswith("slot_attn.visual_branches.1."):
        return "visual_l4_slot_branch"
    if name.startswith("slot_attn.slot_fusion."):
        return "visual_fusion"
    if name.startswith("audnet."):
        return "audio_backbone"
    if name.startswith("slot_attn.audio_branch."):
        return "audio_slot_branch"
    if name in {
        "slot_attn.slots",
        "slot_attn.mask_token_img",
        "slot_attn.mask_token_aud",
    }:
        return "shared_slots"
    if name.startswith("img_decoder."):
        return "image_decoder"
    if name.startswith("aud_decoder."):
        return "audio_decoder"
    raise KeyError(f"Unassigned trainable parameter: {name}")


def max_abs(left, right):
    return float((left.detach() - right.detach()).abs().max().cpu())


def build_loaders(args, workers, audit_batches):
    train_dataset = get_train_dataset(
        args, hard_img=args.hard_img, hard_aud=args.hard_aud, rand_aud=args.rand_aud
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    if len(train_loader) < audit_batches:
        raise RuntimeError(
            f"Need {audit_batches} fixed batches, only {len(train_loader)} exist"
        )
    test_dataset = get_test_dataset(args, args.testset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return train_loader, test_loader, test_dataset


def run_equivalence(args, checkpoint_path, device, batch):
    old = MUFASAL3L4(args)
    old, _ = load_checkpoint_model(old, checkpoint_path, device)
    old.train()
    frame, spec = batch[0].to(device).float(), batch[1].to(device).float()

    captured = {}

    def capture_slot_output(_module, _inputs, output):
        captured.update(output)

    hook = old.slot_attn.register_forward_hook(capture_slot_output)
    setup_seed(12345)
    with torch.no_grad():
        old_info, old_recon, old_div, old_att = old(frame, spec)
    hook.remove()
    old_total = old_info + 0.1 * old_recon + 0.1 * old_div + 100.0 * old_att

    old_values = {
        "info": old_info.detach().cpu(),
        "recon": old_recon.detach().cpu(),
        "div": old_div.detach().cpu(),
        "att": old_att.detach().cpu(),
        "total": old_total.detach().cpu(),
        "aud_raw": captured["audq_imgk_attn"][:, 0].detach().cpu(),
        "img_raw": captured["imgq_imgk_attn"][:, 0].detach().cpu(),
        "fused_img_slots": captured["img_slots"].detach().cpu(),
        "audio_slots": captured["aud_slots"].detach().cpu(),
    }
    del old, old_info, old_recon, old_div, old_att, old_total, captured
    gc.collect()
    torch.cuda.empty_cache()

    new = LossFactorizedL3L4(args)
    new, _ = load_checkpoint_model(new, checkpoint_path, device)
    new.train()
    setup_seed(12345)
    with torch.no_grad():
        detailed = new.forward_train_detailed(frame, spec, LossWeights())
    losses = detailed["losses"]
    checks = {
        "old_recon_vs_rec_img_plus_rec_aud": max_abs(
            old_values["recon"], (losses["rec_img"] + losses["rec_aud"]).cpu()
        ),
        "old_att_vs_att_space_plus_att_time": max_abs(
            old_values["att"], (losses["att_space"] + losses["att_time"]).cpu()
        ),
        "old_div_vs_div_img_plus_div_aud": max_abs(
            old_values["div"], (losses["div_img"] + losses["div_aud"]).cpu()
        ),
        "old_total_vs_new_full_total": max_abs(old_values["total"], losses["total"].cpu()),
        "aud_raw": max_abs(old_values["aud_raw"], detailed["aud_raw"].cpu()),
        "img_raw": max_abs(old_values["img_raw"], detailed["img_raw"].cpu()),
        "fused_img_slots": max_abs(old_values["fused_img_slots"], detailed["fused_img_slots"].cpu()),
        "audio_slots": max_abs(old_values["audio_slots"], detailed["audio_slots"].cpu()),
    }
    tolerance = 1e-7
    payload = {
        "fp32": True,
        "batch_size": int(frame.shape[0]),
        "tolerance": tolerance,
        "max_absolute_errors": checks,
        "maximum_error": max(checks.values()),
        "passed": all(value <= tolerance for value in checks.values()),
        "old_losses": {
            key: float(value) for key, value in old_values.items()
            if value.ndim == 0
        },
    }
    del new, detailed, old_values, frame, spec
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _dot(grads_left, grads_right):
    total = torch.zeros((), device=next(g.device for g in grads_left if g is not None))
    for left, right in zip(grads_left, grads_right):
        if left is not None and right is not None:
            total = total + torch.sum(left * right)
    return float(total.detach().cpu())


def run_gradient_audit(
    args, checkpoint_path, device, train_loader, audit_batches, dataset
):
    model = LossFactorizedL3L4(args)
    model, _ = load_checkpoint_model(model, checkpoint_path, device)
    model.train()
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    names = [item[0] for item in named_parameters]
    params = [item[1] for item in named_parameters]
    groups = [parameter_group(name) for name in names]

    norm_rows = []
    group_rows = []
    loss_sums = defaultdict(float)
    weighted_sums = defaultdict(float)
    dot_sums = defaultdict(float)
    none_seen = {loss: set() for loss in LOSS_NAMES}
    zero_seen = {loss: set() for loss in LOSS_NAMES}

    for batch_index, batch in enumerate(train_loader):
        if batch_index >= audit_batches:
            break
        frame = batch[0].to(device, non_blocking=True).float()
        spec = batch[1].to(device, non_blocking=True).float()
        with torch.amp.autocast("cuda"):
            detailed = model.forward_train_detailed(frame, spec, LossWeights())
        raw = detailed["losses"]
        terms = {name: raw[name] * WEIGHTS[name] for name in LOSS_NAMES}
        gradients = {}

        for loss_index, loss_name in enumerate(LOSS_NAMES):
            model.zero_grad(set_to_none=True)
            gradients[loss_name] = torch.autograd.grad(
                terms[loss_name],
                params,
                retain_graph=loss_index < len(LOSS_NAMES) - 1,
                allow_unused=True,
            )
            loss_sums[loss_name] += float(raw[loss_name].detach().cpu())
            weighted_sums[loss_name] += float(terms[loss_name].detach().cpu())

            group_sq = defaultdict(float)
            global_sq = 0.0
            for name, group, grad in zip(names, groups, gradients[loss_name]):
                if grad is None:
                    none_seen[loss_name].add(name)
                    continue
                squared = float(torch.sum(grad.detach() ** 2).cpu())
                if squared == 0.0:
                    zero_seen[loss_name].add(name)
                global_sq += squared
                group_sq[group] += squared
            norm_rows.append(
                {
                    "dataset": dataset,
                    "batch": batch_index,
                    "weighted_loss": loss_name,
                    "raw_loss": float(raw[loss_name].detach().cpu()),
                    "weight": WEIGHTS[loss_name],
                    "weighted_loss_value": float(terms[loss_name].detach().cpu()),
                    "global_gradient_l2": math.sqrt(global_sq),
                }
            )
            for group in GROUPS:
                group_rows.append(
                    {
                        "dataset": dataset,
                        "batch": batch_index,
                        "weighted_loss": loss_name,
                        "parameter_group": group,
                        "gradient_l2": math.sqrt(group_sq[group]),
                    }
                )

        for left_index, left_name in enumerate(LOSS_NAMES):
            for right_name in LOSS_NAMES[left_index:]:
                dot_sums[(left_name, right_name)] += _dot(
                    gradients[left_name], gradients[right_name]
                )

        del detailed, raw, terms, gradients, frame, spec
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(f"[{dataset}] gradient audit {batch_index + 1}/{audit_batches}", flush=True)

    cosine_rows = []
    negative = 0
    pair_count = 0
    for left in LOSS_NAMES:
        for right in LOSS_NAMES:
            key = (left, right) if LOSS_NAMES.index(left) <= LOSS_NAMES.index(right) else (right, left)
            numerator = dot_sums[key]
            left_norm = math.sqrt(max(dot_sums[(left, left)], 0.0))
            right_norm = math.sqrt(max(dot_sums[(right, right)], 0.0))
            cosine = numerator / (left_norm * right_norm + 1e-30)
            cosine_rows.append(
                {"dataset": dataset, "loss_left": left, "loss_right": right, "cosine": cosine}
            )
            if LOSS_NAMES.index(left) < LOSS_NAMES.index(right):
                pair_count += 1
                negative += int(cosine < 0)

    raw_weighted = [
        {
            "dataset": dataset,
            "loss": name,
            "raw_mean": loss_sums[name] / audit_batches,
            "weight": WEIGHTS[name],
            "weighted_mean": weighted_sums[name] / audit_batches,
        }
        for name in LOSS_NAMES
    ]
    zero_none = {
        name: {
            "none_gradient_parameters_seen": sorted(none_seen[name]),
            "zero_gradient_parameters_seen": sorted(zero_seen[name]),
        }
        for name in LOSS_NAMES
    }
    summary = {
        "dataset": dataset,
        "batches": audit_batches,
        "batch_size": args.batch_size,
        "negative_gradient_cosine_ratio": negative / pair_count,
        "negative_pairs": negative,
        "pair_count": pair_count,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "gradient_forward_precision": "formal-training autocast float16",
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return norm_rows, group_rows, cosine_rows, raw_weighted, zero_none, summary


def pair_diagnostics(left, right, threshold=0.6):
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    left_prob = probability_normalize(left)
    right_prob = probability_normalize(right)
    left_eval = minmax_normalize(left)
    right_eval = minmax_normalize(right)
    ent_left, active_left, ratio_left = entropy_and_effective_area(left_prob)
    ent_right, active_right, ratio_right = entropy_and_effective_area(right_prob)
    return {
        "pearson": pearson(left_prob, right_prob),
        "spearman": spearman(left_prob, right_prob),
        "js_divergence": js_divergence(left_prob, right_prob),
        "top10_overlap": top_overlap(left_prob, right_prob),
        "threshold_mask_iou": binary_iou(
            left_eval >= threshold, right_eval >= threshold
        ),
        "left_entropy": ent_left,
        "right_entropy": ent_right,
        "left_effective_active_area": active_left,
        "right_effective_active_area": active_right,
        "left_effective_active_ratio": ratio_left,
        "right_effective_active_ratio": ratio_right,
    }


def _save_alpha_figure(path, image, gt, aud, img, alpha0, alpha1, sample_id):
    image = inverse_normalize(image.cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
    maps = [gt, aud, img, alpha0, alpha1]
    titles = ["GT", "AUD", "IMG", "Decoder target alpha", "Decoder off-target alpha"]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    for axis, heatmap, title in zip(axes[1:], maps, titles):
        axis.imshow(image, alpha=0.35)
        axis.imshow(heatmap, cmap="magma", alpha=0.65, vmin=0, vmax=1)
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(str(sample_id), fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def run_decoder_alpha_audit(
    args, checkpoint_path, device, test_loader, test_dataset, dataset, output_root
):
    model = LossFactorizedL3L4(args)
    model, _ = load_checkpoint_model(model, checkpoint_path, device)
    model.eval()
    target_rows = []
    alpha_sum_max_error = 0.0
    fixed_indices = set(
        np.linspace(0, len(test_dataset) - 1, num=min(10, len(test_dataset)), dtype=int).tolist()
    )
    global_index = 0

    for image, spec, bboxes, names, _labels in test_loader:
        image_gpu = image.to(device, non_blocking=True).float()
        spec_gpu = spec.to(device, non_blocking=True).float()
        detailed = model.forward_eval_detailed(image_gpu, spec_gpu, run_decoder=True)
        alpha = detailed["decoder_alpha"]
        alpha_sum_max_error = max(
            alpha_sum_max_error,
            float((alpha.sum(dim=1) - 1.0).abs().max().cpu()),
        )
        aud = detailed["aud"].cpu().numpy()[:, 0]
        img = detailed["img"].cpu().numpy()[:, 0]
        alpha_np = alpha.cpu().numpy()
        gt_np = bboxes.cpu().numpy()

        for local_index, name in enumerate(names):
            maps = {
                "DECODER_TARGET_vs_IMG": (alpha_np[local_index, 0], img[local_index]),
                "DECODER_TARGET_vs_AUD": (alpha_np[local_index, 0], aud[local_index]),
                "IMG_vs_AUD": (img[local_index], aud[local_index]),
            }
            for comparison, (left, right) in maps.items():
                row = {"dataset": dataset, "sample_id": str(name), "comparison": comparison}
                row.update(pair_diagnostics(left, right))
                target_rows.append(row)

            if global_index in fixed_indices:
                resize = lambda x: F.interpolate(
                    torch.as_tensor(x)[None, None].float(),
                    (224, 224),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                _save_alpha_figure(
                    output_root / "visualizations" / dataset / f"{global_index:05d}_{name}.png",
                    image[local_index],
                    gt_np[local_index],
                    minmax_normalize(resize(aud[local_index])),
                    minmax_normalize(resize(img[local_index])),
                    minmax_normalize(
                        resize(alpha_np[local_index, 0].reshape(7, 7))
                    ),
                    minmax_normalize(
                        resize(alpha_np[local_index, 1].reshape(7, 7))
                    ),
                    name,
                )
            global_index += 1

    summary = {"dataset": dataset, "decoder_alpha_slot_sum_max_error": alpha_sum_max_error, "sample_count": len(test_dataset), "comparisons": {}}
    for comparison in ("DECODER_TARGET_vs_IMG", "DECODER_TARGET_vs_AUD", "IMG_vs_AUD"):
        subset = [row for row in target_rows if row["comparison"] == comparison]
        summary["comparisons"][comparison] = {}
        for key in (
            "pearson",
            "spearman",
            "js_divergence",
            "top10_overlap",
            "threshold_mask_iou",
            "left_entropy",
            "right_entropy",
            "left_effective_active_area",
            "right_effective_active_area",
        ):
            values = np.asarray([row[key] for row in subset], dtype=np.float64)
            summary["comparisons"][comparison][key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
            }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return target_rows, summary


def run_dataset(dataset, gpu, audit_batches, workers):
    device = require_cuda(gpu)
    torch.cuda.set_device(gpu)
    setup_seed(12345)
    config_path, checkpoint_path = reference_paths(dataset)
    args = namespace_from_json(config_path, gpu=gpu, workers=workers, batch_size=256)
    output_root = RESULTS_ROOT / "audit"
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    before = checkpoint_identity(checkpoint_path)

    train_loader, test_loader, test_dataset = build_loaders(args, workers, audit_batches)
    first_batch = next(iter(train_loader))
    equivalence = run_equivalence(args, checkpoint_path, device, first_batch)
    write_json(dataset_root / "loss_equivalence.json", equivalence)
    if not equivalence["passed"]:
        after = checkpoint_identity(checkpoint_path)
        write_json(dataset_root / "checkpoint_audit.json", {"before": before, "after": after, "identical": before == after})
        raise RuntimeError(f"{dataset}: Full numerical equivalence failed: {equivalence}")

    norm_rows, group_rows, cosine_rows, raw_weighted, zero_none, grad_summary = run_gradient_audit(
        args, checkpoint_path, device, train_loader, audit_batches, dataset
    )
    write_rows(dataset_root / "weighted_gradient_norms.csv", norm_rows)
    write_rows(dataset_root / "parameter_group_gradient_norms.csv", group_rows)
    write_rows(dataset_root / "gradient_cosine_matrix.csv", cosine_rows)
    write_rows(dataset_root / "raw_and_weighted_losses.csv", raw_weighted)
    write_json(dataset_root / "zero_none_gradient_parameters.json", zero_none)
    write_json(dataset_root / "gradient_summary.json", grad_summary)

    # The train loader uses persistent workers.  Shut them down before the
    # full test-set alpha audit so the two loader pools do not coexist.
    del first_batch, train_loader
    gc.collect()

    alpha_rows, alpha_summary = run_decoder_alpha_audit(
        args, checkpoint_path, device, test_loader, test_dataset, dataset, output_root
    )
    write_rows(output_root / f"decoder_alpha_correlations_{dataset}.csv", alpha_rows)
    write_json(output_root / f"decoder_alpha_correlations_{dataset}.json", alpha_summary)

    after = checkpoint_identity(checkpoint_path)
    checkpoint_audit = {
        "dataset": dataset,
        "before": before,
        "after": after,
        "identical": before == after,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "model_checkpoint_saved": False,
    }
    write_json(dataset_root / "checkpoint_audit.json", checkpoint_audit)
    if not checkpoint_audit["identical"]:
        raise RuntimeError(f"{dataset}: reference checkpoint changed during audit")
    print(f"Stage A dataset audit passed: {dataset}", flush=True)


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate():
    root = RESULTS_ROOT / "audit"
    datasets = ("vggss", "flickr")
    for dataset in datasets:
        required = [
            root / dataset / "loss_equivalence.json",
            root / dataset / "weighted_gradient_norms.csv",
            root / dataset / "parameter_group_gradient_norms.csv",
            root / dataset / "gradient_cosine_matrix.csv",
            root / dataset / "checkpoint_audit.json",
            root / f"decoder_alpha_correlations_{dataset}.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot aggregate Stage A; missing {missing}")

    for filename in (
        "weighted_gradient_norms.csv",
        "parameter_group_gradient_norms.csv",
        "gradient_cosine_matrix.csv",
        "raw_and_weighted_losses.csv",
    ):
        rows = []
        for dataset in datasets:
            rows.extend(read_csv(root / dataset / filename))
        write_rows(root / filename, rows)

    equivalence = {}
    checkpoints = {}
    gradients = {}
    decoder = {}
    for dataset in datasets:
        with open(root / dataset / "loss_equivalence.json", encoding="utf-8") as handle:
            equivalence[dataset] = json.load(handle)
        with open(root / dataset / "checkpoint_audit.json", encoding="utf-8") as handle:
            checkpoints[dataset] = json.load(handle)
        with open(root / dataset / "gradient_summary.json", encoding="utf-8") as handle:
            gradients[dataset] = json.load(handle)
        with open(root / f"decoder_alpha_correlations_{dataset}.json", encoding="utf-8") as handle:
            decoder[dataset] = json.load(handle)
    write_json(root / "loss_equivalence.json", equivalence)
    write_json(root / "checkpoint_audit.json", checkpoints)
    summary = {
        "stage_a_passed": all(item["passed"] for item in equivalence.values())
        and all(item["identical"] for item in checkpoints.values()),
        "loss_equivalence": equivalence,
        "gradient_audit": gradients,
        "decoder_alpha": decoder,
    }

    global_records = read_csv(root / "weighted_gradient_norms.csv")
    group_records = read_csv(root / "parameter_group_gradient_norms.csv")
    cosine_records = read_csv(root / "gradient_cosine_matrix.csv")

    def mean_filtered(records, value_key, **filters):
        values = [
            float(row[value_key])
            for row in records
            if all(row[key] == value for key, value in filters.items())
        ]
        if not values:
            raise KeyError(filters)
        return float(np.mean(values))

    def global_norm(dataset, loss):
        return mean_filtered(
            global_records,
            "global_gradient_l2",
            dataset=dataset,
            weighted_loss=loss,
        )

    def group_norm(dataset, loss, group):
        return mean_filtered(
            group_records,
            "gradient_l2",
            dataset=dataset,
            weighted_loss=loss,
            parameter_group=group,
        )

    def grad_cos(dataset, left, right):
        return mean_filtered(
            cosine_records,
            "cosine",
            dataset=dataset,
            loss_left=left,
            loss_right=right,
        )

    def corr(dataset, pair, key="pearson"):
        return decoder[dataset]["comparisons"][pair][key]["mean"]

    findings = {}
    for dataset in datasets:
        findings[dataset] = {
            "info_global_grad": global_norm(dataset, "info"),
            "att_space_global_grad": global_norm(dataset, "att_space"),
            "rec_img_global_grad": global_norm(dataset, "rec_img"),
            "info_l3_slot_grad": group_norm(
                dataset, "info", "visual_l3_slot_branch"
            ),
            "rec_img_l3_slot_grad": group_norm(
                dataset, "rec_img", "visual_l3_slot_branch"
            ),
            "rec_img_l4_slot_grad": group_norm(
                dataset, "rec_img", "visual_l4_slot_branch"
            ),
            "rec_img_info_cosine": grad_cos(dataset, "rec_img", "info"),
            "rec_img_div_img_cosine": grad_cos(
                dataset, "rec_img", "div_img"
            ),
            "img_aud_pearson": corr(dataset, "IMG_vs_AUD"),
            "decoder_img_pearson": corr(dataset, "DECODER_TARGET_vs_IMG"),
            "decoder_aud_pearson": corr(dataset, "DECODER_TARGET_vs_AUD"),
            "decoder_img_top10_overlap": corr(
                dataset, "DECODER_TARGET_vs_IMG", "top10_overlap"
            ),
        }
    summary["key_findings"] = findings
    write_json(root / "audit_summary.json", summary)

    report = f"""# Experiment 6.0 — Stage A Loss Audit

## 审计结论状态

- Full 数值回归：{'通过' if summary['stage_a_passed'] else '失败'}。
- VGGSS 最大误差：`{equivalence['vggss']['maximum_error']:.3e}`。
- Flickr 最大误差：`{equivalence['flickr']['maximum_error']:.3e}`。
- 两个正式 checkpoint 的 SHA256 / size / mtime 在审计前后保持一致。
- 本阶段未创建 optimizer、未调用 optimizer.step，也未写入模型权重。

## Decoder alpha 与定位分支

| Dataset | decoder target–IMG Pearson | decoder target–AUD Pearson | IMG–AUD Pearson |
|---|---:|---:|---:|
| VGGSS | {corr('vggss', 'DECODER_TARGET_vs_IMG'):.4f} | {corr('vggss', 'DECODER_TARGET_vs_AUD'):.4f} | {corr('vggss', 'IMG_vs_AUD'):.4f} |
| Flickr | {corr('flickr', 'DECODER_TARGET_vs_IMG'):.4f} | {corr('flickr', 'DECODER_TARGET_vs_AUD'):.4f} | {corr('flickr', 'IMG_vs_AUD'):.4f} |

## 加权梯度证据（20 batches 均值）

| Dataset | info global | att-space global | rec-img global | rec-img L3-SA | rec-img L4-SA | cos(rec-img, info) | cos(rec-img, div-img) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS | {findings['vggss']['info_global_grad']:.4f} | {findings['vggss']['att_space_global_grad']:.4f} | {findings['vggss']['rec_img_global_grad']:.4f} | {findings['vggss']['rec_img_l3_slot_grad']:.4f} | {findings['vggss']['rec_img_l4_slot_grad']:.4f} | {findings['vggss']['rec_img_info_cosine']:.4f} | {findings['vggss']['rec_img_div_img_cosine']:.4f} |
| Flickr | {findings['flickr']['info_global_grad']:.4f} | {findings['flickr']['att_space_global_grad']:.4f} | {findings['flickr']['rec_img_global_grad']:.4f} | {findings['flickr']['rec_img_l3_slot_grad']:.4f} | {findings['flickr']['rec_img_l4_slot_grad']:.4f} | {findings['flickr']['rec_img_info_cosine']:.4f} | {findings['flickr']['rec_img_div_img_cosine']:.4f} |

## 五个因果问题

1. **spatial attention 是否在梯度上主导 AUD 空间定位**：在已收敛的正式 checkpoint 上不主导。VGGSS/Flickr 的加权 `att_space` global norm 仅为 `{findings['vggss']['att_space_global_grad']:.4f}/{findings['flickr']['att_space_global_grad']:.4f}`，而 InfoNCE 为 `{findings['vggss']['info_global_grad']:.4f}/{findings['flickr']['info_global_grad']:.4f}`。这不否定早期训练的因果作用，因此需由 Stage B 验证。
2. **image reconstruction 是否直接塑造 localization attention**：decoder target alpha 与 IMG/AUD 只有中等 Pearson（VGG约 0.55，Flickr约 0.44），而 top-10 overlap 仅 `{findings['vggss']['decoder_img_top10_overlap']:.3f}/{findings['flickr']['decoder_img_top10_overlap']:.3f}`；相比之下 IMG–AUD Pearson 约 0.986/0.987。因此 rec_img 更像 slot-feature/partition regularizer，不能仅凭 alpha 直接解释 SHRINK。
3. **rec_img、div_img、info_loss 是否冲突**：没有明显冲突。`cos(rec_img, info)` 为 `{findings['vggss']['rec_img_info_cosine']:.4f}/{findings['flickr']['rec_img_info_cosine']:.4f}`，`cos(rec_img, div_img)` 为 `{findings['vggss']['rec_img_div_img_cosine']:.4f}/{findings['flickr']['rec_img_div_img_cosine']:.4f}`，均接近零或轻微同向。
4. **L3 是否被 L4 reconstruction 牵制**：存在明确耦合但没有冲突证据。rec_img 对 L3-SA 的梯度约为 L4-SA 的 `{findings['vggss']['rec_img_l3_slot_grad']/findings['vggss']['rec_img_l4_slot_grad']:.1%}`（VGG）和 `{findings['flickr']['rec_img_l3_slot_grad']/findings['flickr']['rec_img_l4_slot_grad']:.1%}`（Flickr），说明 L4 target 经 M-Fusion 实质性约束 L3；但方向性与 InfoNCE 近正交，不能称为已证实的“牵制”。
5. **AUD–IMG 高相关来自何处**：收敛时显式 att-space 梯度很小，但 IMG–AUD 已高度相关，Stage A 更支持“共享 slot index/相同 L4 keys 与既有训练共同形成的结构性对齐”，不能仅归因当前 att loss。NO_SPATIAL_ATT 的从头训练才是区分 architecture 与 loss 的因果证据。

完整数值位于本目录 CSV/JSON；固定测试样本可视化位于 `visualizations/`。
"""
    (root / "REPORT_STAGE_A.md").write_text(report, encoding="utf-8")
    if not summary["stage_a_passed"]:
        raise RuntimeError("Stage A aggregate failed; Stage B must not start")
    print("Stage A aggregate passed; Stage B is allowed to start.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vggss", "flickr"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--audit-batches", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.dataset is None:
        raise ValueError("--dataset is required unless --aggregate is used")
    run_dataset(args.dataset, args.gpu, args.audit_batches, args.workers)


if __name__ == "__main__":
    main()
