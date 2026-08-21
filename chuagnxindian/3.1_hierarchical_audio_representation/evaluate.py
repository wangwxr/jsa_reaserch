#!/usr/bin/env python3
"""Best-checkpoint evaluation and mechanism diagnostics for Experiment 3.1."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
BASELINE_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
for path in (PROJECT_ROOT, V11_ROOT, BASELINE_ROOT, HERE):
    sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

import test_model  # noqa: E402
import train_slot  # noqa: E402
import utils  # noqa: E402
from dataset import get_test_dataset, inverse_normalize  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402

import common  # noqa: E402
from model import HierarchicalAudioStage1, temporal_token_diagnostics  # noqa: E402
from visualize import save_panel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=sorted(common.SETTINGS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--qualitative-count", type=int, default=12)
    return parser.parse_args()


def object_prior_model(gpu: int):
    model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        train_slot.NormReducer(dim=1),
        train_slot.Unsqueeze(1),
    )
    return model.cuda(gpu).eval()


def load_models(setting: str, gpu: int):
    args = common.load_baseline_config(setting, gpu=gpu)
    baseline = MUFASAL3L4(args)
    baseline_ckpt = torch.load(
        common.baseline_dir(setting) / common.registry(setting)["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    baseline.load_state_dict(
        {key.removeprefix("module."): value for key, value in baseline_ckpt["model"].items()},
        strict=True,
    )
    hierarchical = HierarchicalAudioStage1(args)
    checkpoint = torch.load(
        common.experiment_dir(setting) / common.registry(setting)["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    hierarchical.load_state_dict(
        {key.removeprefix("module."): value for key, value in checkpoint["model"].items()},
        strict=True,
    )
    return (
        args,
        baseline.cuda(gpu).eval(),
        hierarchical.cuda(gpu).eval(),
        checkpoint,
    )


def flatten_batch(image, spec, bboxes, names):
    if image.ndim == 3:
        image = image.unsqueeze(0)
        spec = spec.unsqueeze(0)
        bboxes = bboxes.unsqueeze(0)
    if image.ndim == 5:
        batch, views, channels, height, width = image.shape
        image = image.reshape(batch * views, channels, height, width)
        spec = spec.reshape(batch * views, *spec.shape[2:])
        bboxes = bboxes.reshape(batch * views, *bboxes.shape[2:]).squeeze(1)
        names = [name for name in names for _ in range(views)]
    return image, spec, bboxes, list(names)


def resize_attention(attention):
    return F.interpolate(
        attention, size=(224, 224), mode="bicubic", align_corners=False
    )


@torch.inference_mode()
def compute_batch_maps(baseline, hierarchical, object_prior, image, spec):
    audio, representations = hierarchical.diagnostic_representations(image, spec)
    slot_attention = hierarchical.slot_attn
    image_attention = slot_attention._attention(
        representations["visual_queries"][-1],
        representations["visual_keys"][-1],
        hierarchical.infer_sharpening,
    )
    audio_attention = slot_attention._attention(
        representations["audio_query_a4"],
        representations["visual_keys"][-1],
        hierarchical.infer_sharpening,
    )
    a3_attention = slot_attention._attention(
        representations["audio_query_a3"],
        representations["visual_keys"][-1],
        hierarchical.infer_sharpening,
    )
    batch = image.shape[0]
    new_img = image_attention[:, 0].reshape(batch, 1, 7, 7)
    new_aud = audio_attention[:, 0].reshape(batch, 1, 7, 7)
    a3_aud = a3_attention[:, 0].reshape(batch, 1, 7, 7)
    baseline_img, baseline_aud = baseline(image, spec)
    object_map = object_prior(image)
    tensors = {
        "BASE_IMG": baseline_img,
        "BASE_AUD": baseline_aud,
        "NEW_IMG": new_img,
        "NEW_AUD": new_aud,
        "A3_QUERY_AUD": a3_aud,
        "OBJ_PRIOR": object_map,
    }
    maps = {
        key: resize_attention(value).cpu().numpy()[:, 0]
        for key, value in tensors.items()
    }
    return maps, audio, representations


def normalized_maps(raw: dict[str, np.ndarray], index: int, alpha: float):
    maps = {key: utils.normalize_img(value[index]) for key, value in raw.items()}
    maps["BASE_IQR"] = utils.normalize_img(
        alpha * maps["BASE_AUD"] + (1.0 - alpha) * maps["BASE_IMG"]
    )
    maps["NEW_IQR"] = utils.normalize_img(
        alpha * maps["NEW_AUD"] + (1.0 - alpha) * maps["NEW_IMG"]
    )
    maps["NEW_OGL"] = utils.normalize_img(
        alpha * maps["NEW_AUD"] + (1.0 - alpha) * maps["OBJ_PRIOR"]
    )
    maps["NEW_EXTRA"] = utils.normalize_img(
        alpha * maps["NEW_AUD"]
        + (1.0 - alpha) * 0.5 * maps["NEW_IMG"]
        + (1.0 - alpha) * 0.5 * maps["OBJ_PRIOR"]
    )
    return maps


def evaluator_metric(evaluator):
    return {
        "cIoU": float(evaluator.finalize_AP50()),
        "AUC": float(evaluator.finalize_AUC()),
        "mean_sample_IoU": float(evaluator.finalize_cIoU()),
        "num_samples": len(evaluator.ciou),
    }


def accumulate_scalar(sums, counts, name, values):
    values = values.detach().float().reshape(-1)
    sums[name] = sums.get(name, 0.0) + float(values.sum())
    counts[name] = counts.get(name, 0) + values.numel()


def semantic_diagnostics(representations, sums, counts):
    visual = representations["fused_visual_slots"][:, 0]
    for name, key in (
        ("A3", "audio_slots_a3"),
        ("A4", "audio_slots_a4"),
        ("FUSED", "fused_audio_slots"),
    ):
        audio = representations[key][:, 0]
        positive = F.cosine_similarity(visual, audio, dim=-1)
        negative = F.cosine_similarity(visual, audio.roll(1, dims=0), dim=-1)
        accumulate_scalar(sums, counts, f"{name}_positive", positive)
        accumulate_scalar(sums, counts, f"{name}_shuffled_negative", negative)


def representation_diagnostics(audio, representations, sums, counts):
    batch = audio["a3_tokens"].shape[0]
    a3 = representations["audio_slots_a3"]
    a4 = representations["audio_slots_a4"]
    fused = representations["fused_audio_slots"]
    delta = representations["audio_delta"]
    values = {
        "cos_a3_a4_slot0": F.cosine_similarity(a3[:, 0], a4[:, 0], dim=-1),
        "cos_a3_a4_slot1": F.cosine_similarity(a3[:, 1], a4[:, 1], dim=-1),
        "cos_fused_a4_slot0": F.cosine_similarity(fused[:, 0], a4[:, 0], dim=-1),
        "cos_fused_a4_slot1": F.cosine_similarity(fused[:, 1], a4[:, 1], dim=-1),
        "delta_norm_over_a4_norm": (
            delta.norm(dim=-1) / a4.norm(dim=-1).clamp_min(1e-8)
        ).mean(dim=1),
    }
    for name, value in values.items():
        accumulate_scalar(sums, counts, name, value)
    for level in ("a3", "a4"):
        temporal = temporal_token_diagnostics(audio[f"{level}_tokens"])
        for name, value in temporal.items():
            sums[f"{level}_{name}"] = sums.get(f"{level}_{name}", 0.0) + float(value) * batch
            counts[f"{level}_{name}"] = counts.get(f"{level}_{name}", 0) + batch


def choose_category(row):
    base_aud = row["baseline_aud_iou"] >= 0.5
    new_aud = row["new_aud_iou"] >= 0.5
    base_iqr = row["baseline_iqr_iou"] >= 0.5
    new_iqr = row["new_iqr_iou"] >= 0.5
    if not base_aud and new_aud:
        return "AUD_IMPROVE"
    if base_aud and not new_aud:
        return "AUD_HURT"
    if not base_iqr and new_iqr:
        return "IQR_IMPROVE"
    if base_iqr and not new_iqr:
        return "IQR_HURT"
    if new_aud or new_iqr:
        return "STABLE_SUCCESS"
    return "ALL_FAIL"


def select_qualitative(rows, count):
    categories = (
        "AUD_IMPROVE",
        "AUD_HURT",
        "IQR_IMPROVE",
        "IQR_HURT",
        "STABLE_SUCCESS",
        "ALL_FAIL",
    )
    by_category = {category: [] for category in categories}
    for row in rows:
        by_category[row["category"]].append(row)
    selected = []
    cursor = {category: 0 for category in categories}
    while len(selected) < count:
        advanced = False
        for category in categories:
            index = cursor[category]
            if index < len(by_category[category]) and len(selected) < count:
                selected.append(by_category[category][index])
                cursor[category] += 1
                advanced = True
        if not advanced:
            break
    selected_ids = {row["sample_id"] for row in selected}
    for row in rows:
        if len(selected) >= count:
            break
        if row["sample_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["sample_id"])
    return selected


@torch.inference_mode()
def save_qualitative_panels(
    loader,
    baseline,
    hierarchical,
    object_prior,
    selected,
    output_dir,
    gpu,
    alpha,
):
    selected_by_id = {row["sample_id"]: row for row in selected}
    completed = set()
    for image, spec, bboxes, names, _labels in tqdm(loader, desc="qualitative"):
        image, spec, bboxes, names = flatten_batch(image, spec, bboxes, names)
        batch_ids = set(names) & set(selected_by_id)
        if not batch_ids:
            continue
        image_gpu = image.cuda(gpu, non_blocking=True).float()
        spec_gpu = spec.cuda(gpu, non_blocking=True).float()
        raw, _audio, _representations = compute_batch_maps(
            baseline, hierarchical, object_prior, image_gpu, spec_gpu
        )
        for index, sample_id in enumerate(names):
            if sample_id not in selected_by_id or sample_id in completed:
                continue
            maps = normalized_maps(raw, index, alpha)
            maps["OGL"] = maps["NEW_OGL"]
            maps["AUD_ABS_DELTA"] = utils.normalize_img(
                np.abs(maps["NEW_AUD"] - maps["BASE_AUD"])
            )
            maps["IQR_ABS_DELTA"] = utils.normalize_img(
                np.abs(maps["NEW_IQR"] - maps["BASE_IQR"])
            )
            denormalized = inverse_normalize(image[index]).permute(1, 2, 0).numpy()
            denormalized = np.clip(denormalized, 0.0, 1.0)
            row = selected_by_id[sample_id]
            save_panel(
                {
                    "sample_id": sample_id,
                    "category": row["category"],
                    "image": denormalized,
                    "gt": bboxes[index].numpy(),
                    "maps": maps,
                    "row": row,
                },
                output_dir / f"{len(completed) + 1:02d}_{sample_id}.png",
            )
            completed.add(sample_id)
        if len(completed) == len(selected_by_id):
            break
    if len(completed) != len(selected_by_id):
        raise RuntimeError(
            f"Saved {len(completed)}/{len(selected_by_id)} qualitative panels"
        )


def run(setting: str, gpu: int, qualitative_count: int):
    torch.cuda.set_device(gpu)
    args, baseline, hierarchical, checkpoint = load_models(setting, gpu)
    dataset = get_test_dataset(args, args.testset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    object_prior = object_prior_model(gpu)

    evaluators = {
        name: utils.Evaluator()
        for name in (
            "BASE_AUD",
            "BASE_IMG",
            "BASE_IQR",
            "NEW_AUD",
            "NEW_IMG",
            "NEW_IQR",
            "OBJ_PRIOR",
            "NEW_OGL",
            "NEW_EXTRA",
            "A3_QUERY_AUD",
        )
    }
    semantic_sums: dict[str, float] = {}
    semantic_counts: dict[str, int] = {}
    representation_sums: dict[str, float] = {}
    representation_counts: dict[str, int] = {}
    rows = []

    with torch.inference_mode():
        for image, spec, bboxes, names, _labels in tqdm(loader, desc=setting):
            image, spec, bboxes, names = flatten_batch(image, spec, bboxes, names)
            image = image.cuda(gpu, non_blocking=True).float()
            spec = spec.cuda(gpu, non_blocking=True).float()
            raw, audio, representations = compute_batch_maps(
                baseline, hierarchical, object_prior, image, spec
            )
            semantic_diagnostics(representations, semantic_sums, semantic_counts)
            representation_diagnostics(
                audio, representations, representation_sums, representation_counts
            )
            gt = bboxes.numpy()
            for index, sample_id in enumerate(names):
                maps = normalized_maps(raw, index, args.alpha)
                ious = {}
                for method, key in (
                    ("BASE_AUD", "BASE_AUD"),
                    ("BASE_IMG", "BASE_IMG"),
                    ("BASE_IQR", "BASE_IQR"),
                    ("NEW_AUD", "NEW_AUD"),
                    ("NEW_IMG", "NEW_IMG"),
                    ("NEW_IQR", "NEW_IQR"),
                    ("OBJ_PRIOR", "OBJ_PRIOR"),
                    ("NEW_OGL", "NEW_OGL"),
                    ("NEW_EXTRA", "NEW_EXTRA"),
                    ("A3_QUERY_AUD", "A3_QUERY_AUD"),
                ):
                    iou, *_ = evaluators[method].cal_CIOU(
                        maps[key], gt[index], sample_id, 0.6
                    )
                    ious[method] = float(iou)
                row = {
                    "sample_id": sample_id,
                    "baseline_aud_iou": ious["BASE_AUD"],
                    "baseline_img_iou": ious["BASE_IMG"],
                    "baseline_iqr_iou": ious["BASE_IQR"],
                    "new_aud_iou": ious["NEW_AUD"],
                    "new_img_iou": ious["NEW_IMG"],
                    "new_iqr_iou": ious["NEW_IQR"],
                    "new_ogl_iou": ious["NEW_OGL"],
                    "a3_query_iou": ious["A3_QUERY_AUD"],
                }
                row["category"] = choose_category(row)
                rows.append(row)

    metrics = {name: evaluator_metric(value) for name, value in evaluators.items()}
    semantic = {}
    for level in ("A3", "A4", "FUSED"):
        positive = semantic_sums[f"{level}_positive"] / semantic_counts[f"{level}_positive"]
        negative = semantic_sums[f"{level}_shuffled_negative"] / semantic_counts[
            f"{level}_shuffled_negative"
        ]
        semantic[level] = {
            "positive_cosine": positive,
            "shuffled_negative_cosine": negative,
            "margin": positive - negative,
        }
    representation = {
        name: representation_sums[name] / representation_counts[name]
        for name in representation_sums
    }

    selected_epoch = common.selected_row(
        common.experiment_dir(setting) / "epoch_metrics.csv"
    )
    baseline_row = common.baseline_best_row(setting)
    expected_new = common.metric_block(selected_epoch)
    method_mapping = {
        "AUD": "NEW_AUD",
        "IMG_QUERY": "NEW_IMG",
        "IQR": "NEW_IQR",
        "OBJ_PRIOR": "OBJ_PRIOR",
        "OGL": "NEW_OGL",
        "EXTRA_IQR_OGL": "NEW_EXTRA",
    }
    reproduction_errors = {
        method: {
            key: abs(metrics[source][key] - expected_new[method][key])
            for key in ("cIoU", "AUC")
        }
        for method, source in method_mapping.items()
    }
    max_reproduction_error = max(
        error
        for method in reproduction_errors.values()
        for error in method.values()
    )
    if max_reproduction_error > 1e-12:
        raise RuntimeError(f"Best checkpoint reproduction failed: {reproduction_errors}")

    baseline_metrics = common.metric_block(baseline_row)
    new_metrics = {
        method: {key: metrics[source][key] for key in ("cIoU", "AUC")}
        for method, source in method_mapping.items()
    }
    deltas = {
        method: {
            key: new_metrics[method][key] - baseline_metrics[method][key]
            for key in ("cIoU", "AUC")
        }
        for method in ("AUD", "IMG_QUERY", "IQR")
    }
    gaps = {
        "baseline": {
            "OGL_minus_AUD": baseline_metrics["OGL"]["cIoU"]
            - baseline_metrics["AUD"]["cIoU"],
            "OGL_minus_IQR": baseline_metrics["OGL"]["cIoU"]
            - baseline_metrics["IQR"]["cIoU"],
        },
        "hierarchical": {
            "OGL_minus_AUD": new_metrics["OGL"]["cIoU"]
            - new_metrics["AUD"]["cIoU"],
            "OGL_minus_IQR": new_metrics["OGL"]["cIoU"]
            - new_metrics["IQR"]["cIoU"],
        },
    }
    gaps["reduction"] = {
        key: gaps["baseline"][key] - gaps["hierarchical"][key]
        for key in gaps["baseline"]
    }
    gaps["best_no_OGL_main"] = max(
        new_metrics["AUD"]["cIoU"], new_metrics["IQR"]["cIoU"]
    )

    selected = select_qualitative(rows, qualitative_count)
    output_dir = common.result_dir(setting)
    save_qualitative_panels(
        loader,
        baseline,
        hierarchical,
        object_prior,
        selected,
        output_dir / "qualitative",
        gpu,
        args.alpha,
    )
    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    common.write_csv(
        output_dir / "qualitative" / "selection_manifest.csv", selected
    )
    summary = {
        "experiment": "3.1 Hierarchical Audio Representation",
        "setting": setting,
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_score": checkpoint.get("selection_score"),
        "eval_audio_query_source": checkpoint.get(
            "eval_audio_query_source", hierarchical.eval_audio_query_source
        ),
        "baseline_best_epoch": baseline_row["epoch"],
        "baseline_metrics_from_epoch_csv": baseline_metrics,
        "baseline_checkpoint_re_evaluation": {
            "AUD": metrics["BASE_AUD"],
            "IMG_QUERY": metrics["BASE_IMG"],
            "IQR": metrics["BASE_IQR"],
        },
        "hierarchical_metrics": new_metrics,
        "deltas": deltas,
        "ogl_gaps": gaps,
        "A3_QUERY_AUD": metrics["A3_QUERY_AUD"],
        "semantic_alignment": semantic,
        "representation_diagnostics": representation,
        "best_checkpoint_reproduction": {
            "errors": reproduction_errors,
            "max_error": max_reproduction_error,
            "passed": True,
        },
        "qualitative_ids": [row["sample_id"] for row in selected],
        "num_samples": len(rows),
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.setting, arguments.gpu, arguments.qualitative_count)
