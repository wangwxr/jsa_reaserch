#!/usr/bin/env python3
"""Experiment 3.2: frozen A4 temporal grounding mechanism probe."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import common

visualize = common.load_module("experiment_32_visualize", common.HERE / "visualize.py")


EPS = 1e-8
RAW_CONSENSUS = ("T2_RAW_MEAN", "T2_RAW_GEO", "T4_RAW_MEAN", "T4_RAW_GEO")
METHODS = (
    "ORIGINAL_AUD",
    "T2_CHUNK1",
    "T2_CHUNK2",
    "T2_RAW_MEAN",
    "T2_RAW_GEO",
    "T2_NORM_MEAN",
    "T2_NORM_GEO",
    "T4_CHUNK1",
    "T4_CHUNK2",
    "T4_CHUNK3",
    "T4_CHUNK4",
    "T4_RAW_MEAN",
    "T4_RAW_GEO",
    "T4_NORM_MEAN",
    "T4_NORM_GEO",
    "OGL_REFERENCE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument("--qualitative-count", type=int, default=12)
    return parser.parse_args()


def contiguous_boundaries(length: int, chunks: int) -> list[tuple[int, int]]:
    if length < chunks:
        raise ValueError(f"Cannot split T={length} into {chunks} non-empty chunks")
    width, remainder = divmod(length, chunks)
    boundaries = []
    start = 0
    for index in range(chunks):
        stop = start + width + (1 if index < remainder else 0)
        boundaries.append((start, stop))
        start = stop
    if start != length:
        raise RuntimeError((length, boundaries))
    return boundaries


def query_map(teacher, tokens, initial_slots, k34):
    slots, query, keys = teacher.slot_attn.audio_branch(tokens, initial_slots)
    attention = teacher.slot_attn._attention(query, k34, teacher.infer_sharpening)
    target = attention[:, 0].reshape(tokens.shape[0], 1, 14, 14)
    return {"slots": slots, "query": query, "keys": keys, "map": target}


@torch.inference_mode()
def extract_temporal(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, Any]:
    teacher = model.teacher
    audio_feature = teacher.audnet(audio)
    audio_tokens = teacher._audio_tokens(audio_feature)
    batch, temporal_length, _channels = audio_tokens.shape
    initial_slots = teacher.slot_attn.slots.expand(batch, -1, -1)

    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    f34, f3_spatial, f4_up, delta_f3 = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    visual_slots_l4, visual_query_l4, _visual_keys_l4 = l4_branch(
        image_levels[-1], initial_slots
    )

    full = query_map(teacher, audio_tokens, initial_slots, k34)
    boundaries = {
        "T2": contiguous_boundaries(temporal_length, 2),
        "T4": contiguous_boundaries(temporal_length, 4),
    }
    chunks: dict[str, list[dict[str, torch.Tensor]]] = {}
    for scale, scale_boundaries in boundaries.items():
        chunks[scale] = [
            query_map(teacher, audio_tokens[:, start:stop], initial_slots, k34)
            for start, stop in scale_boundaries
        ]

    raw_maps: dict[str, torch.Tensor] = {"ORIGINAL_AUD": full["map"]}
    for scale in ("T2", "T4"):
        maps = [item["map"] for item in chunks[scale]]
        for index, value in enumerate(maps, start=1):
            raw_maps[f"{scale}_CHUNK{index}"] = value
        stack = torch.stack(maps, dim=0)
        raw_maps[f"{scale}_RAW_MEAN"] = stack.mean(dim=0)
        raw_maps[f"{scale}_RAW_GEO"] = torch.exp(
            torch.log(stack.clamp_min(EPS)).mean(dim=0)
        )

    return {
        "audio_feature": audio_feature,
        "audio_tokens": audio_tokens,
        "full": full,
        "chunks": chunks,
        "boundaries": boundaries,
        "raw_maps": raw_maps,
        "visual_slots_l4": visual_slots_l4,
        "visual_query_l4": visual_query_l4,
        "F34": f34,
        "F3_SPATIAL": f3_spatial,
        "F4_UP": f4_up,
        "DELTA_F3": delta_f3,
        "K34": k34,
        "f4_token_error": (
            f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]
        ).abs().max(),
    }


def evaluator_maps(output: dict[str, Any], object_prior: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    resized = {
        name: common.resize_tensor(value).detach().cpu().numpy()[:, 0]
        for name, value in output["raw_maps"].items()
    }
    resized_object = common.resize_tensor(object_prior).detach().cpu().numpy()[:, 0]
    batch = resized["ORIGINAL_AUD"].shape[0]
    maps = {name: [] for name in METHODS}
    stability = {name: [] for name in ("TEMP_MEAN", "TEMP_STD", "TEMP_CV")}

    for sample_index in range(batch):
        normalized = {
            name: common.normalize_map(values[sample_index])
            for name, values in resized.items()
        }
        for scale, count in (("T2", 2), ("T4", 4)):
            chunk_stack = np.stack(
                [normalized[f"{scale}_CHUNK{index}"] for index in range(1, count + 1)],
                axis=0,
            )
            normalized[f"{scale}_NORM_MEAN"] = common.normalize_map(chunk_stack.mean(axis=0))
            normalized[f"{scale}_NORM_GEO"] = common.normalize_map(
                np.exp(np.log(np.clip(chunk_stack, EPS, None)).mean(axis=0))
            )
        normalized["OGL_REFERENCE"] = common.normalize_map(
            0.6 * normalized["ORIGINAL_AUD"]
            + 0.4 * common.normalize_map(resized_object[sample_index])
        )
        for method in METHODS:
            maps[method].append(common.normalize_map(normalized[method]))

        t4_stack = np.stack(
            [normalized[f"T4_CHUNK{index}"] for index in range(1, 5)], axis=0
        )
        temporal_mean = t4_stack.mean(axis=0)
        temporal_std = t4_stack.std(axis=0)
        stability["TEMP_MEAN"].append(temporal_mean)
        stability["TEMP_STD"].append(temporal_std)
        stability["TEMP_CV"].append(temporal_std / np.clip(temporal_mean, EPS, None))

    return (
        {name: np.stack(values, axis=0) for name, values in maps.items()},
        {name: np.stack(values, axis=0) for name, values in stability.items()},
    )


def all_finite(output: dict[str, Any]) -> bool:
    tensors = [
        output["audio_feature"],
        output["audio_tokens"],
        output["full"]["slots"],
        output["full"]["query"],
        output["full"]["keys"],
        output["visual_slots_l4"],
        output["visual_query_l4"],
        output["F34"],
        output["F3_SPATIAL"],
        output["F4_UP"],
        output["DELTA_F3"],
        output["K34"],
        *output["raw_maps"].values(),
    ]
    for scale in ("T2", "T4"):
        for item in output["chunks"][scale]:
            tensors.extend((item["slots"], item["query"], item["keys"], item["map"]))
    return all(torch.isfinite(value).all().item() for value in tensors)


@torch.inference_mode()
def tensor_audit(loader, model, device: torch.device) -> dict[str, Any]:
    image, spec, bboxes, names, _labels = next(iter(loader))
    image, spec, _bboxes, names = common.flatten_batch(image, spec, bboxes, names)
    image = image.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    official = model(image, spec)["AUD_FINE"]

    captured: dict[str, torch.Tensor] = {}

    def capture_raw_a4(_module, inputs):
        captured["raw_A4"] = inputs[0]

    handle = model.teacher.audnet.proj.register_forward_pre_hook(capture_raw_a4)
    try:
        output = extract_temporal(model, image, spec)
    finally:
        handle.remove()

    full_error = float((output["raw_maps"]["ORIGINAL_AUD"] - official).abs().max())
    spatial_sum_errors = {
        name: float((value.flatten(start_dim=2).sum(dim=-1) - 1.0).abs().max())
        for name, value in output["raw_maps"].items()
        if "CHUNK" in name or name == "ORIGINAL_AUD" or name.endswith("RAW_MEAN")
    }
    audit = {
        "sample_ids": names[:4],
        "raw_A4_shape": list(captured["raw_A4"].shape),
        "A4_feature_shape": list(output["audio_feature"].shape),
        "A4_tokens_shape": list(output["audio_tokens"].shape),
        "T": int(output["audio_tokens"].shape[1]),
        "T2_boundaries": [list(value) for value in output["boundaries"]["T2"]],
        "T4_boundaries": [list(value) for value in output["boundaries"]["T4"]],
        "full_query_shape": list(output["full"]["query"].shape),
        "T2_query_shapes": [list(value["query"].shape) for value in output["chunks"]["T2"]],
        "T4_query_shapes": [list(value["query"].shape) for value in output["chunks"]["T4"]],
        "T2_map_shapes": [list(value["map"].shape) for value in output["chunks"]["T2"]],
        "T4_map_shapes": [list(value["map"].shape) for value in output["chunks"]["T4"]],
        "F34_shape": list(output["F34"].shape),
        "K34_shape": list(output["K34"].shape),
        "visual_query_l4_shape": list(output["visual_query_l4"].shape),
        "FULL_AUD_tensor_reproduction_max_error": full_error,
        "spatial_sum_max_errors": spatial_sum_errors,
        "f4_token_error": float(output["f4_token_error"]),
        "no_nan_or_inf": all_finite(output),
        "attention_implementation": "teacher.slot_attn._attention with original infer_sharpening",
        "target_slot": 0,
    }
    expected = {
        "raw_A4_shape": [512, 9, 16],
        "A4_feature_shape": [512, 16],
        "A4_tokens_shape": [16, 512],
        "full_query_shape": [2, 512],
        "F34_shape": [512, 14, 14],
        "K34_shape": [196, 512],
        "visual_query_l4_shape": [2, 512],
    }
    for key, expected_nonbatch in expected.items():
        if audit[key][1:] != expected_nonbatch:
            raise RuntimeError(f"Unexpected {key}: {audit[key]}")
    if audit["T2_boundaries"] != [[0, 8], [8, 16]]:
        raise RuntimeError(audit)
    if audit["T4_boundaries"] != [[0, 4], [4, 8], [8, 12], [12, 16]]:
        raise RuntimeError(audit)
    if full_error > 1e-6 or audit["f4_token_error"] > 1e-6 or not audit["no_nan_or_inf"]:
        raise RuntimeError(audit)
    if max(spatial_sum_errors.values()) > 1e-6:
        raise RuntimeError(audit)
    audit["passed"] = True
    return audit


def append_distribution(target: dict[str, list[float]], name: str, tensor: torch.Tensor) -> None:
    target.setdefault(name, []).extend(tensor.detach().float().cpu().reshape(-1).tolist())


def accumulate_query_diagnostics(output: dict[str, Any], semantic, consistency) -> None:
    visual = output["visual_query_l4"][:, 0]
    query_groups = {"FULL": [output["full"]["query"]]}
    query_groups.update(
        {scale: [item["query"] for item in output["chunks"][scale]] for scale in ("T2", "T4")}
    )
    for scale, queries in query_groups.items():
        for index, query in enumerate(queries, start=1):
            positive = F.cosine_similarity(query[:, 0], visual, dim=-1)
            negative = F.cosine_similarity(query[:, 0], visual.roll(1, dims=0), dim=-1)
            key = f"{scale}_CHUNK{index}" if scale != "FULL" else "FULL"
            append_distribution(semantic, f"{key}_positive", positive)
            append_distribution(semantic, f"{key}_negative", negative)

    for scale in ("T2", "T4"):
        q0 = [item["query"][:, 0] for item in output["chunks"][scale]]
        pairwise = [F.cosine_similarity(first, second, dim=-1) for first, second in itertools.combinations(q0, 2)]
        pairwise_mean = torch.stack(pairwise, dim=0).mean(dim=0)
        variance = torch.stack(q0, dim=0).var(dim=0, unbiased=False).mean(dim=-1)
        append_distribution(consistency, f"{scale}_pairwise_cosine", pairwise_mean)
        append_distribution(consistency, f"{scale}_query_variance", variance)


def sample_map_similarity(chunk_maps: list[np.ndarray]) -> dict[str, float]:
    pearson, spearman, js = [], [], []
    for first, second in itertools.combinations(chunk_maps, 2):
        pearson.append(common.safe_pearson(first, second))
        spearman.append(common.spearman(first, second))
        js.append(common.js_divergence(first, second))
    return {
        "pearson": float(np.nanmean(pearson)),
        "spearman": float(np.nanmean(spearman)),
        "js_divergence": float(np.nanmean(js)),
    }


def select_qualitative(rows: list[dict[str, Any]], count: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    categories = (
        "TEMPORAL_RESCUE",
        "TEMPORAL_HURT",
        "OGL_RESCUE_TEMPORAL_RESCUE",
        "OGL_RESCUE_TEMPORAL_FAIL",
        "BASELINE_SUCCESS_STABLE",
        "VGG_OVER_EXPANSION",
    )
    candidates = {category: [] for category in categories}
    memberships: dict[str, list[str]] = {}
    for row in rows:
        original = row["IoU_ORIGINAL_AUD"] >= 0.5
        consensus_success = [row[f"IoU_{method}"] >= 0.5 for method in RAW_CONSENSUS]
        ogl_rescue = not original and row["IoU_OGL_REFERENCE"] >= 0.5
        labels = []
        if not original and any(consensus_success):
            labels.append("TEMPORAL_RESCUE")
        if original and not all(consensus_success):
            labels.append("TEMPORAL_HURT")
        if ogl_rescue and any(consensus_success):
            labels.append("OGL_RESCUE_TEMPORAL_RESCUE")
        if ogl_rescue and not any(consensus_success):
            labels.append("OGL_RESCUE_TEMPORAL_FAIL")
        if original and all(consensus_success):
            labels.append("BASELINE_SUCCESS_STABLE")
        if row["is_over_expansion"]:
            labels.append("VGG_OVER_EXPANSION")
        memberships[row["sample_id"]] = labels
        for label in labels:
            candidates[label].append(row["sample_id"])

    selected: list[str] = []
    rank = 0
    while len(selected) < count:
        advanced = False
        for category in categories:
            if rank < len(candidates[category]):
                sample_id = candidates[category][rank]
                if sample_id not in selected:
                    selected.append(sample_id)
                    advanced = True
                    if len(selected) == count:
                        break
        rank += 1
        if not advanced and rank >= max((len(value) for value in candidates.values()), default=0):
            break
    for row in rows:
        if len(selected) >= count:
            break
        if row["sample_id"] not in selected:
            selected.append(row["sample_id"])
    selection = [
        {
            "sample_id": sample_id,
            "categories": "|".join(memberships[sample_id]) or "DETERMINISTIC_FILL",
            "selection_rule": "first-in-test-order round-robin over fixed Experiment 3.2 categories",
        }
        for sample_id in selected
    ]
    return selection, {category: len(candidates[category]) for category in categories}


@torch.inference_mode()
def save_qualitative(loader, model, object_model, selection, rows, output_dir, device) -> None:
    wanted = {row["sample_id"]: row for row in selection}
    row_lookup = {row["sample_id"]: row for row in rows}
    found = {}
    for image, spec, bboxes, names, _labels in tqdm(loader, desc="qualitative", dynamic_ncols=True):
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        needed = [index for index, sample_id in enumerate(names) if sample_id in wanted]
        if not needed:
            continue
        image_gpu = image.to(device, non_blocking=True).float()
        spec_gpu = spec.to(device, non_blocking=True).float()
        output = extract_temporal(model, image_gpu, spec_gpu)
        maps, stability = evaluator_maps(output, object_model(image_gpu))
        gt = bboxes.numpy()
        for index in needed:
            sample_id = names[index]
            rgb = common.inverse_normalize(image[index]).permute(1, 2, 0).numpy()
            payload = {
                "sample_id": sample_id,
                "categories": wanted[sample_id]["categories"],
                "image": np.clip(rgb, 0.0, 1.0),
                "GT": gt[index],
                "row": row_lookup[sample_id],
            }
            for name in (
                "ORIGINAL_AUD",
                "T4_CHUNK1",
                "T4_CHUNK2",
                "T4_CHUNK3",
                "T4_CHUNK4",
                "T4_RAW_MEAN",
                "T4_RAW_GEO",
                "T4_NORM_GEO",
                "OGL_REFERENCE",
            ):
                payload[name] = maps[name][index]
            payload["TEMP_STD"] = common.normalize_map(stability["TEMP_STD"][index])
            found[sample_id] = payload
        if len(found) == len(wanted):
            break
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"Missing qualitative samples: {sorted(missing)}")
    for index, row in enumerate(selection, start=1):
        visualize.save_panel(found[row["sample_id"]], output_dir / f"{index:02d}_{row['sample_id']}.png")
    common.write_csv(output_dir / "selection_manifest.csv", selection)


def audit_models(models: dict[str, Any]) -> dict[str, Any]:
    trainable = []
    gradients = []
    eval_modes = {}
    for model_name, model in models.items():
        eval_modes[model_name] = not model.training
        for parameter_name, parameter in model.named_parameters():
            full_name = f"{model_name}.{parameter_name}"
            if parameter.requires_grad:
                trainable.append(full_name)
            if parameter.grad is not None:
                gradients.append(full_name)
    return {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": sum(
            parameter.numel()
            for model in models.values()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_names": trainable,
        "parameters_with_grad": gradients,
        "all_models_eval": all(eval_modes.values()),
        "model_eval_modes": eval_modes,
        "torch_inference_mode_used": True,
        "object_prior_used_for_evaluation_only": True,
    }


def metric_error(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    return {key: abs(float(actual[key]) - float(expected[key])) for key in ("cIoU", "AUC")}


@torch.inference_mode()
def run(arguments: argparse.Namespace) -> None:
    started = time.time()
    registry = common.EXPERIMENTS[arguments.experiment]
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)

    snapshots_before = common.snapshot_files(common.checkpoint_paths(registry))
    config = common.load_config(registry["stage1"])
    loader = common.build_loader(config, registry)
    model = common.load_original_g(registry, device)
    object_model = common.object_prior_model().to(device).eval()

    audit = tensor_audit(loader, model, device)
    common.write_json(output_dir / "tensor_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)

    values: dict[str, list[float]] = {method: [] for method in METHODS}
    rows: list[dict[str, Any]] = []
    semantic: dict[str, list[float]] = {}
    query_consistency: dict[str, list[float]] = {}
    map_similarity = {
        scale: {name: [] for name in ("pearson", "spearman", "js_divergence")}
        for scale in ("T2", "T4")
    }
    regions = {
        region: {name: [] for name in ("TEMP_STD", "TEMP_CV")}
        for region in ("GT_REGION", "AUD_FP_REGION")
    }
    over_regions = {
        region: {name: [] for name in ("TEMP_STD", "TEMP_CV")}
        for region in ("GT_REGION", "OVER_EXPANSION_REGION")
    }
    fp_gt_std_flags: list[float] = []
    fp_gt_cv_flags: list[float] = []
    over_std_flags: list[float] = []
    over_cv_flags: list[float] = []
    full_tensor_max_error = audit["FULL_AUD_tensor_reproduction_max_error"]
    no_nan_or_inf = True

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(
        tqdm(loader, desc=arguments.experiment, dynamic_ncols=True)
    ):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_temporal(model, image, spec)
        object_prior = object_model(image)
        if not all_finite(output) or not torch.isfinite(object_prior).all().item():
            no_nan_or_inf = False
            raise RuntimeError(f"NaN/Inf detected in batch {batch_index}")
        accumulate_query_diagnostics(output, semantic, query_consistency)
        maps, stability = evaluator_maps(output, object_prior)
        raw_chunks = {
            scale: [
                item["map"].detach().cpu().numpy()[:, 0]
                for item in output["chunks"][scale]
            ]
            for scale in ("T2", "T4")
        }
        gt_batch = bboxes.numpy()

        for sample_index, sample_id in enumerate(names):
            row: dict[str, Any] = {"sample_id": sample_id}
            gt = gt_batch[sample_index] > 0
            for method in METHODS:
                iou = common.sample_iou(maps[method][sample_index], gt_batch[sample_index])
                values[method].append(iou)
                row[f"IoU_{method}"] = iou

            original_pred = maps["ORIGINAL_AUD"][sample_index] >= 0.6
            fp = original_pred & (~gt)
            row["original_pred_area"] = int(original_pred.sum())
            row["gt_area"] = int(gt.sum())
            row["is_over_expansion"] = bool(original_pred.sum() > gt.sum())
            sample_std = stability["TEMP_STD"][sample_index]
            sample_cv = stability["TEMP_CV"][sample_index]
            if gt.any():
                regions["GT_REGION"]["TEMP_STD"].append(float(sample_std[gt].mean()))
                regions["GT_REGION"]["TEMP_CV"].append(float(sample_cv[gt].mean()))
            if fp.any():
                regions["AUD_FP_REGION"]["TEMP_STD"].append(float(sample_std[fp].mean()))
                regions["AUD_FP_REGION"]["TEMP_CV"].append(float(sample_cv[fp].mean()))
            if gt.any() and fp.any():
                fp_gt_std_flags.append(float(sample_std[fp].mean() > sample_std[gt].mean()))
                fp_gt_cv_flags.append(float(sample_cv[fp].mean() > sample_cv[gt].mean()))
            if row["is_over_expansion"] and gt.any() and fp.any():
                gt_std = float(sample_std[gt].mean())
                gt_cv = float(sample_cv[gt].mean())
                over_std = float(sample_std[fp].mean())
                over_cv = float(sample_cv[fp].mean())
                over_regions["GT_REGION"]["TEMP_STD"].append(gt_std)
                over_regions["GT_REGION"]["TEMP_CV"].append(gt_cv)
                over_regions["OVER_EXPANSION_REGION"]["TEMP_STD"].append(over_std)
                over_regions["OVER_EXPANSION_REGION"]["TEMP_CV"].append(over_cv)
                over_std_flags.append(float(over_std > gt_std))
                over_cv_flags.append(float(over_cv > gt_cv))

            for scale in ("T2", "T4"):
                similarity = sample_map_similarity(
                    [chunk[sample_index] for chunk in raw_chunks[scale]]
                )
                for name, value in similarity.items():
                    map_similarity[scale][name].append(value)
                    row[f"{scale}_map_{name}"] = value
            for method in RAW_CONSENSUS:
                row[f"delta_{method}"] = row[f"IoU_{method}"] - row["IoU_ORIGINAL_AUD"]
            row["delta_OGL"] = row["IoU_OGL_REFERENCE"] - row["IoU_ORIGINAL_AUD"]
            rows.append(row)

    metrics = [
        {"dataset": arguments.experiment, "method": method, **common.summarize(values[method])}
        for method in METHODS
    ]
    metric_lookup = {row["method"]: row for row in metrics}

    semantic_rows = []
    for scale, count in (("FULL", 1), ("T2", 2), ("T4", 4)):
        scale_positive, scale_negative = [], []
        for index in range(1, count + 1):
            key = "FULL" if scale == "FULL" else f"{scale}_CHUNK{index}"
            positive = semantic[f"{key}_positive"]
            negative = semantic[f"{key}_negative"]
            scale_positive.extend(positive)
            scale_negative.extend(negative)
            semantic_rows.append(
                {
                    "dataset": arguments.experiment,
                    "scale": scale,
                    "chunk": index,
                    "positive_cosine": float(np.mean(positive)),
                    "negative_cosine": float(np.mean(negative)),
                    "margin": float(np.mean(positive) - np.mean(negative)),
                }
            )
        semantic_rows.append(
            {
                "dataset": arguments.experiment,
                "scale": scale,
                "chunk": "MEAN",
                "positive_cosine": float(np.mean(scale_positive)),
                "negative_cosine": float(np.mean(scale_negative)),
                "margin": float(np.mean(scale_positive) - np.mean(scale_negative)),
            }
        )

    consistency_rows = []
    for scale in ("T2", "T4"):
        consistency_rows.append(
            {
                "dataset": arguments.experiment,
                "scale": scale,
                "query_pairwise_cosine": common.distribution(
                    query_consistency[f"{scale}_pairwise_cosine"]
                ),
                "query_variance": common.distribution(
                    query_consistency[f"{scale}_query_variance"]
                ),
            }
        )
    map_rows = [
        {
            "dataset": arguments.experiment,
            "scale": scale,
            **{name: common.distribution(values_) for name, values_ in measures.items()},
        }
        for scale, measures in map_similarity.items()
    ]

    transition_rows = []
    for method in RAW_CONSENSUS:
        shift = common.transition(values["ORIGINAL_AUD"], values[method])
        original_success = np.asarray(values["ORIGINAL_AUD"]) >= 0.5
        candidate_success = np.asarray(values[method]) >= 0.5
        ogl_success = np.asarray(values["OGL_REFERENCE"]) >= 0.5
        temporal_rescue = (~original_success) & candidate_success
        ogl_rescue = (~original_success) & ogl_success
        transition_rows.append(
            {
                "dataset": arguments.experiment,
                "method": method,
                "rescue": shift["rescue"],
                "hurt": shift["hurt"],
                "net": shift["net"],
                "rescue_intersect_OGL_rescue": int((temporal_rescue & ogl_rescue).sum()),
                "oracle_cIoU": shift["oracle"]["cIoU"],
                "oracle_AUC": shift["oracle"]["AUC"],
            }
        )

    original_success = np.asarray(values["ORIGINAL_AUD"]) >= 0.5
    ogl_success = np.asarray(values["OGL_REFERENCE"]) >= 0.5
    ogl_pool = (~original_success) & ogl_success
    capture_rows = []
    for method in RAW_CONSENSUS:
        candidate_success = np.asarray(values[method]) >= 0.5
        captured = int((ogl_pool & candidate_success).sum())
        capture_rows.append(
            {
                "dataset": arguments.experiment,
                "method": method,
                "OGL_rescue_total": int(ogl_pool.sum()),
                "consensus_captured": captured,
                "capture_rate": captured / max(int(ogl_pool.sum()), 1),
            }
        )

    region_rows = []
    for region, measures in regions.items():
        region_rows.append(
            {
                "dataset": arguments.experiment,
                "region": region,
                **{name: common.distribution(raw) for name, raw in measures.items()},
            }
        )
    region_comparison = {
        "samples_with_GT_and_FP": len(fp_gt_std_flags),
        "fraction_FP_STD_gt_GT_STD": float(np.mean(fp_gt_std_flags)) if fp_gt_std_flags else math.nan,
        "fraction_FP_CV_gt_GT_CV": float(np.mean(fp_gt_cv_flags)) if fp_gt_cv_flags else math.nan,
    }
    over_rows = []
    for region, measures in over_regions.items():
        over_rows.append(
            {
                "dataset": arguments.experiment,
                "region": region,
                **{name: common.distribution(raw) for name, raw in measures.items()},
            }
        )
    over_comparison = {
        "over_expansion_samples": len(over_std_flags),
        "fraction_OVER_STD_gt_GT_STD": float(np.mean(over_std_flags)) if over_std_flags else math.nan,
        "fraction_OVER_CV_gt_GT_CV": float(np.mean(over_cv_flags)) if over_cv_flags else math.nan,
    }

    delta_rows = []
    ogl_delta = np.asarray([row["delta_OGL"] for row in rows])
    for method in RAW_CONSENSUS:
        temporal_delta = np.asarray([row[f"delta_{method}"] for row in rows])
        delta_rows.append(
            {
                "dataset": arguments.experiment,
                "method": method,
                "pearson_delta_temporal_vs_delta_OGL": common.safe_pearson(temporal_delta, ogl_delta),
                "spearman_delta_temporal_vs_delta_OGL": common.spearman(temporal_delta, ogl_delta),
                "temporal_delta": common.distribution(temporal_delta.tolist()),
                "OGL_delta": common.distribution(ogl_delta.tolist()),
            }
        )

    reference_reproduction = {"skipped_for_partial_run": True}
    if arguments.max_batches is None:
        reference = common.reference_summary(registry)
        reference_reproduction = {
            "ORIGINAL_AUD": metric_error(
                metric_lookup["ORIGINAL_AUD"], common.reference_metric(reference, "AUD_FINE")
            ),
            "OGL_REFERENCE": metric_error(
                metric_lookup["OGL_REFERENCE"], common.reference_metric(reference, "OGL")
            ),
        }
        reference_reproduction["max_error"] = max(
            value
            for errors in reference_reproduction.values()
            if isinstance(errors, dict)
            for value in errors.values()
        )
        reference_reproduction["passed"] = reference_reproduction["max_error"] <= 1e-12
        if not reference_reproduction["passed"]:
            raise RuntimeError(f"Evaluator reproduction failed: {reference_reproduction}")

    selection, category_counts = select_qualitative(rows, arguments.qualitative_count)
    if arguments.max_batches is None and not arguments.skip_qualitative:
        save_qualitative(
            loader, model, object_model, selection, rows, output_dir / "qualitative", device
        )

    snapshots_after = common.verify_snapshots(snapshots_before)
    zero_training = audit_models({"original_G": model, "evaluation_only_object_prior": object_model})
    zero_training.update(
        {
            "checkpoint_snapshots": snapshots_after,
            "all_checkpoint_hashes_and_mtimes_unchanged": snapshots_after["all_unchanged"],
            "no_nan_or_inf": no_nan_or_inf,
            "full_tensor_reproduction_max_error": full_tensor_max_error,
        }
    )
    if (
        zero_training["new_trainable_params"] != 0
        or zero_training["trainable_parameter_names"]
        or zero_training["parameters_with_grad"]
        or not zero_training["all_models_eval"]
        or not zero_training["all_checkpoint_hashes_and_mtimes_unchanged"]
        or not zero_training["no_nan_or_inf"]
    ):
        raise RuntimeError(f"Zero-training audit failed: {zero_training}")

    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    common.write_csv(output_dir / "method_metrics.csv", metrics)
    common.write_csv(output_dir / "temporal_query_semantics.csv", semantic_rows)
    common.write_json(output_dir / "query_consistency.json", consistency_rows)
    common.write_json(output_dir / "temporal_map_similarity.json", map_rows)
    common.write_json(output_dir / "region_stability.json", {"rows": region_rows, "comparison": region_comparison})
    common.write_json(output_dir / "overexpansion_stability.json", {"rows": over_rows, "comparison": over_comparison})
    common.write_csv(output_dir / "rescue_hurt_oracle.csv", transition_rows)
    common.write_csv(output_dir / "ogl_rescue_capture.csv", capture_rows)
    common.write_json(output_dir / "delta_correlation.json", delta_rows)
    common.write_json(output_dir / "zero_training_audit.json", zero_training)
    if arguments.max_batches is None and not arguments.skip_qualitative:
        common.write_csv(output_dir / "qualitative" / "selection_manifest.csv", selection)

    summary = {
        "experiment": "3.2 A4 Temporal Grounding Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": arguments.max_batches is None,
        "tensor_audit": audit,
        "method_metrics": metrics,
        "temporal_query_semantics": semantic_rows,
        "query_consistency": consistency_rows,
        "temporal_map_similarity": map_rows,
        "region_stability": {"rows": region_rows, "comparison": region_comparison},
        "overexpansion_stability": {"rows": over_rows, "comparison": over_comparison},
        "rescue_hurt_oracle": transition_rows,
        "ogl_rescue_capture": capture_rows,
        "delta_correlation": delta_rows,
        "qualitative_category_counts": category_counts,
        "qualitative_ids": [row["sample_id"] for row in selection],
        "reference_reproduction": reference_reproduction,
        "zero_training_audit": zero_training,
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())

