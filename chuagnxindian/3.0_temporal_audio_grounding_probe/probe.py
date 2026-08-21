#!/usr/bin/env python3
"""Experiment 3.0: frozen temporal audio grounding decision probe."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import common

visualize = common.load_module(
    "experiment_30_visualize", common.HERE / "visualize.py"
)


EPS = 1e-8
ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
PRIMARY_FUSIONS = ("FULL_TEMP_MEAN_4", "FULL_TEMP_GEO_4")
TEMPORAL_METHODS = (
    "CHUNK_1",
    "CHUNK_2",
    "CHUNK_3",
    "CHUNK_4",
    "TEMP_MEAN_4",
    "TEMP_GEO_4",
    "FULL_TEMP_MEAN_4",
    "FULL_TEMP_GEO_4",
    "TEMP_MEAN_2",
    "TEMP_GEO_2",
)
LOCALIZATION_METHODS = (
    "FULL_AUD",
    "CHUNK_1",
    "CHUNK_2",
    "CHUNK_3",
    "CHUNK_4",
    "TEMP_MEAN_4",
    "TEMP_GEO_4",
    "FULL_TEMP_MEAN_4",
    "FULL_TEMP_GEO_4",
    "TEMP_MEAN_2",
    "TEMP_GEO_2",
    "OGL",
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


def spatial_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.flatten(start_dim=2).sum(dim=-1, keepdim=True).view(
        value.shape[0], value.shape[1], 1, 1
    ).clamp_min(EPS)


def query_map(teacher, tokens: torch.Tensor, initial_slots: torch.Tensor, keys: torch.Tensor):
    _slots, query, _audio_keys = teacher.slot_attn.audio_branch(tokens, initial_slots)
    attention = teacher.slot_attn._attention(
        query, keys, teacher.infer_sharpening
    )
    return query, attention[:, 0].reshape(-1, 1, 14, 14)


@torch.inference_mode()
def extract_temporal(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, Any]:
    teacher = model.teacher
    audio_feature = teacher.audnet(audio)
    audio_tokens = teacher._audio_tokens(audio_feature)
    batch, temporal_length, channels = audio_tokens.shape
    initial_slots = teacher.slot_attn.slots.expand(batch, -1, -1)

    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    f34, _f3_spatial, _f4_up, _delta = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))

    full_query, full_map = query_map(teacher, audio_tokens, initial_slots, k34)
    boundaries4 = contiguous_boundaries(temporal_length, 4)
    boundaries2 = contiguous_boundaries(temporal_length, 2)

    chunk_queries = []
    chunk_maps = []
    chunk_quality = []
    slot_cosines = []
    for start, stop in boundaries4:
        chunk_tokens = audio_tokens[:, start:stop]
        chunk_query, chunk_map = query_map(
            teacher, chunk_tokens, initial_slots, k34
        )
        chunk_queries.append(chunk_query)
        chunk_maps.append(chunk_map)
        chunk_quality.append(
            {
                "token_mean_norm": chunk_tokens.norm(dim=-1).mean(dim=1),
                "token_temporal_variance": chunk_tokens.var(
                    dim=1, unbiased=False
                ).mean(dim=-1),
                "query_q0_norm": chunk_query[:, 0].norm(dim=-1),
                "query_q1_norm": chunk_query[:, 1].norm(dim=-1),
            }
        )
        slot_cosines.append(
            {
                "chunk_q0_full_q0": F.cosine_similarity(
                    chunk_query[:, 0], full_query[:, 0], dim=-1
                ),
                "chunk_q0_full_q1": F.cosine_similarity(
                    chunk_query[:, 0], full_query[:, 1], dim=-1
                ),
                "chunk_q1_full_q0": F.cosine_similarity(
                    chunk_query[:, 1], full_query[:, 0], dim=-1
                ),
                "chunk_q1_full_q1": F.cosine_similarity(
                    chunk_query[:, 1], full_query[:, 1], dim=-1
                ),
            }
        )

    half_maps = []
    for start, stop in boundaries2:
        _query, half_map = query_map(
            teacher, audio_tokens[:, start:stop], initial_slots, k34
        )
        half_maps.append(half_map)

    stacked4 = torch.stack(chunk_maps, dim=0)
    temporal_mean4 = stacked4.mean(dim=0)
    temporal_geo4 = spatial_normalize(
        torch.exp(torch.log(stacked4.clamp_min(EPS)).mean(dim=0))
    )
    temporal_mean2 = torch.stack(half_maps, dim=0).mean(dim=0)
    temporal_geo2 = spatial_normalize(
        torch.exp(
            torch.log(torch.stack(half_maps, dim=0).clamp_min(EPS)).mean(dim=0)
        )
    )
    temporal_std = stacked4.std(dim=0, unbiased=False)
    temporal_cv = temporal_std / temporal_mean4.clamp_min(EPS)

    maps = {
        "FULL_AUD": full_map,
        **{f"CHUNK_{index + 1}": value for index, value in enumerate(chunk_maps)},
        "TEMP_MEAN_4": temporal_mean4,
        "TEMP_GEO_4": temporal_geo4,
        "FULL_TEMP_MEAN_4": 0.6 * full_map + 0.4 * temporal_mean4,
        "FULL_TEMP_GEO_4": 0.6 * full_map + 0.4 * temporal_geo4,
        "TEMP_MEAN_2": temporal_mean2,
        "TEMP_GEO_2": temporal_geo2,
    }
    for alpha in ALPHAS:
        maps[f"FULL_TEMP_MEAN_4_A{alpha:.1f}"] = (
            alpha * full_map + (1.0 - alpha) * temporal_mean4
        )
        maps[f"FULL_TEMP_GEO_4_A{alpha:.1f}"] = (
            alpha * full_map + (1.0 - alpha) * temporal_geo4
        )
    return {
        "audio_feature": audio_feature,
        "audio_tokens": audio_tokens,
        "full_query": full_query,
        "chunk_queries": chunk_queries,
        "chunk_maps": chunk_maps,
        "chunk_quality": chunk_quality,
        "slot_cosines": slot_cosines,
        "K34": k34,
        "F34": f34,
        "boundaries4": boundaries4,
        "boundaries2": boundaries2,
        "maps": maps,
        "TEMP_STD": temporal_std,
        "TEMP_CV": temporal_cv,
        "f4_token_error": (
            f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]
        ).abs().max(),
    }


def all_finite(output: dict[str, Any]) -> bool:
    tensors = [
        output["audio_feature"],
        output["audio_tokens"],
        output["full_query"],
        output["K34"],
        output["F34"],
        output["TEMP_STD"],
        output["TEMP_CV"],
        *output["chunk_queries"],
        *output["chunk_maps"],
        *output["maps"].values(),
    ]
    return all(torch.isfinite(value).all().item() for value in tensors)


@torch.inference_mode()
def tensor_audit(loader, model, device: torch.device) -> dict[str, Any]:
    image, spec, bboxes, names, _labels = next(iter(loader))
    image, spec, _bboxes, names = common.flatten_batch(image, spec, bboxes, names)
    image = image.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    official = model(image, spec)["AUD_FINE"]
    output = extract_temporal(model, image, spec)
    maps = output["maps"]
    spatial_sum_errors = {
        name: float((value.flatten(start_dim=2).sum(dim=-1) - 1.0).abs().max())
        for name, value in maps.items()
        if name == "FULL_AUD"
        or name.startswith("CHUNK_")
        or name in {"TEMP_MEAN_4", "TEMP_GEO_4", "TEMP_MEAN_2", "TEMP_GEO_2"}
    }
    audit = {
        "sample_ids": names[:4],
        "original_audio_feature_shape": list(output["audio_feature"].shape),
        "audio_tokens_shape": list(output["audio_tokens"].shape),
        "T": int(output["audio_tokens"].shape[1]),
        "full_audio_query_shape": list(output["full_query"].shape),
        "K34_shape": list(output["K34"].shape),
        "F34_shape": list(output["F34"].shape),
        "four_chunk_boundaries": [list(value) for value in output["boundaries4"]],
        "two_chunk_boundaries": [list(value) for value in output["boundaries2"]],
        "chunk_map_shapes": [list(value.shape) for value in output["chunk_maps"]],
        "FULL_AUD_tensor_reproduction_max_error": float(
            (maps["FULL_AUD"] - official).abs().max()
        ),
        "spatial_sum_max_errors": spatial_sum_errors,
        "f4_token_error": float(output["f4_token_error"]),
        "no_nan_or_inf": all_finite(output),
    }
    expected = {
        "original_audio_feature_shape": [512, 16],
        "audio_tokens_shape": [16, 512],
        "full_audio_query_shape": [2, 512],
        "K34_shape": [196, 512],
        "F34_shape": [512, 14, 14],
    }
    for key, nonbatch in expected.items():
        if audit[key][1:] != nonbatch:
            raise RuntimeError(f"Unexpected {key}: {audit[key]}")
    if audit["FULL_AUD_tensor_reproduction_max_error"] > 1e-6:
        raise RuntimeError(audit)
    if max(spatial_sum_errors.values()) > 1e-6:
        raise RuntimeError(audit)
    if audit["f4_token_error"] > 1e-6 or not audit["no_nan_or_inf"]:
        raise RuntimeError(audit)
    audit["passed"] = True
    return audit


def append_iou(values, row, method, prediction, ground_truth) -> None:
    iou = common.sample_iou(prediction, ground_truth)
    values.setdefault(method, []).append(iou)
    row[f"IoU_{method}"] = iou


def sample_pairwise_agreement(chunk_maps: list[np.ndarray]) -> dict[str, float]:
    pearson = []
    spearman = []
    js = []
    for first, second in itertools.combinations(chunk_maps, 2):
        pearson.append(common.safe_pearson(first, second))
        spearman.append(common.spearman(first, second))
        js.append(common.js_divergence(first, second))
    return {
        "pearson": float(np.nanmean(pearson)),
        "spearman": float(np.nanmean(spearman)),
        "js_divergence": float(np.nanmean(js)),
    }


def aggregate_named(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        f"{name}_{stat}": value
        for name, raw in values.items()
        for stat, value in common.aggregate_distribution(raw).items()
    }


def region_row(dataset: str, region: str, values: dict[str, list[float]]) -> dict[str, Any]:
    return {"dataset": dataset, "region": region, **aggregate_named(values)}


def reference_metric(summary: dict[str, Any], method: str) -> dict[str, Any]:
    for row in summary["method_metrics"]:
        if row["method"] == method:
            return row
    raise KeyError(method)


def metric_error(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    return {key: abs(float(actual[key]) - float(expected[key])) for key in ("cIoU", "AUC")}


def select_qualitative(rows: list[dict[str, Any]], count: int) -> list[dict[str, str]]:
    categories = {
        "AUD_SUCCESS": [],
        "OGL_RESCUE": [],
        "TEMPORAL_RESCUE": [],
        "TEMPORAL_HURT": [],
        "ALL_FAIL": [],
    }
    membership = {}
    for row in rows:
        full = row["IoU_FULL_AUD"] >= 0.5
        ogl = row["IoU_OGL"] >= 0.5
        mean = row["IoU_FULL_TEMP_MEAN_4"] >= 0.5
        geo = row["IoU_FULL_TEMP_GEO_4"] >= 0.5
        labels = []
        if full:
            labels.append("AUD_SUCCESS")
        if not full and ogl:
            labels.append("OGL_RESCUE")
        if not full and (mean or geo):
            labels.append("TEMPORAL_RESCUE")
        if full and (not mean or not geo):
            labels.append("TEMPORAL_HURT")
        if not full and not ogl and not mean and not geo:
            labels.append("ALL_FAIL")
        membership[row["sample_id"]] = labels
        for label in labels:
            categories[label].append(row["sample_id"])

    selected = []
    for rank in range(count):
        for label in categories:
            candidates = categories[label]
            if rank < len(candidates) and candidates[rank] not in selected:
                selected.append(candidates[rank])
                if len(selected) == count:
                    break
        if len(selected) == count:
            break
    if len(selected) < count:
        for row in rows:
            if row["sample_id"] not in selected:
                selected.append(row["sample_id"])
            if len(selected) == count:
                break
    rule = (
        "first-in-test-order round-robin over predefined AUD_SUCCESS, OGL_RESCUE, "
        "TEMPORAL_RESCUE, TEMPORAL_HURT, ALL_FAIL categories; test-order fill"
    )
    return [
        {
            "sample_id": sample_id,
            "categories": "|".join(membership[sample_id]) or "FILL",
            "selection_rule": rule,
        }
        for sample_id in selected
    ]


@torch.inference_mode()
def save_qualitative(
    loader,
    model,
    object_model,
    selection,
    rows,
    output_dir,
    device,
) -> None:
    wanted = {row["sample_id"]: row for row in selection}
    row_lookup = {row["sample_id"]: row for row in rows}
    found = {}
    for image, spec, bboxes, names, _labels in tqdm(
        loader, desc="qualitative", dynamic_ncols=True
    ):
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        needed = [index for index, name in enumerate(names) if name in wanted]
        if not needed:
            continue
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_temporal(model, image, spec)
        object_prior = object_model(image)
        tensors = {
            name: output["maps"][name]
            for name in (
                "FULL_AUD",
                "CHUNK_1",
                "CHUNK_2",
                "CHUNK_3",
                "CHUNK_4",
                "TEMP_MEAN_4",
                "TEMP_GEO_4",
                "FULL_TEMP_GEO_4",
            )
        }
        resized = common.resize_maps(tensors)
        resized["OBJ_PRIOR"] = common.resize_maps({"OBJ": object_prior})["OBJ"]
        resized["TEMP_STD"] = common.resize_maps(
            {"STD": output["TEMP_STD"]}, mode="bilinear"
        )["STD"]
        ground_truth = bboxes.numpy()
        for index in needed:
            maps = {
                name: common.normalize_map(value[index]) for name, value in resized.items()
            }
            maps["OGL"] = common.fuse_normalized(
                maps["FULL_AUD"], maps["OBJ_PRIOR"], 0.6
            )
            sample_id = names[index]
            rgb = common.inverse_normalize(image[index].cpu()).permute(1, 2, 0).numpy()
            found[sample_id] = {
                "sample_id": sample_id,
                "categories": wanted[sample_id]["categories"],
                "image": np.clip(rgb, 0.0, 1.0),
                "GT": ground_truth[index],
                "row": row_lookup[sample_id],
                **maps,
            }
        if len(found) == len(wanted):
            break
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"Missing qualitative samples: {sorted(missing)}")
    for index, item in enumerate(selection, start=1):
        visualize.save_panel(
            found[item["sample_id"]],
            output_dir / f"{index:02d}_{item['sample_id']}.png",
        )
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
        "evaluation_only_object_prior_not_used_in_temporal_maps": True,
    }


@torch.inference_mode()
def run(arguments: argparse.Namespace) -> None:
    started = time.time()
    registry = common.EXPERIMENTS[arguments.experiment]
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)

    snapshots_before = common.snapshot_files(
        {
            **common.checkpoint_paths(registry),
            **common.source_artifact_paths(registry),
        }
    )
    config = common.load_config(registry["stage1"])
    loader = common.build_loader(config, registry)
    model = common.load_original_g(registry, device)
    object_model = common.object_prior_model().to(device).eval()

    audit = tensor_audit(loader, model, device)
    common.write_json(output_dir / "tensor_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)

    values: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    slot_values = [
        {
            name: []
            for name in (
                "chunk_q0_full_q0",
                "chunk_q0_full_q1",
                "chunk_q1_full_q0",
                "chunk_q1_full_q1",
            )
        }
        for _ in range(4)
    ]
    slot_closer = [
        {"full_slot0": 0, "full_slot1": 0, "ties": 0, "total": 0}
        for _ in range(4)
    ]
    chunk_quality = [
        {
            name: []
            for name in (
                "token_mean_norm",
                "token_temporal_variance",
                "query_q0_norm",
                "query_q1_norm",
                "query_q0_full_q0_cosine",
            )
        }
        for _ in range(4)
    ]
    region_values = {
        region: {name: [] for name in ("temporal_mean", "temporal_std", "temporal_cv")}
        for region in ("GT_FOREGROUND", "CORRECT_AUD_FOREGROUND", "AUD_FALSE_POSITIVE_CONTEXT")
    }
    agreement_values = {
        group: {name: [] for name in ("pearson", "spearman", "js_divergence")}
        for group in ("ALL", "FULL_AUD_SUCCESS", "FULL_AUD_FAILURE", "OGL_RESCUE")
    }
    max_spatial_sum_error = 0.0
    full_no_nan_or_inf = True

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
        batch_finite = all_finite(output) and torch.isfinite(object_prior).all().item()
        full_no_nan_or_inf = full_no_nan_or_inf and batch_finite
        if not batch_finite:
            raise RuntimeError(f"NaN/Inf detected in batch {batch_index}")

        for name, value in output["maps"].items():
            if name == "FULL_AUD" or name.startswith("CHUNK_") or name.startswith("TEMP_"):
                max_spatial_sum_error = max(
                    max_spatial_sum_error,
                    float((value.flatten(start_dim=2).sum(dim=-1) - 1.0).abs().max()),
                )

        resized = common.resize_maps(output["maps"])
        resized["OBJ_PRIOR"] = common.resize_maps({"OBJ": object_prior})["OBJ"]
        stability = common.resize_maps(
            {
                "TEMP_MEAN_RAW": output["maps"]["TEMP_MEAN_4"],
                "TEMP_STD_RAW": output["TEMP_STD"],
                "TEMP_CV_RAW": output["TEMP_CV"],
            },
            mode="bilinear",
        )
        raw_chunks = [value.detach().cpu().numpy()[:, 0] for value in output["chunk_maps"]]
        ground_truth = bboxes.numpy()

        for chunk_index in range(4):
            cosines = output["slot_cosines"][chunk_index]
            quality = output["chunk_quality"][chunk_index]
            for name, tensor in cosines.items():
                slot_values[chunk_index][name].extend(tensor.cpu().tolist())
            closer0 = cosines["chunk_q0_full_q0"] > cosines["chunk_q0_full_q1"]
            closer1 = cosines["chunk_q0_full_q1"] > cosines["chunk_q0_full_q0"]
            slot_closer[chunk_index]["full_slot0"] += int(closer0.sum())
            slot_closer[chunk_index]["full_slot1"] += int(closer1.sum())
            slot_closer[chunk_index]["ties"] += int((~closer0 & ~closer1).sum())
            slot_closer[chunk_index]["total"] += int(closer0.numel())
            for name, tensor in quality.items():
                chunk_quality[chunk_index][name].extend(tensor.cpu().tolist())
            chunk_quality[chunk_index]["query_q0_full_q0_cosine"].extend(
                cosines["chunk_q0_full_q0"].cpu().tolist()
            )

        for sample_index, sample_id in enumerate(names):
            maps = {
                name: common.normalize_map(value[sample_index])
                for name, value in resized.items()
            }
            maps["OGL"] = common.fuse_normalized(
                maps["FULL_AUD"], maps["OBJ_PRIOR"], 0.6
            )
            row: dict[str, Any] = {"sample_id": sample_id}
            for method in LOCALIZATION_METHODS:
                append_iou(values, row, method, maps[method], ground_truth[sample_index])
            for family in ("MEAN", "GEO"):
                for alpha in ALPHAS:
                    method = f"FULL_TEMP_{family}_4_A{alpha:.1f}"
                    append_iou(values, row, method, maps[method], ground_truth[sample_index])

            gt_mask = ground_truth[sample_index] > 0
            full_pred = maps["FULL_AUD"] >= 0.6
            masks = {
                "GT_FOREGROUND": gt_mask,
                "CORRECT_AUD_FOREGROUND": full_pred & gt_mask,
                "AUD_FALSE_POSITIVE_CONTEXT": full_pred & (~gt_mask),
            }
            raw_stability = {
                "temporal_mean": np.clip(stability["TEMP_MEAN_RAW"][sample_index], 0.0, None),
                "temporal_std": np.clip(stability["TEMP_STD_RAW"][sample_index], 0.0, None),
                "temporal_cv": np.clip(stability["TEMP_CV_RAW"][sample_index], 0.0, None),
            }
            for region, mask in masks.items():
                if mask.any():
                    for name, value in raw_stability.items():
                        region_values[region][name].append(float(value[mask].mean()))

            agreement = sample_pairwise_agreement(
                [chunk[sample_index] for chunk in raw_chunks]
            )
            for name, value in agreement.items():
                row[f"temporal_agreement_{name}"] = value
                agreement_values["ALL"][name].append(value)
            full_success = row["IoU_FULL_AUD"] >= 0.5
            ogl_rescue = not full_success and row["IoU_OGL"] >= 0.5
            group = "FULL_AUD_SUCCESS" if full_success else "FULL_AUD_FAILURE"
            for name, value in agreement.items():
                agreement_values[group][name].append(value)
                if ogl_rescue:
                    agreement_values["OGL_RESCUE"][name].append(value)
            rows.append(row)

    if max_spatial_sum_error > 1e-6:
        raise RuntimeError(f"Spatial probability sum error: {max_spatial_sum_error}")

    metrics = [
        {"dataset": arguments.experiment, "method": method, **common.summarize(values[method])}
        for method in LOCALIZATION_METHODS
    ]
    metric_lookup = {row["method"]: row for row in metrics}
    transitions = []
    for method in PRIMARY_FUSIONS:
        shift = common.transition(values["FULL_AUD"], values[method])
        transitions.append(
            {
                "dataset": arguments.experiment,
                "method": method,
                "rescue": shift["rescue"],
                "hurt": shift["hurt"],
                "net": shift["net"],
                "oracle_cIoU": shift["oracle"]["cIoU"],
                "oracle_AUC": shift["oracle"]["AUC"],
            }
        )

    full_success = np.asarray(values["FULL_AUD"]) >= 0.5
    ogl_success = np.asarray(values["OGL"]) >= 0.5
    ogl_pool = (~full_success) & ogl_success
    other_failures = (~full_success) & (~ogl_success)
    capture_rows = []
    for method in TEMPORAL_METHODS:
        candidate_success = np.asarray(values[method]) >= 0.5
        captured = int((ogl_pool & candidate_success).sum())
        hurt = int((full_success & (~candidate_success)).sum())
        other_captured = int((other_failures & candidate_success).sum())
        capture_rows.append(
            {
                "dataset": arguments.experiment,
                "method": method,
                "OGL_RESCUE_TOTAL": int(ogl_pool.sum()),
                "OGL_RESCUE_CAPTURED": captured,
                "OGL_RESCUE_CAPTURE_RATE": captured / max(int(ogl_pool.sum()), 1),
                "NEW_HURT": hurt,
                "CAPTURE_MINUS_HURT": captured - hurt,
                "OTHER_FULL_FAILURE_TOTAL": int(other_failures.sum()),
                "OTHER_FULL_FAILURE_CAPTURED": other_captured,
                "OTHER_FULL_FAILURE_CAPTURE_RATE": other_captured
                / max(int(other_failures.sum()), 1),
            }
        )

    baseline_gap = metric_lookup["OGL"]["cIoU"] - metric_lookup["FULL_AUD"]["cIoU"]
    gap_rows = []
    for method in (
        "TEMP_MEAN_4",
        "TEMP_GEO_4",
        "FULL_TEMP_MEAN_4",
        "FULL_TEMP_GEO_4",
        "TEMP_MEAN_2",
        "TEMP_GEO_2",
    ):
        new_gap = metric_lookup["OGL"]["cIoU"] - metric_lookup[method]["cIoU"]
        reduction = baseline_gap - new_gap
        gap_rows.append(
            {
                "dataset": arguments.experiment,
                "method": method,
                "original_OGL_FULL_gap": baseline_gap,
                "new_OGL_method_gap": new_gap,
                "gap_reduction": reduction,
                "gap_reduction_percent": reduction / baseline_gap if baseline_gap else 0.0,
            }
        )

    region_rows = [
        region_row(arguments.experiment, region, value)
        for region, value in region_values.items()
    ]
    agreement_rows = [
        {"dataset": arguments.experiment, "group": group, **aggregate_named(value)}
        for group, value in agreement_values.items()
    ]
    slot_rows = []
    for chunk_index in range(4):
        counts = slot_closer[chunk_index]
        slot_rows.append(
            {
                "dataset": arguments.experiment,
                "chunk": chunk_index + 1,
                **aggregate_named(slot_values[chunk_index]),
                "slot0_closer_to_full_slot0": counts["full_slot0"],
                "slot0_closer_to_full_slot1": counts["full_slot1"],
                "ties": counts["ties"],
                "total": counts["total"],
                "slot0_to_slot0_rate": counts["full_slot0"] / counts["total"],
                "slot0_to_slot1_rate": counts["full_slot1"] / counts["total"],
            }
        )
    overall_total = sum(row["total"] for row in slot_rows)
    slot_overall = {
        "dataset": arguments.experiment,
        "chunk": "OVERALL",
        "slot0_closer_to_full_slot0": sum(row["slot0_closer_to_full_slot0"] for row in slot_rows),
        "slot0_closer_to_full_slot1": sum(row["slot0_closer_to_full_slot1"] for row in slot_rows),
        "ties": sum(row["ties"] for row in slot_rows),
        "total": overall_total,
    }
    slot_overall["slot0_to_slot0_rate"] = slot_overall["slot0_closer_to_full_slot0"] / overall_total
    slot_overall["slot0_to_slot1_rate"] = slot_overall["slot0_closer_to_full_slot1"] / overall_total
    slot_rows.append(slot_overall)

    quality_rows = [
        {
            "dataset": arguments.experiment,
            "chunk": chunk_index + 1,
            **aggregate_named(chunk_quality[chunk_index]),
        }
        for chunk_index in range(4)
    ]
    alpha_rows = []
    for family in ("MEAN", "GEO"):
        for alpha in ALPHAS:
            method = f"FULL_TEMP_{family}_4_A{alpha:.1f}"
            metric = common.summarize(values[method])
            shift = common.transition(values["FULL_AUD"], values[method])
            alpha_rows.append(
                {
                    "dataset": arguments.experiment,
                    "family": family,
                    "alpha_full": alpha,
                    **metric,
                    "rescue": shift["rescue"],
                    "hurt": shift["hurt"],
                    "net": shift["net"],
                    "formal": alpha == 0.6,
                }
            )

    reproduction = {}
    if arguments.max_batches is None:
        reference = json.loads(
            (
                common.R22_ROOT / registry["reference_key"] / "summary.json"
            ).read_text(encoding="utf-8")
        )
        reproduction = {
            "FULL_AUD": metric_error(
                metric_lookup["FULL_AUD"], reference_metric(reference, "AUD_FINE")
            ),
            "OGL": metric_error(metric_lookup["OGL"], reference_metric(reference, "OGL")),
        }
        reproduction["max_error"] = max(
            value for errors in reproduction.values() for value in errors.values()
        )
        reproduction["passed"] = reproduction["max_error"] <= 1e-12
        if not reproduction["passed"]:
            raise RuntimeError(f"Evaluator reproduction failed: {reproduction}")
    else:
        reproduction = {"skipped_for_partial_run": True}

    selection = []
    if arguments.max_batches is None and not arguments.skip_qualitative:
        selection = select_qualitative(rows, arguments.qualitative_count)
        save_qualitative(
            loader,
            model,
            object_model,
            selection,
            rows,
            output_dir / "qualitative",
            device,
        )

    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    common.write_csv(output_dir / "method_metrics.csv", metrics)
    common.write_csv(output_dir / "primary_rescue_hurt_oracle.csv", transitions)
    common.write_csv(output_dir / "ogl_rescue_capture.csv", capture_rows)
    common.write_csv(output_dir / "ogl_gap.csv", gap_rows)
    common.write_csv(output_dir / "temporal_region_stability.csv", region_rows)
    common.write_csv(output_dir / "sample_temporal_agreement.csv", agreement_rows)
    common.write_csv(output_dir / "chunk_slot_identity.csv", slot_rows)
    common.write_csv(output_dir / "chunk_quality.csv", quality_rows)
    common.write_csv(output_dir / "alpha_diagnostic.csv", alpha_rows)

    zero_training = audit_models({"original_G": model, "evaluation_only_object_prior": object_model})
    snapshot_verification = common.verify_snapshots(snapshots_before)
    zero_training.update(
        {
            "checkpoint_and_source_snapshots": snapshot_verification,
            "all_checkpoint_hashes_and_mtimes_unchanged": snapshot_verification["all_unchanged"],
            "max_spatial_probability_sum_error": max_spatial_sum_error,
            "no_nan_or_inf": full_no_nan_or_inf,
        }
    )
    if (
        zero_training["new_trainable_params"] != 0
        or zero_training["trainable_parameter_names"]
        or zero_training["parameters_with_grad"]
        or not zero_training["all_models_eval"]
        or not zero_training["all_checkpoint_hashes_and_mtimes_unchanged"]
    ):
        raise RuntimeError(f"Zero-training audit failed: {zero_training}")
    common.write_json(output_dir / "zero_training_audit.json", zero_training)

    summary = {
        "experiment": "3.0 Temporal Audio Grounding Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": arguments.max_batches is None,
        "tensor_audit": audit,
        "method_metrics": metrics,
        "primary_rescue_hurt_oracle": transitions,
        "ogl_rescue_capture": capture_rows,
        "ogl_gap": gap_rows,
        "temporal_region_stability": region_rows,
        "sample_temporal_agreement": agreement_rows,
        "chunk_slot_identity": slot_rows,
        "chunk_quality": quality_rows,
        "alpha_diagnostic": alpha_rows,
        "reference_reproduction": reproduction,
        "zero_training_audit": zero_training,
        "qualitative_ids": [row["sample_id"] for row in selection],
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
