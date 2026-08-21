#!/usr/bin/env python3
"""Frozen extraction for Experiment 5.3 AUD-only leakage cue probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

import common


BENEFICIAL_GAIN = 0.01
DOMINANCE_MARGIN = 0.01
PROBE_PER_CLASS_PER_SAMPLE = 6
ROUTING_PER_SAMPLE = 12
EPS = 1e-6

PREDICTION_FEATURES = (
    "AUD_raw_score",
    "IMG_raw_score",
    "AUD_norm_score",
    "IMG_norm_score",
    "AUD_minus_IMG",
    "AUD_IMG_log_ratio",
    "AUD_threshold_distance",
    "IMG_threshold_distance",
)
MULTILEVEL_FEATURES = (
    "AUD_L3_response",
    "AUD_L4_response",
    "IMG_L3_response",
    "IMG_L4_response",
    "AUD_L3_L4_consistency",
    "AUD_L3_L4_disagreement",
    "IMG_L3_L4_consistency",
    "IMG_L3_L4_disagreement",
)
SLOT_FEATURES = (
    "L4_target_ownership",
    "L4_other_max_ownership",
    "L4_target_other_margin",
    "L4_ownership_entropy",
    "HR14_target_ownership",
    "HR14_other_max_ownership",
    "HR14_target_other_margin",
    "HR14_ownership_entropy",
)
LOCAL_FEATURES = (
    "AUD_neighbor_consistency",
    "IMG_neighbor_consistency",
    "F34_local_similarity",
    "K34_local_similarity",
    "distance_to_AUD_boundary",
    "distance_to_IMG_boundary",
)
PROTOTYPE_FEATURES = (
    "F34_agreement_core_similarity",
    "K34_agreement_core_similarity",
    "F34_AUD_core_similarity",
    "K34_AUD_core_similarity",
    "F34_IMG_core_similarity",
    "K34_IMG_core_similarity",
)
WITHOUT_PROTOTYPE_FEATURES = PREDICTION_FEATURES + MULTILEVEL_FEATURES + SLOT_FEATURES + LOCAL_FEATURES
WITH_PROTOTYPE_FEATURES = WITHOUT_PROTOTYPE_FEATURES + PROTOTYPE_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def references(setting: str) -> dict[str, Any]:
    return {
        "rows52": load_csv(common.reference_52_dir(setting) / "per_sample_diagnosis.csv"),
        "raw41": np.load(common.reference_41_dir(setting) / "raw_maps.npz"),
        "raw51": np.load(common.reference_51_dir(setting) / "raw_propagation_maps.npz"),
    }


def normalize_batch(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(start_dim=1)
    minimum = flat.min(dim=1).values[:, None, None, None]
    maximum = flat.max(dim=1).values[:, None, None, None]
    span = maximum - minimum
    return torch.where(span != 0, (value - minimum) / span, value)


def stable_top_mask(value: np.ndarray, count: int = 10) -> np.ndarray:
    flat = np.asarray(value).reshape(-1)
    order = np.argsort(-flat, kind="stable")
    output = np.zeros(flat.size, dtype=bool)
    output[order[:count]] = True
    return output.reshape(value.shape)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    value = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(value, size, mode="nearest")[0, 0].numpy() >= 0.5


def deterministic_choice(coords: np.ndarray, count: int, sample_id: str, role: str) -> np.ndarray:
    if count <= 0 or coords.shape[0] == 0:
        return coords[:0]
    digest = hashlib.sha256(f"JSA-5.3::{sample_id}::{role}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    selected = rng.choice(coords.shape[0], size=min(count, coords.shape[0]), replace=False)
    return coords[selected]


def sample_fold(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"JSA-5.3-FOLD::{seed}::{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % 5


def propagate(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(tokens, dim=-1)
    weights = masks.flatten(start_dim=1).float()
    counts = weights.sum(dim=1, keepdim=True)
    prototype = torch.einsum("bn,bnd->bd", weights, normalized) / counts.clamp_min(1.0)
    prototype = F.normalize(prototype, dim=-1)
    raw = torch.einsum("bd,bnd->bn", prototype, normalized)
    raw = torch.where(counts > 0, raw, torch.zeros_like(raw))
    return raw.reshape(tokens.shape[0], 1, 14, 14)


def local_similarity(tokens: torch.Tensor) -> torch.Tensor:
    batch, count, channels = tokens.shape
    side = int(round(math.sqrt(count)))
    if side * side != count:
        raise ValueError(count)
    value = F.normalize(tokens, dim=-1).transpose(1, 2).reshape(batch, channels, side, side)
    total = torch.zeros((batch, side, side), device=value.device, dtype=value.dtype)
    weight = torch.zeros_like(total)
    vertical = (value[:, :, 1:] * value[:, :, :-1]).sum(dim=1)
    horizontal = (value[:, :, :, 1:] * value[:, :, :, :-1]).sum(dim=1)
    total[:, 1:] += vertical
    total[:, :-1] += vertical
    weight[:, 1:] += 1
    weight[:, :-1] += 1
    total[:, :, 1:] += horizontal
    total[:, :, :-1] += horizontal
    weight[:, :, 1:] += 1
    weight[:, :, :-1] += 1
    return (total / weight.clamp_min(1)).unsqueeze(1)


@torch.inference_mode()
def extract_all(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = model.teacher
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    scale = teacher.infer_sharpening
    l4 = teacher.slot_attn._l4_attentions(encoded, scale_multiplier=scale)
    aud_l3 = teacher.slot_attn._attention(encoded["audio_query"], encoded["visual_keys"][0], scale)
    img_l3 = teacher.slot_attn._attention(encoded["visual_queries"][0], encoded["visual_keys"][0], scale)

    ownership = {}
    for name, query, key in (
        ("L4", encoded["visual_queries"][1], encoded["visual_keys"][1]),
    ):
        logits = torch.einsum("bsd,bnd->bsn", query, key) * teacher.slot_attn.scale
        ownership[name] = logits.softmax(dim=1)

    f34, f3_spatial, f4_up, delta_f3 = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    f34_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(f34_tokens))
    aud_fine_all = teacher.slot_attn._attention(encoded["audio_query"], k34, scale)
    hr_logits = torch.einsum("bsd,bnd->bsn", encoded["visual_queries"][-1], k34) * teacher.slot_attn.scale
    ownership["HR14"] = hr_logits.softmax(dim=1)
    batch = image.shape[0]
    return {
        "AUD_L3": aud_l3[:, 0].reshape(batch, 1, 7, 7),
        "IMG_L3": img_l3[:, 0].reshape(batch, 1, 7, 7),
        "AUD_L4": l4["audq_imgk_attn"][:, 0].reshape(batch, 1, 7, 7),
        "IMG_L4": l4["imgq_imgk_attn"][:, 0].reshape(batch, 1, 7, 7),
        "AUD_FINE": aud_fine_all[:, 0].reshape(batch, 1, 14, 14),
        "OWN_L4": ownership["L4"].reshape(batch, 2, 7, 7),
        "OWN_HR14": ownership["HR14"].reshape(batch, 2, 14, 14),
        "F34": f34,
        "K34": k34,
        "F3_SPATIAL": f3_spatial,
        "F4_UP": f4_up,
        "DELTA_F3": delta_f3,
        "f4_token_error": (f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]).abs().max(),
    }


def bilinear_sample(value: np.ndarray, coords: np.ndarray, output_size: int = 224) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    height, width = value.shape
    source_y = (coords[:, 0].astype(np.float64) + 0.5) * height / output_size - 0.5
    source_x = (coords[:, 1].astype(np.float64) + 0.5) * width / output_size - 0.5
    y0 = np.floor(source_y).astype(np.int64)
    x0 = np.floor(source_x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = source_y - y0
    wx = source_x - x0
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    return (
        (1 - wy) * (1 - wx) * value[y0, x0]
        + (1 - wy) * wx * value[y0, x1]
        + wy * (1 - wx) * value[y1, x0]
        + wy * wx * value[y1, x1]
    )


def normalize_native(value: np.ndarray) -> np.ndarray:
    return common.normalize_map(np.asarray(value))


def ownership_maps(value: np.ndarray) -> dict[str, np.ndarray]:
    probability = np.clip(np.asarray(value, dtype=np.float64), EPS, 1.0)
    target = probability[0]
    other = np.max(probability[1:], axis=0)
    entropy = -np.sum(probability * np.log(probability), axis=0) / math.log(probability.shape[0])
    return {"target": target, "other": other, "margin": target - other, "entropy": entropy}


def intrinsic_label(expand_gain: float, shrink_gain: float) -> str:
    if expand_gain < BENEFICIAL_GAIN and shrink_gain < BENEFICIAL_GAIN:
        return "KEEP"
    if expand_gain >= BENEFICIAL_GAIN and expand_gain - shrink_gain >= DOMINANCE_MARGIN:
        return "INTRINSIC_EXPAND"
    if shrink_gain >= BENEFICIAL_GAIN and shrink_gain - expand_gain >= DOMINANCE_MARGIN:
        return "INTRINSIC_SHRINK"
    return "MIXED_AMBIGUOUS"


def summarize(values: list[float]) -> dict[str, Any]:
    return common.distribution(values)


@torch.inference_mode()
def run(arguments: argparse.Namespace) -> None:
    started = time.time()
    registry = common.EXPERIMENTS[arguments.experiment]
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = common.load_config(registry["stage1"])
    config.gpu = arguments.gpu
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    loader = common.build_loader(config, registry)
    ref = references(arguments.experiment)
    checkpoints_before = common.snapshot_files(
        {"formal_stage1": common.stage1_checkpoint_path(registry), "formal_original_1_3G": common.g_checkpoint_path(registry)}
    )
    model = common.load_original_g(registry, device)
    parameters_with_grad = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    sample_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    raw_errors = {"AUD_L4": 0.0, "IMG_L4": 0.0, "AUD_FINE": 0.0, "PROP_F34": 0.0, "PROP_K34": 0.0}
    metric_errors = {"AUD": 0.0, "IMG": 0.0, "PROP_F34": 0.0, "PROP_K34": 0.0}
    sample_mismatches = 0
    no_nan_or_inf = True
    global_index = 0
    tensor_audit: dict[str, Any] | None = None

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader)):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_all(model, image, spec)
        batch = len(names)
        if tensor_audit is None:
            tensor_audit = {key: list(value.shape) for key, value in output.items() if isinstance(value, torch.Tensor) and value.ndim > 0}
            tensor_audit["f4_token_error"] = float(output["f4_token_error"])
        no_nan_or_inf = no_nan_or_inf and all(
            torch.isfinite(value).all().item() for value in output.values() if isinstance(value, torch.Tensor)
        )

        raw_aud_l4 = output["AUD_L4"].cpu().numpy()[:, 0]
        raw_img_l4 = output["IMG_L4"].cpu().numpy()[:, 0]
        raw_aud_fine = output["AUD_FINE"].cpu().numpy()[:, 0]
        seeds_aud = []
        seeds_img = []
        seeds_agreement = []
        for local in range(batch):
            aud_seed = stable_top_mask(normalize_native(raw_aud_l4[local]))
            img_seed = stable_top_mask(normalize_native(raw_img_l4[local]))
            seeds_aud.append(resize_mask(aud_seed, (14, 14)))
            seeds_img.append(resize_mask(img_seed, (14, 14)))
            seeds_agreement.append(resize_mask(aud_seed & img_seed, (14, 14)))
        seed_tensors = {
            "AUD": torch.from_numpy(np.stack(seeds_aud).astype(np.float32)).to(device),
            "IMG": torch.from_numpy(np.stack(seeds_img).astype(np.float32)).to(device),
            "AGREEMENT": torch.from_numpy(np.stack(seeds_agreement).astype(np.float32)).to(device),
        }
        f34_tokens = output["F34"].flatten(start_dim=2).transpose(1, 2)
        k34_tokens = output["K34"]
        prototype_maps = {}
        for seed_name, mask in seed_tensors.items():
            prototype_maps[f"F34_{seed_name}"] = propagate(f34_tokens, mask).cpu().numpy()[:, 0]
            prototype_maps[f"K34_{seed_name}"] = propagate(k34_tokens, mask).cpu().numpy()[:, 0]
        local_f34 = local_similarity(f34_tokens).cpu().numpy()[:, 0]
        local_k34 = local_similarity(k34_tokens).cpu().numpy()[:, 0]

        raw_aud_eval = F.interpolate(output["AUD_FINE"], (224, 224), mode="bicubic", align_corners=False)
        raw_img_eval = F.interpolate(output["IMG_L4"], (224, 224), mode="bicubic", align_corners=False)
        aud_eval = normalize_batch(raw_aud_eval).cpu().numpy()[:, 0]
        img_eval = normalize_batch(raw_img_eval).cpu().numpy()[:, 0]
        raw_aud_eval_np = raw_aud_eval.cpu().numpy()[:, 0]
        raw_img_eval_np = raw_img_eval.cpu().numpy()[:, 0]
        prop_f_eval = normalize_batch(
            F.interpolate(
                torch.from_numpy(ref["raw51"]["PROP_F34_RAW"][global_index : global_index + batch]).to(device)[:, None],
                (224, 224), mode="bicubic", align_corners=False,
            )
        ).cpu().numpy()[:, 0]
        prop_k_eval = normalize_batch(
            F.interpolate(
                torch.from_numpy(ref["raw51"]["PROP_K34_RAW"][global_index : global_index + batch]).to(device)[:, None],
                (224, 224), mode="bicubic", align_corners=False,
            )
        ).cpu().numpy()[:, 0]
        gt_raw_batch = bboxes.cpu().numpy()
        gt_batch = gt_raw_batch >= 0.5

        for local, sample_id in enumerate(names):
            ref52 = ref["rows52"][global_index]
            sample_mismatches += int(
                ref52["sample_id"] != sample_id
                or str(ref["raw41"]["sample_id"][global_index]) != sample_id
                or str(ref["raw51"]["sample_id"][global_index]) != sample_id
            )
            raw_errors["AUD_L4"] = max(raw_errors["AUD_L4"], float(np.max(np.abs(raw_aud_l4[local] - ref["raw41"]["AUD_L4"][global_index]))))
            raw_errors["IMG_L4"] = max(raw_errors["IMG_L4"], float(np.max(np.abs(raw_img_l4[local] - ref["raw41"]["IMG_L4"][global_index]))))
            raw_errors["AUD_FINE"] = max(raw_errors["AUD_FINE"], float(np.max(np.abs(raw_aud_fine[local] - ref["raw41"]["AUD_FINE"][global_index]))))
            raw_errors["PROP_F34"] = max(raw_errors["PROP_F34"], float(np.max(np.abs(prototype_maps["F34_AGREEMENT"][local] - ref["raw51"]["PROP_F34_RAW"][global_index]))))
            raw_errors["PROP_K34"] = max(raw_errors["PROP_K34"], float(np.max(np.abs(prototype_maps["K34_AGREEMENT"][local] - ref["raw51"]["PROP_K34_RAW"][global_index]))))

            gt = gt_batch[local]
            gt_raw = gt_raw_batch[local]
            aud = aud_eval[local]
            img = img_eval[local]
            aud_mask = aud >= 0.6
            img_mask = img >= 0.6
            aud_only = aud_mask & ~img_mask
            true_extent = aud_only & gt
            leakage = aud_only & ~gt
            tp = int((aud_mask & gt).sum())
            fp = int((aud_mask & ~gt).sum())
            fn = int((~aud_mask & gt).sum())
            gt_count = tp + fn
            intrinsic_iou_aud = tp / max(tp + fp + fn, 1)
            iou_expand_star = gt_count / max(gt_count + fp, 1)
            iou_shrink_star = tp / max(tp + fn, 1)
            intrinsic_expand_gain = iou_expand_star - intrinsic_iou_aud
            intrinsic_shrink_gain = iou_shrink_star - intrinsic_iou_aud
            intrinsic = intrinsic_label(intrinsic_expand_gain, intrinsic_shrink_gain)
            iou_aud = common.sample_iou(aud, gt_raw)
            iou_img = common.sample_iou(img, gt_raw)
            iou_prop_f = common.sample_iou(prop_f_eval[local], gt_raw)
            iou_prop_k = common.sample_iou(prop_k_eval[local], gt_raw)
            for key, current, reference in (
                ("AUD", iou_aud, ref52["IoU_AUD"]),
                ("IMG", iou_img, ref52["IoU_IMG"]),
                ("PROP_F34", iou_prop_f, ref52["IoU_PROP_F34"]),
                ("PROP_K34", iou_prop_k, ref52["IoU_PROP_K34"]),
            ):
                metric_errors[key] = max(metric_errors[key], abs(float(current) - float(reference)))

            aud_only_count = int(aud_only.sum())
            extent_count = int(true_extent.sum())
            leakage_count = int(leakage.sum())
            sample_row: dict[str, Any] = {
                "sample_index": global_index,
                "sample_id": sample_id,
                "dataset": arguments.experiment,
                "fold": sample_fold(sample_id, int(config.seed)),
                "IoU_AUD": iou_aud,
                "intrinsic_IoU_AUD_binary": intrinsic_iou_aud,
                "IoU_IMG": iou_img,
                "IoU_PROP_F34": iou_prop_f,
                "IoU_PROP_K34": iou_prop_k,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "IoU_expand_star": iou_expand_star,
                "IoU_shrink_star": iou_shrink_star,
                "intrinsic_expand_gain": intrinsic_expand_gain,
                "intrinsic_shrink_gain": intrinsic_shrink_gain,
                "intrinsic_error_type": intrinsic,
                "candidate_5_2_error_type": ref52["error_type"],
                "candidate_5_2_expand_gain": float(ref52["expand_gain"]),
                "candidate_5_2_shrink_gain": float(ref52["shrink_gain"]),
                "AUD_ONLY_pixels": aud_only_count,
                "TRUE_EXTENT_pixels": extent_count,
                "CONTEXT_LEAKAGE_pixels": leakage_count,
                "AUD_ONLY_true_extent_ratio": extent_count / aud_only_count if aud_only_count else math.nan,
                "AUD_ONLY_leakage_ratio": leakage_count / aud_only_count if aud_only_count else math.nan,
            }
            for group in (
                "group_IMG_ONLY",
                "group_AUD_ONLY",
                "group_BOTH_SUCCESS",
                "group_BOTH_FAIL",
                "group_OGL_RESCUE",
                "group_IMG_ONLY_SHRINK",
                "AUD_OVER_EXPANSION",
            ):
                sample_row[group] = ref52[group] == "True"
            sample_row["group_PROP_ONLY"] = iou_aud < 0.5 and max(iou_prop_f, iou_prop_k) >= 0.5
            sample_row["group_PROP_HURT"] = iou_aud >= 0.5 and iou_prop_f < 0.5 and iou_prop_k < 0.5

            extent_coords = np.argwhere(true_extent)
            leakage_coords = np.argwhere(leakage)
            probe_count = min(PROBE_PER_CLASS_PER_SAMPLE, extent_coords.shape[0], leakage_coords.shape[0])
            probe_extent = deterministic_choice(extent_coords, probe_count, sample_id, "probe-extent")
            probe_leakage = deterministic_choice(leakage_coords, probe_count, sample_id, "probe-leakage")
            routing_coords = deterministic_choice(np.argwhere(aud_only), ROUTING_PER_SAMPLE, sample_id, "routing")
            selected: dict[tuple[int, int], dict[str, bool]] = {}
            for coord in probe_extent:
                selected.setdefault(tuple(coord), {"probe_selected": False, "routing_selected": False})["probe_selected"] = True
            for coord in probe_leakage:
                selected.setdefault(tuple(coord), {"probe_selected": False, "routing_selected": False})["probe_selected"] = True
            for coord in routing_coords:
                selected.setdefault(tuple(coord), {"probe_selected": False, "routing_selected": False})["routing_selected"] = True
            coords = np.asarray(list(selected), dtype=np.int64) if selected else np.empty((0, 2), dtype=np.int64)
            sample_row["probe_sampled_pixels"] = int(sum(flags["probe_selected"] for flags in selected.values()))
            sample_row["routing_sampled_pixels"] = int(sum(flags["routing_selected"] for flags in selected.values()))

            if coords.shape[0]:
                aud_neighbor = ndimage.uniform_filter(aud_mask.astype(np.float32), size=3, mode="constant")
                img_neighbor = ndimage.uniform_filter(img_mask.astype(np.float32), size=3, mode="constant")
                aud_distance = ndimage.distance_transform_edt(aud_mask) / math.sqrt(gt.shape[0] ** 2 + gt.shape[1] ** 2)
                img_distance = ndimage.distance_transform_edt(~img_mask) / math.sqrt(gt.shape[0] ** 2 + gt.shape[1] ** 2)

                aud_l3 = normalize_native(output["AUD_L3"][local, 0].cpu().numpy())
                aud_l4 = normalize_native(raw_aud_l4[local])
                img_l3 = normalize_native(output["IMG_L3"][local, 0].cpu().numpy())
                img_l4 = normalize_native(raw_img_l4[local])
                own_l4 = ownership_maps(output["OWN_L4"][local].cpu().numpy())
                own_hr = ownership_maps(output["OWN_HR14"][local].cpu().numpy())
                sampled_native = {
                    "AUD_L3_response": bilinear_sample(aud_l3, coords),
                    "AUD_L4_response": bilinear_sample(aud_l4, coords),
                    "IMG_L3_response": bilinear_sample(img_l3, coords),
                    "IMG_L4_response": bilinear_sample(img_l4, coords),
                    "L4_target_ownership": bilinear_sample(own_l4["target"], coords),
                    "L4_other_max_ownership": bilinear_sample(own_l4["other"], coords),
                    "L4_target_other_margin": bilinear_sample(own_l4["margin"], coords),
                    "L4_ownership_entropy": bilinear_sample(own_l4["entropy"], coords),
                    "HR14_target_ownership": bilinear_sample(own_hr["target"], coords),
                    "HR14_other_max_ownership": bilinear_sample(own_hr["other"], coords),
                    "HR14_target_other_margin": bilinear_sample(own_hr["margin"], coords),
                    "HR14_ownership_entropy": bilinear_sample(own_hr["entropy"], coords),
                    "F34_local_similarity": bilinear_sample(local_f34[local], coords),
                    "K34_local_similarity": bilinear_sample(local_k34[local], coords),
                    "F34_agreement_core_similarity": bilinear_sample(prototype_maps["F34_AGREEMENT"][local], coords),
                    "K34_agreement_core_similarity": bilinear_sample(prototype_maps["K34_AGREEMENT"][local], coords),
                    "F34_AUD_core_similarity": bilinear_sample(prototype_maps["F34_AUD"][local], coords),
                    "K34_AUD_core_similarity": bilinear_sample(prototype_maps["K34_AUD"][local], coords),
                    "F34_IMG_core_similarity": bilinear_sample(prototype_maps["F34_IMG"][local], coords),
                    "K34_IMG_core_similarity": bilinear_sample(prototype_maps["K34_IMG"][local], coords),
                }
                sampled_native["AUD_L3_L4_disagreement"] = np.abs(sampled_native["AUD_L3_response"] - sampled_native["AUD_L4_response"])
                sampled_native["AUD_L3_L4_consistency"] = 1.0 - sampled_native["AUD_L3_L4_disagreement"]
                sampled_native["IMG_L3_L4_disagreement"] = np.abs(sampled_native["IMG_L3_response"] - sampled_native["IMG_L4_response"])
                sampled_native["IMG_L3_L4_consistency"] = 1.0 - sampled_native["IMG_L3_L4_disagreement"]

                for pixel_index, (y, x) in enumerate(coords):
                    flags = selected[(int(y), int(x))]
                    label = int(not gt[y, x])
                    aud_value = float(aud[y, x])
                    img_value = float(img[y, x])
                    pixel_row: dict[str, Any] = {
                        "dataset": arguments.experiment,
                        "sample_index": global_index,
                        "sample_id": sample_id,
                        "fold": sample_row["fold"],
                        "y": int(y),
                        "x": int(x),
                        "gt_type": "CONTEXT_LEAKAGE" if label else "TRUE_EXTENT",
                        "label_context_leakage": label,
                        "probe_selected": flags["probe_selected"],
                        "routing_selected": flags["routing_selected"],
                        "intrinsic_error_type": intrinsic,
                        "candidate_5_2_error_type": ref52["error_type"],
                        "AUD_raw_score": float(raw_aud_eval_np[local, y, x]),
                        "IMG_raw_score": float(raw_img_eval_np[local, y, x]),
                        "AUD_norm_score": aud_value,
                        "IMG_norm_score": img_value,
                        "AUD_minus_IMG": aud_value - img_value,
                        "AUD_IMG_log_ratio": math.log((aud_value + EPS) / (img_value + EPS)),
                        "AUD_threshold_distance": aud_value - 0.6,
                        "IMG_threshold_distance": 0.6 - img_value,
                        "AUD_neighbor_consistency": float(aud_neighbor[y, x]),
                        "IMG_neighbor_consistency": float(img_neighbor[y, x]),
                        "distance_to_AUD_boundary": float(aud_distance[y, x]),
                        "distance_to_IMG_boundary": float(img_distance[y, x]),
                    }
                    pixel_row.update({key: float(value[pixel_index]) for key, value in sampled_native.items()})
                    no_nan_or_inf = no_nan_or_inf and all(math.isfinite(float(pixel_row[key])) for key in WITH_PROTOTYPE_FEATURES)
                    pixel_rows.append(pixel_row)

            sample_rows.append(sample_row)
            global_index += 1

    completed_full = global_index == len(ref["rows52"])
    reproduction = {
        "raw_tensor_max_errors": raw_errors,
        "per_sample_metric_max_errors": metric_errors,
        "sample_mismatches": sample_mismatches,
        "processed_samples": global_index,
        "reference_samples": len(ref["rows52"]),
        "passed": max(raw_errors.values()) == 0.0 and max(metric_errors.values()) == 0.0 and sample_mismatches == 0,
    }
    if not reproduction["passed"]:
        raise RuntimeError(reproduction)

    checkpoint_after = common.verify_snapshots(checkpoints_before)
    zero = {
        "model_eval": not model.training,
        "inference_mode": True,
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": len(parameters_with_grad),
        "parameters_with_grad": parameters_with_grad,
        "checkpoint_hashes_and_mtimes_unchanged": checkpoint_after["all_unchanged"],
        "checkpoint_files": checkpoint_after["files"],
        "no_nan_or_inf": no_nan_or_inf,
    }
    if parameters_with_grad or not checkpoint_after["all_unchanged"] or not no_nan_or_inf:
        raise RuntimeError(zero)

    common.write_csv(output_dir / "per_sample_intrinsic_diagnosis.csv", sample_rows)
    common.write_csv(output_dir / "sampled_pixels.csv", pixel_rows)
    intrinsic_types = ("INTRINSIC_EXPAND", "INTRINSIC_SHRINK", "MIXED_AMBIGUOUS", "KEEP")
    composition = {}
    for error_type in ("ALL",) + intrinsic_types:
        selected = sample_rows if error_type == "ALL" else [row for row in sample_rows if row["intrinsic_error_type"] == error_type]
        total_pixels = sum(int(row["AUD_ONLY_pixels"]) for row in selected)
        extent_pixels = sum(int(row["TRUE_EXTENT_pixels"]) for row in selected)
        leakage_pixels = sum(int(row["CONTEXT_LEAKAGE_pixels"]) for row in selected)
        composition[error_type] = {
            "samples": len(selected),
            "AUD_ONLY_pixels": total_pixels,
            "TRUE_EXTENT_pixels": extent_pixels,
            "CONTEXT_LEAKAGE_pixels": leakage_pixels,
            "TRUE_EXTENT_fraction": extent_pixels / total_pixels if total_pixels else math.nan,
            "CONTEXT_LEAKAGE_fraction": leakage_pixels / total_pixels if total_pixels else math.nan,
            "sample_leakage_ratio": summarize([float(row["AUD_ONLY_leakage_ratio"]) for row in selected if math.isfinite(float(row["AUD_ONLY_leakage_ratio"]))]),
        }
    summary = {
        "experiment": "5.3_aud_only_leakage_cue_probe",
        "setting": arguments.experiment,
        "completed_full_dataset": completed_full,
        "audit": reproduction,
        "zero_training_audit": zero,
        "tensor_audit": tensor_audit,
        "intrinsic_definition": {
            "beneficial_threshold": BENEFICIAL_GAIN,
            "dominance_margin": DOMINANCE_MARGIN,
            "IoU_expand_star": "|GT| / (|GT| + FP)",
            "IoU_shrink_star": "TP / (TP + FN)",
        },
        "intrinsic_distribution": {
            name: {"count": sum(row["intrinsic_error_type"] == name for row in sample_rows), "fraction": sum(row["intrinsic_error_type"] == name for row in sample_rows) / max(len(sample_rows), 1)}
            for name in intrinsic_types
        },
        "AUD_ONLY_composition": composition,
        "sampling": {
            "seed": int(config.seed),
            "fold_rule": "sha256(seed, sample_id) mod 5",
            "probe_per_class_per_mixed_sample": PROBE_PER_CLASS_PER_SAMPLE,
            "routing_per_sample": ROUTING_PER_SAMPLE,
            "saved_pixel_rows": len(pixel_rows),
            "probe_selected_rows": sum(bool(row["probe_selected"]) for row in pixel_rows),
            "routing_selected_rows": sum(bool(row["routing_selected"]) for row in pixel_rows),
            "probe_TRUE_EXTENT": sum(bool(row["probe_selected"]) and int(row["label_context_leakage"]) == 0 for row in pixel_rows),
            "probe_CONTEXT_LEAKAGE": sum(bool(row["probe_selected"]) and int(row["label_context_leakage"]) == 1 for row in pixel_rows),
        },
        "feature_groups": {
            "PREDICTION": PREDICTION_FEATURES,
            "WITHOUT_PROTOTYPE": WITHOUT_PROTOTYPE_FEATURES,
            "WITH_PROTOTYPE": WITH_PROTOTYPE_FEATURES,
            "PROTOTYPE_ONLY": PROTOTYPE_FEATURES,
        },
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "extraction_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
