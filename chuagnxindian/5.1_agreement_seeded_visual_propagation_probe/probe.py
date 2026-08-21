#!/usr/bin/env python3
"""Experiment 5.1: zero-training agreement-seeded visual propagation probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

import common

visualize = common.load_module("experiment_51_visualize", common.HERE / "visualize.py")


FRACTIONS = (0.10, 0.20, 0.30)
SPACES = ("F34", "K34")
CONTROL_SEEDS = ("RANDOM", "AUD", "IMG", "AGREEMENT")
GROUPS = ("IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE", "IMG_ONLY_SHRINK")
CONNECTIVITY8 = np.ones((3, 3), dtype=np.uint8)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_top_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(value).reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.size)))
    order = np.argsort(-flat, kind="stable")
    output = np.zeros(flat.size, dtype=bool)
    output[order[:count]] = True
    return output.reshape(value.shape)


def deterministic_random_mask(
    shape: tuple[int, int], count: int, sample_id: str, role: str, allowed: np.ndarray | None = None
) -> np.ndarray:
    output = np.zeros(shape[0] * shape[1], dtype=bool)
    if count == 0:
        return output.reshape(shape)
    digest = hashlib.sha256(f"JSA-5.1::{sample_id}::{role}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))
    candidates = np.arange(output.size) if allowed is None else np.flatnonzero(np.asarray(allowed).reshape(-1))
    count = min(count, candidates.size)
    output[rng.choice(candidates, size=count, replace=False)] = True
    return output.reshape(shape)


def seed_bundle(aud_raw: np.ndarray, img_raw: np.ndarray, fraction: float, sample_id: str) -> dict[str, np.ndarray]:
    aud = common.normalize_map(aud_raw)
    img = common.normalize_map(img_raw)
    a = stable_top_mask(aud, fraction)
    i = stable_top_mask(img, fraction)
    p = a & i
    random = deterministic_random_mask(a.shape, int(p.sum()), sample_id, f"P-{fraction}")
    return {"AUD_NORM": aud, "IMG_NORM": img, "AUD": a, "IMG": i, "AGREEMENT": p, "RANDOM": random}


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = mask.shape
    out_h, out_w = shape
    if out_h % in_h == 0 and out_w % in_w == 0:
        return np.repeat(np.repeat(mask, out_h // in_h, axis=0), out_w // in_w, axis=1)
    value = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(value, shape, mode="nearest")[0, 0].numpy() >= 0.5


def precision_recall(mask: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    area = int(mask.sum())
    intersection = int(np.logical_and(mask, gt).sum())
    gt_area = int(gt.sum())
    return {
        "area": area,
        "precision": float(intersection / area) if area else math.nan,
        "recall": float(intersection / gt_area) if gt_area else math.nan,
        "fg": intersection,
        "fp": area - intersection,
    }


def scalar_error(first: float, second: float) -> float:
    if math.isnan(first) and math.isnan(second):
        return 0.0
    if not math.isfinite(first) or not math.isfinite(second):
        return math.inf
    return abs(first - second)


def top_fraction_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(value).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    selected = np.argpartition(flat, -count)[-count:]
    output = np.zeros(flat.size, dtype=bool)
    output[selected] = True
    return output.reshape(value.shape)


def map_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = np.asarray(first)
    second = np.asarray(second)
    mask_first = first >= 0.6
    mask_second = second >= 0.6
    union = np.logical_or(mask_first, mask_second).sum()
    top_first = top_fraction_mask(first, 0.20)
    top_second = top_fraction_mask(second, 0.20)
    top_union = np.logical_or(top_first, top_second).sum()
    return {
        "Pearson": common.safe_pearson(first, second),
        "Spearman": common.spearman(first, second),
        "JS": common.js_divergence(first, second),
        "Top20_overlap": float(np.logical_and(top_first, top_second).sum() / max(top_union, 1)),
        "Mask_IoU": float(np.logical_and(mask_first, mask_second).sum() / max(union, 1)),
    }


def binary_iou(mask: np.ndarray, gt: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = np.logical_and(mask, gt).sum()
    denominator = gt.sum() + np.logical_and(mask, ~gt).sum()
    return float(intersection / max(denominator, 1))


def region_oracle_iou(aud: np.ndarray, prop: np.ndarray, gt: np.ndarray) -> float:
    aud_mask = np.asarray(aud) >= 0.6
    prop_mask = np.asarray(prop) >= 0.6
    disagreement = np.logical_xor(aud_mask, prop_mask)
    labels, count = ndimage.label(disagreement.astype(np.uint8), structure=CONNECTIVITY8)
    output = aud_mask.copy()
    for component_index in range(1, count + 1):
        component = labels == component_index
        correct_aud = np.sum(aud_mask[component] == gt[component])
        correct_prop = np.sum(prop_mask[component] == gt[component])
        if correct_prop > correct_aud:
            output[component] = prop_mask[component]
    return binary_iou(output, gt)


@torch.inference_mode()
def extract_all(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = model.teacher
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    attention = teacher.slot_attn._l4_attentions(encoded, scale_multiplier=teacher.infer_sharpening)
    batch = image.shape[0]
    aud_l4 = attention["audq_imgk_attn"].reshape(batch, 2, 7, 7)[:, 0:1]
    img_l4 = attention["imgq_imgk_attn"].reshape(batch, 2, 7, 7)[:, 0:1]
    f34, f3_spatial, f4_up, delta_f3 = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    f34_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(f34_tokens))
    aud_fine_all = teacher.slot_attn._attention(encoded["audio_query"], k34, teacher.infer_sharpening)
    aud_fine = aud_fine_all[:, 0].reshape(batch, 1, 14, 14)
    return {
        "Qa": encoded["audio_query"],
        "Qv": encoded["visual_queries"][-1],
        "K4": encoded["visual_keys"][-1],
        "AUD_L4": aud_l4,
        "IMG_L4": img_l4,
        "F3_SPATIAL": f3_spatial,
        "F4_UP": f4_up,
        "F34": f34,
        "DELTA_F3": delta_f3,
        "K34": k34,
        "AUD_FINE": aud_fine,
    }


def propagate(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(tokens, dim=-1)
    weights = masks.flatten(start_dim=1).float()
    counts = weights.sum(dim=1, keepdim=True)
    prototype = torch.einsum("bn,bnd->bd", weights, normalized) / counts.clamp_min(1.0)
    prototype = F.normalize(prototype, dim=-1)
    raw = torch.einsum("bd,bnd->bn", prototype, normalized)
    raw = torch.where(counts > 0, raw, torch.zeros_like(raw))
    return raw.reshape(tokens.shape[0], 1, 14, 14)


def reference_data(setting: str) -> dict[str, Any]:
    rows50 = load_csv(common.reference_50_dir(setting) / "per_sample_purity.csv")
    summary50 = json.loads((common.reference_50_dir(setting) / "summary.json").read_text(encoding="utf-8"))
    rows41 = load_csv(common.reference_41_dir(setting) / "per_sample_metrics.csv")
    summary41 = json.loads((common.reference_41_dir(setting) / "summary.json").read_text(encoding="utf-8"))
    raw = np.load(common.reference_41_dir(setting) / "raw_maps.npz")
    return {
        "rows50": rows50,
        "summary50": summary50,
        "rows41": rows41,
        "summary41": summary41,
        "sample_id": raw["sample_id"],
        "AUD_L4": raw["AUD_L4"],
        "IMG_L4": raw["IMG_L4"],
        "AUD_FINE": raw["AUD_FINE"],
    }


def group_flags(reference: dict[str, str]) -> dict[str, bool]:
    return {
        "IMG_ONLY": reference["group_IMG_ONLY"] == "True",
        "AUD_ONLY": reference["group_AUD_ONLY"] == "True",
        "BOTH_SUCCESS": reference["group_BOTH_SUCCESS"] == "True",
        "BOTH_FAIL": reference["group_BOTH_FAIL"] == "True",
        "OGL_RESCUE": reference["group_OGL_RESCUE"] == "True",
        "IMG_ONLY_SHRINK": reference["group_IMG_ONLY_SHRINK"] == "True",
    }


def aggregate_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return common.summarize([float(row[key]) for row in rows])


def aggregate_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return common.distribution([float(row[key]) for row in rows])


def success_decomposition(rows: list[dict[str, Any]], prop_key: str) -> dict[str, Any]:
    aud = np.asarray([float(row["IoU_AUD"]) for row in rows]) >= 0.5
    prop = np.asarray([float(row[prop_key]) for row in rows]) >= 0.5
    return {
        "BOTH_SUCCESS": int((aud & prop).sum()),
        "AUD_ONLY": int((aud & ~prop).sum()),
        "PROP_ONLY": int((~aud & prop).sum()),
        "BOTH_FAIL": int((~aud & ~prop).sum()),
    }


def pair_oracle(rows: list[dict[str, Any]], prop_key: str) -> dict[str, Any]:
    aud = np.asarray([float(row["IoU_AUD"]) for row in rows])
    prop = np.asarray([float(row[prop_key]) for row in rows])
    oracle = common.summarize(np.maximum(aud, prop).tolist())
    aud_summary = common.summarize(aud.tolist())
    return {
        "metrics": oracle,
        "gain_cIoU": oracle["cIoU"] - aud_summary["cIoU"],
        "gain_AUC": oracle["AUC"] - aud_summary["AUC"],
    }


def finite_mean(values: list[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def quality_summary(rows: list[dict[str, Any]], space: str) -> dict[str, Any]:
    prefix = f"PROP_{space}"
    return {
        "FG_similarity": aggregate_distribution(rows, f"{prefix}_FG_similarity"),
        "BG_similarity": aggregate_distribution(rows, f"{prefix}_BG_similarity"),
        "FG_BG_margin": aggregate_distribution(rows, f"{prefix}_FG_BG_margin"),
        "seed_precision": aggregate_distribution(rows, "SEED_precision"),
        "seed_recall": aggregate_distribution(rows, "SEED_recall"),
        "prop_precision": aggregate_distribution(rows, f"{prefix}_precision"),
        "prop_recall": aggregate_distribution(rows, f"{prefix}_recall"),
        "recall_gain": aggregate_distribution(rows, f"{prefix}_recall_gain"),
        "precision_loss": aggregate_distribution(rows, f"{prefix}_precision_loss"),
        "expansion_FG_purity": aggregate_distribution(rows, f"{prefix}_EXPAND_precision"),
        "random_expansion_FG_purity": aggregate_distribution(rows, f"{prefix}_RANDOM_EXPAND_precision"),
        "expansion_non_empty_rate": float(np.mean([int(row[f"{prefix}_EXPAND_area"]) > 0 for row in rows])),
    }


def similarity_summary(rows: list[dict[str, Any]], pair: str) -> dict[str, Any]:
    return {
        metric: aggregate_distribution(rows, f"SIM_{pair}_{metric}")
        for metric in ("Pearson", "Spearman", "JS", "Top20_overlap", "Mask_IoU")
    }


def hard_group_summary(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    chosen = [row for row in rows if row[f"group_{group}"]]
    methods = ("AUD", "IMG", "PROP_F34", "PROP_K34")
    result: dict[str, Any] = {"count": len(chosen)}
    for method in methods:
        result[method] = {
            "IoU": aggregate_distribution(chosen, f"IoU_{method}"),
            "area_fraction": aggregate_distribution(chosen, f"{method}_area_fraction"),
            "FP_area_fraction": aggregate_distribution(chosen, f"{method}_FP_area_fraction"),
            "FG_recall": aggregate_distribution(chosen, f"{method}_FG_recall"),
        }
    return result


def confidence_quartiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = np.argsort(np.asarray([float(row["SEED_CONF"]) for row in rows]), kind="stable")
    output = []
    for index, chunk in enumerate(np.array_split(order, 4), start=1):
        chosen = [rows[int(item)] for item in chunk]
        item: dict[str, Any] = {
            "quartile": f"Q{index}",
            "count": len(chosen),
            "seed_confidence": aggregate_distribution(chosen, "SEED_CONF"),
            "seed_purity": aggregate_distribution(chosen, "SEED_precision"),
        }
        for space in SPACES:
            prop_key = f"IoU_PROP_{space}"
            oracle = pair_oracle(chosen, prop_key)
            item[space] = {
                "propagation": aggregate_metric(chosen, prop_key),
                "oracle_gain_cIoU": oracle["gain_cIoU"],
                "FG_BG_margin": aggregate_distribution(chosen, f"PROP_{space}_FG_BG_margin"),
            }
        output.append(item)
    return output


def correlation(rows: list[dict[str, Any]], first_key: str, second_key: str) -> dict[str, Any]:
    first = np.asarray([float(row[first_key]) for row in rows], dtype=np.float64)
    second = np.asarray([float(row[second_key]) for row in rows], dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    return {
        "num_samples": int(valid.sum()),
        "Pearson": common.safe_pearson(first[valid], second[valid]) if valid.sum() >= 2 else math.nan,
        "Spearman": common.spearman(first[valid], second[valid]) if valid.sum() >= 2 else math.nan,
    }


def qualitative_categories(row: dict[str, Any]) -> dict[str, float | str]:
    output: dict[str, float | str] = {}
    for group in ("IMG_ONLY_SHRINK", "OGL_RESCUE", "AUD_ONLY"):
        if row[f"group_{group}"]:
            output[group] = "LEXICOGRAPHIC"
    if row["AUD_OVER_EXPANSION"]:
        output["AUD_OVER_EXPANSION"] = "LEXICOGRAPHIC"
    if row["IoU_AUD"] < 0.5 and max(row["IoU_PROP_F34"], row["IoU_PROP_K34"]) >= 0.5:
        output["PROP_ONLY_SUCCESS"] = "LEXICOGRAPHIC"
    if row["IoU_AUD"] >= 0.5 and row["IoU_PROP_F34"] < 0.5 and row["IoU_PROP_K34"] < 0.5:
        output["PROP_HURT"] = "LEXICOGRAPHIC"
    output["HIGH_CONFIDENCE"] = float(row["SEED_CONF"])
    output["LOW_CONFIDENCE"] = -float(row["SEED_CONF"])
    return output


def update_qualitative(selected: dict[str, dict[str, Any]], categories: dict[str, float | str], payload: dict[str, Any]) -> None:
    for category, value in categories.items():
        sort_key = (0.0, payload["sample_id"]) if value == "LEXICOGRAPHIC" else (-float(value), payload["sample_id"])
        if category not in selected or sort_key < selected[category]["sort_key"]:
            selected[category] = {**payload, "sort_key": sort_key, "category": category}


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
    references = reference_data(arguments.experiment)
    checkpoints_before = common.snapshot_files(
        {
            "formal_stage1": common.stage1_checkpoint_path(registry),
            "formal_original_1_3G": common.g_checkpoint_path(registry),
            "evaluation_only_object_prior": common.OBJECT_CHECKPOINT,
        }
    )
    model = common.load_original_g(registry, device)
    object_model = common.object_prior_model().to(device).eval()
    parameters_with_grad = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters_with_grad += [f"object.{name}" for name, parameter in object_model.named_parameters() if parameter.requires_grad]

    rows: list[dict[str, Any]] = []
    qualitative: dict[str, dict[str, Any]] = {}
    raw_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    raw_maps["sample_id"] = []
    raw_errors = {"AUD_L4": 0.0, "IMG_L4": 0.0, "AUD_FINE": 0.0}
    metric_errors = {"Stage1_AUD": 0.0, "Stage1_IMG": 0.0, "Stage2_AUD": 0.0, "Stage2_IMG": 0.0, "OGL": 0.0}
    seed_errors = {"area": 0.0, "fg": 0.0, "bg": 0.0, "fg_purity": 0.0, "fg_recall": 0.0}
    sample_mismatches = 0
    no_nan_or_inf = True
    tensor_audit: dict[str, Any] | None = None
    global_index = 0

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader)):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_all(model, image, spec)
        object_prior = object_model(image)
        if tensor_audit is None:
            tensor_audit = {
                "Qa_shape": list(output["Qa"].shape),
                "Qv_shape": list(output["Qv"].shape),
                "K4_shape": list(output["K4"].shape),
                "AUD_L4_shape": list(output["AUD_L4"].shape),
                "IMG_L4_shape": list(output["IMG_L4"].shape),
                "F3_SPATIAL_shape": list(output["F3_SPATIAL"].shape),
                "F4_UP_shape": list(output["F4_UP"].shape),
                "F34_shape": list(output["F34"].shape),
                "K34_shape": list(output["K34"].shape),
                "AUD_FINE_shape": list(output["AUD_FINE"].shape),
                "feature_space_audit": {
                    "F34": "visual feature space; cosine is only F34 prototype vs F34 token",
                    "K34": "frozen L4 key-projection space; cosine is only K34 prototype vs K34 token",
                    "forbidden_cross_space_cosines_used": False,
                    "audio_cosine_used": False,
                },
            }
        no_nan_or_inf = no_nan_or_inf and all(
            torch.isfinite(value).all().item()
            for value in (*output.values(), object_prior)
            if isinstance(value, torch.Tensor)
        )

        raw_aud_l4 = output["AUD_L4"].cpu().numpy()[:, 0]
        raw_img_l4 = output["IMG_L4"].cpu().numpy()[:, 0]
        raw_aud_fine = output["AUD_FINE"].cpu().numpy()[:, 0]
        seed_batches: dict[float, list[dict[str, np.ndarray]]] = {
            fraction: [seed_bundle(raw_aud_l4[i], raw_img_l4[i], fraction, names[i]) for i in range(len(names))]
            for fraction in FRACTIONS
        }
        f34_tokens = output["F34"].flatten(start_dim=2).transpose(1, 2)
        k34_tokens = output["K34"]
        propagation_raw: dict[str, torch.Tensor] = {}
        for fraction in FRACTIONS:
            percent = int(round(100 * fraction))
            seed_names = CONTROL_SEEDS if fraction == 0.20 else ("AGREEMENT",)
            for seed_name in seed_names:
                native = np.stack([item[seed_name] for item in seed_batches[fraction]])
                mask = torch.from_numpy(native.astype(np.float32)).to(device)[:, None]
                mask14 = F.interpolate(mask, (14, 14), mode="nearest")[:, 0] >= 0.5
                propagation_raw[f"{seed_name}_F34_TOP{percent}"] = propagate(f34_tokens, mask14)
                propagation_raw[f"{seed_name}_K34_TOP{percent}"] = propagate(k34_tokens, mask14)

        resized_raw = {
            key: F.interpolate(value, (224, 224), mode="bicubic", align_corners=False).cpu().numpy()[:, 0]
            for key, value in propagation_raw.items()
        }
        eval_base = {
            "STAGE1_AUD": common.resize_tensor(output["AUD_L4"]).cpu().numpy()[:, 0],
            "STAGE1_IMG": common.resize_tensor(output["IMG_L4"]).cpu().numpy()[:, 0],
            "AUD": common.resize_tensor(output["AUD_FINE"]).cpu().numpy()[:, 0],
            "IMG": common.resize_tensor(output["IMG_L4"]).cpu().numpy()[:, 0],
            "OBJ": common.resize_tensor(object_prior).cpu().numpy()[:, 0],
        }
        gt_raw_batch = bboxes.cpu().numpy()
        gt_batch = gt_raw_batch >= 0.5

        for index, sample_id in enumerate(names):
            ref50 = references["rows50"][global_index]
            ref41 = references["rows41"][global_index]
            sample_mismatches += int(
                ref50["sample_id"] != sample_id
                or ref41["sample_id"] != sample_id
                or str(references["sample_id"][global_index]) != sample_id
            )
            raw_errors["AUD_L4"] = max(raw_errors["AUD_L4"], float(np.max(np.abs(raw_aud_l4[index] - references["AUD_L4"][global_index]))))
            raw_errors["IMG_L4"] = max(raw_errors["IMG_L4"], float(np.max(np.abs(raw_img_l4[index] - references["IMG_L4"][global_index]))))
            raw_errors["AUD_FINE"] = max(raw_errors["AUD_FINE"], float(np.max(np.abs(raw_aud_fine[index] - references["AUD_FINE"][global_index]))))

            gt_raw = gt_raw_batch[index]
            gt = gt_batch[index]
            stage1_aud = common.normalize_map(eval_base["STAGE1_AUD"][index])
            stage1_img = common.normalize_map(eval_base["STAGE1_IMG"][index])
            aud = common.normalize_map(eval_base["AUD"][index])
            img = common.normalize_map(eval_base["IMG"][index])
            obj = common.normalize_map(eval_base["OBJ"][index])
            ogl = common.normalize_map(0.6 * aud + 0.4 * obj)
            ious = {
                "Stage1_AUD": common.sample_iou(stage1_aud, gt_raw),
                "Stage1_IMG": common.sample_iou(stage1_img, gt_raw),
                "Stage2_AUD": common.sample_iou(aud, gt_raw),
                "Stage2_IMG": common.sample_iou(img, gt_raw),
                "OGL": common.sample_iou(ogl, gt_raw),
            }
            metric_errors["Stage1_AUD"] = max(metric_errors["Stage1_AUD"], abs(ious["Stage1_AUD"] - float(ref50["Stage1_AUD"])))
            metric_errors["Stage1_IMG"] = max(metric_errors["Stage1_IMG"], abs(ious["Stage1_IMG"] - float(ref50["Stage1_IMG"])))
            metric_errors["Stage2_AUD"] = max(metric_errors["Stage2_AUD"], abs(ious["Stage2_AUD"] - float(ref50["Stage2_AUD"])))
            metric_errors["Stage2_IMG"] = max(metric_errors["Stage2_IMG"], abs(ious["Stage2_IMG"] - float(ref50["Stage2_IMG"])))
            metric_errors["OGL"] = max(metric_errors["OGL"], abs(ious["OGL"] - float(ref41["IoU_OGL"])))

            primary = seed_batches[0.20][index]
            p14 = resize_mask(primary["AGREEMENT"], (14, 14))
            p224 = resize_mask(p14, gt.shape)
            seed_quality = precision_recall(p224, gt)
            seed_reproduction_values = {
                "area": seed_quality["area"],
                "fg": seed_quality["fg"],
                "bg": seed_quality["fp"],
                "fg_purity": seed_quality["precision"],
                "fg_recall": seed_quality["recall"],
            }
            for key, current in seed_reproduction_values.items():
                reference = float(ref50[f"Stage1_top20_P_{key}"])
                seed_errors[key] = max(seed_errors[key], scalar_error(float(current), reference))
            seed_conf = float(np.mean(np.minimum(primary["AUD_NORM"], primary["IMG_NORM"])[primary["AGREEMENT"]]))

            row: dict[str, Any] = {
                "sample_index": global_index,
                "sample_id": sample_id,
                "IoU_AUD": ious["Stage2_AUD"],
                "IoU_IMG": ious["Stage2_IMG"],
                "IoU_OGL": ious["OGL"],
                "SEED_CONF": seed_conf,
                "SEED_pixel_count_14": int(p14.sum()),
                "SEED_precision": seed_quality["precision"],
                "SEED_recall": seed_quality["recall"],
                "SEED_area": seed_quality["area"],
            }
            flags = group_flags(ref50)
            row.update({f"group_{key}": value for key, value in flags.items()})

            seed_only_iou = common.sample_iou(p224.astype(np.float32), gt_raw)
            row["IoU_SEED_ONLY"] = seed_only_iou
            normalized_props: dict[str, np.ndarray] = {}
            raw_primary: dict[str, np.ndarray] = {}
            for fraction in FRACTIONS:
                percent = int(round(100 * fraction))
                for space in SPACES:
                    raw_key = f"AGREEMENT_{space}_TOP{percent}"
                    prop = common.normalize_map(resized_raw[raw_key][index])
                    method = f"PROP_{space}_TOP{percent}"
                    normalized_props[method] = prop
                    row[f"IoU_{method}"] = common.sample_iou(prop, gt_raw)
                    seed = seed_batches[fraction][index]["AGREEMENT"]
                    seed224 = resize_mask(seed, gt.shape)
                    seed_record = precision_recall(seed224, gt)
                    row[f"TOP{percent}_SEED_precision"] = seed_record["precision"]
                    row[f"TOP{percent}_SEED_recall"] = seed_record["recall"]
            for space in SPACES:
                row[f"IoU_PROP_{space}"] = row[f"IoU_PROP_{space}_TOP20"]
                normalized_props[f"PROP_{space}"] = normalized_props[f"PROP_{space}_TOP20"]
                raw_primary[space] = resized_raw[f"AGREEMENT_{space}_TOP20"][index]
                for seed_name in CONTROL_SEEDS:
                    key = f"{seed_name}_{space}_TOP20"
                    prop = common.normalize_map(resized_raw[key][index])
                    method = f"{seed_name}_SEED_PROP_{space}"
                    normalized_props[method] = prop
                    row[f"IoU_{method}"] = common.sample_iou(prop, gt_raw)

            method_maps = {"AUD": aud, "IMG": img, "PROP_F34": normalized_props["PROP_F34"], "PROP_K34": normalized_props["PROP_K34"]}
            for method, value in method_maps.items():
                quality = precision_recall(value >= 0.6, gt)
                row[f"{method}_area_fraction"] = float(quality["area"] / gt.size)
                row[f"{method}_FP_area_fraction"] = float(quality["fp"] / gt.size)
                row[f"{method}_FG_recall"] = quality["recall"]

            for space in SPACES:
                prefix = f"PROP_{space}"
                prop = normalized_props[prefix]
                raw_prop = raw_primary[space]
                fg_similarity = float(raw_prop[gt].mean()) if gt.any() else math.nan
                bg_similarity = float(raw_prop[~gt].mean()) if (~gt).any() else math.nan
                row[f"{prefix}_FG_similarity"] = fg_similarity
                row[f"{prefix}_BG_similarity"] = bg_similarity
                row[f"{prefix}_FG_BG_margin"] = fg_similarity - bg_similarity
                prop_quality = precision_recall(prop >= 0.6, gt)
                row[f"{prefix}_precision"] = prop_quality["precision"]
                row[f"{prefix}_recall"] = prop_quality["recall"]
                row[f"{prefix}_recall_gain"] = prop_quality["recall"] - float(seed_quality["recall"])
                row[f"{prefix}_precision_loss"] = float(seed_quality["precision"]) - float(prop_quality["precision"])
                expansion = np.logical_and(prop >= 0.6, ~p224)
                expansion_quality = precision_recall(expansion, gt)
                random_expansion = deterministic_random_mask(
                    gt.shape,
                    int(expansion.sum()),
                    sample_id,
                    f"EXPAND-{space}",
                    allowed=~p224,
                )
                random_quality = precision_recall(random_expansion, gt)
                row[f"{prefix}_EXPAND_area"] = expansion_quality["area"]
                row[f"{prefix}_EXPAND_precision"] = expansion_quality["precision"]
                row[f"{prefix}_RANDOM_EXPAND_precision"] = random_quality["precision"]
                similarity = map_similarity(aud, prop)
                row.update({f"SIM_AUD_{space}_{key}": value for key, value in similarity.items()})
                row[f"IoU_REGION_ORACLE_{space}"] = region_oracle_iou(aud, prop, gt)

            for pair, second in (("IMG", img), ("OBJ", obj)):
                similarity = map_similarity(aud, second)
                row.update({f"SIM_AUD_{pair}_{key}": value for key, value in similarity.items()})

            aud_mask = aud >= 0.6
            row["AUD_OVER_EXPANSION"] = int(aud_mask.sum()) > int(gt.sum())
            row["GT_area_fraction"] = float(gt.mean())

            raw_maps["sample_id"].append(sample_id)
            raw_maps["PROP_F34_RAW"].append(propagation_raw["AGREEMENT_F34_TOP20"][index, 0].cpu().numpy())
            raw_maps["PROP_K34_RAW"].append(propagation_raw["AGREEMENT_K34_TOP20"][index, 0].cpu().numpy())
            raw_maps["P14"].append(p14.astype(np.uint8))

            if not arguments.skip_qualitative:
                rgb = common.inverse_normalize(image[index].detach().cpu()).permute(1, 2, 0).numpy()
                display = {
                    "sample_id": sample_id,
                    "image": np.clip(rgb, 0.0, 1.0),
                    "GT": gt.astype(np.float32),
                    "STAGE1_AUD": stage1_aud,
                    "STAGE1_IMG": stage1_img,
                    "P": p224.astype(np.float32),
                    "AUD": aud,
                    "PROP_F34": normalized_props["PROP_F34"],
                    "PROP_K34": normalized_props["PROP_K34"],
                    "IMG": img,
                    "OGL": ogl,
                    "RAW_F34": common.normalize_map(raw_primary["F34"]),
                    "RAW_K34": common.normalize_map(raw_primary["K34"]),
                    "seed_purity": float(seed_quality["precision"]),
                    "seed_recall": float(seed_quality["recall"]),
                    "seed_confidence": seed_conf,
                    "iou_aud": row["IoU_AUD"],
                    "iou_f34": row["IoU_PROP_F34"],
                    "iou_k34": row["IoU_PROP_K34"],
                    "iou_img": row["IoU_IMG"],
                    "iou_ogl": row["IoU_OGL"],
                    "group": ",".join(key for key, value in flags.items() if value),
                }
                update_qualitative(qualitative, qualitative_categories(row), display)

            rows.append(row)
            global_index += 1

    completed_full = global_index == len(references["rows50"])
    reproduction = {
        "raw_tensor_max_errors": raw_errors,
        "per_sample_metric_max_errors": metric_errors,
        "stage1_P_max_errors": seed_errors,
        "sample_mismatches": sample_mismatches,
        "processed_samples": global_index,
        "reference_samples": len(references["rows50"]),
        "passed": max(raw_errors.values()) == 0.0 and max(metric_errors.values()) == 0.0 and max(seed_errors.values()) == 0.0 and sample_mismatches == 0,
    }
    if not reproduction["passed"]:
        raise RuntimeError(reproduction)

    stage1_p_macro = finite_mean([float(row["SEED_precision"]) for row in rows])
    reference_p_macro = finite_mean(
        [float(row["Stage1_top20_P_fg_purity"]) for row in references["rows50"][:global_index]]
    )
    aggregate_p_error = abs(stage1_p_macro - reference_p_macro)
    reproduction["aggregate_stage1_P_purity_error"] = aggregate_p_error
    reproduction["passed"] = reproduction["passed"] and aggregate_p_error == 0.0
    if not reproduction["passed"]:
        raise RuntimeError(reproduction)

    standalone = {
        "AUD_FINE": aggregate_metric(rows, "IoU_AUD"),
        "IMG": aggregate_metric(rows, "IoU_IMG"),
        "SEED_ONLY": aggregate_metric(rows, "IoU_SEED_ONLY"),
        "PROP_F34": aggregate_metric(rows, "IoU_PROP_F34"),
        "PROP_K34": aggregate_metric(rows, "IoU_PROP_K34"),
        "OGL": aggregate_metric(rows, "IoU_OGL"),
    }
    quality = {space: quality_summary(rows, space) for space in SPACES}
    similarity = {
        "AUD_vs_PROP_F34": similarity_summary(rows, "AUD_F34"),
        "AUD_vs_PROP_K34": similarity_summary(rows, "AUD_K34"),
        "AUD_vs_IMG": similarity_summary(rows, "AUD_IMG"),
        "AUD_vs_OBJ": similarity_summary(rows, "AUD_OBJ"),
    }
    complementarity = {}
    for space in SPACES:
        prop_key = f"IoU_PROP_{space}"
        ogl_rescue = [row for row in rows if row["group_OGL_RESCUE"]]
        captured = sum(float(row[prop_key]) >= 0.5 for row in ogl_rescue)
        better = sum(float(row[prop_key]) > float(row["IoU_AUD"]) for row in ogl_rescue)
        complementarity[space] = {
            "success_decomposition": success_decomposition(rows, prop_key),
            "pair_oracle": pair_oracle(rows, prop_key),
            "region_oracle": {
                "success_rate": float(np.mean([float(row[f"IoU_REGION_ORACLE_{space}"]) >= 0.5 for row in rows])),
                "mean_sample_IoU": finite_mean([float(row[f"IoU_REGION_ORACLE_{space}"]) for row in rows]),
                "AUC": None,
            },
            "OGL_rescue_total": len(ogl_rescue),
            "OGL_rescue_captured": int(captured),
            "OGL_rescue_capture_rate": float(captured / max(len(ogl_rescue), 1)),
            "OGL_rescue_PROP_better_than_AUD": int(better),
            "OGL_rescue_PROP_better_rate": float(better / max(len(ogl_rescue), 1)),
        }

    controls = {}
    for space in SPACES:
        controls[space] = {}
        for seed in CONTROL_SEEDS:
            key = f"IoU_{seed}_SEED_PROP_{space}"
            controls[space][seed] = {
                "metrics": aggregate_metric(rows, key),
                "pair_oracle": pair_oracle(rows, key),
            }

    topk = {}
    for percent in (10, 20, 30):
        topk[str(percent)] = {
            "seed_purity": aggregate_distribution(rows, f"TOP{percent}_SEED_precision"),
            "seed_recall": aggregate_distribution(rows, f"TOP{percent}_SEED_recall"),
        }
        for space in SPACES:
            key = f"IoU_PROP_{space}_TOP{percent}"
            topk[str(percent)][space] = {
                "metrics": aggregate_metric(rows, key),
                "pair_oracle": pair_oracle(rows, key),
            }

    over_expansion = [row for row in rows if row["AUD_OVER_EXPANSION"]]
    over_expansion_summary: dict[str, Any] = {"count": len(over_expansion)}
    for method in ("AUD", "PROP_F34", "PROP_K34"):
        over_expansion_summary[method] = {
            "FP_area_fraction": aggregate_distribution(over_expansion, f"{method}_FP_area_fraction"),
            "FG_recall": aggregate_distribution(over_expansion, f"{method}_FG_recall"),
        }
    aud_only = [row for row in rows if row["group_AUD_ONLY"]]
    aud_only_risk = {"count": len(aud_only)}
    for space in SPACES:
        prop_key = f"IoU_PROP_{space}"
        aud_only_risk[space] = {
            "prop_success": int(sum(float(row[prop_key]) >= 0.5 for row in aud_only)),
            "prop_hurt": int(sum(float(row[prop_key]) < 0.5 for row in aud_only)),
            "IoU_delta": common.distribution([float(row[prop_key]) - float(row["IoU_AUD"]) for row in aud_only]),
            "recall_delta": common.distribution([float(row[f"PROP_{space}_FG_recall"]) - float(row["AUD_FG_recall"]) for row in aud_only]),
        }

    hard_cases = {group: hard_group_summary(rows, group) for group in GROUPS}
    quartiles = confidence_quartiles(rows)
    purity_gain_correlation = {}
    for space in SPACES:
        delta_key = f"PROP_{space}_IoU_delta"
        for row in rows:
            row[delta_key] = float(row[f"IoU_PROP_{space}"]) - float(row["IoU_AUD"])
        purity_gain_correlation[space] = correlation(rows, "SEED_precision", delta_key)

    zero = common.verify_snapshots(checkpoints_before)
    zero_training = {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": len(parameters_with_grad),
        "parameters_with_grad": parameters_with_grad,
        "model_eval": not model.training and not object_model.training,
        "inference_mode": True,
        "no_nan_or_inf": no_nan_or_inf,
        "checkpoint_audit": zero,
        "all_checkpoint_hashes_and_mtimes_unchanged": zero["all_unchanged"],
    }
    if parameters_with_grad or not zero["all_unchanged"] or not no_nan_or_inf:
        raise RuntimeError(zero_training)
    if completed_full:
        expected = {"vggss_144k": (357, 168, 125), "flickr_144k": (19, 13, 10)}[arguments.experiment]
        observed = (
            sum(row["group_OGL_RESCUE"] for row in rows),
            sum(row["group_IMG_ONLY"] for row in rows),
            sum(row["group_IMG_ONLY_SHRINK"] for row in rows),
        )
        if observed != expected:
            raise RuntimeError({"expected_groups": expected, "observed": observed})

    for category, payload in qualitative.items():
        clean = {key: value for key, value in payload.items() if key != "sort_key"}
        visualize.save_panel(clean, output_dir / "qualitative" / f"{category}.png")

    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    np.savez_compressed(
        output_dir / "raw_propagation_maps.npz",
        sample_id=np.asarray(raw_maps["sample_id"]),
        PROP_F34_RAW=np.asarray(raw_maps["PROP_F34_RAW"], dtype=np.float32),
        PROP_K34_RAW=np.asarray(raw_maps["PROP_K34_RAW"], dtype=np.float32),
        P14=np.asarray(raw_maps["P14"], dtype=np.uint8),
    )
    summary = {
        "experiment": "5.1 Agreement-Seeded Visual Propagation Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": completed_full,
        "reproduction_5_0": reproduction,
        "tensor_and_feature_space_audit": tensor_audit,
        "seed_audit": {
            "source": "Stage1 AUD_L4/IMG_L4 only",
            "native_shape": [7, 7],
            "top20_k_each_branch": 10,
            "resize_to_14": "nearest, binary",
            "P14_pixel_count": common.distribution([float(row["SEED_pixel_count_14"]) for row in rows]),
            "non_empty_rate": float(np.mean([int(row["SEED_pixel_count_14"]) > 0 for row in rows])),
            "P_FG_purity": common.distribution([float(row["SEED_precision"]) for row in rows]),
            "P_FG_recall": common.distribution([float(row["SEED_recall"]) for row in rows]),
        },
        "standalone": standalone,
        "propagation_quality": quality,
        "map_similarity": similarity,
        "complementarity": complementarity,
        "seed_controls": controls,
        "topk_diagnostic": topk,
        "hard_cases": hard_cases,
        "aud_over_expansion": over_expansion_summary,
        "aud_only_risk": aud_only_risk,
        "agreement_confidence_quartiles": quartiles,
        "seed_purity_vs_propagation_gain": purity_gain_correlation,
        "reference_AUD_IMG_sample_oracle": references["summary41"]["capacity"]["SAMPLE_ORACLE"],
        "qualitative_selection": {category: payload["sample_id"] for category, payload in qualitative.items()},
        "zero_training_audit": zero_training,
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    run(parse_args())
