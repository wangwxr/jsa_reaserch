#!/usr/bin/env python3
"""Experiment 5.0: zero-training agreement/disagreement pseudo-label purity probe."""

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
from tqdm import tqdm

import common

visualize = common.load_module("experiment_50_visualize", common.HERE / "visualize.py")


FRACTIONS = (0.10, 0.20, 0.30)
SEED_NAMES = ("A", "I", "P", "NA", "NI", "RANDOM_A", "RANDOM_P", "RANDOM_NA")
GROUPS = (
    "ALL",
    "AUD_SUCCESS",
    "IMG_ONLY",
    "AUD_ONLY",
    "BOTH_SUCCESS",
    "BOTH_FAIL",
    "OGL_RESCUE",
    "IMG_ONLY_SHRINK",
)
QUALITATIVE_CATEGORIES = (
    "IMG_ONLY_SHRINK",
    "OGL_RESCUE",
    "AUD_ONLY",
    "BOTH_SUCCESS",
    "BOTH_FAIL",
    "HIGH_PURITY_P",
    "LOW_PURITY_P",
    "HIGH_PURITY_NA",
    "NA_TRUE_EXTENT",
)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def stable_top_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(value).reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.size)))
    order = np.argsort(-flat, kind="stable")
    output = np.zeros(flat.size, dtype=bool)
    output[order[:count]] = True
    return output.reshape(value.shape)


def nearest_resize_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = mask.shape
    out_h, out_w = output_shape
    if out_h % height == 0 and out_w % width == 0:
        return np.repeat(np.repeat(mask, out_h // height, axis=0), out_w // width, axis=1)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(tensor, output_shape, mode="nearest")[0, 0].numpy() >= 0.5


def deterministic_random_mask(
    shape: tuple[int, int], count: int, sample_id: str, role: str
) -> np.ndarray:
    output = np.zeros(shape[0] * shape[1], dtype=bool)
    if count == 0:
        return output.reshape(shape)
    digest = hashlib.sha256(f"JSA-5.0::{sample_id}::{role}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    output[rng.choice(output.size, size=count, replace=False)] = True
    return output.reshape(shape)


def make_seeds(
    aud_raw: np.ndarray, img_raw: np.ndarray, fraction: float, sample_id: str
) -> dict[str, Any]:
    aud = common.normalize_map(aud_raw)
    img = common.normalize_map(img_raw)
    a = stable_top_mask(aud, fraction)
    i = stable_top_mask(img, fraction)
    p = a & i
    na = a & ~i
    ni = i & ~a
    return {
        "AUD_NORM": aud,
        "IMG_NORM": img,
        "A": a,
        "I": i,
        "P": p,
        "NA": na,
        "NI": ni,
        "RANDOM_A": deterministic_random_mask(a.shape, int(a.sum()), sample_id, f"A-{fraction}"),
        "RANDOM_P": deterministic_random_mask(a.shape, int(p.sum()), sample_id, f"P-{fraction}"),
        "RANDOM_NA": deterministic_random_mask(a.shape, int(na.sum()), sample_id, f"NA-{fraction}"),
    }


def seed_record(mask: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    gt = np.asarray(gt) >= 0.5
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    fg = int(np.logical_and(mask, gt).sum())
    bg = area - fg
    gt_area = int(gt.sum())
    bg_area = int(gt.size - gt_area)
    return {
        "area": area,
        "fg": fg,
        "bg": bg,
        "fg_purity": float(fg / area) if area else math.nan,
        "bg_purity": float(bg / area) if area else math.nan,
        "fg_recall": float(fg / gt_area) if gt_area else math.nan,
        "bg_coverage": float(bg / bg_area) if bg_area else math.nan,
    }


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def aggregate_seed_records(records: list[dict[str, float | int]]) -> dict[str, Any]:
    valid = [row for row in records if int(row["area"]) > 0]
    areas = np.asarray([int(row["area"]) for row in records], dtype=np.float64)
    pooled_area = int(sum(int(row["area"]) for row in records))
    pooled_fg = int(sum(int(row["fg"]) for row in records))
    pooled_bg = int(sum(int(row["bg"]) for row in records))
    return {
        "num_samples": len(records),
        "valid_samples": len(valid),
        "non_empty_rate": float(len(valid) / max(len(records), 1)),
        "empty_rate": float(1.0 - len(valid) / max(len(records), 1)),
        "mean_area": float(areas.mean()) if areas.size else math.nan,
        "median_area": float(np.median(areas)) if areas.size else math.nan,
        "macro_fg_purity": finite_mean([float(row["fg_purity"]) for row in valid]),
        "macro_bg_purity": finite_mean([float(row["bg_purity"]) for row in valid]),
        "macro_fg_recall": finite_mean([float(row["fg_recall"]) for row in records]),
        "macro_bg_coverage": finite_mean([float(row["bg_coverage"]) for row in records]),
        "micro_fg_purity": float(pooled_fg / pooled_area) if pooled_area else math.nan,
        "micro_bg_purity": float(pooled_bg / pooled_area) if pooled_area else math.nan,
        "pooled_area": pooled_area,
        "pooled_fg": pooled_fg,
        "pooled_bg": pooled_bg,
    }


def resize_float(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value, size=size, mode="bicubic", align_corners=False)


@torch.inference_mode()
def extract_maps(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = model.teacher
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    attention = teacher.slot_attn._l4_attentions(
        encoded, scale_multiplier=teacher.infer_sharpening
    )
    batch = image.shape[0]
    img_l4_all = attention["imgq_imgk_attn"].reshape(batch, 2, 7, 7)
    aud_l4_all = attention["audq_imgk_attn"].reshape(batch, 2, 7, 7)
    f34, _f3_spatial, _f4_up, _delta_f3 = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    aud_fine_all = teacher.slot_attn._attention(
        encoded["audio_query"], k34, teacher.infer_sharpening
    )
    aud_fine = aud_fine_all[:, 0].reshape(batch, 1, 14, 14)
    return {
        "Qa": encoded["audio_query"],
        "Qv": encoded["visual_queries"][-1],
        "K4": encoded["visual_keys"][-1],
        "K34": k34,
        "AUD_L4": aud_l4_all[:, 0:1],
        "IMG_L4": img_l4_all[:, 0:1],
        "AUD_FINE": aud_fine,
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reference_data(setting: str) -> dict[str, Any]:
    rows40 = load_csv(common.reference_40_dir(setting) / "per_sample_metrics.csv")
    rows41 = load_csv(common.reference_41_dir(setting) / "per_sample_metrics.csv")
    rows42 = load_csv(common.reference_42_dir(setting) / "per_sample_metrics.csv")
    raw41 = np.load(common.reference_41_dir(setting) / "raw_maps.npz")
    correction = load_csv(common.reference_41_dir(setting) / "correction_types.csv")
    shrink = {
        row["sample_id"]
        for row in correction
        if row["group"] == "IMG_ONLY" and row["correction_type"] == "SHRINK"
    }
    return {
        "rows40": rows40,
        "rows41": rows41,
        "rows42": rows42,
        "raw_sample_id": raw41["sample_id"],
        "raw_AUD_L4": raw41["AUD_L4"],
        "raw_IMG_L4": raw41["IMG_L4"],
        "raw_AUD_FINE": raw41["AUD_FINE"],
        "shrink": shrink,
    }


def group_flags(reference: dict[str, str], shrink: set[str]) -> dict[str, bool]:
    aud_success = float(reference["IoU_AUD"]) >= 0.5
    return {
        "ALL": True,
        "AUD_SUCCESS": aud_success,
        "IMG_ONLY": reference["group_IMG_ONLY"] == "True",
        "AUD_ONLY": reference["group_AUD_ONLY"] == "True",
        "BOTH_SUCCESS": reference["group_BOTH_SUCCESS"] == "True",
        "BOTH_FAIL": reference["group_BOTH_FAIL"] == "True",
        "OGL_RESCUE": reference["group_OGL_RESCUE"] == "True",
        "IMG_ONLY_SHRINK": reference["sample_id"] in shrink,
    }


def cell_score_records(
    seed_native: np.ndarray,
    score_native: np.ndarray,
    gt: np.ndarray,
) -> list[tuple[float, int, int]]:
    output = []
    for row, column in np.argwhere(seed_native):
        one = np.zeros_like(seed_native, dtype=bool)
        one[row, column] = True
        resized = nearest_resize_mask(one, gt.shape)
        area = int(resized.sum())
        fg = int(np.logical_and(resized, gt).sum())
        output.append((float(score_native[row, column]), fg, area))
    return output


def quartile_summary(records: list[tuple[float, int, int]], target: str) -> list[dict[str, Any]]:
    if not records:
        return []
    records = sorted(records, key=lambda row: row[0])
    chunks = np.array_split(np.arange(len(records)), 4)
    output = []
    for index, chunk in enumerate(chunks, start=1):
        chosen = [records[int(item)] for item in chunk]
        fg = sum(row[1] for row in chosen)
        area = sum(row[2] for row in chosen)
        purity = fg / area if target == "FG" else (area - fg) / area
        output.append(
            {
                "quartile": f"Q{index}",
                "num_native_cells": len(chosen),
                "score_min": chosen[0][0],
                "score_max": chosen[-1][0],
                "purity": float(purity),
            }
        )
    return output


def correlation(first: list[float], second: list[float]) -> dict[str, Any]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    return {
        "num_samples": int(valid.sum()),
        "Pearson": common.safe_pearson(a[valid], b[valid]) if valid.sum() >= 2 else math.nan,
        "Spearman": common.spearman(a[valid], b[valid]) if valid.sum() >= 2 else math.nan,
    }


def qualitative_categories(
    flags: dict[str, bool], p_purity: float, na_bg_purity: float
) -> dict[str, float | str]:
    categories: dict[str, float | str] = {}
    for group in ("IMG_ONLY_SHRINK", "OGL_RESCUE", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL"):
        if flags[group]:
            categories[group] = "LEXICOGRAPHIC"
    if math.isfinite(p_purity):
        categories["HIGH_PURITY_P"] = p_purity
        categories["LOW_PURITY_P"] = -p_purity
    if math.isfinite(na_bg_purity):
        categories["HIGH_PURITY_NA"] = na_bg_purity
        categories["NA_TRUE_EXTENT"] = -na_bg_purity
    return categories


def update_qualitative(
    selected: dict[str, dict[str, Any]],
    categories: dict[str, float | str],
    payload: dict[str, Any],
) -> None:
    for category, score in categories.items():
        if score == "LEXICOGRAPHIC":
            sort_key = (0.0, payload["sample_id"])
            better = category not in selected or sort_key < selected[category]["sort_key"]
        else:
            sort_key = (-float(score), payload["sample_id"])
            better = category not in selected or sort_key < selected[category]["sort_key"]
        if better:
            selected[category] = {**payload, "sort_key": sort_key, "category": category}


def summarize_level(
    rows: list[dict[str, Any]], level: str, fraction: float = 0.20
) -> dict[str, Any]:
    prefix = f"{level}_top{int(round(fraction * 100))}_"
    output = {}
    for seed in SEED_NAMES:
        records = [row[prefix + seed] for row in rows]
        output[seed] = aggregate_seed_records(records)
    return output


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
    checkpoint_before = common.snapshot_files(
        {
            "formal_stage1": common.stage1_checkpoint_path(registry),
            "formal_original_1_3G": common.g_checkpoint_path(registry),
            "evaluation_only_object_prior": common.OBJECT_CHECKPOINT,
        }
    )
    model = common.load_original_g(registry, device)
    object_model = common.object_prior_model().to(device).eval()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable += sum(parameter.numel() for parameter in object_model.parameters() if parameter.requires_grad)

    rows: list[dict[str, Any]] = []
    qualitative: dict[str, dict[str, Any]] = {}
    raw_errors = {"AUD_L4": 0.0, "IMG_L4": 0.0, "AUD_FINE": 0.0}
    metric_errors = {"Stage1_AUD": 0.0, "Stage1_IMG": 0.0, "Stage2_AUD": 0.0, "Stage2_IMG": 0.0}
    metric_errors_42 = {"Stage2_AUD": 0.0, "Stage2_IMG": 0.0}
    sample_mismatches = 0
    no_nan_or_inf = True
    tensor_audit: dict[str, Any] | None = None
    p_cells: list[tuple[float, int, int]] = []
    na_cells: list[tuple[float, int, int]] = []
    correctness = defaultdict(int)
    global_index = 0

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader)):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_maps(model, image, spec)
        object_prior = object_model(image)
        if tensor_audit is None:
            tensor_audit = {
                "Qa_shape": list(output["Qa"].shape),
                "Qv_shape": list(output["Qv"].shape),
                "K4_shape": list(output["K4"].shape),
                "K34_shape": list(output["K34"].shape),
                "Stage1_AUD_L4_shape": list(output["AUD_L4"].shape),
                "Stage1_IMG_L4_shape": list(output["IMG_L4"].shape),
                "Stage2_AUD_FINE_shape": list(output["AUD_FINE"].shape),
                "Stage2_IMG_native_shape": list(output["IMG_L4"].shape),
                "Stage2_IMG_aligned_shape": list(resize_float(output["IMG_L4"], (14, 14)).shape),
                "GT_shape": list(bboxes.shape),
            }
        no_nan_or_inf = no_nan_or_inf and all(
            torch.isfinite(value).all().item()
            for value in (*output.values(), object_prior)
            if isinstance(value, torch.Tensor)
        )

        raw_aud_l4 = output["AUD_L4"].cpu().numpy()[:, 0]
        raw_img_l4 = output["IMG_L4"].cpu().numpy()[:, 0]
        raw_aud_fine = output["AUD_FINE"].cpu().numpy()[:, 0]
        raw_img_14 = resize_float(output["IMG_L4"], (14, 14)).cpu().numpy()[:, 0]
        eval_maps = {
            "S1_AUD": common.resize_tensor(output["AUD_L4"]).cpu().numpy()[:, 0],
            "S1_IMG": common.resize_tensor(output["IMG_L4"]).cpu().numpy()[:, 0],
            "S2_AUD": common.resize_tensor(output["AUD_FINE"]).cpu().numpy()[:, 0],
            "S2_IMG": common.resize_tensor(output["IMG_L4"]).cpu().numpy()[:, 0],
            "OBJ": common.resize_tensor(object_prior).cpu().numpy()[:, 0],
        }
        gt_raw_batch = bboxes.cpu().numpy()
        gt_batch = gt_raw_batch >= 0.5

        for index, sample_id in enumerate(names):
            if global_index >= len(references["rows41"]):
                raise RuntimeError("Current loader is longer than the formal reference")
            ref40 = references["rows40"][global_index]
            ref41 = references["rows41"][global_index]
            ref42 = references["rows42"][global_index]
            sample_mismatches += int(
                ref40["sample_id"] != sample_id
                or ref41["sample_id"] != sample_id
                or ref42["sample_id"] != sample_id
                or str(references["raw_sample_id"][global_index]) != sample_id
            )
            raw_errors["AUD_L4"] = max(
                raw_errors["AUD_L4"],
                float(np.max(np.abs(raw_aud_l4[index] - references["raw_AUD_L4"][global_index]))),
            )
            raw_errors["IMG_L4"] = max(
                raw_errors["IMG_L4"],
                float(np.max(np.abs(raw_img_l4[index] - references["raw_IMG_L4"][global_index]))),
            )
            raw_errors["AUD_FINE"] = max(
                raw_errors["AUD_FINE"],
                float(np.max(np.abs(raw_aud_fine[index] - references["raw_AUD_FINE"][global_index]))),
            )

            gt_raw = gt_raw_batch[index]
            gt = gt_batch[index]
            s1_aud = common.normalize_map(eval_maps["S1_AUD"][index])
            s1_img = common.normalize_map(eval_maps["S1_IMG"][index])
            s2_aud = common.normalize_map(eval_maps["S2_AUD"][index])
            s2_img = common.normalize_map(eval_maps["S2_IMG"][index])
            obj = common.normalize_map(eval_maps["OBJ"][index])
            ogl = common.normalize_map(0.6 * s2_aud + 0.4 * obj)
            ious = {
                "Stage1_AUD": common.sample_iou(s1_aud, gt_raw),
                "Stage1_IMG": common.sample_iou(s1_img, gt_raw),
                "Stage2_AUD": common.sample_iou(s2_aud, gt_raw),
                "Stage2_IMG": common.sample_iou(s2_img, gt_raw),
            }
            metric_errors["Stage1_AUD"] = max(metric_errors["Stage1_AUD"], abs(ious["Stage1_AUD"] - float(ref40["Stage1_IoU_AUD"])))
            metric_errors["Stage1_IMG"] = max(metric_errors["Stage1_IMG"], abs(ious["Stage1_IMG"] - float(ref40["Stage1_IoU_IMG"])))
            metric_errors["Stage2_AUD"] = max(metric_errors["Stage2_AUD"], abs(ious["Stage2_AUD"] - float(ref41["IoU_AUD"])))
            metric_errors["Stage2_IMG"] = max(metric_errors["Stage2_IMG"], abs(ious["Stage2_IMG"] - float(ref41["IoU_IMG"])))
            metric_errors_42["Stage2_AUD"] = max(metric_errors_42["Stage2_AUD"], abs(ious["Stage2_AUD"] - float(ref42["IoU_AUD"])))
            metric_errors_42["Stage2_IMG"] = max(metric_errors_42["Stage2_IMG"], abs(ious["Stage2_IMG"] - float(ref42["IoU_IMG"])))

            flags = group_flags(ref41, references["shrink"])
            row: dict[str, Any] = {
                "sample_index": global_index,
                "sample_id": sample_id,
                **ious,
                **{f"group_{key}": value for key, value in flags.items()},
            }
            per_fraction_native: dict[tuple[str, float], dict[str, Any]] = {}
            for level, aud_native, img_native in (
                ("Stage1", raw_aud_l4[index], raw_img_l4[index]),
                ("Stage2", raw_aud_fine[index], raw_img_14[index]),
            ):
                for fraction in FRACTIONS:
                    seeds = make_seeds(aud_native, img_native, fraction, sample_id)
                    per_fraction_native[(level, fraction)] = seeds
                    for seed in SEED_NAMES:
                        resized_seed = nearest_resize_mask(seeds[seed], gt.shape)
                        record = seed_record(resized_seed, gt)
                        key = f"{level}_top{int(round(fraction * 100))}_{seed}"
                        row[key] = record
                        for metric, value in record.items():
                            row[f"{key}_{metric}"] = value

            primary = per_fraction_native[("Stage1", 0.20)]
            primary_records = {seed: row[f"Stage1_top20_{seed}"] for seed in SEED_NAMES}
            p_cells.extend(cell_score_records(primary["P"], np.minimum(primary["AUD_NORM"], primary["IMG_NORM"]), gt))
            na_cells.extend(cell_score_records(primary["NA"], primary["AUD_NORM"] - primary["IMG_NORM"], gt))
            for region in ("P", "NA", "NI"):
                correctness[f"{region}_FG"] += int(primary_records[region]["fg"])
                correctness[f"{region}_BG"] += int(primary_records[region]["bg"])

            if not arguments.skip_qualitative:
                rgb = common.inverse_normalize(image[index].detach().cpu()).permute(1, 2, 0).numpy()
                display = {
                    "sample_id": sample_id,
                    "image": np.clip(rgb, 0.0, 1.0),
                    "GT": gt.astype(np.float32),
                    "STAGE1_AUD": s1_aud,
                    "STAGE1_IMG": s1_img,
                    "A20": nearest_resize_mask(primary["A"], gt.shape).astype(np.float32),
                    "I20": nearest_resize_mask(primary["I"], gt.shape).astype(np.float32),
                    "P": nearest_resize_mask(primary["P"], gt.shape).astype(np.float32),
                    "NA": nearest_resize_mask(primary["NA"], gt.shape).astype(np.float32),
                    "NI": nearest_resize_mask(primary["NI"], gt.shape).astype(np.float32),
                    "STAGE2_AUD": s2_aud,
                    "STAGE2_IMG": s2_img,
                    "OGL": ogl,
                    "p_fg_purity": float(primary_records["P"]["fg_purity"]),
                    "na_bg_purity": float(primary_records["NA"]["bg_purity"]),
                    "p_area": int(primary_records["P"]["area"]),
                    "na_area": int(primary_records["NA"]["area"]),
                    "group": ",".join(key for key, value in flags.items() if value and key != "ALL"),
                }
                categories = qualitative_categories(
                    flags, display["p_fg_purity"], display["na_bg_purity"]
                )
                update_qualitative(qualitative, categories, display)

            rows.append(row)
            global_index += 1

    completed_full = global_index == len(references["rows41"])
    reproduction = {
        "raw_map_max_errors": raw_errors,
        "per_sample_metric_max_errors": metric_errors,
        "per_sample_4_2_metric_max_errors": metric_errors_42,
        "sample_order_mismatches": sample_mismatches,
        "processed_samples": global_index,
        "reference_samples": len(references["rows41"]),
        "passed": max(raw_errors.values()) == 0.0 and max(metric_errors.values()) == 0.0 and max(metric_errors_42.values()) == 0.0 and sample_mismatches == 0,
    }
    if not reproduction["passed"]:
        raise RuntimeError(reproduction)

    primary = summarize_level(rows, "Stage1", 0.20)
    diagnostic = summarize_level(rows, "Stage2", 0.20)
    topk = {}
    for level in ("Stage1", "Stage2"):
        topk[level] = {
            str(int(fraction * 100)): summarize_level(rows, level, fraction)
            for fraction in FRACTIONS
        }

    hard_cases = {}
    hard_cases_stage2 = {}
    for group in GROUPS:
        chosen = [row for row in rows if row[f"group_{group}"]]
        hard_cases[group] = {
            "count": len(chosen),
            "P": aggregate_seed_records([row["Stage1_top20_P"] for row in chosen]),
            "NA": aggregate_seed_records([row["Stage1_top20_NA"] for row in chosen]),
        }
        hard_cases_stage2[group] = {
            "count": len(chosen),
            "P": aggregate_seed_records([row["Stage2_top20_P"] for row in chosen]),
            "NA": aggregate_seed_records([row["Stage2_top20_NA"] for row in chosen]),
        }

    ranking_delta = []
    for row in rows:
        p = row["Stage1_top20_P"]
        na = row["Stage1_top20_NA"]
        if int(p["area"]) and int(na["area"]):
            ranking_delta.append(float(p["fg_purity"]) - float(na["fg_purity"]))
    ranking = {
        **common.distribution(ranking_delta),
        "fraction_positive": float(np.mean(np.asarray(ranking_delta) > 0)) if ranking_delta else math.nan,
    }

    transfer = {
        "P_FG_purity": correlation(
            [float(row["Stage1_top20_P"]["fg_purity"]) for row in rows],
            [float(row["Stage2_top20_P"]["fg_purity"]) for row in rows],
        ),
        "NA_BG_purity": correlation(
            [float(row["Stage1_top20_NA"]["bg_purity"]) for row in rows],
            [float(row["Stage2_top20_NA"]["bg_purity"]) for row in rows],
        ),
    }

    correctness_rows = []
    for region in ("P", "NA", "NI"):
        total = correctness[f"{region}_FG"] + correctness[f"{region}_BG"]
        for label in ("FG", "BG"):
            count = correctness[f"{region}_{label}"]
            correctness_rows.append(
                {"region": region, "GT": label, "count": count, "fraction_within_region": count / max(total, 1)}
            )

    enrichment = {
        "macro": {
            "FG_P_minus_A": primary["P"]["macro_fg_purity"] - primary["A"]["macro_fg_purity"],
            "FG_P_minus_I": primary["P"]["macro_fg_purity"] - primary["I"]["macro_fg_purity"],
            "BG_NA_minus_A": primary["NA"]["macro_bg_purity"] - primary["A"]["macro_bg_purity"],
            "BG_NA_minus_random_NA": primary["NA"]["macro_bg_purity"] - primary["RANDOM_NA"]["macro_bg_purity"],
        },
        "micro": {
            "FG_P_minus_A": primary["P"]["micro_fg_purity"] - primary["A"]["micro_fg_purity"],
            "FG_P_minus_I": primary["P"]["micro_fg_purity"] - primary["I"]["micro_fg_purity"],
            "BG_NA_minus_A": primary["NA"]["micro_bg_purity"] - primary["A"]["micro_bg_purity"],
            "BG_NA_minus_random_NA": primary["NA"]["micro_bg_purity"] - primary["RANDOM_NA"]["micro_bg_purity"],
        },
    }

    stage1_metrics = {
        "AUD": common.summarize([float(row["Stage1_AUD"]) for row in rows]),
        "IMG": common.summarize([float(row["Stage1_IMG"]) for row in rows]),
    }
    stage2_metrics = {
        "AUD_FINE": common.summarize([float(row["Stage2_AUD"]) for row in rows]),
        "IMG": common.summarize([float(row["Stage2_IMG"]) for row in rows]),
    }
    checkpoint_after = common.verify_snapshots(checkpoint_before)
    zero_training = {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": trainable,
        "model_eval": not model.training and not object_model.training,
        "inference_mode": True,
        "no_nan_or_inf": no_nan_or_inf,
        "checkpoint_audit": checkpoint_after,
        "all_checkpoint_hashes_and_mtimes_unchanged": checkpoint_after["all_unchanged"],
    }
    if trainable != 0 or not zero_training["all_checkpoint_hashes_and_mtimes_unchanged"] or not no_nan_or_inf:
        raise RuntimeError(zero_training)

    if completed_full:
        expected = {"vggss_144k": (357, 168, 125), "flickr_144k": (19, 13, 10)}[arguments.experiment]
        observed = (
            hard_cases["OGL_RESCUE"]["count"],
            hard_cases["IMG_ONLY"]["count"],
            hard_cases["IMG_ONLY_SHRINK"]["count"],
        )
        if observed != expected:
            raise RuntimeError({"expected_hard_groups": expected, "observed": observed})
        expected_area = 10240
        for row in rows:
            for level in ("Stage1", "Stage2"):
                for seed in ("A", "I", "RANDOM_A"):
                    area = int(row[f"{level}_top20_{seed}"]["area"])
                    if area != expected_area:
                        raise RuntimeError({"sample_id": row["sample_id"], "level": level, "seed": seed, "area": area})
                for seed, reference_seed in (("RANDOM_P", "P"), ("RANDOM_NA", "NA")):
                    area = int(row[f"{level}_top20_{seed}"]["area"])
                    reference_area = int(row[f"{level}_top20_{reference_seed}"]["area"])
                    if area != reference_area:
                        raise RuntimeError({"sample_id": row["sample_id"], "level": level, "seed": seed, "area": area, "expected": reference_area})

    flat_rows = []
    for row in rows:
        flat_rows.append({key: value for key, value in row.items() if not isinstance(value, dict)})
    common.write_csv(output_dir / "per_sample_purity.csv", flat_rows)
    common.write_csv(output_dir / "correctness_matrix.csv", correctness_rows)
    topk_rows = []
    for level, fractions in topk.items():
        for fraction, result in fractions.items():
            for seed in ("P", "NA"):
                topk_rows.append({"level": level, "top_percent": fraction, "seed": seed, **result[seed]})
    common.write_csv(output_dir / "topk_diagnostic.csv", topk_rows)
    quartiles = {
        "P_agreement_confidence": quartile_summary(p_cells, "FG"),
        "NA_disagreement_magnitude": quartile_summary(na_cells, "BG"),
    }
    quartile_rows = []
    for score_name, result in quartiles.items():
        for item in result:
            quartile_rows.append({"score": score_name, **item})
    common.write_csv(output_dir / "quartile_diagnostic.csv", quartile_rows)
    hard_rows = []
    for group, value in hard_cases.items():
        hard_rows.append(
            {
                "group": group,
                "count": value["count"],
                "P_FG_purity": value["P"]["macro_fg_purity"],
                "P_FG_recall": value["P"]["macro_fg_recall"],
                "P_non_empty_rate": value["P"]["non_empty_rate"],
                "NA_BG_purity": value["NA"]["macro_bg_purity"],
                "NA_BG_coverage": value["NA"]["macro_bg_coverage"],
                "NA_non_empty_rate": value["NA"]["non_empty_rate"],
            }
        )
    common.write_csv(output_dir / "hard_case_purity.csv", hard_rows)

    for category, payload in qualitative.items():
        clean = {key: value for key, value in payload.items() if key != "sort_key"}
        visualize.save_panel(clean, output_dir / "qualitative" / f"{category}.png")

    summary = {
        "experiment": "5.0 Agreement-Disagreement Pseudo-Label Purity Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": completed_full,
        "reproduction": reproduction,
        "tensor_audit": tensor_audit,
        "top20_seed_audit": {
            "Stage1_native_shape": [7, 7],
            "Stage1_k": 10,
            "Stage1_A_I_resized_area": 10240,
            "Stage2_native_shape": [14, 14],
            "Stage2_k": 40,
            "Stage2_A_I_resized_area": 10240,
            "resize": "nearest-neighbor to binary 224x224 GT",
            "tie_breaking": "stable descending native flat index",
        },
        "formal_metrics": {"Stage1": stage1_metrics, "Stage2": stage2_metrics},
        "stage1_primary_top20": primary,
        "stage2_diagnostic_top20": diagnostic,
        "enrichment": enrichment,
        "topk_diagnostic": topk,
        "confidence_quartiles": quartiles,
        "hard_cases_stage1": hard_cases,
        "hard_cases_stage2_diagnostic": hard_cases_stage2,
        "stage1_vs_stage2_transfer": transfer,
        "correctness_matrix": correctness_rows,
        "ranking_viability": ranking,
        "qualitative_selection": {category: payload["sample_id"] for category, payload in qualitative.items()},
        "zero_training_audit": zero_training,
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    run(parse_args())
