#!/usr/bin/env python3
"""Experiment 4.1: zero-training selective fusion capacity and evidence probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm

import common

visualize = common.load_module("experiment_41_visualize", common.HERE / "visualize.py")


EPS = 1e-8
EVIDENCES = (
    "CTRL_RAW_PEAK",
    "CTRL_NEG_ENTROPY",
    "CTRL_NEG_AREA",
    "CTRL_TOP20_CONCENTRATION",
    "CTRL_NEG_COMPONENTS",
    "SEMANTIC_SLOT",
    "RECIPROCAL_L4",
)
LOCAL_EVIDENCES = ("SEMANTIC_SLOT", "RECIPROCAL_L4")
BASE_METHODS = ("AUD", "IMG", "IQR", "OBJ", "OGL")
QUALITATIVE_CATEGORIES = (
    "IMG_ONLY",
    "AUD_ONLY",
    "FIXED_IQR_HURT",
    "FIXED_IQR_RESCUE",
    "SELECTOR_CORRECT_IMG",
    "SELECTOR_WRONG_IMG",
    "SELECTOR_CORRECT_AUD",
    "OGL_RESCUE_CAPTURED",
    "OGL_RESCUE_MISSED",
)
CONNECTIVITY8 = np.ones((3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def binary_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(np.asarray(mask, dtype=np.uint8), structure=CONNECTIVITY8)


def component_count(mask: np.ndarray) -> int:
    return int(binary_components(mask)[1])


def centroid(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return np.asarray([111.5, 111.5], dtype=np.float64)
    return coordinates.mean(axis=0)


def top_fraction_mask(value: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(value).ravel()
    count = max(1, int(math.ceil(flat.size * fraction)))
    indices = np.argpartition(flat, -count)[-count:]
    output = np.zeros(flat.size, dtype=bool)
    output[indices] = True
    return output.reshape(value.shape)


def top_mass(value: np.ndarray, fraction: float) -> float:
    value = np.asarray(value, dtype=np.float64)
    total = value.sum()
    if total <= 0:
        return 0.0
    return float(value[top_fraction_mask(value, fraction)].sum() / total)


def normalized_entropy(value: np.ndarray) -> float:
    values = np.clip(np.asarray(value, dtype=np.float64).ravel(), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 1.0
    probability = values / total
    return float(-np.sum(probability * np.log(probability + EPS)) / np.log(probability.size))


def top20_concentration(value: np.ndarray) -> float:
    values = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0
    return float(values[top_fraction_mask(values, 0.20)].sum() / total)


def map_controls(raw: np.ndarray, normalized: np.ndarray) -> dict[str, float]:
    mask = normalized >= 0.6
    return {
        "CTRL_RAW_PEAK": float(np.max(raw)),
        "CTRL_NEG_ENTROPY": -normalized_entropy(normalized),
        "CTRL_NEG_AREA": -float(mask.mean()),
        "CTRL_TOP20_CONCENTRATION": top20_concentration(normalized),
        "CTRL_NEG_COMPONENTS": -float(component_count(mask)),
    }


def soft_contrast(candidate: np.ndarray, support: np.ndarray) -> float:
    candidate = np.clip(np.asarray(candidate, dtype=np.float64), 0.0, 1.0)
    support = np.asarray(support, dtype=np.float64)
    foreground = candidate / max(candidate.sum(), EPS)
    background_values = 1.0 - candidate
    background = background_values / max(background_values.sum(), EPS)
    return float(np.sum(foreground * support) - np.sum(background * support))


def capacity_oracles(
    aud: np.ndarray, img: np.ndarray, gt: np.ndarray
) -> dict[str, Any]:
    choose_img = np.abs(img - gt) < np.abs(aud - gt)
    soft_pixel = common.normalize_map(np.where(choose_img, img, aud))

    mask_aud = aud >= 0.6
    mask_img = img >= 0.6
    disagreement = np.logical_xor(mask_aud, mask_img)
    binary_pixel = mask_aud.copy()
    binary_pixel[disagreement] = gt[disagreement] >= 0.5

    labels, components = binary_components(disagreement)
    region = mask_aud.copy()
    for component_index in range(1, components + 1):
        component = labels == component_index
        correct_aud = np.sum(mask_aud[component] == (gt[component] >= 0.5))
        correct_img = np.sum(mask_img[component] == (gt[component] >= 0.5))
        if correct_img > correct_aud:
            region[component] = mask_img[component]
    return {
        "SOFT_PIXEL_MAP": soft_pixel,
        "BINARY_PIXEL_MASK": binary_pixel.astype(np.float32),
        "REGION_MASK": region.astype(np.float32),
        "disagreement_components": components,
    }


def binary_iou(mask: np.ndarray, gt: np.ndarray) -> float:
    mask = np.asarray(mask) >= 0.5
    gt = np.asarray(gt) >= 0.5
    intersection = np.logical_and(mask, gt).sum()
    denominator = gt.sum() + np.logical_and(mask, ~gt).sum()
    return float(intersection / denominator)


def boundary_energy(disagreement: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    gt_mask = np.asarray(gt) >= 0.5
    eroded = ndimage.binary_erosion(gt_mask, structure=CONNECTIVITY8, iterations=1)
    dilated = ndimage.binary_dilation(gt_mask, structure=CONNECTIVITY8, iterations=1)
    interior = eroded
    boundary = np.logical_and(dilated, ~eroded)
    exterior = ~dilated
    total = float(np.asarray(disagreement, dtype=np.float64).sum())
    if total <= 0:
        return {"GT_INTERIOR": 0.0, "GT_BOUNDARY_1PX": 0.0, "GT_EXTERIOR": 0.0}
    return {
        "GT_INTERIOR": float(disagreement[interior].sum() / total),
        "GT_BOUNDARY_1PX": float(disagreement[boundary].sum() / total),
        "GT_EXTERIOR": float(disagreement[exterior].sum() / total),
    }


def correction_type(area_aud: float, area_img: float, centroid_distance: float) -> str:
    if area_img < 0.9 * area_aud:
        return "SHRINK"
    if area_img > 1.1 * area_aud:
        return "EXPAND"
    if centroid_distance > 0.10 * math.sqrt(224**2 + 224**2):
        return "RELOCATE"
    return "MIXED"


@torch.inference_mode()
def extract_all(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = model.teacher
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    eval_attentions = teacher.slot_attn._l4_attentions(
        encoded, scale_multiplier=teacher.infer_sharpening
    )

    batch = image.shape[0]
    img_l4_all = eval_attentions["imgq_imgk_attn"].reshape(batch, 2, 7, 7)
    aud_l4_all = eval_attentions["audq_imgk_attn"].reshape(batch, 2, 7, 7)

    qv = encoded["visual_queries"][-1]
    k4 = encoded["visual_keys"][-1]
    ownership_logits = (
        torch.einsum("bsd,bnd->bsn", qv, k4)
        * teacher.infer_sharpening
        * teacher.slot_attn.scale
    )
    ownership_pixel = ownership_logits.softmax(dim=1).reshape(batch, 2, 7, 7)

    fused_visual_slots = teacher.slot_attn.slot_fusion(encoded["visual_slots"])
    audio_slots = encoded["audio_slots"]
    semantic_cosines = F.cosine_similarity(
        fused_visual_slots,
        audio_slots[:, 0:1].expand_as(fused_visual_slots),
        dim=-1,
    )
    semantic_support = torch.einsum(
        "bs,bshw->bhw", semantic_cosines, ownership_pixel
    ).unsqueeze(1)

    # Reciprocal support combines the audio-to-visual target attention with
    # how closely each visual slot's audio-token attention matches audio slot0.
    img_to_audio = eval_attentions["imgq_audk_attn"]
    aud_to_audio_target = eval_attentions["audq_audk_attn"][:, 0:1]
    midpoint = 0.5 * (img_to_audio + aud_to_audio_target)
    temporal_js = 0.5 * (
        (
            img_to_audio
            * (img_to_audio.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())
        ).sum(dim=-1)
        + (
            aud_to_audio_target
            * (
                aud_to_audio_target.clamp_min(EPS).log()
                - midpoint.clamp_min(EPS).log()
            )
        ).sum(dim=-1)
    )
    temporal_reciprocity = 1.0 - temporal_js / math.log(2.0)
    visual_to_audio_support = torch.einsum(
        "bs,bshw->bhw", temporal_reciprocity, ownership_pixel
    ).unsqueeze(1)
    reciprocal_support = 0.5 * (
        aud_l4_all[:, 0:1] + visual_to_audio_support
    )

    f34, f3_spatial, f4_up, delta_f3 = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    aud_fine_all = teacher.slot_attn._attention(
        encoded["audio_query"], k34, teacher.infer_sharpening
    )
    aud_fine = aud_fine_all[:, 0].reshape(batch, 1, 14, 14)
    return {
        "Qa": encoded["audio_query"],
        "Qv": qv,
        "K4": k4,
        "K34": k34,
        "AUD_FINE": aud_fine,
        "AUD_L4_ALL": aud_l4_all,
        "IMG_L4_ALL": img_l4_all,
        "OWNERSHIP_PIXEL": ownership_pixel,
        "FUSED_VISUAL_SLOTS": fused_visual_slots,
        "AUDIO_SLOTS": audio_slots,
        "SEMANTIC_COSINES": semantic_cosines,
        "SEMANTIC_SUPPORT": semantic_support,
        "TEMPORAL_RECIPROCITY": temporal_reciprocity,
        "VISUAL_TO_AUDIO_SUPPORT": visual_to_audio_support,
        "RECIPROCAL_SUPPORT": reciprocal_support,
        "F34": f34,
        "F3_SPATIAL": f3_spatial,
        "F4_UP": f4_up,
        "DELTA_F3": delta_f3,
        "f4_token_error": (
            f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]
        ).abs().max(),
    }


@torch.inference_mode()
def tensor_audit(loader, model, object_model, device: torch.device) -> dict[str, Any]:
    image, spec, bboxes, names, _labels = next(iter(loader))
    image, spec, _bboxes, names = common.flatten_batch(image, spec, bboxes, names)
    image = image.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    official_img, _official_aud_l4 = model.teacher(image, spec)
    official_aud = model(image, spec)["AUD_FINE"]
    output = extract_all(model, image, spec)
    object_prior = object_model(image)
    img_local = output["IMG_L4_ALL"][:, 0:1]
    errors = {
        "AUD_FINE": float((output["AUD_FINE"] - official_aud).abs().max()),
        "IMG_QUERY": float((img_local - official_img).abs().max()),
        "f4_tokens": float(output["f4_token_error"]),
    }
    finite = all(
        torch.isfinite(value).all().item()
        for value in (*output.values(), object_prior)
        if isinstance(value, torch.Tensor)
    )
    audit = {
        "sample_ids": names[:4],
        "Qa_shape": list(output["Qa"].shape),
        "Qv_shape": list(output["Qv"].shape),
        "K4_shape": list(output["K4"].shape),
        "K34_shape": list(output["K34"].shape),
        "F34_shape": list(output["F34"].shape),
        "AUD_FINE_shape": list(output["AUD_FINE"].shape),
        "IMG_L4_ALL_shape": list(output["IMG_L4_ALL"].shape),
        "AUD_L4_ALL_shape": list(output["AUD_L4_ALL"].shape),
        "ownership_pixel_shape": list(output["OWNERSHIP_PIXEL"].shape),
        "ownership_slot_sum_error": float(
            (output["OWNERSHIP_PIXEL"].sum(dim=1) - 1.0).abs().max()
        ),
        "fused_visual_slots_shape": list(output["FUSED_VISUAL_SLOTS"].shape),
        "audio_slots_shape": list(output["AUDIO_SLOTS"].shape),
        "semantic_support_shape": list(output["SEMANTIC_SUPPORT"].shape),
        "temporal_reciprocity_shape": list(output["TEMPORAL_RECIPROCITY"].shape),
        "reciprocal_support_shape": list(output["RECIPROCAL_SUPPORT"].shape),
        "object_prior_shape": list(object_prior.shape),
        "reconstruction_errors": errors,
        "no_nan_or_inf": finite,
    }
    if max(errors.values()) > 1e-6 or audit["ownership_slot_sum_error"] > 1e-6 or not finite:
        raise RuntimeError(audit)
    audit["passed"] = True
    return audit


def safe_classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        return {"AUROC": math.nan, "AUPRC": math.nan}
    return {
        "AUROC": float(sklearn_metrics.roc_auc_score(labels, scores)),
        "AUPRC": float(sklearn_metrics.average_precision_score(labels, scores)),
    }


def selector_failure(labels: np.ndarray, choices_img: np.ndarray) -> dict[str, int]:
    labels = labels.astype(bool)
    choices_img = choices_img.astype(bool)
    return {
        "true_AUD_choice": int((~labels & ~choices_img).sum()),
        "true_IMG_choice": int((labels & choices_img).sum()),
        "false_AUD_choice": int((labels & ~choices_img).sum()),
        "false_IMG_choice": int((~labels & choices_img).sum()),
    }


def selector_summary(
    aud: list[float],
    img: list[float],
    candidate: list[float],
    deltas: list[float],
    img_only: np.ndarray,
    ogl_rescue: np.ndarray,
    img_rescue: np.ndarray,
    pixel_switch_rates: list[float] | None = None,
    sample_choice_defined: bool = True,
) -> dict[str, Any]:
    aud_array = np.asarray(aud)
    img_array = np.asarray(img)
    candidate_array = np.asarray(candidate)
    delta_array = np.asarray(deltas)
    choices_img = delta_array > 0
    shift = common.transition(aud, candidate)
    success = candidate_array >= 0.5
    return {
        "metrics": common.summarize(candidate),
        "rescue": shift["rescue"],
        "hurt": shift["hurt"],
        "net": shift["net"],
        "IMG_selection_rate": (
            float(choices_img.mean()) if sample_choice_defined else None
        ),
        "IMG_only_total": int(img_only.sum()),
        "IMG_only_captured": int((img_only & success).sum()),
        "IMG_only_capture_rate": float((img_only & success).sum() / max(img_only.sum(), 1)),
        "IMG_rescue_total": int(img_rescue.sum()),
        "IMG_rescue_retained": int((img_rescue & success).sum()),
        "IMG_rescue_retention": float(
            (img_rescue & success).sum() / max(img_rescue.sum(), 1)
        ),
        "OGL_rescue_total": int(ogl_rescue.sum()),
        "OGL_rescue_captured": int((ogl_rescue & success).sum()),
        "OGL_rescue_capture_rate": float((ogl_rescue & success).sum() / max(ogl_rescue.sum(), 1)),
        "failure_decomposition": (
            selector_failure(img_array > aud_array, choices_img)
            if sample_choice_defined
            else None
        ),
        "mean_pixel_switch_rate": (
            float(np.mean(pixel_switch_rates)) if pixel_switch_rates is not None else None
        ),
    }


def optimal_threshold(deltas: list[float], aud: list[float], img: list[float]) -> dict[str, Any]:
    delta = np.asarray(deltas)
    aud_array = np.asarray(aud)
    img_array = np.asarray(img)
    unique = np.unique(delta)
    epsilon = np.finfo(np.float64).eps
    thresholds = np.concatenate(([unique.max() + epsilon], unique))
    rows = []
    for threshold in thresholds:
        selected = np.where(delta > threshold, img_array, aud_array)
        summary = common.summarize(selected.tolist())
        rows.append((float(threshold), summary))
    best_c = max(row[1]["cIoU"] for row in rows)
    candidates = [row for row in rows if row[1]["cIoU"] == best_c]
    threshold, summary = min(candidates, key=lambda row: abs(row[0]))
    return {"threshold": threshold, **summary, "num_thresholds": len(rows)}


def aggregate_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = ("IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE")
    metrics = (
        "disagreement_mean",
        "disagreement_max",
        "disagreement_std",
        "disagreement_top20_mass",
        "AUD_IMG_Pearson",
        "mask_IoU",
        "area_AUD",
        "area_IMG",
        "area_ratio_IMG_AUD",
        "centroid_distance",
    )
    output = []
    for group in groups:
        selected = [row for row in rows if row[f"group_{group}"]]
        output.append(
            {
                "group": group,
                "count": len(selected),
                **{
                    metric: common.distribution([float(row[metric]) for row in selected])
                    for metric in metrics
                },
            }
        )
    return output


def aggregate_boundary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = ("ALL", "IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE")
    output = []
    for group in groups:
        selected = rows if group == "ALL" else [row for row in rows if row[f"group_{group}"]]
        output.append(
            {
                "group": group,
                "count": len(selected),
                **{
                    region: common.distribution([float(row[f"energy_{region}"]) for row in selected])
                    for region in ("GT_INTERIOR", "GT_BOUNDARY_1PX", "GT_EXTERIOR")
                },
            }
        )
    return output


def qualitative_categories(row: dict[str, Any]) -> list[str]:
    aud_success = row["IoU_AUD"] >= 0.5
    img_success = row["IoU_IMG"] >= 0.5
    iqr_success = row["IoU_IQR"] >= 0.5
    ogl_success = row["IoU_OGL"] >= 0.5
    y_img = row["IoU_IMG"] > row["IoU_AUD"]
    choice_img = row["DELTA_SEMANTIC_SLOT"] > 0
    selected_success = row["IoU_SAMPLE_SEMANTIC_SLOT"] >= 0.5
    labels = []
    if not aud_success and img_success:
        labels.append("IMG_ONLY")
    if aud_success and not img_success:
        labels.append("AUD_ONLY")
    if aud_success and not iqr_success:
        labels.append("FIXED_IQR_HURT")
    if not aud_success and iqr_success:
        labels.append("FIXED_IQR_RESCUE")
    if choice_img and y_img:
        labels.append("SELECTOR_CORRECT_IMG")
    if choice_img and not y_img:
        labels.append("SELECTOR_WRONG_IMG")
    if not choice_img and not y_img:
        labels.append("SELECTOR_CORRECT_AUD")
    if not aud_success and ogl_success:
        labels.append("OGL_RESCUE_CAPTURED" if selected_success else "OGL_RESCUE_MISSED")
    return labels


def update_qualitative(
    selected: dict[str, dict[str, Any]],
    sort_key: str,
    categories: list[str],
    image: torch.Tensor,
    gt: np.ndarray,
    maps: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    for category in categories:
        current = selected.get(category)
        if current is not None and current["sort_key"] <= sort_key:
            continue
        rgb = common.inverse_normalize(image.detach().cpu()).permute(1, 2, 0).numpy()
        selected[category] = {
            "sort_key": sort_key,
            "sample_id": row["sample_id"],
            "category": category,
            "image": np.clip(rgb, 0.0, 1.0),
            "GT": gt,
            "AUD": maps["AUD"],
            "IMG": maps["IMG"],
            "DISAGREEMENT": common.normalize_map(np.abs(maps["AUD"] - maps["IMG"])),
            "IQR": maps["IQR"],
            "SELECTED": maps["SAMPLE_SEMANTIC_SLOT"],
            "OGL": maps["OGL"],
            "iou_aud": row["IoU_AUD"],
            "iou_img": row["IoU_IMG"],
            "iou_iqr": row["IoU_IQR"],
            "iou_selected": row["IoU_SAMPLE_SEMANTIC_SLOT"],
            "iou_ogl": row["IoU_OGL"],
            "e_aud": row["E_AUD_SEMANTIC_SLOT"],
            "e_img": row["E_IMG_SEMANTIC_SLOT"],
            "delta": row["DELTA_SEMANTIC_SLOT"],
        }


def reproduce_40(setting: str, rows: list[dict[str, Any]], summaries: dict[str, Any]) -> dict[str, Any]:
    path = common.reference_40_dir(setting) / "per_sample_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    if len(reference) != len(rows):
        raise RuntimeError((len(reference), len(rows)))
    max_error = 0.0
    id_mismatches = 0
    for current, old in zip(rows, reference):
        id_mismatches += int(
            int(current["sample_index"]) != int(old["sample_index"])
            or current["sample_id"] != old["sample_id"]
        )
        for method in BASE_METHODS:
            max_error = max(
                max_error,
                abs(float(current[f"IoU_{method}"]) - float(old[f"Stage2_IoU_{method}"])),
            )
    old_summary = json.loads((common.reference_40_dir(setting) / "summary.json").read_text())
    old_stage = old_summary["stage_summaries"]["Stage2"]
    old_pair = next(row for row in old_stage["pairs"] if row["pair"] == "AUD+IMG")
    checks = {
        "sample_oracle_cIoU": abs(
            summaries["capacity"]["SAMPLE_ORACLE"]["cIoU"]
            - old_pair["pair_oracle"]["cIoU"]
        ),
        "sample_oracle_AUC": abs(
            summaries["capacity"]["SAMPLE_ORACLE"]["AUC"]
            - old_pair["pair_oracle"]["AUC"]
        ),
        "IMG_only": abs(
            summaries["counts"]["IMG_ONLY"]
            - old_pair["success_decomposition"]["AUX_ONLY"]
        ),
        "OGL_rescue_total": abs(
            summaries["counts"]["OGL_RESCUE"]
            - old_stage["OGL_rescue_decomposition"]["OGL_rescue_total"]
        ),
        "IMG_OGL_rescue_capture": abs(
            summaries["counts"]["IMG_CAPTURED_OGL_RESCUE"]
            - old_stage["OGL_rescue_decomposition"]["IMG_captured"]
        ),
    }
    passed = max_error == 0.0 and id_mismatches == 0 and max(checks.values()) == 0
    result = {
        "per_sample_max_error": max_error,
        "sample_order_mismatches": id_mismatches,
        "aggregate_errors": checks,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(result)
    return result


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
    checkpoints_before = common.snapshot_files(
        {
            "formal_stage1": common.stage1_checkpoint_path(registry),
            "formal_original_1_3G": common.g_checkpoint_path(registry),
            "evaluation_only_object_prior": common.OBJECT_CHECKPOINT,
        }
    )
    model = common.load_original_g(registry, device)
    object_model = common.object_prior_model().to(device).eval()
    audit = tensor_audit(loader, model, object_model, device)

    method_ious: dict[str, list[float]] = {name: [] for name in BASE_METHODS}
    method_ious.update({"SAMPLE_ORACLE": [], "PIXEL_ORACLE": []})
    binary_ious = {"BINARY_PIXEL_ORACLE": [], "REGION_ORACLE": []}
    evidence_values = {
        evidence: {
            "E_AUD": [],
            "E_IMG": [],
            "DELTA": [],
            "SAMPLE": [],
            "DISAGREE20": [],
            "DISAGREE20_SWITCH_RATE": [],
        }
        for evidence in EVIDENCES
    }
    for evidence in LOCAL_EVIDENCES:
        evidence_values[evidence]["LOCAL20"] = []
        evidence_values[evidence]["LOCAL20_SWITCH_RATE"] = []

    rows: list[dict[str, Any]] = []
    raw_maps = defaultdict(list)
    qualitative: dict[str, dict[str, Any]] = {}
    correction_rows = []
    no_nan_or_inf = True
    global_index = 0

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader)):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_all(model, image, spec)
        object_prior = object_model(image)
        no_nan_or_inf = no_nan_or_inf and all(
            torch.isfinite(value).all().item()
            for value in (*output.values(), object_prior)
            if isinstance(value, torch.Tensor)
        )

        raw_native = {
            "AUD_FINE": output["AUD_FINE"].cpu().numpy()[:, 0],
            "IMG_L4": output["IMG_L4_ALL"].cpu().numpy()[:, 0],
            "AUD_L4": output["AUD_L4_ALL"].cpu().numpy()[:, 0],
        }
        for key, values in raw_native.items():
            raw_maps[key].extend(values)
        raw_maps["sample_id"].extend(names)

        resized = {
            "AUD": common.resize_tensor(output["AUD_FINE"]).cpu().numpy()[:, 0],
            "IMG": common.resize_tensor(output["IMG_L4_ALL"][:, 0:1]).cpu().numpy()[:, 0],
            "AUD_L4": common.resize_tensor(output["AUD_L4_ALL"][:, 0:1]).cpu().numpy()[:, 0],
            "OBJ": common.resize_tensor(object_prior).cpu().numpy()[:, 0],
            "SEMANTIC": common.resize_tensor(output["SEMANTIC_SUPPORT"]).cpu().numpy()[:, 0],
            "RECIPROCAL": common.resize_tensor(output["RECIPROCAL_SUPPORT"]).cpu().numpy()[:, 0],
        }
        gt_batch = bboxes.cpu().numpy()

        for index, sample_id in enumerate(names):
            raw_aud = resized["AUD"][index]
            raw_img = resized["IMG"][index]
            aud = common.normalize_map(raw_aud)
            img = common.normalize_map(raw_img)
            obj = common.normalize_map(resized["OBJ"][index])
            iqr = common.normalize_map(0.6 * aud + 0.4 * img)
            ogl = common.normalize_map(0.6 * aud + 0.4 * obj)
            gt = gt_batch[index]
            maps = {"AUD": aud, "IMG": img, "IQR": iqr, "OBJ": obj, "OGL": ogl}
            row: dict[str, Any] = {"sample_index": global_index, "sample_id": sample_id}
            for method in BASE_METHODS:
                value = common.sample_iou(maps[method], gt)
                method_ious[method].append(value)
                row[f"IoU_{method}"] = value

            oracles = capacity_oracles(aud, img, gt)
            sample_oracle_iou = max(row["IoU_AUD"], row["IoU_IMG"])
            pixel_iou = common.sample_iou(oracles["SOFT_PIXEL_MAP"], gt)
            binary_pixel_iou = binary_iou(oracles["BINARY_PIXEL_MASK"], gt)
            region_iou = binary_iou(oracles["REGION_MASK"], gt)
            method_ious["SAMPLE_ORACLE"].append(sample_oracle_iou)
            method_ious["PIXEL_ORACLE"].append(pixel_iou)
            binary_ious["BINARY_PIXEL_ORACLE"].append(binary_pixel_iou)
            binary_ious["REGION_ORACLE"].append(region_iou)
            row.update(
                {
                    "IoU_SAMPLE_ORACLE": sample_oracle_iou,
                    "IoU_PIXEL_ORACLE": pixel_iou,
                    "IoU_BINARY_PIXEL_ORACLE": binary_pixel_iou,
                    "IoU_REGION_ORACLE": region_iou,
                    "region_oracle_components": oracles["disagreement_components"],
                }
            )

            disagreement = np.abs(aud - img)
            mask_aud = aud >= 0.6
            mask_img = img >= 0.6
            area_aud = float(mask_aud.mean())
            area_img = float(mask_img.mean())
            centroid_distance = float(np.linalg.norm(centroid(mask_aud) - centroid(mask_img)))
            mask_union = np.logical_or(mask_aud, mask_img).sum()
            mask_overlap = (
                float(np.logical_and(mask_aud, mask_img).sum() / mask_union)
                if mask_union
                else 1.0
            )
            row.update(
                {
                    "disagreement_mean": float(disagreement.mean()),
                    "disagreement_max": float(disagreement.max()),
                    "disagreement_std": float(disagreement.std()),
                    "disagreement_top10_mass": top_mass(disagreement, 0.10),
                    "disagreement_top20_mass": top_mass(disagreement, 0.20),
                    "disagreement_top30_mass": top_mass(disagreement, 0.30),
                    "AUD_IMG_Pearson": common.safe_pearson(aud, img),
                    "mask_IoU": mask_overlap,
                    "area_AUD": area_aud,
                    "area_IMG": area_img,
                    "area_ratio_IMG_AUD": area_img / max(area_aud, EPS),
                    "centroid_distance": centroid_distance,
                    "components_AUD": component_count(mask_aud),
                    "components_IMG": component_count(mask_img),
                    "GT_overlap_AUD": float(np.logical_and(mask_aud, gt >= 0.5).sum()),
                    "GT_overlap_IMG": float(np.logical_and(mask_img, gt >= 0.5).sum()),
                }
            )
            for region, value in boundary_energy(disagreement, gt).items():
                row[f"energy_{region}"] = value

            aud_success = row["IoU_AUD"] >= 0.5
            img_success = row["IoU_IMG"] >= 0.5
            ogl_success = row["IoU_OGL"] >= 0.5
            group_flags = {
                "IMG_ONLY": (not aud_success) and img_success,
                "AUD_ONLY": aud_success and (not img_success),
                "BOTH_SUCCESS": aud_success and img_success,
                "BOTH_FAIL": (not aud_success) and (not img_success),
                "OGL_RESCUE": (not aud_success) and ogl_success,
            }
            for group, flag in group_flags.items():
                row[f"group_{group}"] = flag
            for correction_group in ("IMG_ONLY", "OGL_RESCUE"):
                if not group_flags[correction_group]:
                    continue
                correction_rows.append(
                    {
                        "sample_index": global_index,
                        "sample_id": sample_id,
                        "group": correction_group,
                        "correction_type": correction_type(area_aud, area_img, centroid_distance),
                        "area_AUD": area_aud,
                        "area_IMG": area_img,
                        "centroid_distance": centroid_distance,
                        "components_AUD": row["components_AUD"],
                        "components_IMG": row["components_IMG"],
                        "GT_overlap_AUD": row["GT_overlap_AUD"],
                        "GT_overlap_IMG": row["GT_overlap_IMG"],
                    }
                )

            controls_aud = map_controls(raw_aud, aud)
            controls_img = map_controls(raw_img, img)
            semantic_support = common.normalize_map(resized["SEMANTIC"][index])
            reciprocal_support = common.normalize_map(resized["RECIPROCAL"][index])
            evidence_aud = {**controls_aud}
            evidence_img = {**controls_img}
            evidence_aud["SEMANTIC_SLOT"] = soft_contrast(aud, semantic_support)
            evidence_img["SEMANTIC_SLOT"] = soft_contrast(img, semantic_support)
            evidence_aud["RECIPROCAL_L4"] = soft_contrast(aud, reciprocal_support)
            evidence_img["RECIPROCAL_L4"] = soft_contrast(img, reciprocal_support)
            local_supports = {
                "SEMANTIC_SLOT": semantic_support,
                "RECIPROCAL_L4": reciprocal_support,
            }
            disagree20 = top_fraction_mask(disagreement, 0.20)
            for evidence in EVIDENCES:
                e_aud = evidence_aud[evidence]
                e_img = evidence_img[evidence]
                delta = e_img - e_aud
                sample_selected = img if delta > 0 else aud
                d20_selected = aud.copy()
                if delta > 0:
                    d20_selected[disagree20] = img[disagree20]
                d20_selected = common.normalize_map(d20_selected)
                sample_iou = common.sample_iou(sample_selected, gt)
                d20_iou = common.sample_iou(d20_selected, gt)
                evidence_values[evidence]["E_AUD"].append(e_aud)
                evidence_values[evidence]["E_IMG"].append(e_img)
                evidence_values[evidence]["DELTA"].append(delta)
                evidence_values[evidence]["SAMPLE"].append(sample_iou)
                evidence_values[evidence]["DISAGREE20"].append(d20_iou)
                evidence_values[evidence]["DISAGREE20_SWITCH_RATE"].append(
                    float(disagree20.mean()) if delta > 0 else 0.0
                )
                row[f"E_AUD_{evidence}"] = e_aud
                row[f"E_IMG_{evidence}"] = e_img
                row[f"DELTA_{evidence}"] = delta
                row[f"IoU_SAMPLE_{evidence}"] = sample_iou
                row[f"IoU_D20_{evidence}"] = d20_iou
                if evidence in LOCAL_EVIDENCES:
                    support = local_supports[evidence]
                    local_delta = np.abs(aud - support) - np.abs(img - support)
                    local_selected = aud.copy()
                    use_img_pixel = np.logical_and(disagree20, local_delta > 0)
                    local_selected[use_img_pixel] = img[use_img_pixel]
                    local_selected = common.normalize_map(local_selected)
                    local_iou = common.sample_iou(local_selected, gt)
                    evidence_values[evidence]["LOCAL20"].append(local_iou)
                    evidence_values[evidence]["LOCAL20_SWITCH_RATE"].append(
                        float(use_img_pixel.mean())
                    )
                    row[f"IoU_LOCAL20_{evidence}"] = local_iou

            maps["SAMPLE_SEMANTIC_SLOT"] = img if row["DELTA_SEMANTIC_SLOT"] > 0 else aud
            categories = qualitative_categories(row)
            if not arguments.skip_qualitative:
                update_qualitative(
                    qualitative,
                    f"{sample_id}::{global_index:06d}",
                    categories,
                    image[index],
                    gt,
                    maps,
                    row,
                )
            rows.append(row)
            global_index += 1

    base_metrics = {method: common.summarize(method_ious[method]) for method in BASE_METHODS}
    capacity = {
        "SAMPLE_ORACLE": common.summarize(method_ious["SAMPLE_ORACLE"]),
        "PIXEL_ORACLE": common.summarize(method_ious["PIXEL_ORACLE"]),
        "BINARY_PIXEL_ORACLE": {
            "success_rate": float(np.mean(np.asarray(binary_ious["BINARY_PIXEL_ORACLE"]) >= 0.5)),
            "mean_sample_IoU": float(np.mean(binary_ious["BINARY_PIXEL_ORACLE"])),
            "AUC": None,
        },
        "REGION_ORACLE": {
            "success_rate": float(np.mean(np.asarray(binary_ious["REGION_ORACLE"]) >= 0.5)),
            "mean_sample_IoU": float(np.mean(binary_ious["REGION_ORACLE"])),
            "AUC": None,
        },
    }
    capacity_gaps = {
        "SampleOracle_minus_AUD": capacity["SAMPLE_ORACLE"]["cIoU"] - base_metrics["AUD"]["cIoU"],
        "RegionOracle_minus_SampleOracle": capacity["REGION_ORACLE"]["success_rate"] - capacity["SAMPLE_ORACLE"]["cIoU"],
        "PixelOracle_minus_RegionOracle": capacity["PIXEL_ORACLE"]["cIoU"] - capacity["REGION_ORACLE"]["success_rate"],
        "OGL_minus_SampleOracle": base_metrics["OGL"]["cIoU"] - capacity["SAMPLE_ORACLE"]["cIoU"],
        "OGL_minus_RegionOracle": base_metrics["OGL"]["cIoU"] - capacity["REGION_ORACLE"]["success_rate"],
        "OGL_minus_PixelOracle": base_metrics["OGL"]["cIoU"] - capacity["PIXEL_ORACLE"]["cIoU"],
    }

    aud_array = np.asarray(method_ious["AUD"])
    img_array = np.asarray(method_ious["IMG"])
    ogl_array = np.asarray(method_ious["OGL"])
    labels_better = img_array > aud_array
    img_only = (aud_array < 0.5) & (img_array >= 0.5)
    ogl_rescue = (aud_array < 0.5) & (ogl_array >= 0.5)
    img_rescue = ogl_rescue & (img_array >= 0.5)
    fixed_shift = common.transition(method_ious["AUD"], method_ious["IQR"])
    evidence_summaries = []
    selector_summaries = []
    for evidence in EVIDENCES:
        values = evidence_values[evidence]
        delta = np.asarray(values["DELTA"])
        better_metrics = safe_classification_metrics(labels_better.astype(int), delta)
        img_only_metrics = safe_classification_metrics(img_only.astype(int), delta)
        balanced = float(
            sklearn_metrics.balanced_accuracy_score(labels_better, delta > 0)
        )
        evidence_summaries.append(
            {
                "evidence": evidence,
                "AUROC_IMG_better": better_metrics["AUROC"],
                "AUPRC_IMG_better": better_metrics["AUPRC"],
                "AUROC_IMG_only": img_only_metrics["AUROC"],
                "balanced_accuracy_threshold_0": balanced,
                "positive_rate_IMG_better": float(labels_better.mean()),
                "mean_delta_IMG_better": float(delta[labels_better].mean()),
                "mean_delta_AUD_better_or_tie": float(delta[~labels_better].mean()),
                "optimal_threshold_diagnostic": optimal_threshold(
                    values["DELTA"], method_ious["AUD"], method_ious["IMG"]
                ),
            }
        )
        selector_summaries.append(
            {
                "evidence": evidence,
                "mode": "SAMPLE",
                **selector_summary(
                    method_ious["AUD"], method_ious["IMG"], values["SAMPLE"],
                    values["DELTA"], img_only, ogl_rescue,
                    img_rescue,
                ),
            }
        )
        selector_summaries.append(
            {
                "evidence": evidence,
                "mode": "DISAGREE20_SCALAR",
                **selector_summary(
                    method_ious["AUD"], method_ious["IMG"], values["DISAGREE20"],
                    values["DELTA"], img_only, ogl_rescue,
                    img_rescue,
                    values["DISAGREE20_SWITCH_RATE"],
                ),
            }
        )
        if evidence in LOCAL_EVIDENCES:
            selector_summaries.append(
                {
                    "evidence": evidence,
                    "mode": "DISAGREE20_LOCAL",
                    **selector_summary(
                        method_ious["AUD"], method_ious["IMG"], values["LOCAL20"],
                        values["DELTA"], img_only, ogl_rescue,
                        img_rescue,
                        values["LOCAL20_SWITCH_RATE"],
                        sample_choice_defined=False,
                    ),
                }
            )

    counts = {
        "num_samples": len(rows),
        "IMG_BETTER": int(labels_better.sum()),
        "IMG_ONLY": int(img_only.sum()),
        "AUD_ONLY": int(((aud_array >= 0.5) & (img_array < 0.5)).sum()),
        "OGL_RESCUE": int(ogl_rescue.sum()),
        "IMG_CAPTURED_OGL_RESCUE": int((ogl_rescue & (img_array >= 0.5)).sum()),
    }
    correction_summary = {}
    for group in ("IMG_ONLY", "OGL_RESCUE"):
        selected_rows = [row for row in correction_rows if row["group"] == group]
        correction_summary[group] = {
            "count": len(selected_rows),
            "types": dict(Counter(row["correction_type"] for row in selected_rows)),
            "area_AUD": common.distribution([row["area_AUD"] for row in selected_rows]),
            "area_IMG": common.distribution([row["area_IMG"] for row in selected_rows]),
            "centroid_distance": common.distribution([row["centroid_distance"] for row in selected_rows]),
            "components_AUD": common.distribution([row["components_AUD"] for row in selected_rows]),
            "components_IMG": common.distribution([row["components_IMG"] for row in selected_rows]),
            "GT_overlap_delta_IMG_minus_AUD": common.distribution(
                [row["GT_overlap_IMG"] - row["GT_overlap_AUD"] for row in selected_rows]
            ),
        }

    protocol_summary = {
        "base_metrics": base_metrics,
        "capacity": capacity,
        "capacity_gaps": capacity_gaps,
        "counts": counts,
    }
    reproduction = (
        {"skipped_for_partial_run": True}
        if arguments.max_batches is not None
        else reproduce_40(arguments.experiment, rows, protocol_summary)
    )

    for category in QUALITATIVE_CATEGORIES:
        payload = qualitative.get(category)
        if payload is not None and not arguments.skip_qualitative:
            visualize.save_panel(payload, output_dir / "qualitative" / f"{category}.png")
    if qualitative and not arguments.skip_qualitative:
        common.write_csv(
            output_dir / "qualitative" / "selection_manifest.csv",
            [
                {"category": category, "sample_id": payload["sample_id"], "sort_key": payload["sort_key"]}
                for category, payload in qualitative.items()
            ],
        )

    np.savez_compressed(
        output_dir / "raw_maps.npz",
        sample_id=np.asarray(raw_maps["sample_id"]),
        AUD_FINE=np.asarray(raw_maps["AUD_FINE"], dtype=np.float32),
        IMG_L4=np.asarray(raw_maps["IMG_L4"], dtype=np.float32),
        AUD_L4=np.asarray(raw_maps["AUD_L4"], dtype=np.float32),
    )
    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    correction_path = output_dir / "correction_types.csv"
    if correction_rows:
        common.write_csv(correction_path, correction_rows)
    else:
        correction_path.write_text(
            "sample_index,sample_id,group,correction_type,area_AUD,area_IMG,"
            "centroid_distance,components_AUD,components_IMG,GT_overlap_AUD,"
            "GT_overlap_IMG\n",
            encoding="utf-8",
        )
    common.write_json(output_dir / "evidence_prediction.json", evidence_summaries)
    common.write_json(output_dir / "selector_results.json", selector_summaries)

    checkpoints_after = common.verify_snapshots(checkpoints_before)
    parameters_with_grad = [
        f"model.{name}"
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ] + [
        f"object_model.{name}"
        for name, parameter in object_model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel()
        for module in (model, object_model)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    zero_training = {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": trainable_parameter_count,
        "trainable_parameter_names": parameters_with_grad,
        "parameters_with_grad": parameters_with_grad,
        "all_models_eval": not model.training and not object_model.training,
        "torch_inference_mode": True,
        "checkpoint_snapshots": checkpoints_after,
        "all_checkpoint_hashes_and_mtimes_unchanged": checkpoints_after["all_unchanged"],
        "no_nan_or_inf": no_nan_or_inf,
    }
    if not all(
        (
            zero_training["new_trainable_params"] == 0,
            not zero_training["parameters_with_grad"],
            zero_training["all_models_eval"],
            zero_training["all_checkpoint_hashes_and_mtimes_unchanged"],
            zero_training["no_nan_or_inf"],
        )
    ):
        raise RuntimeError(zero_training)

    metric_space_audit = {
        "explicitly_cross_modal_aligned": (
            "fused visual slots and audio slots; InfoNCE directly compares normalized slot0-to-slot0"
        ),
        "not_legal_for_direct_audio_cosine": (
            "F34, K34, K4, raw visual tokens, and img_to_v outputs are attention/key/value spaces, not InfoNCE metric embeddings"
        ),
        "direct_token_region_semantic_verifier": "N/A: no trained token-level projection into the InfoNCE slot metric space",
        "SEMANTIC_SLOT": (
            "H_sem(x)=sum_s ownership_s(x)*cos(fused_visual_slot_s,audio_slot0); "
            "E(M)=soft_fg_mean(H_sem)-soft_bg_mean(H_sem). No new projection."
        ),
        "RECIPROCAL_L4": (
            "C_av=Qa0->K4. For each visual slot s, r_s=1-JS(Qv_s->Ka, Qa0->Ka)/log(2); "
            "H_va=sum_s ownership_s*r_s; H_recip=0.5*(C_av+H_va); "
            "E(M)=soft_fg_mean(H_recip)-soft_bg_mean(H_recip)."
        ),
        "circularity": {
            "AUD_FINE_regeneration": "not used as a verifier tensor",
            "SEMANTIC_SLOT": "not identical to AUD_FINE; uses InfoNCE slot similarity and visual ownership",
            "RECIPROCAL_L4": (
                "partially circular: C_av is the coarse precursor of AUD; the reciprocal H_va term "
                "uses the separately trained Qv->audio-key versus Qa->audio-key agreement"
            ),
        },
        "local_evidence": (
            "Within DISAGREE20, choose the candidate value closer to H_sem or H_recip; common 80% remains AUD."
        ),
    }
    summary = {
        "experiment": "4.1 Selective Fusion Capacity & Evidence Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": arguments.max_batches is None,
        "tensor_audit": audit,
        "reproduction_4_0": reproduction,
        "base_metrics": base_metrics,
        "capacity": capacity,
        "capacity_gaps": capacity_gaps,
        "counts": counts,
        "disagreement_groups": aggregate_group_rows(rows),
        "boundary_disagreement": aggregate_boundary(rows),
        "correction_types": correction_summary,
        "metric_space_audit": metric_space_audit,
        "evidence_prediction": evidence_summaries,
        "selector_results": selector_summaries,
        "fixed_IQR_transition": {
            "rescue": fixed_shift["rescue"],
            "hurt": fixed_shift["hurt"],
            "net": fixed_shift["net"],
        },
        "qualitative_selection": {
            category: payload["sample_id"] for category, payload in qualitative.items()
        },
        "zero_training_audit": zero_training,
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    run(parse_args())
