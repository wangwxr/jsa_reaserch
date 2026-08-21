#!/usr/bin/env python3
"""Experiment 4.2: zero-training counterfactual reliability probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

import common

visualize = common.load_module("experiment_42_visualize", common.HERE / "visualize.py")


EPS = 1e-8
LOW_SIZE = 14
LOW_PIXELS = LOW_SIZE * LOW_SIZE
TOP_FRACTION = 0.20
TOP_K = math.ceil(TOP_FRACTION * LOW_PIXELS)
INPUT_SIZE = 224
BLUR_KERNEL = (31, 31)
BLUR_SIGMA = (10.0, 10.0)
RANDOM_SEED = 42020
BASE_METHODS = ("AUD", "IMG", "IQR", "OBJ", "OGL")
EVIDENCES = (
    "DELTA_CF_BLUR",
    "DELTA_CF_MEAN",
    "DELTA_KEEP_BLUR",
    "DELTA_KEEP_MEAN",
    "DELTA_DROP_BLUR",
    "DELTA_DROP_MEAN",
)
QUALITATIVE_CATEGORIES = (
    "IMG_ONLY_CF_CORRECT",
    "IMG_ONLY_CF_FAIL",
    "AUD_ONLY_CF_CORRECT",
    "AUD_ONLY_CF_FAIL",
    "SHRINK_SUCCESS",
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
    parser.add_argument("--cf-microbatch", type=int, default=256)
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def tensor_minmax(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(start_dim=1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    span = maximum - minimum
    return torch.where(span > 0, (value - minimum) / span, value)


def topk_mask(value: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
    flat = value.flatten(start_dim=1)
    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    selected = order[:, :k]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask.reshape(value.shape)


def deterministic_random_mask(names: list[str], device: torch.device) -> torch.Tensor:
    masks = []
    for name in names:
        digest = hashlib.sha256(f"{RANDOM_SEED}:{name}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        indices = rng.choice(LOW_PIXELS, size=TOP_K, replace=False)
        mask = np.zeros(LOW_PIXELS, dtype=np.bool_)
        mask[indices] = True
        masks.append(mask.reshape(1, LOW_SIZE, LOW_SIZE))
    return torch.from_numpy(np.stack(masks)).to(device=device)


def input_mask(mask14: torch.Tensor) -> torch.Tensor:
    return F.interpolate(mask14.float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")


def make_intervention(
    image: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor, keep: bool
) -> torch.Tensor:
    if keep:
        return image * mask + baseline * (1.0 - mask)
    return baseline * mask + image * (1.0 - mask)


def perturbation(original: torch.Tensor, variant: torch.Tensor) -> dict[str, torch.Tensor]:
    difference = variant - original
    flat = difference.flatten(start_dim=1)
    return {
        "mean_abs": flat.abs().mean(dim=1),
        "L1": flat.abs().sum(dim=1),
        "L2": flat.square().sum(dim=1).sqrt(),
    }


@torch.inference_mode()
def visual_target_representation(model, image: torch.Tensor) -> torch.Tensor:
    teacher = model.teacher
    image_levels = teacher.imgnet(image)
    model.feature_hooks.pop()
    initial_slots = teacher.slot_attn.slots.expand(image.shape[0], -1, -1)
    visual_slots = []
    for branch, tokens in zip(teacher.slot_attn.visual_branches, image_levels):
        slots, _query, _keys = branch(tokens, initial_slots)
        visual_slots.append(slots)
    fused = teacher.slot_attn.slot_fusion(visual_slots)
    return F.normalize(fused[:, 0], dim=-1)


@torch.inference_mode()
def intervention_scores(
    model,
    image: torch.Tensor,
    audio_target: torch.Tensor,
    baseline: torch.Tensor,
    masks: dict[str, torch.Tensor],
    microbatch: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    specifications = (
        ("KEEP_A", "A", True),
        ("KEEP_I", "I", True),
        ("REMOVE_A", "A", False),
        ("REMOVE_I", "I", False),
        ("REMOVE_A_EXTRA", "A_EXTRA", False),
        ("REMOVE_I_EXTRA", "I_EXTRA", False),
        ("KEEP_RANDOM", "RANDOM", True),
        ("REMOVE_RANDOM", "RANDOM", False),
    )
    variants = []
    perturbations: dict[str, dict[str, torch.Tensor]] = {}
    for name, mask_name, keep in specifications:
        variant = make_intervention(image, baseline, masks[mask_name], keep)
        variants.append(variant)
        perturbations[name] = perturbation(image, variant)
    stacked = torch.stack(variants, dim=0)
    variant_count, batch, channels, height, width = stacked.shape
    flattened = stacked.reshape(variant_count * batch, channels, height, width)
    repeated_audio = (
        audio_target.unsqueeze(0)
        .expand(variant_count, batch, -1)
        .reshape(variant_count * batch, -1)
    )
    score_chunks = []
    for start in range(0, flattened.shape[0], microbatch):
        stop = min(start + microbatch, flattened.shape[0])
        visual_target = visual_target_representation(model, flattened[start:stop])
        score_chunks.append((visual_target * repeated_audio[start:stop]).sum(dim=-1))
    score_matrix = torch.cat(score_chunks).reshape(variant_count, batch)
    scores = {
        specification[0]: score_matrix[index]
        for index, specification in enumerate(specifications)
    }
    return scores, perturbations


@torch.inference_mode()
def extract_original(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = model.teacher
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    attentions = teacher.slot_attn._l4_attentions(
        encoded, scale_multiplier=teacher.infer_sharpening
    )
    fused_visual_slots = teacher.slot_attn.slot_fusion(encoded["visual_slots"])
    visual_target = F.normalize(fused_visual_slots[:, 0], dim=-1)
    audio_target = F.normalize(encoded["audio_slots"][:, 0], dim=-1)
    semantic_score = (visual_target * audio_target).sum(dim=-1)

    f34, _f3, _f4_up, _delta = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    aud_fine_all = teacher.slot_attn._attention(
        encoded["audio_query"], k34, teacher.infer_sharpening
    )
    batch = image.shape[0]
    return {
        "AUD_FINE": aud_fine_all[:, 0].reshape(batch, 1, 14, 14),
        "IMG_L4": attentions["imgq_imgk_attn"][:, 0].reshape(batch, 1, 7, 7),
        "AUD_L4": attentions["audq_imgk_attn"][:, 0].reshape(batch, 1, 7, 7),
        "VISUAL_TARGET": visual_target,
        "AUDIO_TARGET": audio_target,
        "S_ORIGINAL": semantic_score,
        "K34": k34,
        "F34": f34,
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
    local = extract_original(model, image, spec)
    local_visual = visual_target_representation(model, image)
    official_img, _official_aud_l4 = model.teacher(image, spec)
    model.feature_hooks.pop()
    official_aud = model(image, spec)["AUD_FINE"]
    object_prior = object_model(image)
    errors = {
        "AUD_FINE": float((local["AUD_FINE"] - official_aud).abs().max()),
        "IMG_QUERY": float((local["IMG_L4"] - official_img).abs().max()),
        "visual_target_reconstruction": float(
            (local["VISUAL_TARGET"] - local_visual).abs().max()
        ),
        "f4_tokens": float(local["f4_token_error"]),
    }
    finite = all(
        torch.isfinite(value).all().item()
        for value in (*local.values(), object_prior)
        if isinstance(value, torch.Tensor)
    )
    result = {
        "sample_ids": names[:4],
        "AUD_FINE_shape": list(local["AUD_FINE"].shape),
        "IMG_L4_shape": list(local["IMG_L4"].shape),
        "F34_shape": list(local["F34"].shape),
        "K34_shape": list(local["K34"].shape),
        "visual_target_shape": list(local["VISUAL_TARGET"].shape),
        "audio_target_shape": list(local["AUDIO_TARGET"].shape),
        "semantic_score_shape": list(local["S_ORIGINAL"].shape),
        "reconstruction_errors": errors,
        "no_nan_or_inf": finite,
    }
    if max(errors.values()) > 1e-6 or not finite:
        raise RuntimeError(result)
    result["passed"] = True
    return result


def binary_centroid(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return np.asarray([111.5, 111.5], dtype=np.float64)
    return coordinates.mean(axis=0)


def correction_type(aud: np.ndarray, img: np.ndarray) -> str:
    mask_a = aud >= 0.6
    mask_i = img >= 0.6
    area_a = float(mask_a.mean())
    area_i = float(mask_i.mean())
    if area_i < 0.9 * area_a:
        return "SHRINK"
    if area_i > 1.1 * area_a:
        return "EXPAND"
    distance = float(np.linalg.norm(binary_centroid(mask_a) - binary_centroid(mask_i)))
    if distance > 0.10 * math.sqrt(224**2 + 224**2):
        return "RELOCATE"
    return "MIXED"


def safe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size < 2:
        return {"AUROC": math.nan, "AUPRC": math.nan}
    return {
        "AUROC": float(sklearn_metrics.roc_auc_score(labels, scores)),
        "AUPRC": float(sklearn_metrics.average_precision_score(labels, scores)),
    }


def balanced_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    return float(sklearn_metrics.balanced_accuracy_score(labels, prediction))


def selector_failure(labels: np.ndarray, choices_img: np.ndarray) -> dict[str, int]:
    labels = labels.astype(bool)
    choices_img = choices_img.astype(bool)
    return {
        "true_AUD_choice": int((~labels & ~choices_img).sum()),
        "true_IMG_choice": int((labels & choices_img).sum()),
        "false_AUD_choice": int((labels & ~choices_img).sum()),
        "false_IMG_choice": int((~labels & choices_img).sum()),
    }


def optimal_threshold(delta: np.ndarray, aud: np.ndarray, img: np.ndarray) -> dict[str, Any]:
    unique = np.unique(delta)
    thresholds = np.concatenate(([unique.max() + np.finfo(np.float64).eps], unique))
    candidates = []
    for threshold in thresholds:
        metrics = common.summarize(np.where(delta > threshold, img, aud).tolist())
        candidates.append((float(threshold), metrics))
    best = max(row[1]["cIoU"] for row in candidates)
    threshold, metrics = min(
        (row for row in candidates if row[1]["cIoU"] == best),
        key=lambda row: abs(row[0]),
    )
    return {"threshold": threshold, **metrics, "num_thresholds": len(candidates)}


def distribution_from_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return common.distribution([float(row[key]) for row in rows])


def group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = ("IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE")
    output = []
    for group in groups:
        selected = [row for row in rows if row[f"group_{group}"]]
        output.append(
            {
                "group": group,
                "count": len(selected),
                "DELTA_CF_BLUR": distribution_from_rows(selected, "DELTA_CF_BLUR"),
                "DELTA_CF_MEAN": distribution_from_rows(selected, "DELTA_CF_MEAN"),
                "fraction_BLUR_IMG": float(
                    np.mean([row["DELTA_CF_BLUR"] > 0 for row in selected])
                ) if selected else math.nan,
                "fraction_MEAN_IMG": float(
                    np.mean([row["DELTA_CF_MEAN"] > 0 for row in selected])
                ) if selected else math.nan,
                "A20_I20_overlap": distribution_from_rows(selected, "A20_I20_IoU"),
                "drop_A_BLUR": distribution_from_rows(selected, "DROP_A_BLUR"),
                "drop_I_BLUR": distribution_from_rows(selected, "DROP_I_BLUR"),
                "drop_A_MEAN": distribution_from_rows(selected, "DROP_A_MEAN"),
                "drop_I_MEAN": distribution_from_rows(selected, "DROP_I_MEAN"),
                "drop_A_EXTRA_BLUR": distribution_from_rows(selected, "DROP_A_EXTRA_BLUR"),
                "drop_density_A_EXTRA_BLUR": distribution_from_rows(
                    selected, "DROP_DENSITY_A_BLUR"
                ),
                "drop_A_EXTRA_MEAN": distribution_from_rows(selected, "DROP_A_EXTRA_MEAN"),
                "drop_density_A_EXTRA_MEAN": distribution_from_rows(
                    selected, "DROP_DENSITY_A_MEAN"
                ),
            }
        )
    return output


def correction_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["Y_IMG_BETTER"]]
    output: dict[str, Any] = {"counts": dict(Counter(row["correction_type"] for row in selected))}
    for correction in ("SHRINK", "EXPAND", "RELOCATE", "MIXED"):
        subset = [row for row in selected if row["correction_type"] == correction]
        output[correction] = {
            "count": len(subset),
            "DELTA_CF_BLUR": distribution_from_rows(subset, "DELTA_CF_BLUR"),
            "DELTA_CF_MEAN": distribution_from_rows(subset, "DELTA_CF_MEAN"),
            "fraction_BLUR_IMG": float(
                np.mean([row["DELTA_CF_BLUR"] > 0 for row in subset])
            ) if subset else math.nan,
            "fraction_MEAN_IMG": float(
                np.mean([row["DELTA_CF_MEAN"] > 0 for row in subset])
            ) if subset else math.nan,
        }
    img_only_shrink = [
        row for row in rows
        if row["group_IMG_ONLY"] and row["correction_type"] == "SHRINK"
    ]
    output["IMG_ONLY_SHRINK_signed_disagreement"] = {
        "count": len(img_only_shrink),
        "AUD_extra_exterior_fraction": distribution_from_rows(
            img_only_shrink, "AUD_EXTRA_EXTERIOR_FRACTION"
        ),
        "DROP_A_EXTRA_BLUR": distribution_from_rows(img_only_shrink, "DROP_A_EXTRA_BLUR"),
        "DROP_DENSITY_A_BLUR": distribution_from_rows(
            img_only_shrink, "DROP_DENSITY_A_BLUR"
        ),
        "DROP_A_EXTRA_MEAN": distribution_from_rows(img_only_shrink, "DROP_A_EXTRA_MEAN"),
        "DROP_DENSITY_A_MEAN": distribution_from_rows(
            img_only_shrink, "DROP_DENSITY_A_MEAN"
        ),
    }
    return output


def evidence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels_better = np.asarray([row["Y_IMG_BETTER"] for row in rows], dtype=bool)
    labels_only = np.asarray([row["group_IMG_ONLY"] for row in rows], dtype=bool)
    output = []
    for evidence in EVIDENCES:
        values = np.asarray([row[evidence] for row in rows], dtype=np.float64)
        task = safe_metrics(labels_better.astype(int), values)
        only = safe_metrics(labels_only.astype(int), values)
        output.append(
            {
                "evidence": evidence,
                "AUROC_IMG_better": task["AUROC"],
                "AUPRC_IMG_better": task["AUPRC"],
                "AUROC_IMG_only": only["AUROC"],
                "AUPRC_IMG_only": only["AUPRC"],
                "balanced_accuracy_threshold_0": balanced_accuracy(
                    labels_better, values > 0
                ),
                "fraction_delta_positive": float((values > 0).mean()),
                "mean_IMG_better": float(values[labels_better].mean()),
                "mean_AUD_better_or_tie": float(values[~labels_better].mean()),
                "optimal_threshold_diagnostic": optimal_threshold(
                    values,
                    np.asarray([row["IoU_AUD"] for row in rows]),
                    np.asarray([row["IoU_IMG"] for row in rows]),
                ),
            }
        )
    return output


def selector_summary(
    rows: list[dict[str, Any]], name: str, choices_img: np.ndarray
) -> dict[str, Any]:
    aud = np.asarray([row["IoU_AUD"] for row in rows])
    img = np.asarray([row["IoU_IMG"] for row in rows])
    ogl = np.asarray([row["IoU_OGL"] for row in rows])
    selected = np.where(choices_img, img, aud)
    transition = common.transition(aud.tolist(), selected.tolist())
    img_only = (aud < 0.5) & (img >= 0.5)
    ogl_rescue = (aud < 0.5) & (ogl >= 0.5)
    img_rescue = ogl_rescue & (img >= 0.5)
    success = selected >= 0.5
    aud_only = (aud >= 0.5) & (img < 0.5)
    return {
        "method": name,
        "metrics": common.summarize(selected.tolist()),
        "rescue": transition["rescue"],
        "hurt": transition["hurt"],
        "net": transition["net"],
        "IMG_selection_rate": float(choices_img.mean()),
        "IMG_rescue_total": int(img_rescue.sum()),
        "IMG_rescue_retained": int((img_rescue & choices_img & success).sum()),
        "IMG_rescue_retention": float(
            (img_rescue & choices_img & success).sum() / max(img_rescue.sum(), 1)
        ),
        "OGL_rescue_total": int(ogl_rescue.sum()),
        "OGL_rescue_captured": int((ogl_rescue & success).sum()),
        "OGL_rescue_capture_rate": float(
            (ogl_rescue & success).sum() / max(ogl_rescue.sum(), 1)
        ),
        "failure_decomposition": selector_failure(img > aud, choices_img),
        "IMG_ONLY_wrong_AUD": int((img_only & ~choices_img).sum()),
        "AUD_ONLY_wrong_IMG": int((aud_only & choices_img).sum()),
    }


def qualitative_categories(row: dict[str, Any]) -> list[str]:
    choice_img = row["DELTA_CF_BLUR"] > 0
    labels = []
    if row["group_IMG_ONLY"]:
        labels.append("IMG_ONLY_CF_CORRECT" if choice_img else "IMG_ONLY_CF_FAIL")
    if row["group_AUD_ONLY"]:
        labels.append("AUD_ONLY_CF_FAIL" if choice_img else "AUD_ONLY_CF_CORRECT")
    if row["Y_IMG_BETTER"] and row["correction_type"] == "SHRINK" and choice_img:
        labels.append("SHRINK_SUCCESS")
    if row["group_OGL_RESCUE"]:
        labels.append("OGL_RESCUE_CAPTURED" if row["IoU_CF_BLUR"] >= 0.5 else "OGL_RESCUE_MISSED")
    return labels


def display_image(value: torch.Tensor) -> np.ndarray:
    return np.clip(
        common.inverse_normalize(value.detach().cpu()).permute(1, 2, 0).numpy(),
        0.0,
        1.0,
    )


def update_qualitative(
    selected: dict[str, dict[str, Any]],
    categories: list[str],
    sort_key: str,
    sample_id: str,
    image: torch.Tensor,
    blurred: torch.Tensor,
    mask_a: torch.Tensor,
    mask_i: torch.Tensor,
    gt: np.ndarray,
    maps: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    for category in categories:
        current = selected.get(category)
        if current is not None and current["sort_key"] <= sort_key:
            continue
        keep_a = make_intervention(image[None], blurred[None], mask_a[None], True)[0]
        keep_i = make_intervention(image[None], blurred[None], mask_i[None], True)[0]
        remove_a = make_intervention(image[None], blurred[None], mask_a[None], False)[0]
        remove_i = make_intervention(image[None], blurred[None], mask_i[None], False)[0]
        choice_img = row["DELTA_CF_BLUR"] > 0
        selected[category] = {
            "sort_key": sort_key,
            "sample_id": sample_id,
            "category": category,
            "image": display_image(image),
            "GT": gt,
            "AUD": maps["AUD"],
            "IMG": maps["IMG"],
            "DISAGREEMENT": common.normalize_map(np.abs(maps["AUD"] - maps["IMG"])),
            "MASK_A20": mask_a[0].detach().cpu().numpy(),
            "MASK_I20": mask_i[0].detach().cpu().numpy(),
            "KEEP_A_BLUR": display_image(keep_a),
            "KEEP_I_BLUR": display_image(keep_i),
            "REMOVE_A_BLUR": display_image(remove_a),
            "REMOVE_I_BLUR": display_image(remove_i),
            "SELECTED": maps["IMG"] if choice_img else maps["AUD"],
            "OGL": maps["OGL"],
            "iou_aud": row["IoU_AUD"],
            "iou_img": row["IoU_IMG"],
            "iou_selected": row["IoU_CF_BLUR"],
            "iou_ogl": row["IoU_OGL"],
            "S_original": row["S_ORIGINAL"],
            "S_keep_A": row["S_KEEP_A_BLUR"],
            "S_keep_I": row["S_KEEP_I_BLUR"],
            "S_remove_A": row["S_REMOVE_A_BLUR"],
            "S_remove_I": row["S_REMOVE_I_BLUR"],
            "CF_A": row["CF_A_BLUR"],
            "CF_I": row["CF_I_BLUR"],
            "DELTA_CF": row["DELTA_CF_BLUR"],
            "true_branch": "IMG" if row["Y_IMG_BETTER"] else "AUD",
            "selected_branch": "IMG" if choice_img else "AUD",
        }


def reproduce_41(
    setting: str,
    rows: list[dict[str, Any]],
    raw_maps: dict[str, list[np.ndarray]],
    base_metrics: dict[str, Any],
    sample_oracle: dict[str, Any],
) -> dict[str, Any]:
    reference_dir = common.reference_41_dir(setting)
    with (reference_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    if len(reference_rows) != len(rows):
        raise RuntimeError((len(reference_rows), len(rows)))
    metric_error = 0.0
    order_mismatches = 0
    for current, reference in zip(rows, reference_rows):
        order_mismatches += int(
            int(current["sample_index"]) != int(reference["sample_index"])
            or current["sample_id"] != reference["sample_id"]
        )
        for method in BASE_METHODS:
            metric_error = max(
                metric_error,
                abs(float(current[f"IoU_{method}"]) - float(reference[f"IoU_{method}"])),
            )
    with np.load(reference_dir / "raw_maps.npz") as reference_maps:
        raw_errors = {
            "AUD_FINE": float(
                np.max(np.abs(np.asarray(raw_maps["AUD_FINE"]) - reference_maps["AUD_FINE"]))
            ),
            "IMG_L4": float(
                np.max(np.abs(np.asarray(raw_maps["IMG_L4"]) - reference_maps["IMG_L4"]))
            ),
            "AUD_L4": float(
                np.max(np.abs(np.asarray(raw_maps["AUD_L4"]) - reference_maps["AUD_L4"]))
            ),
        }
    reference_summary = json.loads((reference_dir / "summary.json").read_text())
    aggregate_errors = {
        method: max(
            abs(base_metrics[method][metric] - reference_summary["base_metrics"][method][metric])
            for metric in ("cIoU", "AUC")
        )
        for method in BASE_METHODS
    }
    aggregate_errors["SAMPLE_ORACLE"] = max(
        abs(sample_oracle[metric] - reference_summary["capacity"]["SAMPLE_ORACLE"][metric])
        for metric in ("cIoU", "AUC")
    )
    count_errors = {
        "IMG_ONLY": abs(
            sum(row["group_IMG_ONLY"] for row in rows)
            - reference_summary["counts"]["IMG_ONLY"]
        ),
        "AUD_ONLY": abs(
            sum(row["group_AUD_ONLY"] for row in rows)
            - reference_summary["counts"]["AUD_ONLY"]
        ),
        "OGL_RESCUE": abs(
            sum(row["group_OGL_RESCUE"] for row in rows)
            - reference_summary["counts"]["OGL_RESCUE"]
        ),
    }
    passed = (
        metric_error == 0.0
        and order_mismatches == 0
        and max(raw_errors.values()) == 0.0
        and max(aggregate_errors.values()) == 0.0
        and max(count_errors.values()) == 0
    )
    result = {
        "per_sample_metric_max_error": metric_error,
        "raw_tensor_max_errors": raw_errors,
        "sample_order_mismatches": order_mismatches,
        "aggregate_errors": aggregate_errors,
        "count_errors": count_errors,
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

    rows: list[dict[str, Any]] = []
    raw_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    mask_archive: dict[str, list[np.ndarray]] = defaultdict(list)
    qualitative: dict[str, dict[str, Any]] = {}
    no_nan_or_inf = True
    global_index = 0

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader)):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_original(model, image, spec)
        object_prior = object_model(image)

        img14_raw = F.interpolate(
            output["IMG_L4"], size=(14, 14), mode="bicubic", align_corners=False
        )
        aud14 = tensor_minmax(output["AUD_FINE"])
        img14 = tensor_minmax(img14_raw)
        disagreement14 = (aud14 - img14).abs()
        mask_a14 = topk_mask(aud14)
        mask_i14 = topk_mask(img14)
        mask_d14 = topk_mask(disagreement14)
        mask_a_extra14 = mask_d14 & (aud14 > img14)
        mask_i_extra14 = mask_d14 & (img14 > aud14)
        mask_random14 = deterministic_random_mask(names, device)
        masks = {
            "A": input_mask(mask_a14),
            "I": input_mask(mask_i14),
            "A_EXTRA": input_mask(mask_a_extra14),
            "I_EXTRA": input_mask(mask_i_extra14),
            "RANDOM": input_mask(mask_random14),
        }
        blurred = gaussian_blur(image, kernel_size=list(BLUR_KERNEL), sigma=list(BLUR_SIGMA))
        mean_fill = torch.zeros_like(image)
        blur_scores, blur_perturb = intervention_scores(
            model, image, output["AUDIO_TARGET"], blurred, masks, arguments.cf_microbatch
        )
        mean_scores, mean_perturb = intervention_scores(
            model, image, output["AUDIO_TARGET"], mean_fill, masks, arguments.cf_microbatch
        )

        tensors = (
            *output.values(),
            object_prior,
            aud14,
            img14,
            disagreement14,
            *blur_scores.values(),
            *mean_scores.values(),
        )
        no_nan_or_inf = no_nan_or_inf and all(
            torch.isfinite(value).all().item()
            for value in tensors
            if isinstance(value, torch.Tensor)
        )

        raw_native = {
            "AUD_FINE": output["AUD_FINE"].cpu().numpy()[:, 0],
            "IMG_L4": output["IMG_L4"].cpu().numpy()[:, 0],
            "AUD_L4": output["AUD_L4"].cpu().numpy()[:, 0],
        }
        for key, values in raw_native.items():
            raw_maps[key].extend(values)
        raw_maps["sample_id"].extend(names)
        for key, value in (
            ("MASK_A20", mask_a14),
            ("MASK_I20", mask_i14),
            ("D20", mask_d14),
            ("R_A_PLUS", mask_a_extra14),
            ("R_I_PLUS", mask_i_extra14),
            ("RANDOM20", mask_random14),
        ):
            mask_archive[key].extend(value.cpu().numpy()[:, 0])
        mask_archive["sample_id"].extend(names)

        resized = {
            "AUD": common.resize_tensor(output["AUD_FINE"]).cpu().numpy()[:, 0],
            "IMG": common.resize_tensor(output["IMG_L4"]).cpu().numpy()[:, 0],
            "OBJ": common.resize_tensor(object_prior).cpu().numpy()[:, 0],
        }
        gt_batch = bboxes.cpu().numpy()
        gt14 = F.interpolate(
            bboxes[:, None].to(device).float(), size=(14, 14), mode="nearest"
        )[:, 0] >= 0.5

        score_cpu = {
            "S_ORIGINAL": output["S_ORIGINAL"].cpu().numpy(),
            **{f"{key}_BLUR": value.cpu().numpy() for key, value in blur_scores.items()},
            **{f"{key}_MEAN": value.cpu().numpy() for key, value in mean_scores.items()},
        }
        perturb_cpu: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for baseline_name, values in (("BLUR", blur_perturb), ("MEAN", mean_perturb)):
            perturb_cpu[baseline_name] = {
                variant: {metric: tensor.cpu().numpy() for metric, tensor in metrics.items()}
                for variant, metrics in values.items()
            }

        for index, sample_id in enumerate(names):
            aud = common.normalize_map(resized["AUD"][index])
            img = common.normalize_map(resized["IMG"][index])
            obj = common.normalize_map(resized["OBJ"][index])
            iqr = common.normalize_map(0.6 * aud + 0.4 * img)
            ogl = common.normalize_map(0.6 * aud + 0.4 * obj)
            gt = gt_batch[index]
            maps = {"AUD": aud, "IMG": img, "IQR": iqr, "OBJ": obj, "OGL": ogl}
            row: dict[str, Any] = {"sample_index": global_index, "sample_id": sample_id}
            for method in BASE_METHODS:
                row[f"IoU_{method}"] = common.sample_iou(maps[method], gt)

            aud_success = row["IoU_AUD"] >= 0.5
            img_success = row["IoU_IMG"] >= 0.5
            ogl_success = row["IoU_OGL"] >= 0.5
            row.update(
                {
                    "Y_IMG_BETTER": row["IoU_IMG"] > row["IoU_AUD"],
                    "group_IMG_ONLY": (not aud_success) and img_success,
                    "group_AUD_ONLY": aud_success and (not img_success),
                    "group_BOTH_SUCCESS": aud_success and img_success,
                    "group_BOTH_FAIL": (not aud_success) and (not img_success),
                    "group_OGL_RESCUE": (not aud_success) and ogl_success,
                    "correction_type": correction_type(aud, img),
                }
            )

            row["S_ORIGINAL"] = float(score_cpu["S_ORIGINAL"][index])
            for baseline in ("BLUR", "MEAN"):
                for name in (
                    "KEEP_A", "KEEP_I", "REMOVE_A", "REMOVE_I",
                    "REMOVE_A_EXTRA", "REMOVE_I_EXTRA", "KEEP_RANDOM", "REMOVE_RANDOM",
                ):
                    row[f"S_{name}_{baseline}"] = float(score_cpu[f"{name}_{baseline}"][index])
                row[f"DROP_A_{baseline}"] = row["S_ORIGINAL"] - row[f"S_REMOVE_A_{baseline}"]
                row[f"DROP_I_{baseline}"] = row["S_ORIGINAL"] - row[f"S_REMOVE_I_{baseline}"]
                row[f"CF_A_{baseline}"] = row[f"S_KEEP_A_{baseline}"] - row[f"S_REMOVE_A_{baseline}"]
                row[f"CF_I_{baseline}"] = row[f"S_KEEP_I_{baseline}"] - row[f"S_REMOVE_I_{baseline}"]
                row[f"DELTA_CF_{baseline}"] = row[f"CF_I_{baseline}"] - row[f"CF_A_{baseline}"]
                row[f"DELTA_KEEP_{baseline}"] = row[f"S_KEEP_I_{baseline}"] - row[f"S_KEEP_A_{baseline}"]
                row[f"DELTA_DROP_{baseline}"] = row[f"DROP_I_{baseline}"] - row[f"DROP_A_{baseline}"]
                row[f"DROP_A_EXTRA_{baseline}"] = row["S_ORIGINAL"] - row[f"S_REMOVE_A_EXTRA_{baseline}"]
                row[f"DROP_I_EXTRA_{baseline}"] = row["S_ORIGINAL"] - row[f"S_REMOVE_I_EXTRA_{baseline}"]
                area_a_extra = int(mask_a_extra14[index].sum().item())
                area_i_extra = int(mask_i_extra14[index].sum().item())
                row[f"DROP_DENSITY_A_{baseline}"] = (
                    row[f"DROP_A_EXTRA_{baseline}"] / area_a_extra
                    if area_a_extra > 0
                    else 0.0
                )
                row[f"DROP_DENSITY_I_{baseline}"] = (
                    row[f"DROP_I_EXTRA_{baseline}"] / area_i_extra
                    if area_i_extra > 0
                    else 0.0
                )
                row[f"CF_RANDOM_{baseline}"] = row[f"S_KEEP_RANDOM_{baseline}"] - row[f"S_REMOVE_RANDOM_{baseline}"]
                for variant in ("KEEP_A", "KEEP_I", "REMOVE_A", "REMOVE_I"):
                    for metric in ("mean_abs", "L1", "L2"):
                        row[f"PERTURB_{baseline}_{variant}_{metric}"] = float(
                            perturb_cpu[baseline][variant][metric][index]
                        )

            mask_a_np = mask_a14[index, 0].cpu().numpy()
            mask_i_np = mask_i14[index, 0].cpu().numpy()
            intersection = np.logical_and(mask_a_np, mask_i_np).sum()
            union = np.logical_or(mask_a_np, mask_i_np).sum()
            area_a_extra = int(mask_a_extra14[index].sum().item())
            area_i_extra = int(mask_i_extra14[index].sum().item())
            aud_extra_exterior = np.logical_and(
                mask_a_extra14[index, 0].cpu().numpy(),
                ~gt14[index].cpu().numpy(),
            ).sum()
            row.update(
                {
                    "A20_area_14": int(mask_a14[index].sum().item()),
                    "I20_area_14": int(mask_i14[index].sum().item()),
                    "A20_area_input": int(masks["A"][index].sum().item()),
                    "I20_area_input": int(masks["I"][index].sum().item()),
                    "A20_I20_IoU": float(intersection / max(union, 1)),
                    "R_A_PLUS_area": area_a_extra,
                    "R_I_PLUS_area": area_i_extra,
                    "AUD_EXTRA_EXTERIOR_AREA": int(aud_extra_exterior),
                    "AUD_EXTRA_EXTERIOR_FRACTION": float(
                        aud_extra_exterior / max(area_a_extra, 1)
                    ),
                }
            )

            choice_blur = row["DELTA_CF_BLUR"] > 0
            choice_mean = row["DELTA_CF_MEAN"] > 0
            choice_consensus = choice_blur and choice_mean
            row["IoU_CF_BLUR"] = row["IoU_IMG"] if choice_blur else row["IoU_AUD"]
            row["IoU_CF_MEAN"] = row["IoU_IMG"] if choice_mean else row["IoU_AUD"]
            row["IoU_CF_CONSENSUS"] = row["IoU_IMG"] if choice_consensus else row["IoU_AUD"]

            if not arguments.skip_qualitative:
                update_qualitative(
                    qualitative,
                    qualitative_categories(row),
                    f"{sample_id}::{global_index:06d}",
                    sample_id,
                    image[index],
                    blurred[index],
                    masks["A"][index],
                    masks["I"][index],
                    gt,
                    maps,
                    row,
                )
            rows.append(row)
            global_index += 1

    base_metrics = {
        method: common.summarize([row[f"IoU_{method}"] for row in rows])
        for method in BASE_METHODS
    }
    sample_oracle = common.summarize(
        [max(row["IoU_AUD"], row["IoU_IMG"]) for row in rows]
    )
    fixed_transition = common.transition(
        [row["IoU_AUD"] for row in rows], [row["IoU_IQR"] for row in rows]
    )
    evidence = evidence_summary(rows)
    choices_blur = np.asarray([row["DELTA_CF_BLUR"] > 0 for row in rows])
    choices_mean = np.asarray([row["DELTA_CF_MEAN"] > 0 for row in rows])
    selector_results = [
        selector_summary(rows, "CF_BLUR", choices_blur),
        selector_summary(rows, "CF_MEAN", choices_mean),
        selector_summary(rows, "CF_CONSENSUS", choices_blur & choices_mean),
    ]

    reproduction = (
        {"skipped_for_partial_run": True}
        if arguments.max_batches is not None
        else reproduce_41(
            arguments.experiment, rows, raw_maps, base_metrics, sample_oracle
        )
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

    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    np.savez_compressed(
        output_dir / "candidate_masks.npz",
        **{
            key: np.asarray(value, dtype=np.bool_) if key != "sample_id" else np.asarray(value)
            for key, value in mask_archive.items()
        },
    )

    intervention_strength = {}
    for baseline in ("BLUR", "MEAN"):
        intervention_strength[baseline] = {}
        for variant in ("KEEP_A", "KEEP_I", "REMOVE_A", "REMOVE_I"):
            intervention_strength[baseline][variant] = {
                metric: distribution_from_rows(
                    rows, f"PERTURB_{baseline}_{variant}_{metric}"
                )
                for metric in ("mean_abs", "L1", "L2")
            }
    semantic_score_stability = {
        "S_ORIGINAL": distribution_from_rows(rows, "S_ORIGINAL"),
        **{
            f"S_{name}_{baseline}": distribution_from_rows(rows, f"S_{name}_{baseline}")
            for baseline in ("BLUR", "MEAN")
            for name in ("KEEP_A", "KEEP_I", "REMOVE_A", "REMOVE_I")
        },
        "CF_RANDOM_BLUR": distribution_from_rows(rows, "CF_RANDOM_BLUR"),
        "CF_RANDOM_MEAN": distribution_from_rows(rows, "CF_RANDOM_MEAN"),
    }
    mask_audit = {
        "native_shape": [1, 14, 14],
        "native_pixels": LOW_PIXELS,
        "top_fraction": TOP_FRACTION,
        "top_k": TOP_K,
        "A20_area_14": common.distribution([row["A20_area_14"] for row in rows]),
        "I20_area_14": common.distribution([row["I20_area_14"] for row in rows]),
        "A20_area_input": common.distribution([row["A20_area_input"] for row in rows]),
        "I20_area_input": common.distribution([row["I20_area_input"] for row in rows]),
        "input_area_difference": common.distribution(
            [row["I20_area_input"] - row["A20_area_input"] for row in rows]
        ),
        "R_A_PLUS_area": common.distribution([row["R_A_PLUS_area"] for row in rows]),
        "R_I_PLUS_area": common.distribution([row["R_I_PLUS_area"] for row in rows]),
    }
    overlap_groups = []
    for group in ("ALL", "IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL"):
        subset = rows if group == "ALL" else [row for row in rows if row[f"group_{group}"]]
        overlap_groups.append(
            {
                "group": group,
                "count": len(subset),
                "A20_I20_IoU": distribution_from_rows(subset, "A20_I20_IoU"),
            }
        )

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
    trainable_count = sum(
        parameter.numel()
        for module in (model, object_model)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    zero_training = {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": trainable_count,
        "parameters_with_grad": parameters_with_grad,
        "all_models_eval": not model.training and not object_model.training,
        "torch_inference_mode": True,
        "checkpoint_snapshots": checkpoints_after,
        "all_checkpoint_hashes_and_mtimes_unchanged": checkpoints_after["all_unchanged"],
        "no_nan_or_inf": no_nan_or_inf,
    }
    if not all(
        (
            trainable_count == 0,
            not parameters_with_grad,
            zero_training["all_models_eval"],
            zero_training["all_checkpoint_hashes_and_mtimes_unchanged"],
            no_nan_or_inf,
        )
    ):
        raise RuntimeError(zero_training)

    semantic_metric_audit = {
        "visual_representation": "slot_fusion([L3 visual slots, L4 visual slots])[:,0]",
        "audio_representation": "audio_slots[:,0]",
        "normalization": "torch.nn.functional.normalize(..., dim=2) in training; target vectors use dim=-1 equivalently",
        "projection": "none after slot fusion/audio Slot Attention",
        "similarity": "dot product of unit-normalized target slots, exactly cosine similarity",
        "temperature": float(model.teacher.tau),
        "training_logit": "S / tau",
        "counterfactual_score": "S uses the pre-temperature normalized cosine; scaling by fixed tau would not change Delta direction/AUROC",
        "invalid_direct_metric_spaces": "F34, K34, K4, raw visual tokens are not directly compared with audio slots",
    }
    summary = {
        "experiment": "4.2 Counterfactual Cross-Modal Reliability Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": arguments.max_batches is None,
        "tensor_audit": audit,
        "reproduction_4_1": reproduction,
        "semantic_metric_audit": semantic_metric_audit,
        "intervention_config": {
            "mask": "equal-area top20% at 14x14",
            "top_k": TOP_K,
            "mask_resize": "nearest 14x14 -> 224x224",
            "blur_kernel": list(BLUR_KERNEL),
            "blur_sigma": list(BLUR_SIGMA),
            "mean_fill_normalized_value": [0.0, 0.0, 0.0],
            "mean_fill_reason": "ImageNet channel means map exactly to zero after formal normalization",
            "random_seed": RANDOM_SEED,
        },
        "mask_audit": mask_audit,
        "candidate_overlap_groups": overlap_groups,
        "intervention_strength": intervention_strength,
        "semantic_score_stability": semantic_score_stability,
        "base_metrics": base_metrics,
        "sample_oracle": sample_oracle,
        "fixed_IQR_transition": {
            "rescue": fixed_transition["rescue"],
            "hurt": fixed_transition["hurt"],
            "net": fixed_transition["net"],
        },
        "evidence_prediction": evidence,
        "selector_results": selector_results,
        "mechanism_groups": group_summaries(rows),
        "correction_type_analysis": correction_summaries(rows),
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
