#!/usr/bin/env python3
"""Experiment 2.1: zero-training audio-guided Slot reliability diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_slot_reliability_mpl")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PROBE20_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.0_slot_objectness_probe"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe20 = _load_module("slot_objectness_probe20", PROBE20_ROOT / "probe.py")
viz = _load_module("slot_reliability_visualize", HERE / "visualize.py")


EXPERIMENT_KEYS = ("vggss_144k", "flickr_144k")
FEATURES = (
    "semantic_margin",
    "ownership_confidence",
    "soft_containment",
    "seed_containment_top10",
    "seed_containment_top20",
    "centroid_distance",
    "js_divergence",
    "extent_ratio",
    "R1",
    "R2",
    "R3",
)
SCORE_DIRECTIONS = {
    "semantic_margin": "higher",
    "ownership_confidence": "higher",
    "soft_containment": "higher",
    "seed_containment_top10": "higher",
    "seed_containment_top20": "higher",
    "centroid_distance": "lower",
    "js_divergence": "lower",
    # A priori, controlled extent is represented by closeness to 1.  The raw
    # ratio is preserved in every row and group statistic.
    "extent_ratio": "closer_to_one",
    "R1": "higher",
    "R2": "higher",
    "R3": "higher",
}
METHODS = (
    "AUD",
    "SLOT_L4_FIXED_SLOT0",
    "SLOT_L4_AUDIO_SELECTED",
    "AUD_FIXED_SLOT0",
    "AUD_AUDIO_SELECTED_SLOT",
    "OGL",
    "ORACLE_AUD_VS_FIXED_SLOT0",
    "ORACLE_AUD_VS_AUDIO_SELECTED_SLOT",
)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_KEYS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--qualitative-count", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_internal(
    refinement: torch.nn.Module,
    image: torch.Tensor,
    audio: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extract actual trained semantic slots and final L4 slot ownership."""
    teacher = refinement.teacher
    image_levels = teacher.imgnet(image)
    refinement.feature_hooks.pop()
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)

    # This is the visual representation explicitly used by training InfoNCE.
    fused_visual_slots = teacher.slot_attn.slot_fusion(encoded["visual_slots"])
    audio_slots = encoded["audio_slots"]
    normalized_audio_target = F.normalize(audio_slots[:, 0], dim=-1)
    normalized_fused_visual = F.normalize(fused_visual_slots, dim=-1)
    fused_semantic_similarity = torch.einsum(
        "bd,bsd->bs", normalized_audio_target, normalized_fused_visual
    )

    # The requested probe explicitly compares the two L4 visual slots.  These
    # S4 slots are trained end-to-end through MFusion, reconstruction and
    # divergence, although only their fused descendants receive direct InfoNCE.
    # We therefore use raw S4 cosine for the literal L4 selection and retain
    # fused-slot cosine as the direct-InfoNCE control.
    normalized_raw_l4 = F.normalize(encoded["visual_slots"][-1], dim=-1)
    semantic_similarity = torch.einsum(
        "bd,bsd->bs", normalized_audio_target, normalized_raw_l4
    )

    l4_query = encoded["visual_queries"][-1]
    l4_keys = encoded["visual_keys"][-1]
    logits = torch.einsum("bid,bjd->bij", l4_query, l4_keys) * (
        teacher.slot_attn.slot_dim**-0.5
    )
    ownership = logits.softmax(dim=1)
    if ownership.shape[1:] != (2, 49):
        raise RuntimeError(f"Expected L4 ownership [B,2,49], got {ownership.shape}")
    selected_slot = semantic_similarity.argmax(dim=1)
    selected_ownership = ownership[
        torch.arange(ownership.shape[0], device=ownership.device), selected_slot
    ]
    return {
        "ownership": ownership,
        "slot0_map": ownership[:, 0].reshape(-1, 1, 7, 7),
        "slot1_map": ownership[:, 1].reshape(-1, 1, 7, 7),
        "selected_map": selected_ownership.reshape(-1, 1, 7, 7),
        "semantic_similarity": semantic_similarity,
        "fused_semantic_similarity": fused_semantic_similarity,
        "selected_slot": selected_slot,
        "audio_target_slot": audio_slots[:, 0],
        "fused_visual_slots": fused_visual_slots,
    }


def ownership_diagnostics(ownership: torch.Tensor) -> dict[str, np.ndarray]:
    probability = ownership.float().clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(dim=1).mean(dim=-1)
    confidence = 1.0 - entropy / math.log(2.0)
    maxima = ownership.max(dim=1).values
    return {
        "ownership_confidence": confidence.detach().cpu().numpy(),
        "mean_max_ownership_probability": maxima.mean(dim=-1).cpu().numpy(),
        "fraction_max_ownership_gt_0_7": (maxima > 0.7).float().mean(dim=-1).cpu().numpy(),
        "fraction_max_ownership_gt_0_8": (maxima > 0.8).float().mean(dim=-1).cpu().numpy(),
        "fraction_max_ownership_gt_0_9": (maxima > 0.9).float().mean(dim=-1).cpu().numpy(),
    }


def _spatial_probability(value: np.ndarray) -> np.ndarray:
    flat = np.clip(value.astype(np.float64, copy=False).ravel(), 0.0, None)
    total = flat.sum()
    if total <= EPS:
        return np.full_like(flat, 1.0 / flat.size)
    return flat / total


def _centroid(probability: np.ndarray, height: int, width: int) -> tuple[float, float]:
    y, x = np.mgrid[0:height, 0:width]
    x = x.astype(np.float64) / max(width - 1, 1)
    y = y.astype(np.float64) / max(height - 1, 1)
    spatial = probability.reshape(height, width)
    return float((spatial * x).sum()), float((spatial * y).sum())


def _extent_fraction(probability: np.ndarray, mass: float = 0.8) -> float:
    descending = np.sort(probability)[::-1]
    count = int(np.searchsorted(np.cumsum(descending), mass, side="left") + 1)
    return count / probability.size


def reliability_features(
    aud: np.ndarray,
    slot: np.ndarray,
    semantic_margin: float,
    ownership_values: dict[str, float],
) -> dict[str, float]:
    if aud.shape != slot.shape:
        raise ValueError(f"Map shape mismatch: {aud.shape} vs {slot.shape}")
    flat_aud = aud.ravel()
    flat_slot = slot.ravel()
    num_pixels = flat_aud.size
    top10 = np.argpartition(flat_aud, -max(1, math.ceil(0.10 * num_pixels)))[
        -max(1, math.ceil(0.10 * num_pixels)) :
    ]
    top20 = np.argpartition(flat_aud, -max(1, math.ceil(0.20 * num_pixels)))[
        -max(1, math.ceil(0.20 * num_pixels)) :
    ]
    aud_probability = _spatial_probability(aud)
    slot_probability = _spatial_probability(slot)
    aud_centroid = _centroid(aud_probability, *aud.shape)
    slot_centroid = _centroid(slot_probability, *slot.shape)
    centroid_distance = math.dist(aud_centroid, slot_centroid) / math.sqrt(2.0)
    extent_aud = _extent_fraction(aud_probability)
    extent_slot = _extent_fraction(slot_probability)
    extent_ratio = extent_slot / max(extent_aud, EPS)
    seed20 = float(flat_slot[top20].mean())
    result = {
        "semantic_margin": float(semantic_margin),
        **ownership_values,
        "soft_containment": float(np.sum(flat_aud * flat_slot)),
        "seed_containment_top10": float(flat_slot[top10].mean()),
        "seed_containment_top20": seed20,
        "centroid_aud_x": aud_centroid[0],
        "centroid_aud_y": aud_centroid[1],
        "centroid_slot_x": slot_centroid[0],
        "centroid_slot_y": slot_centroid[1],
        "centroid_distance": centroid_distance,
        "js_divergence": probe20.js_divergence(aud, slot),
        "extent_aud": extent_aud,
        "extent_slot": extent_slot,
        "extent_ratio": extent_ratio,
    }
    result["R1"] = result["semantic_margin"] * seed20
    result["R2"] = result["ownership_confidence"] * seed20
    result["R3"] = (
        result["semantic_margin"] * result["ownership_confidence"] * seed20
    )
    return result


def outcome(aud_iou: float, fusion_iou: float) -> str:
    if aud_iou < 0.5 and fusion_iou >= 0.5:
        return "Rescue"
    if aud_iou >= 0.5 and fusion_iou < 0.5:
        return "Hurt"
    return "Neutral"


def metric_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in rows}


def run_20_reproduction(loader, refinement, reference: dict[str, Any], device) -> dict[str, Any]:
    values = {name: [] for name in ("AUD", "SLOT_L4", "AUD_SLOT_L4")}
    with torch.inference_mode():
        for image, spec, bboxes, names, _labels in tqdm(
            loader, desc="Stage 2/4 reproduce 2.0", dynamic_ncols=True
        ):
            image, spec, bboxes, names = probe20.flatten_eval_batch(
                image, spec, bboxes, names
            )
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()
            aud = refinement(image, spec)["AUD_FINE"]
            internal = extract_internal(refinement, image, spec)
            resized = probe20.resize_maps(
                {"AUD": aud, "SLOT_L4": internal["slot0_map"]}
            )
            gt = bboxes.numpy()
            for index in range(len(names)):
                aud_map = probe20.normalize_map(resized["AUD"][index])
                slot_map = probe20.normalize_map(resized["SLOT_L4"][index])
                fusion = probe20.fuse_maps(aud_map, slot_map, 0.6)
                values["AUD"].append(probe20.sample_iou(aud_map, gt[index]))
                values["SLOT_L4"].append(probe20.sample_iou(slot_map, gt[index]))
                values["AUD_SLOT_L4"].append(probe20.sample_iou(fusion, gt[index]))

    observed_metrics = {
        name: probe20.summarize_ious(iou_values) for name, iou_values in values.items()
    }
    aud_success = np.asarray(values["AUD"]) >= 0.5
    fixed_success = np.asarray(values["AUD_SLOT_L4"]) >= 0.5
    observed_rescue = int(((~aud_success) & fixed_success).sum())
    observed_hurt = int((aud_success & (~fixed_success)).sum())
    reference_metrics = metric_lookup(reference["official_metrics"])
    reference_rescue = next(
        row for row in reference["rescue_hurt"] if row["method"] == "AUD_SLOT_L4"
    )
    errors = {
        "AUD_cIoU": abs(observed_metrics["AUD"]["cIoU"] - reference_metrics["AUD_FINE"]["cIoU"]),
        "AUD_AUC": abs(observed_metrics["AUD"]["AUC"] - reference_metrics["AUD_FINE"]["AUC"]),
        "SLOT_L4_cIoU": abs(observed_metrics["SLOT_L4"]["cIoU"] - reference_metrics["SLOT_L4"]["cIoU"]),
        "SLOT_L4_AUC": abs(observed_metrics["SLOT_L4"]["AUC"] - reference_metrics["SLOT_L4"]["AUC"]),
        "AUD_SLOT_L4_cIoU": abs(observed_metrics["AUD_SLOT_L4"]["cIoU"] - reference_metrics["AUD_SLOT_L4"]["cIoU"]),
        "AUD_SLOT_L4_AUC": abs(observed_metrics["AUD_SLOT_L4"]["AUC"] - reference_metrics["AUD_SLOT_L4"]["AUC"]),
        "rescue_count": abs(observed_rescue - int(reference_rescue["rescue_count"])),
        "hurt_count": abs(observed_hurt - int(reference_rescue["hurt_count"])),
    }
    passed = all(value <= 1e-12 for value in errors.values())
    result = {
        "passed": passed,
        "reference_summary": reference,
        "observed_metrics": observed_metrics,
        "observed_rescue": observed_rescue,
        "observed_hurt": observed_hurt,
        "absolute_errors": errors,
    }
    if not passed:
        raise RuntimeError(f"Experiment 2.0 did not reproduce exactly: {errors}")
    return result


def _score_for_auroc(row: dict[str, Any], feature: str) -> float:
    value = float(row[feature])
    direction = SCORE_DIRECTIONS[feature]
    if direction == "lower":
        return -value
    if direction == "closer_to_one":
        return -abs(math.log(max(value, EPS)))
    return value


def reliability_aurocs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled = [row for row in rows if row["outcome"] in {"Rescue", "Hurt"}]
    labels = np.asarray([1 if row["outcome"] == "Rescue" else 0 for row in labelled])
    if np.unique(labels).size != 2:
        raise RuntimeError("Both Rescue and Hurt are required for AUROC")
    output = []
    for feature in FEATURES:
        scores = np.asarray([_score_for_auroc(row, feature) for row in labelled])
        finite = np.isfinite(scores)
        output.append(
            {
                "feature": feature,
                "score_direction": SCORE_DIRECTIONS[feature],
                "AUROC": float(roc_auc_score(labels[finite], scores[finite])),
                "rescue_samples": int((labels[finite] == 1).sum()),
                "hurt_samples": int((labels[finite] == 0).sum()),
                "valid_samples": int(finite.sum()),
            }
        )
    return output


def distribution_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in ("Rescue", "Hurt", "Neutral"):
        chosen = [row for row in rows if row["outcome"] == group]
        for feature in FEATURES:
            values = np.asarray([row[feature] for row in chosen], dtype=np.float64)
            output.append(
                {
                    "outcome": group,
                    "feature": feature,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std()),
                }
            )
    return output


def add_failure_heuristics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians = {
        feature: float(np.median([row[feature] for row in rows]))
        for feature in ("semantic_margin", "seed_containment_top20", "js_divergence")
    }
    for row in rows:
        is_hurt = row["outcome"] == "Hurt"
        selection_related = bool(
            is_hurt
            and row["selection_changed"]
            and row["selected_minus_fixed_iou"] >= 0.05
        )
        extent_related = bool(
            is_hurt
            and not selection_related
            and row["semantic_margin"] >= medians["semantic_margin"]
            and row["seed_containment_top20"] >= medians["seed_containment_top20"]
            and row["js_divergence"] <= medians["js_divergence"]
            and (row["extent_ratio"] >= 1.25 or row["extent_ratio"] <= 0.80)
        )
        row["possible_selection_error"] = selection_related
        row["possible_extent_error"] = extent_related
    hurt = [row for row in rows if row["outcome"] == "Hurt"]
    selection_count = sum(row["possible_selection_error"] for row in hurt)
    extent_count = sum(row["possible_extent_error"] for row in hurt)
    return {
        "warning": "Transparent offline heuristic, not a causal classification and not used by the model.",
        "selection_rule": "selected fusion is still Hurt, selected_slot!=0, and selected IoU improves over fixed by >=0.05",
        "extent_rule": "remaining Hurt with margin>=global median, seed20>=median, JS<=median, and extent_ratio outside [0.80,1.25]",
        "global_medians": medians,
        "hurt_count": len(hurt),
        "possible_selection_error_count": int(selection_count),
        "possible_selection_error_ratio": selection_count / max(len(hurt), 1),
        "possible_extent_error_count": int(extent_count),
        "possible_extent_error_ratio": extent_count / max(len(hurt), 1),
        "unclassified_hurt_count": len(hurt) - selection_count - extent_count,
    }


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [row for row in rows if row["selection_changed"]]
    fused_changed = [row for row in rows if row["fused_selected_slot"] != 0]
    improves = sum(row["selected_minus_fixed_iou"] > 1e-12 for row in changed)
    worsens = sum(row["selected_minus_fixed_iou"] < -1e-12 for row in changed)
    ties = len(changed) - improves - worsens

    def counts(method_key: str) -> tuple[int, int]:
        rescues = sum(
            row["IoU_AUD"] < 0.5 and row[method_key] >= 0.5 for row in rows
        )
        hurts = sum(
            row["IoU_AUD"] >= 0.5 and row[method_key] < 0.5 for row in rows
        )
        return rescues, hurts

    fixed_rescue, fixed_hurt = counts("IoU_AUD_FIXED")
    selected_rescue, selected_hurt = counts("IoU_AUD_SELECTED")
    ogl_rescue, ogl_hurt = counts("IoU_OGL")
    fixed_hurt_resolved = sum(
        row["IoU_AUD"] >= 0.5
        and row["IoU_AUD_FIXED"] < 0.5
        and row["IoU_AUD_SELECTED"] >= 0.5
        for row in rows
    )
    fixed_rescue_lost = sum(
        row["IoU_AUD"] < 0.5
        and row["IoU_AUD_FIXED"] >= 0.5
        and row["IoU_AUD_SELECTED"] < 0.5
        for row in rows
    )
    return {
        "num_samples": len(rows),
        "slot_selection_changed": len(changed),
        "slot_selection_changed_ratio": len(changed) / len(rows),
        "direct_infonce_fused_selection_changed": len(fused_changed),
        "direct_infonce_fused_selection_changed_ratio": len(fused_changed) / len(rows),
        "changed_selection_improves": int(improves),
        "changed_selection_worsens": int(worsens),
        "changed_selection_ties": int(ties),
        "fixed_rescue": int(fixed_rescue),
        "fixed_hurt": int(fixed_hurt),
        "fixed_net_rescue": int(fixed_rescue - fixed_hurt),
        "selected_rescue": int(selected_rescue),
        "selected_hurt": int(selected_hurt),
        "selected_net_rescue": int(selected_rescue - selected_hurt),
        "ogl_rescue": int(ogl_rescue),
        "ogl_hurt": int(ogl_hurt),
        "ogl_net_rescue": int(ogl_rescue - ogl_hurt),
        "fixed_hurt_resolved_by_selection": int(fixed_hurt_resolved),
        "fixed_rescue_lost_by_selection": int(fixed_rescue_lost),
    }


def select_visual_ids(rows: list[dict[str, Any]], count: int) -> list[dict[str, str]]:
    q25 = {
        feature: float(np.quantile([row[feature] for row in rows], 0.25))
        for feature in ("semantic_margin", "seed_containment_top20")
    }
    q75_r3 = float(np.quantile([row["R3"] for row in rows], 0.75))
    categorized: dict[str, list[str]] = {
        "HIGH_RELIABILITY_RESCUE": [],
        "HIGH_RELIABILITY_HURT": [],
        "LOW_MARGIN": [],
        "LOW_SEED_CONTAINMENT": [],
        "EXTENT_OVEREXPANSION_GE_1P25": [],
        "SELECTION_IMPROVES": [],
        "SELECTION_WORSENS": [],
    }
    row_categories: dict[str, list[str]] = {}
    for row in rows:
        categories = []
        if row["outcome"] == "Rescue" and row["R3"] >= q75_r3:
            categories.append("HIGH_RELIABILITY_RESCUE")
        if row["outcome"] == "Hurt" and row["R3"] >= q75_r3:
            categories.append("HIGH_RELIABILITY_HURT")
        if row["semantic_margin"] <= q25["semantic_margin"]:
            categories.append("LOW_MARGIN")
        if row["seed_containment_top20"] <= q25["seed_containment_top20"]:
            categories.append("LOW_SEED_CONTAINMENT")
        if row["extent_ratio"] >= 1.25:
            categories.append("EXTENT_OVEREXPANSION_GE_1P25")
        if row["selection_changed"] and row["selected_minus_fixed_iou"] > 1e-12:
            categories.append("SELECTION_IMPROVES")
        if row["selection_changed"] and row["selected_minus_fixed_iou"] < -1e-12:
            categories.append("SELECTION_WORSENS")
        row_categories[row["sample_id"]] = categories
        for category in categories:
            categorized[category].append(row["sample_id"])

    # Some datasets contain no sample whose selected Slot mask expands beyond
    # 1.25x the AUD 80%-mass extent.  In that case, visualize the real maximum
    # instead of silently omitting the requested diagnostic or inventing an
    # "over-expansion" case.
    extent_category = "EXTENT_OVEREXPANSION_GE_1P25"
    if not categorized[extent_category]:
        max_extent_row = max(rows, key=lambda row: row["extent_ratio"])
        fallback_category = "MAX_EXTENT_FALLBACK_NO_GE_1P25_CASE"
        row_categories[max_extent_row["sample_id"]].append(fallback_category)
        categorized[fallback_category] = [max_extent_row["sample_id"]]

    selected: list[str] = []
    priority = tuple(categorized)
    for rank in range(3):
        for category in priority:
            candidates = categorized[category]
            if rank < len(candidates) and candidates[rank] not in selected:
                selected.append(candidates[rank])
                if len(selected) >= count:
                    break
        if len(selected) >= count:
            break
    selected = selected[:count]
    if len(selected) < count:
        for row in rows:
            if row["sample_id"] not in selected:
                selected.append(row["sample_id"])
            if len(selected) >= count:
                break
    rule = (
        "first-in-test-order per predefined quartile/fixed-threshold category; "
        "round-robin then first-in-test-order fill"
    )
    return [
        {
            "sample_id": sample_id,
            "categories": "|".join(row_categories[sample_id]) or "FILL",
            "selection_rule": rule,
        }
        for sample_id in selected
    ]


def save_qualitative(
    loader,
    refinement,
    object_model,
    selected: list[dict[str, str]],
    rows: list[dict[str, Any]],
    output_dir: Path,
    device,
) -> None:
    wanted = {entry["sample_id"]: entry for entry in selected}
    row_lookup = {row["sample_id"]: row for row in rows}
    found: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for image, spec, bboxes, names, _labels in tqdm(
            loader, desc="Stage 4/4 qualitative", dynamic_ncols=True
        ):
            image, spec, bboxes, names = probe20.flatten_eval_batch(
                image, spec, bboxes, names
            )
            batch_needed = [index for index, name in enumerate(names) if name in wanted]
            if not batch_needed:
                continue
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()
            aud = refinement(image, spec)["AUD_FINE"]
            internal = extract_internal(refinement, image, spec)
            obj = object_model(image)
            resized = probe20.resize_maps(
                {
                    "AUD_FINE": aud,
                    "SLOT0": internal["slot0_map"],
                    "SLOT1": internal["slot1_map"],
                    "SLOT_SELECTED": internal["selected_map"],
                    "OBJ": obj,
                }
            )
            gt = bboxes.numpy()
            for index in batch_needed:
                sample_id = names[index]
                maps = {
                    key: probe20.normalize_map(value[index])
                    for key, value in resized.items()
                }
                maps["AUD_SELECTED"] = probe20.fuse_maps(
                    maps["AUD_FINE"], maps["SLOT_SELECTED"], 0.6
                )
                maps["OGL"] = probe20.fuse_maps(maps["AUD_FINE"], maps["OBJ"], 0.6)
                rgb = (
                    probe20.inverse_normalize(image[index].cpu())
                    .permute(1, 2, 0)
                    .numpy()
                )
                found[sample_id] = {
                    "sample_id": sample_id,
                    "image": np.clip(rgb, 0.0, 1.0),
                    "GT": gt[index],
                    "row": row_lookup[sample_id],
                    "categories": wanted[sample_id]["categories"],
                    **maps,
                }
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"Qualitative IDs not found: {sorted(missing)}")
    for index, entry in enumerate(selected, start=1):
        viz.save_sample_panel(
            found[entry["sample_id"]],
            output_dir / f"{index:02d}_{entry['sample_id']}.png",
        )
    viz.save_selection_manifest(selected, output_dir / "selection_manifest.csv")


def run(arguments: argparse.Namespace) -> None:
    registry = probe20.EXPERIMENTS[arguments.experiment]
    experiment_name = registry["default_experiment"]
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    checkpoint_path = checkpoint_dir / f"{registry['dataset']}_best.pth"
    reference20_path = PROBE20_ROOT / "results" / arguments.experiment / "summary.json"
    if not checkpoint_path.is_file() or not reference20_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/reference: {checkpoint_path}, {reference20_path}")
    reference20 = json.loads(reference20_path.read_text(encoding="utf-8"))
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash_before = sha256(checkpoint_path)
    checkpoint_mtime_before = checkpoint_path.stat().st_mtime_ns
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = probe20.load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    config.alpha = 0.6
    config.model_dir = str(PROJECT_ROOT / "checkpoints")
    config.experiment_name = experiment_name
    probe20.setup_seed(config.seed)

    refinement, base_checkpoint = probe20.build_model(config, registry, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "multi_geometry_equivariant_l3_refine":
        raise RuntimeError(f"Unexpected architecture: {checkpoint.get('architecture')}")
    refinement.student.proj3_spatial.load_state_dict(
        checkpoint["proj3_spatial_state_dict"], strict=True
    )
    refinement.student.adapter.load_state_dict(
        checkpoint["topdown_adapter_state_dict"], strict=True
    )
    for parameter in refinement.parameters():
        parameter.requires_grad = False
    refinement.eval()
    object_model = probe20.object_prior_model().to(device).eval()
    test_dataset = probe20.get_test_dataset(config, registry["dataset"])
    loader = probe20.build_test_loader(test_dataset, config, registry)

    print("Stage 1/4: exact formal 1.3G baseline reproduction", flush=True)
    formal_check = probe20.verify_baseline(
        loader,
        refinement,
        object_model,
        config,
        registry,
        checkpoint_dir,
        output_dir,
    )
    print(json.dumps(formal_check["observed"], indent=2), flush=True)

    print("Stage 2/4: exact Experiment 2.0 reproduction", flush=True)
    reproduction20 = run_20_reproduction(loader, refinement, reference20, device)
    (output_dir / "probe20_reproduction.json").write_text(
        json.dumps(reproduction20, indent=2), encoding="utf-8"
    )
    print(
        f"2.0 exact: rescue={reproduction20['observed_rescue']}, "
        f"hurt={reproduction20['observed_hurt']}",
        flush=True,
    )

    method_ious = {method: [] for method in METHODS}
    rows: list[dict[str, Any]] = []
    ownership_audit = {
        "max_slot_sum_error": 0.0,
        "logits_shape": "[B,2,49]",
        "ownership_definition": "softmax(final L4 SlotAttention logits, dim=slot_dim=1)",
    }
    print("Stage 3/4: audio selection and reliability features", flush=True)
    with ThreadPoolExecutor(max_workers=arguments.feature_workers) as executor:
        with torch.inference_mode():
            for image, spec, bboxes, names, _labels in tqdm(
                loader, desc=f"Reliability {arguments.experiment}", dynamic_ncols=True
            ):
                image, spec, bboxes, names = probe20.flatten_eval_batch(
                    image, spec, bboxes, names
                )
                image = image.to(device, non_blocking=True).float()
                spec = spec.to(device, non_blocking=True).float()
                aud = refinement(image, spec)["AUD_FINE"]
                internal = extract_internal(refinement, image, spec)
                obj = object_model(image)
                ownership_audit["max_slot_sum_error"] = max(
                    ownership_audit["max_slot_sum_error"],
                    float((internal["ownership"].sum(dim=1) - 1.0).abs().max()),
                )
                ownership_stats = ownership_diagnostics(internal["ownership"])
                semantic = internal["semantic_similarity"].cpu().numpy()
                fused_semantic = internal["fused_semantic_similarity"].cpu().numpy()
                selected_slot = internal["selected_slot"].cpu().numpy()
                resized = probe20.resize_maps(
                    {
                        "AUD": aud,
                        "SLOT0": internal["slot0_map"],
                        "SLOT1": internal["slot1_map"],
                        "SLOT_SELECTED": internal["selected_map"],
                        "OBJ": obj,
                    }
                )
                gt = bboxes.numpy()
                batch_feature_args = []
                batch_rows = []
                for index, sample_id in enumerate(names):
                    maps = {
                        key: probe20.normalize_map(value[index])
                        for key, value in resized.items()
                    }
                    maps["AUD_FIXED"] = probe20.fuse_maps(maps["AUD"], maps["SLOT0"], 0.6)
                    maps["AUD_SELECTED"] = probe20.fuse_maps(
                        maps["AUD"], maps["SLOT_SELECTED"], 0.6
                    )
                    maps["OGL"] = probe20.fuse_maps(maps["AUD"], maps["OBJ"], 0.6)
                    ious = {
                        "AUD": probe20.sample_iou(maps["AUD"], gt[index]),
                        "SLOT0": probe20.sample_iou(maps["SLOT0"], gt[index]),
                        "SLOT1": probe20.sample_iou(maps["SLOT1"], gt[index]),
                        "SLOT_SELECTED": probe20.sample_iou(maps["SLOT_SELECTED"], gt[index]),
                        "AUD_FIXED": probe20.sample_iou(maps["AUD_FIXED"], gt[index]),
                        "AUD_SELECTED": probe20.sample_iou(maps["AUD_SELECTED"], gt[index]),
                        "OGL": probe20.sample_iou(maps["OGL"], gt[index]),
                    }
                    oracle_fixed_iou = max(ious["AUD"], ious["AUD_FIXED"])
                    oracle_selected_iou = max(ious["AUD"], ious["AUD_SELECTED"])
                    method_values = {
                        "AUD": ious["AUD"],
                        "SLOT_L4_FIXED_SLOT0": ious["SLOT0"],
                        "SLOT_L4_AUDIO_SELECTED": ious["SLOT_SELECTED"],
                        "AUD_FIXED_SLOT0": ious["AUD_FIXED"],
                        "AUD_AUDIO_SELECTED_SLOT": ious["AUD_SELECTED"],
                        "OGL": ious["OGL"],
                        "ORACLE_AUD_VS_FIXED_SLOT0": oracle_fixed_iou,
                        "ORACLE_AUD_VS_AUDIO_SELECTED_SLOT": oracle_selected_iou,
                    }
                    for method, value in method_values.items():
                        method_ious[method].append(value)
                    ownership_values = {
                        key: float(value[index]) for key, value in ownership_stats.items()
                    }
                    margin = float(abs(semantic[index, 0] - semantic[index, 1]))
                    row = {
                        "sample_id": sample_id,
                        "sim_slot0": float(semantic[index, 0]),
                        "sim_slot1": float(semantic[index, 1]),
                        "fused_sim_slot0": float(fused_semantic[index, 0]),
                        "fused_sim_slot1": float(fused_semantic[index, 1]),
                        "fused_selected_slot": int(fused_semantic[index].argmax()),
                        "selected_slot": int(selected_slot[index]),
                        "selection_changed": bool(selected_slot[index] != 0),
                        "IoU_AUD": ious["AUD"],
                        "IoU_SLOT0": ious["SLOT0"],
                        "IoU_SLOT1": ious["SLOT1"],
                        "IoU_SLOT_SELECTED": ious["SLOT_SELECTED"],
                        "IoU_AUD_FIXED": ious["AUD_FIXED"],
                        "IoU_AUD_SELECTED": ious["AUD_SELECTED"],
                        "IoU_OGL": ious["OGL"],
                        "IoU_ORACLE_FIXED": oracle_fixed_iou,
                        "IoU_ORACLE_SELECTED": oracle_selected_iou,
                        "selected_minus_fixed_iou": ious["AUD_SELECTED"] - ious["AUD_FIXED"],
                    }
                    batch_feature_args.append(
                        (maps["AUD"], maps["SLOT_SELECTED"], margin, ownership_values)
                    )
                    batch_rows.append(row)
                feature_values = list(executor.map(lambda args: reliability_features(*args), batch_feature_args))
                for row, features in zip(batch_rows, feature_values):
                    row.update(features)
                    row["outcome"] = outcome(row["IoU_AUD"], row["IoU_AUD_SELECTED"])
                    rows.append(row)

    metrics_rows = [
        {"method": method, **probe20.summarize_ious(method_ious[method])}
        for method in METHODS
    ]
    select_summary = selection_summary(rows)
    failure_analysis = add_failure_heuristics(rows)
    auroc_rows = reliability_aurocs(rows)
    distribution_rows = distribution_statistics(rows)
    selected_visuals = select_visual_ids(rows, arguments.qualitative_count)

    write_csv(output_dir / "method_metrics.csv", metrics_rows)
    write_csv(output_dir / "per_sample_reliability.csv", rows)
    write_csv(output_dir / "reliability_auroc.csv", auroc_rows)
    write_csv(output_dir / "feature_group_statistics.csv", distribution_rows)
    (output_dir / "selection_summary.json").write_text(
        json.dumps(select_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "failure_mode_heuristic.json").write_text(
        json.dumps(failure_analysis, indent=2), encoding="utf-8"
    )
    semantic_audit = {
        "audio_representation": "normalized target audio Slot Attention output Sa[:,0,:]",
        "visual_representation": "normalized raw L4 visual Slot Attention outputs S4[:,0/1,:], as explicitly requested",
        "similarity": "cosine similarity (dot product after L2 normalization)",
        "selection": "argmax_j cos(Sa0, S4j), then take L4 ownership map with slot index j",
        "training_evidence": "S4 is trained end-to-end through the existing MFusion/base losses; cosine follows normalized Slot InfoNCE semantics, but direct InfoNCE is applied after MFusion",
        "direct_infonce_control": "cos(Sa0,Sfj) is saved per sample as fused_sim_slot0/1 and fused_selected_slot",
        "new_projection_created": False,
        "raw_l4_similarity_used_for_selection": True,
        "fused_similarity_used_for_selection": False,
        "ownership": ownership_audit,
        "forbidden_reliability_inputs": {
            "GT": False,
            "IoU": False,
            "rescue_hurt_label": False,
            "OBJ_PRIOR": False,
            "OGL": False,
        },
        "extent_auroc_direction": "-abs(log(extent_ratio)); high means closer to controlled ratio 1",
    }
    (output_dir / "semantic_space_audit.json").write_text(
        json.dumps(semantic_audit, indent=2), encoding="utf-8"
    )

    viz.save_auroc_figure(auroc_rows, output_dir / "fig_reliability_auroc")
    viz.save_outcome_figure(select_summary, output_dir / "fig_rescue_hurt")
    viz.save_feature_boxplots(rows, output_dir / "fig_feature_distributions")
    save_qualitative(
        loader,
        refinement,
        object_model,
        selected_visuals,
        rows,
        output_dir / "qualitative",
        device,
    )

    checkpoint_hash_after = sha256(checkpoint_path)
    checkpoint_mtime_after = checkpoint_path.stat().st_mtime_ns
    zero_training = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "sha256_before": checkpoint_hash_before,
        "sha256_after": checkpoint_hash_after,
        "mtime_ns_before": checkpoint_mtime_before,
        "mtime_ns_after": checkpoint_mtime_after,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after
        and checkpoint_mtime_before == checkpoint_mtime_after,
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_parameters": 0,
        "model_trainable_parameters_during_probe": sum(
            parameter.numel() for parameter in refinement.parameters() if parameter.requires_grad
        ),
        "base_checkpoint": str(base_checkpoint),
    }
    if not zero_training["checkpoint_unchanged"]:
        raise RuntimeError("Formal checkpoint changed during zero-training probe")
    (output_dir / "zero_training_audit.json").write_text(
        json.dumps(zero_training, indent=2), encoding="utf-8"
    )

    result = {
        "experiment": "2.1 Audio-Guided Slot Reliability Probe",
        "dataset": arguments.experiment,
        "formal_checkpoint": str(checkpoint_path.resolve()),
        "formal_baseline_reproduced": formal_check["passed"],
        "probe20_reproduced": reproduction20["passed"],
        "semantic_space": semantic_audit,
        "method_metrics": metrics_rows,
        "selection_summary": select_summary,
        "reliability_auroc": auroc_rows,
        "feature_group_statistics": distribution_rows,
        "failure_mode_analysis": failure_analysis,
        "qualitative_ids": selected_visuals,
        "zero_training_audit": zero_training,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    refinement.close()


if __name__ == "__main__":
    run(parse_args())
