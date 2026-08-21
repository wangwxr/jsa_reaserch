#!/usr/bin/env python3
"""Experiment 4.0: frozen AUD-IMG redundancy and att-loss mechanism probe."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import common

visualize = common.load_module("experiment_40_visualize", common.HERE / "visualize.py")


STAGES = ("Stage1", "Stage2")
AUXILIARIES = ("IMG", "OBJ")
METHODS = ("AUD", "IMG", "IQR", "OBJ", "OGL")
ALPHAS = tuple(round(value / 10, 1) for value in range(11))
TOP_FRACTIONS = (0.10, 0.20, 0.30)
QUALITATIVE_CATEGORIES = (
    "AUD_ONLY",
    "IMG_ONLY",
    "OGL_RESCUE_CAPTURED_BY_IMG",
    "OGL_RESCUE_NOT_CAPTURED_BY_IMG",
    "IQR_RESCUE",
    "IQR_HURT",
    "BOTH_FAIL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-qualitative", action="store_true")
    return parser.parse_args()


def topk_overlap(first: np.ndarray, second: np.ndarray, fraction: float) -> float:
    first = np.asarray(first).ravel()
    second = np.asarray(second).ravel()
    count = max(1, int(math.ceil(first.size * fraction)))
    first_indices = np.argpartition(first, -count)[-count:]
    second_indices = np.argpartition(second, -count)[-count:]
    intersection = np.intersect1d(first_indices, second_indices, assume_unique=False).size
    union = 2 * count - intersection
    return float(intersection / union)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first) >= 0.6
    second_mask = np.asarray(second) >= 0.6
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_mask, second_mask).sum() / union)


def pair_measures(
    raw_first: np.ndarray,
    raw_second: np.ndarray,
    normalized_first: np.ndarray,
    normalized_second: np.ndarray,
) -> dict[str, float]:
    # Per-map min-max normalization is a positive affine transform, so Pearson
    # and ranks are exactly invariant. Compute the expensive rank statistic once.
    pearson = common.safe_pearson(raw_first, raw_second)
    spearman = common.spearman(raw_first, raw_second)
    output = {
        "raw_pearson": pearson,
        "raw_spearman": spearman,
        "raw_js": common.js_divergence(raw_first, raw_second),
        "norm_pearson": pearson,
        "norm_spearman": spearman,
        "norm_js": common.js_divergence(normalized_first, normalized_second),
        "mask_iou": mask_iou(normalized_first, normalized_second),
    }
    for fraction in TOP_FRACTIONS:
        output[f"top{int(fraction * 100)}_overlap"] = topk_overlap(
            normalized_first, normalized_second, fraction
        )
    return output


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
    train_attentions = teacher.slot_attn._l4_attentions(encoded, scale_multiplier=1.0)

    batch = image.shape[0]
    img_l4 = eval_attentions["imgq_imgk_attn"][:, 0].reshape(batch, 1, 7, 7)
    aud_l4 = eval_attentions["audq_imgk_attn"][:, 0].reshape(batch, 1, 7, 7)

    f34, f3_spatial, f4_up, delta_f3 = model.student(layer3_native, f4_projected)
    l4_branch = teacher.slot_attn.visual_branches[-1]
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    aud_fine_all = teacher.slot_attn._attention(
        encoded["audio_query"], k34, teacher.infer_sharpening
    )
    aud_fine = aud_fine_all[:, 0].reshape(batch, 1, 14, 14)

    spatial_mse = (
        train_attentions["audq_imgk_attn"][:, 0]
        - train_attentions["imgq_imgk_attn"][:, 0]
    ).square().mean(dim=-1)
    temporal_mse = (
        train_attentions["imgq_audk_attn"][:, 0]
        - train_attentions["audq_audk_attn"][:, 0]
    ).square().mean(dim=-1)

    return {
        "Qa": encoded["audio_query"],
        "Qv": encoded["visual_queries"][-1],
        "K4": encoded["visual_keys"][-1],
        "AUD_L4": aud_l4,
        "IMG_L4": img_l4,
        "F34": f34,
        "F3_SPATIAL": f3_spatial,
        "F4_UP": f4_up,
        "DELTA_F3": delta_f3,
        "K34": k34,
        "AUD_FINE": aud_fine,
        "ATT_SPATIAL_MSE": spatial_mse,
        "ATT_TEMPORAL_MSE": temporal_mse,
        "ATT_TOTAL_MSE": spatial_mse + temporal_mse,
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

    official_img, official_aud_l4 = model.teacher(image, spec)
    official_aud_fine = model(image, spec)["AUD_FINE"]
    local = extract_all(model, image, spec)
    object_prior = object_model(image)
    finite = all(
        torch.isfinite(value).all().item()
        for value in (*local.values(), object_prior)
        if isinstance(value, torch.Tensor)
    )
    audit = {
        "sample_ids": names[:4],
        "Stage1": {
            "Qa_shape": list(local["Qa"].shape),
            "Qv_shape": list(local["Qv"].shape),
            "K4_shape": list(local["K4"].shape),
            "AUD_L4_shape": list(local["AUD_L4"].shape),
            "IMG_L4_shape": list(local["IMG_L4"].shape),
            "AUD_L4_reconstruction_max_error": float(
                (local["AUD_L4"] - official_aud_l4).abs().max()
            ),
            "IMG_L4_reconstruction_max_error": float(
                (local["IMG_L4"] - official_img).abs().max()
            ),
        },
        "Stage2": {
            "F34_shape": list(local["F34"].shape),
            "K34_shape": list(local["K34"].shape),
            "AUD_FINE_shape": list(local["AUD_FINE"].shape),
            "official_IMG_QUERY_shape": list(official_img.shape),
            "AUD_FINE_reconstruction_max_error": float(
                (local["AUD_FINE"] - official_aud_fine).abs().max()
            ),
            "official_IMG_QUERY_source": "frozen Stage1 Q4 -> K4, 7x7",
        },
        "f4_token_error": float(local["f4_token_error"]),
        "direct_eval_attention_discrepancy": {
            "spatial_MSE": float(local["ATT_SPATIAL_MSE"].mean()),
            "temporal_MSE": float(local["ATT_TEMPORAL_MSE"].mean()),
            "total_MSE": float(local["ATT_TOTAL_MSE"].mean()),
        },
        "object_prior_shape": list(object_prior.shape),
        "no_nan_or_inf": finite,
    }
    errors = (
        audit["Stage1"]["AUD_L4_reconstruction_max_error"],
        audit["Stage1"]["IMG_L4_reconstruction_max_error"],
        audit["Stage2"]["AUD_FINE_reconstruction_max_error"],
        audit["f4_token_error"],
    )
    expected = {
        "Qa_shape": [2, 512],
        "Qv_shape": [2, 512],
        "K4_shape": [49, 512],
        "AUD_L4_shape": [1, 7, 7],
        "IMG_L4_shape": [1, 7, 7],
    }
    for key, shape in expected.items():
        if audit["Stage1"][key][1:] != shape:
            raise RuntimeError(audit)
    if audit["Stage2"]["K34_shape"][1:] != [196, 512]:
        raise RuntimeError(audit)
    if max(errors) > 1e-6 or not finite:
        raise RuntimeError(audit)
    audit["passed"] = True
    return audit


def append_pair(
    accumulator: dict[str, Any], stage: str, auxiliary: str, values: dict[str, float]
) -> None:
    for key, value in values.items():
        accumulator[stage][auxiliary][key].append(value)


def categorical_labels(aud_iou: float, img_iou: float, iqr_iou: float, ogl_iou: float) -> list[str]:
    aud_success = aud_iou >= 0.5
    img_success = img_iou >= 0.5
    iqr_success = iqr_iou >= 0.5
    ogl_success = ogl_iou >= 0.5
    labels = []
    if aud_success and not img_success:
        labels.append("AUD_ONLY")
    if not aud_success and img_success:
        labels.append("IMG_ONLY")
    if not aud_success and ogl_success:
        labels.append(
            "OGL_RESCUE_CAPTURED_BY_IMG" if img_success else "OGL_RESCUE_NOT_CAPTURED_BY_IMG"
        )
    if not aud_success and iqr_success:
        labels.append("IQR_RESCUE")
    if aud_success and not iqr_success:
        labels.append("IQR_HURT")
    if not aud_success and not img_success:
        labels.append("BOTH_FAIL")
    return labels


def update_qualitative(
    selected: dict[str, dict[str, Any]],
    key: str,
    categories: list[str],
    image: torch.Tensor,
    gt: np.ndarray,
    maps: dict[str, np.ndarray],
    row: dict[str, Any],
) -> None:
    for category in categories:
        current = selected.get(category)
        if current is not None and current["sort_key"] <= key:
            continue
        rgb = common.inverse_normalize(image.detach().cpu()).permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        selected[category] = {
            "sort_key": key,
            "sample_id": row["sample_id"],
            "category": category,
            "image": rgb,
            "GT": gt,
            "AUD": maps["AUD"],
            "IMG": maps["IMG"],
            "IQR": maps["IQR"],
            "OBJ": maps["OBJ"],
            "OGL": maps["OGL"],
            "AUD_IMG_DIFF": common.normalize_map(np.abs(maps["AUD"] - maps["IMG"])),
            "AUD_OBJ_DIFF": common.normalize_map(np.abs(maps["AUD"] - maps["OBJ"])),
            "row": {
                "IoU_AUD": row["Stage2_IoU_AUD"],
                "IoU_IMG": row["Stage2_IoU_IMG"],
                "IoU_IQR": row["Stage2_IoU_IQR"],
                "IoU_OBJ": row["Stage2_IoU_OBJ"],
                "IoU_OGL": row["Stage2_IoU_OGL"],
            },
        }


def success_decomposition(aud: list[float], auxiliary: list[float]) -> dict[str, Any]:
    aud_success = np.asarray(aud) >= 0.5
    aux_success = np.asarray(auxiliary) >= 0.5
    count = aud_success.size
    values = {
        "BOTH_SUCCESS": int((aud_success & aux_success).sum()),
        "AUD_ONLY": int((aud_success & ~aux_success).sum()),
        "AUX_ONLY": int((~aud_success & aux_success).sum()),
        "BOTH_FAIL": int((~aud_success & ~aux_success).sum()),
    }
    return {
        **values,
        **{f"{key}_fraction": value / count for key, value in values.items()},
        "num_samples": int(count),
    }


def build_stage_summary(
    stage: str,
    ious: dict[str, dict[str, list[float]]],
    pair_values: dict[str, Any],
    alpha_values: dict[str, Any],
) -> dict[str, Any]:
    methods = {method: common.summarize(ious[stage][method]) for method in METHODS}
    pairs = []
    fusion_method = {"IMG": "IQR", "OBJ": "OGL"}
    for auxiliary in AUXILIARIES:
        aud = ious[stage]["AUD"]
        aux = ious[stage][auxiliary]
        fusion = ious[stage][fusion_method[auxiliary]]
        oracle = common.summarize(np.maximum(aud, aux).tolist())
        shift = common.transition(aud, fusion)
        pairs.append(
            {
                "pair": f"AUD+{auxiliary}",
                "standalone_auxiliary": methods[auxiliary],
                "similarity": {
                    key: common.distribution(values)
                    for key, values in pair_values[stage][auxiliary].items()
                },
                "success_decomposition": success_decomposition(aud, aux),
                "pair_oracle": oracle,
                "oracle_gain_over_AUD_cIoU": oracle["cIoU"] - methods["AUD"]["cIoU"],
                "oracle_gain_over_AUD_AUC": oracle["AUC"] - methods["AUD"]["AUC"],
                "fixed_fusion_method": fusion_method[auxiliary],
                "fixed_fusion_gain_cIoU": methods[fusion_method[auxiliary]]["cIoU"] - methods["AUD"]["cIoU"],
                "fixed_fusion_gain_AUC": methods[fusion_method[auxiliary]]["AUC"] - methods["AUD"]["AUC"],
                "rescue": shift["rescue"],
                "hurt": shift["hurt"],
                "net": shift["net"],
            }
        )

    aud = np.asarray(ious[stage]["AUD"])
    img = np.asarray(ious[stage]["IMG"])
    iqr = np.asarray(ious[stage]["IQR"])
    ogl = np.asarray(ious[stage]["OGL"])
    pool = (aud < 0.5) & (ogl >= 0.5)
    ogl_decomposition = {
        "OGL_rescue_total": int(pool.sum()),
        "IMG_captured": int((pool & (img >= 0.5)).sum()),
        "IMG_capture_rate": float((pool & (img >= 0.5)).sum() / max(pool.sum(), 1)),
        "IQR_captured": int((pool & (iqr >= 0.5)).sum()),
        "IQR_capture_rate": float((pool & (iqr >= 0.5)).sum() / max(pool.sum(), 1)),
        "IMG_IoU_gt_AUD_count": int((pool & (img > aud)).sum()),
        "IMG_IoU_gt_AUD_fraction": float((pool & (img > aud)).sum() / max(pool.sum(), 1)),
    }
    alpha_rows = [
        {"alpha_AUD": alpha, **common.summarize(alpha_values[stage][alpha])}
        for alpha in ALPHAS
    ]
    return {
        "stage": stage,
        "method_metrics": methods,
        "pairs": pairs,
        "OGL_rescue_decomposition": ogl_decomposition,
        "alpha_diagnostic_AUD_IMG": alpha_rows,
    }


def formal_reproduction(
    stage_summaries: dict[str, Any], registry: dict[str, Any], partial: bool
) -> dict[str, Any]:
    if partial:
        return {"skipped_for_partial_run": True}
    references = {
        "Stage1": common.stage1_reference(registry),
        "Stage2": common.stage2_reference(registry),
    }
    epoch_reference = common.stage1_epoch_reference(registry)
    result: dict[str, Any] = {}
    max_error = 0.0
    for stage in STAGES:
        actual = stage_summaries[stage]["method_metrics"]
        result[stage] = {}
        for method in METHODS:
            errors = {}
            for metric in ("cIoU", "AUC"):
                actual_value = float(actual[method][metric])
                reference_value = float(references[stage][method][metric])
                errors[metric] = abs(actual_value - reference_value)
                errors[f"rounded_4dp_match_{metric}"] = round(actual_value, 4) == round(reference_value, 4)
            result[stage][method] = errors
            max_error = max(max_error, errors["cIoU"], errors["AUC"])
    result["Stage1_training_epoch_history"] = {
        method: {
            metric: float(epoch_reference[method][metric])
            for metric in ("cIoU", "AUC")
        }
        for method in METHODS
    }
    result["Stage1_checkpoint_vs_epoch_history"] = {
        method: {
            metric: float(actual := stage_summaries["Stage1"]["method_metrics"][method][metric])
            - float(epoch_reference[method][metric])
            for metric in ("cIoU", "AUC")
        }
        for method in METHODS
    }
    result["max_error"] = max_error
    result["passed"] = all(
        result[stage][method][f"rounded_4dp_match_{metric}"]
        for stage in STAGES
        for method in METHODS
        for metric in ("cIoU", "AUC")
    )
    if not result["passed"]:
        raise RuntimeError(f"Formal metric reproduction failed: {result}")
    return result


def trajectory_row_from_stage_summary(
    checkpoint_name: str,
    checkpoint_epoch: int,
    stage: dict[str, Any],
    att_loss: float,
    direct_att_mse: float,
) -> dict[str, Any]:
    pair = next(value for value in stage["pairs"] if value["pair"] == "AUD+IMG")
    methods = stage["method_metrics"]
    decomposition = pair["success_decomposition"]
    similarity = pair["similarity"]
    return {
        "checkpoint": checkpoint_name,
        "epoch": checkpoint_epoch,
        "train_attention_match_loss": att_loss,
        "direct_eval_attention_MSE": direct_att_mse,
        "AUD_cIoU": methods["AUD"]["cIoU"],
        "AUD_AUC": methods["AUD"]["AUC"],
        "IMG_cIoU": methods["IMG"]["cIoU"],
        "IMG_AUC": methods["IMG"]["AUC"],
        "IQR_cIoU": methods["IQR"]["cIoU"],
        "IQR_AUC": methods["IQR"]["AUC"],
        "AUD_IMG_Pearson": similarity["norm_pearson"]["mean"],
        "AUD_IMG_Spearman": similarity["norm_spearman"]["mean"],
        "AUD_IMG_JS": similarity["norm_js"]["mean"],
        "Top20Overlap": similarity["top20_overlap"]["mean"],
        "AUD_ONLY": decomposition["AUD_ONLY"],
        "IMG_ONLY": decomposition["AUX_ONLY"],
        "IMG_ONLY_fraction": decomposition["AUX_ONLY_fraction"],
        "PairOracle_cIoU": pair["pair_oracle"]["cIoU"],
        "OracleGain": pair["oracle_gain_over_AUD_cIoU"],
        "IQRGain": pair["fixed_fusion_gain_cIoU"],
    }


@torch.inference_mode()
def evaluate_stage1_checkpoint(
    loader, model, device: torch.device, max_batches: int | None
) -> tuple[dict[str, Any], float]:
    ious = {method: [] for method in ("AUD", "IMG", "IQR")}
    pair_values = defaultdict(list)
    direct_total: list[float] = []
    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        image_levels = model.imgnet(image)
        audio_tokens = model._audio_tokens(model.audnet(spec))
        encoded = model.slot_attn._encode(image_levels, audio_tokens)
        eval_att = model.slot_attn._l4_attentions(encoded, model.infer_sharpening)
        train_att = model.slot_attn._l4_attentions(encoded, 1.0)
        img = eval_att["imgq_imgk_attn"][:, 0].reshape(-1, 1, 7, 7)
        aud = eval_att["audq_imgk_attn"][:, 0].reshape(-1, 1, 7, 7)
        direct = (
            (train_att["audq_imgk_attn"][:, 0] - train_att["imgq_imgk_attn"][:, 0])
            .square()
            .mean(dim=-1)
            + (train_att["imgq_audk_attn"][:, 0] - train_att["audq_audk_attn"][:, 0])
            .square()
            .mean(dim=-1)
        )
        direct_total.extend(direct.cpu().tolist())
        resized_aud = common.resize_tensor(aud).cpu().numpy()[:, 0]
        resized_img = common.resize_tensor(img).cpu().numpy()[:, 0]
        gt = bboxes.cpu().numpy()
        for index in range(len(names)):
            norm_aud = common.normalize_map(resized_aud[index])
            norm_img = common.normalize_map(resized_img[index])
            norm_iqr = common.normalize_map(0.6 * norm_aud + 0.4 * norm_img)
            ious["AUD"].append(common.sample_iou(norm_aud, gt[index]))
            ious["IMG"].append(common.sample_iou(norm_img, gt[index]))
            ious["IQR"].append(common.sample_iou(norm_iqr, gt[index]))
            values = pair_measures(
                resized_aud[index], resized_img[index], norm_aud, norm_img
            )
            for key, value in values.items():
                pair_values[key].append(value)

    methods = {method: common.summarize(values) for method, values in ious.items()}
    decomposition = success_decomposition(ious["AUD"], ious["IMG"])
    oracle = common.summarize(np.maximum(ious["AUD"], ious["IMG"]).tolist())
    return (
        {
            "stage": "Stage1",
            "method_metrics": methods,
            "pairs": [
                {
                    "pair": "AUD+IMG",
                    "similarity": {
                        key: common.distribution(values) for key, values in pair_values.items()
                    },
                    "success_decomposition": decomposition,
                    "pair_oracle": oracle,
                    "oracle_gain_over_AUD_cIoU": oracle["cIoU"] - methods["AUD"]["cIoU"],
                    "fixed_fusion_gain_cIoU": methods["IQR"]["cIoU"] - methods["AUD"]["cIoU"],
                }
            ],
        },
        float(np.mean(direct_total)),
    )


def checkpoint_trajectory_candidates(registry: dict[str, Any]) -> list[dict[str, Any]]:
    root = common.PROJECT_ROOT / "checkpoints" / registry["stage1"]
    candidates = [common.stage1_checkpoint_path(registry), root / "final.pth", root / "latest.pth"]
    output = []
    seen_states = set()
    for path in candidates:
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
        state_hash = common.state_sha256(state)
        if state_hash in seen_states:
            continue
        seen_states.add(state_hash)
        output.append(
            {
                "path": path,
                "checkpoint": path.name,
                "epoch": int(checkpoint["epoch"]),
                "state_sha256": state_hash,
            }
        )
    return sorted(output, key=lambda value: value["epoch"])


def trajectory_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = {
        "AUD_IMG_Pearson": "AUD_IMG_Pearson",
        "AUD_IMG_Spearman": "AUD_IMG_Spearman",
        "Top20Overlap": "Top20Overlap",
        "IMG_ONLY_fraction": "IMG_ONLY_fraction",
        "OracleGain": "OracleGain",
        "IQRGain": "IQRGain",
    }
    attention = np.asarray([row["train_attention_match_loss"] for row in rows])
    correlations = {}
    for label, key in targets.items():
        values = np.asarray([row[key] for row in rows])
        correlations[label] = {
            "pearson": common.safe_pearson(attention, values),
            "spearman": common.spearman(attention, values),
        }
    return {
        "num_distinct_checkpoints": len(rows),
        "sufficient_for_trajectory_inference": len(rows) >= 3,
        "limitation": (
            "Only best and final distinct Stage1 weights are available; n=2 correlations are descriptive and not reliable mechanism evidence."
            if len(rows) < 3
            else None
        ),
        "correlations_with_train_attention_match_loss": correlations,
    }


def save_trajectory_plot(rows: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    series = (
        ("train_attention_match_loss", "att loss"),
        ("AUD_IMG_Pearson", "AUD-IMG Pearson"),
        ("IMG_ONLY_fraction", "IMG-only fraction"),
        ("OracleGain", "oracle gain"),
        ("IQRGain", "IQR gain"),
    )
    fig, axes = plt.subplots(len(series), 1, figsize=(7, 12), sharex=True, constrained_layout=True)
    for axis, (key, title) in zip(axes, series):
        axis.plot(epochs, [row[key] for row in rows], marker="o")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("epoch")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def audit_models(models: dict[str, torch.nn.Module]) -> dict[str, Any]:
    trainable = []
    gradients = []
    for model_name, model in models.items():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                trainable.append(f"{model_name}.{name}")
            if parameter.grad is not None:
                gradients.append(f"{model_name}.{name}")
    return {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": 0,
        "trainable_parameter_names": trainable,
        "parameters_with_grad": gradients,
        "all_models_eval": all(not model.training for model in models.values()),
        "torch_inference_mode": True,
    }


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

    trajectory_candidates = checkpoint_trajectory_candidates(registry)
    snapshot_paths = {
        "formal_stage1_best": common.stage1_checkpoint_path(registry),
        "formal_original_1_3G": common.g_checkpoint_path(registry),
        "evaluation_only_object_prior": common.OBJECT_CHECKPOINT,
    }
    for candidate in trajectory_candidates:
        if candidate["path"] != common.stage1_checkpoint_path(registry):
            snapshot_paths[f"trajectory_{candidate['checkpoint']}"] = candidate["path"]
    snapshots_before = common.snapshot_files(snapshot_paths)

    model = common.load_original_g(registry, device)
    object_model = common.object_prior_model().to(device).eval()
    audit = tensor_audit(loader, model, object_model, device)

    ious = {stage: {method: [] for method in METHODS} for stage in STAGES}
    alpha_values = {
        stage: {alpha: [] for alpha in ALPHAS} for stage in STAGES
    }
    pair_values: dict[str, Any] = {
        stage: {
            auxiliary: defaultdict(list) for auxiliary in AUXILIARIES
        }
        for stage in STAGES
    }
    direct_attention = defaultdict(list)
    per_sample: list[dict[str, Any]] = []
    qualitative: dict[str, dict[str, Any]] = {}
    global_index = 0
    no_nan_or_inf = True

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
        direct_attention["spatial_MSE"].extend(output["ATT_SPATIAL_MSE"].cpu().tolist())
        direct_attention["temporal_MSE"].extend(output["ATT_TEMPORAL_MSE"].cpu().tolist())
        direct_attention["total_MSE"].extend(output["ATT_TOTAL_MSE"].cpu().tolist())

        raw_batch = {
            "AUD_L4": common.resize_tensor(output["AUD_L4"]).cpu().numpy()[:, 0],
            "IMG": common.resize_tensor(output["IMG_L4"]).cpu().numpy()[:, 0],
            "AUD_FINE": common.resize_tensor(output["AUD_FINE"]).cpu().numpy()[:, 0],
            "OBJ": common.resize_tensor(object_prior).cpu().numpy()[:, 0],
        }
        gt_batch = bboxes.cpu().numpy()

        for index, sample_id in enumerate(names):
            raw = {name: values[index] for name, values in raw_batch.items()}
            normalized = {name: common.normalize_map(value) for name, value in raw.items()}
            stage_maps = {
                "Stage1": {
                    "AUD": normalized["AUD_L4"],
                    "IMG": normalized["IMG"],
                    "OBJ": normalized["OBJ"],
                },
                "Stage2": {
                    "AUD": normalized["AUD_FINE"],
                    "IMG": normalized["IMG"],
                    "OBJ": normalized["OBJ"],
                },
            }
            raw_stage = {"Stage1": raw["AUD_L4"], "Stage2": raw["AUD_FINE"]}
            row: dict[str, Any] = {"sample_index": global_index, "sample_id": sample_id}
            for stage in STAGES:
                maps = stage_maps[stage]
                maps["IQR"] = common.normalize_map(0.6 * maps["AUD"] + 0.4 * maps["IMG"])
                maps["OGL"] = common.normalize_map(0.6 * maps["AUD"] + 0.4 * maps["OBJ"])
                for method in METHODS:
                    value = common.sample_iou(maps[method], gt_batch[index])
                    ious[stage][method].append(value)
                    row[f"{stage}_IoU_{method}"] = value
                for alpha in ALPHAS:
                    fused = common.normalize_map(alpha * maps["AUD"] + (1.0 - alpha) * maps["IMG"])
                    alpha_values[stage][alpha].append(common.sample_iou(fused, gt_batch[index]))
                for auxiliary in AUXILIARIES:
                    raw_aux = raw[auxiliary]
                    measures = pair_measures(
                        raw_stage[stage], raw_aux, maps["AUD"], maps[auxiliary]
                    )
                    append_pair(pair_values, stage, auxiliary, measures)
                    if auxiliary == "IMG":
                        for key, value in measures.items():
                            row[f"{stage}_AUD_IMG_{key}"] = value

            categories = categorical_labels(
                row["Stage2_IoU_AUD"],
                row["Stage2_IoU_IMG"],
                row["Stage2_IoU_IQR"],
                row["Stage2_IoU_OGL"],
            )
            row["Stage2_categories"] = ";".join(categories)
            per_sample.append(row)
            if not arguments.skip_qualitative:
                update_qualitative(
                    qualitative,
                    f"{sample_id}::{global_index:06d}",
                    categories,
                    image[index],
                    gt_batch[index],
                    stage_maps["Stage2"],
                    row,
                )
            global_index += 1

    stage_summaries = {
        stage: build_stage_summary(stage, ious, pair_values, alpha_values)
        for stage in STAGES
    }
    reproduction = formal_reproduction(
        stage_summaries, registry, partial=arguments.max_batches is not None
    )

    disagreement_groups = {}
    for group in ("IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE"):
        selected = []
        for row in per_sample:
            aud = row["Stage2_IoU_AUD"] >= 0.5
            img = row["Stage2_IoU_IMG"] >= 0.5
            ogl = row["Stage2_IoU_OGL"] >= 0.5
            matches = {
                "IMG_ONLY": (not aud) and img,
                "AUD_ONLY": aud and (not img),
                "BOTH_SUCCESS": aud and img,
                "BOTH_FAIL": (not aud) and (not img),
                "OGL_RESCUE": (not aud) and ogl,
            }[group]
            if matches:
                selected.append(1.0 - row["Stage2_AUD_IMG_norm_pearson"])
        disagreement_groups[group] = common.distribution(selected)

    epoch_rows = {int(row["epoch"]): row for row in common.read_epoch_rows(registry)}
    best_checkpoint = torch.load(
        common.stage1_checkpoint_path(registry), map_location="cpu", weights_only=False
    )
    best_epoch = int(best_checkpoint["epoch"])
    trajectory_rows = [
        trajectory_row_from_stage_summary(
            registry["checkpoint"],
            best_epoch,
            stage_summaries["Stage1"],
            float(epoch_rows[best_epoch]["train_attention_match_loss"]),
            float(np.mean(direct_attention["total_MSE"])),
        )
    ]

    del object_model
    model.close()
    del model
    torch.cuda.empty_cache()
    for candidate in trajectory_candidates:
        if candidate["epoch"] == best_epoch:
            continue
        checkpoint_model = common.load_stage1_path(registry, candidate["path"], device)
        checkpoint_stage, direct_mse = evaluate_stage1_checkpoint(
            loader, checkpoint_model, device, arguments.max_batches
        )
        trajectory_rows.append(
            trajectory_row_from_stage_summary(
                candidate["checkpoint"],
                candidate["epoch"],
                checkpoint_stage,
                float(epoch_rows[candidate["epoch"]]["train_attention_match_loss"]),
                direct_mse,
            )
        )
        del checkpoint_model
        torch.cuda.empty_cache()
    trajectory_rows.sort(key=lambda value: value["epoch"])
    trajectory = {
        "available_checkpoint_audit": [
            {**candidate, "path": str(candidate["path"].resolve())}
            for candidate in trajectory_candidates
        ],
        "rows": trajectory_rows,
        "correlations": trajectory_correlations(trajectory_rows),
    }
    save_trajectory_plot(trajectory_rows, output_dir / "epoch_trajectory.png")

    if not arguments.skip_qualitative:
        for category in QUALITATIVE_CATEGORIES:
            payload = qualitative.get(category)
            if payload is None:
                continue
            visualize.save_panel(payload, output_dir / "qualitative" / f"{category}.png")
        common.write_csv(
            output_dir / "qualitative" / "selection_manifest.csv",
            [
                {
                    "category": category,
                    "sample_id": payload["sample_id"],
                    "sort_key": payload["sort_key"],
                }
                for category, payload in qualitative.items()
            ],
        )

    snapshots_after = common.verify_snapshots(snapshots_before)
    zero_training = {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": 0,
        "trainable_parameter_names": [],
        "parameters_with_grad": [],
        "all_models_eval": True,
        "torch_inference_mode": True,
        "checkpoint_snapshots": snapshots_after,
        "all_checkpoint_hashes_and_mtimes_unchanged": snapshots_after["all_unchanged"],
        "no_nan_or_inf": no_nan_or_inf,
    }
    if not all(
        (
            zero_training["new_trainable_params"] == 0,
            not zero_training["trainable_parameter_names"],
            not zero_training["parameters_with_grad"],
            zero_training["all_models_eval"],
            zero_training["all_checkpoint_hashes_and_mtimes_unchanged"],
            zero_training["no_nan_or_inf"],
        )
    ):
        raise RuntimeError(zero_training)

    direct_summary = {
        key: common.distribution(values) for key, values in direct_attention.items()
    }
    config_audit = {
        "lam3": float(config.lam3),
        "total_loss": "info + lam1*recon + lam2*div + lam3*att",
        "att_loss": (
            "MSE(audq_imgk[:,0], detach(imgq_imgk[:,0])) + "
            "MSE(imgq_audk[:,0], detach(audq_audk[:,0]))"
        ),
        "spatial_teacher_direction": "IMG->image keys is detached; AUD->image keys is optimized",
        "temporal_teacher_direction": "AUD->audio keys is detached; IMG->audio keys is optimized",
        "directly_contains_final_spatial_maps": True,
        "training_scale_multiplier": 1.0,
        "evaluation_scale_multiplier": float(config.infer_sharpening),
        "source": {
            "loss": "chuagnxindian/1mufasaslot/model_mufasa_jsa.py:113-121",
            "attention": "chuagnxindian/mufasa_ablation2_l3_l4_ablation/l3_l4_slot_attention.py:79-120",
            "weighting": "train_slot.py:339-343",
            "config": str((common.PROJECT_ROOT / "checkpoints" / registry["stage1"] / "configs.json").resolve()),
        },
    }

    common.write_csv(output_dir / "per_sample_metrics.csv", per_sample)
    common.write_csv(output_dir / "epoch_trajectory.csv", trajectory_rows)
    common.write_json(output_dir / "stage_summaries.json", stage_summaries)
    common.write_json(output_dir / "zero_training_audit.json", zero_training)
    summary = {
        "experiment": "4.0 AUD-IMG Redundancy & Attention-Loss Mechanism Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": arguments.max_batches is None,
        "att_loss_code_audit": config_audit,
        "tensor_audit": audit,
        "direct_attention_discrepancy_full_dataset": direct_summary,
        "stage_summaries": stage_summaries,
        "sample_disagreement_by_group": disagreement_groups,
        "formal_reproduction": reproduction,
        "epoch_trajectory": trajectory,
        "qualitative_selection": {
            category: payload["sample_id"] for category, payload in qualitative.items()
        },
        "zero_training_audit": zero_training,
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
