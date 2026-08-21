#!/usr/bin/env python3
"""Experiment 5.2: zero-training expansion/shrink error diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn import metrics as sklearn_metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import common


BENEFICIAL_GAIN = 0.01
DOMINANCE_MARGIN = 0.01
EPS = 1e-12

SPATIAL_SIGNALS = (
    "AUD_area",
    "IMG_area",
    "IMG_AUD_area_ratio",
    "AUD_IMG_mask_IoU",
    "AUD_only_area",
    "IMG_only_area",
    "intersection_area",
    "AUD_only_ratio",
    "IMG_only_ratio",
    "intersection_ratio",
    "disagreement_mean",
    "disagreement_max",
    "disagreement_std",
    "AUD_IMG_Pearson",
)
CONFIDENCE_SIGNALS = tuple(
    f"{branch}_{metric}"
    for branch in ("AUD", "IMG")
    for metric in (
        "raw_max",
        "raw_mean",
        "raw_std",
        "raw_entropy",
        "raw_top20_mass",
        "foreground_concentration",
    )
) + tuple(
    f"DELTA_IMG_MINUS_AUD_{metric}"
    for metric in (
        "raw_max",
        "raw_mean",
        "raw_std",
        "raw_entropy",
        "raw_top20_mass",
        "foreground_concentration",
    )
)
SLOT_SIGNALS = tuple(
    f"{level}_{metric}"
    for level in ("L3", "L4", "HR14")
    for metric in (
        "ownership_entropy",
        "target_confidence",
        "target_other_margin",
        "target_spatial_concentration",
    )
)
CROSS_LEVEL_SIGNALS = (
    "AUD_L3_L4_Pearson",
    "AUD_L3_L4_Spearman",
    "AUD_L3_L4_mask_IoU",
    "AUD_L3_L4_area_ratio",
    "AUD_L3_L4_disagreement",
    "IMG_L3_L4_Pearson",
    "IMG_L3_L4_Spearman",
    "IMG_L3_L4_mask_IoU",
    "IMG_L3_L4_area_ratio",
    "IMG_L3_L4_disagreement",
    "L3_L4_target_ownership_Pearson",
)
REGION_SIGNALS = (
    "AUD_only_IMG_response",
    "AUD_only_PROP_F34_similarity",
    "AUD_only_PROP_K34_similarity",
    "AUD_only_HR14_target_ownership",
    "COMMON_IMG_response",
    "COMMON_PROP_F34_similarity",
    "COMMON_PROP_K34_similarity",
    "COMMON_HR14_target_ownership",
    "REGION_DELTA_IMG_response",
    "REGION_DELTA_PROP_F34_similarity",
    "REGION_DELTA_PROP_K34_similarity",
    "REGION_DELTA_HR14_target_ownership",
    "SEED_CONF",
    "DELTA_SEMANTIC_SLOT",
    "DELTA_RECIPROCAL_L4",
)
ALL_SIGNALS = SPATIAL_SIGNALS + CONFIDENCE_SIGNALS + SLOT_SIGNALS + CROSS_LEVEL_SIGNALS + REGION_SIGNALS
FEATURE_SETS = {
    "SPATIAL": SPATIAL_SIGNALS,
    "INTERNAL": CONFIDENCE_SIGNALS + SLOT_SIGNALS + CROSS_LEVEL_SIGNALS + REGION_SIGNALS,
    "ALL": ALL_SIGNALS,
}


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


def reference_data(setting: str) -> dict[str, Any]:
    rows41 = load_csv(common.reference_41_dir(setting) / "per_sample_metrics.csv")
    rows51 = load_csv(common.reference_51_dir(setting) / "per_sample_metrics.csv")
    raw41 = np.load(common.reference_41_dir(setting) / "raw_maps.npz")
    raw51 = np.load(common.reference_51_dir(setting) / "raw_propagation_maps.npz")
    return {"rows41": rows41, "rows51": rows51, "raw41": raw41, "raw51": raw51}


def normalize_batch(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(start_dim=1)
    minimum = flat.min(dim=1).values[:, None, None, None]
    maximum = flat.max(dim=1).values[:, None, None, None]
    span = maximum - minimum
    return torch.where(span != 0, (value - minimum) / span, value)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def finite_mean(value: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(value)[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else math.nan


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def normalized_entropy(value: np.ndarray) -> float:
    flat = np.clip(np.asarray(value, dtype=np.float64).ravel(), 0.0, None)
    total = flat.sum()
    if total <= 0 or flat.size <= 1:
        return 0.0
    probability = flat / total
    return float(-np.sum(probability * np.log(probability + EPS)) / math.log(flat.size))


def top_mass(value: np.ndarray, fraction: float = 0.20) -> float:
    flat = np.clip(np.asarray(value, dtype=np.float64).ravel(), 0.0, None)
    count = max(1, int(math.ceil(flat.size * fraction)))
    total = flat.sum()
    if total <= 0:
        return 0.0
    return float(np.partition(flat, -count)[-count:].sum() / total)


def activation_stats(raw: np.ndarray, normalized: np.ndarray) -> dict[str, float]:
    mask = normalized >= 0.6
    inside = finite_mean(normalized, mask)
    outside = finite_mean(normalized, ~mask)
    concentration = inside - outside if math.isfinite(inside) and math.isfinite(outside) else math.nan
    return {
        "raw_max": float(np.max(raw)),
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_entropy": normalized_entropy(raw),
        "raw_top20_mass": top_mass(raw),
        "foreground_concentration": concentration,
    }


def ownership_stats(value: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(value, dtype=np.float64), EPS, 1.0)
    entropy = -np.sum(probability * np.log(probability), axis=0) / math.log(probability.shape[0])
    target = probability[0]
    other = np.max(probability[1:], axis=0)
    return {
        "ownership_entropy": float(entropy.mean()),
        "target_confidence": float(target.mean()),
        "target_other_margin": float((target - other).mean()),
        "target_spatial_concentration": top_mass(target),
    }


def cross_level_stats(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = common.normalize_map(first)
    second = common.normalize_map(second)
    mask_first = first >= 0.6
    mask_second = second >= 0.6
    return {
        "Pearson": common.safe_pearson(first, second),
        "Spearman": common.spearman(first, second),
        "mask_IoU": binary_iou(mask_first, mask_second),
        "area_ratio": safe_ratio(float(mask_first.sum()), float(mask_second.sum())),
        "disagreement": float(np.mean(np.abs(first - second))),
    }


@torch.inference_mode()
def extract_all(model, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = model.teacher
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    scale = teacher.infer_sharpening
    l4 = teacher.slot_attn._l4_attentions(encoded, scale_multiplier=scale)
    aud_l3_all = teacher.slot_attn._attention(encoded["audio_query"], encoded["visual_keys"][0], scale)
    img_l3_all = teacher.slot_attn._attention(encoded["visual_queries"][0], encoded["visual_keys"][0], scale)

    ownership = {}
    for name, query, key in (
        ("L3", encoded["visual_queries"][0], encoded["visual_keys"][0]),
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
        "AUD_L3_ALL": aud_l3_all,
        "IMG_L3_ALL": img_l3_all,
        "AUD_L4_ALL": l4["audq_imgk_attn"],
        "IMG_L4_ALL": l4["imgq_imgk_attn"],
        "AUD_FINE": aud_fine_all[:, 0].reshape(batch, 1, 14, 14),
        "OWN_L3": ownership["L3"],
        "OWN_L4": ownership["L4"],
        "OWN_HR14": ownership["HR14"],
        "F34": f34,
        "K34": k34,
        "F3_SPATIAL": f3_spatial,
        "F4_UP": f4_up,
        "DELTA_F3": delta_f3,
        "f4_token_error": (f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]).abs().max(),
    }


def strict_error_type(expand_gain: float, shrink_gain: float) -> str:
    if expand_gain >= BENEFICIAL_GAIN and expand_gain >= shrink_gain + DOMINANCE_MARGIN:
        return "EXPAND"
    if shrink_gain >= BENEFICIAL_GAIN and shrink_gain >= expand_gain + DOMINANCE_MARGIN:
        return "SHRINK"
    return "KEEP_AMBIGUOUS"


def summarize_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return common.distribution([float(row[key]) for row in rows])


def cohen_d(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size < 2 or negative.size < 2:
        return math.nan
    pooled = math.sqrt(((positive.size - 1) * positive.var() + (negative.size - 1) * negative.var()) / (positive.size + negative.size - 2))
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else 0.0


def task_labels(rows: list[dict[str, Any]], task: str) -> tuple[np.ndarray, np.ndarray]:
    types = np.asarray([row["error_type"] for row in rows])
    if task == "EXPAND_vs_SHRINK":
        subset = np.flatnonzero(np.isin(types, ["EXPAND", "SHRINK"]))
        return subset, (types[subset] == "SHRINK").astype(np.int64)
    if task == "SHRINK_vs_OTHERS":
        subset = np.arange(len(rows))
        return subset, (types == "SHRINK").astype(np.int64)
    if task == "EXPAND_vs_OTHERS":
        subset = np.arange(len(rows))
        return subset, (types == "EXPAND").astype(np.int64)
    raise KeyError(task)


def single_variable_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    expand_gain = np.asarray([float(row["expand_gain"]) for row in rows])
    shrink_gain = np.asarray([float(row["shrink_gain"]) for row in rows])
    types = np.asarray([row["error_type"] for row in rows])
    for task in ("EXPAND_vs_SHRINK", "SHRINK_vs_OTHERS", "EXPAND_vs_OTHERS"):
        subset, labels = task_labels(rows, task)
        for signal in ALL_SIGNALS:
            all_scores = np.asarray([float(row[signal]) for row in rows], dtype=np.float64)
            scores = all_scores[subset]
            finite = np.isfinite(scores)
            y = labels[finite]
            x = scores[finite]
            if x.size < 3 or np.unique(y).size < 2:
                raw_auc = directed_auc = auprc = effect = math.nan
                direction = "N/A"
            else:
                raw_auc = float(sklearn_metrics.roc_auc_score(y, x))
                direction = "higher" if raw_auc >= 0.5 else "lower"
                directed = x if direction == "higher" else -x
                directed_auc = float(sklearn_metrics.roc_auc_score(y, directed))
                auprc = float(sklearn_metrics.average_precision_score(y, directed))
                effect = cohen_d(x[y == 1], x[y == 0])
            finite_all = np.isfinite(all_scores)
            output.append(
                {
                    "task": task,
                    "positive_class": "SHRINK" if task != "EXPAND_vs_OTHERS" else "EXPAND",
                    "signal": signal,
                    "direction": direction,
                    "AUROC_raw": raw_auc,
                    "AUROC_directed": directed_auc,
                    "AUPRC_directed": auprc,
                    "effect_size_cohen_d": effect,
                    "positive_mean": float(x[y == 1].mean()) if x.size and np.any(y == 1) else math.nan,
                    "positive_std": float(x[y == 1].std()) if x.size and np.any(y == 1) else math.nan,
                    "negative_mean": float(x[y == 0].mean()) if x.size and np.any(y == 0) else math.nan,
                    "negative_std": float(x[y == 0].std()) if x.size and np.any(y == 0) else math.nan,
                    "Pearson_expand_gain": common.safe_pearson(all_scores[finite_all], expand_gain[finite_all]),
                    "Spearman_expand_gain": common.safe_pearson(rankdata(all_scores[finite_all]), rankdata(expand_gain[finite_all])),
                    "Pearson_shrink_gain": common.safe_pearson(all_scores[finite_all], shrink_gain[finite_all]),
                    "Spearman_shrink_gain": common.safe_pearson(rankdata(all_scores[finite_all]), rankdata(shrink_gain[finite_all])),
                    "EXPAND_mean": float(np.nanmean(all_scores[types == "EXPAND"])),
                    "SHRINK_mean": float(np.nanmean(all_scores[types == "SHRINK"])),
                    "KEEP_mean": float(np.nanmean(all_scores[types == "KEEP_AMBIGUOUS"])),
                    "num_samples": int(x.size),
                }
            )
    return output


def group_ids(setting: str, rows: list[dict[str, Any]]) -> np.ndarray:
    if setting == "vggss_144k":
        return np.asarray([str(row["sample_id"]).rsplit("_", 1)[0] for row in rows])
    return np.asarray([str(row["sample_id"]) for row in rows])


def split_iterator(labels: np.ndarray, groups: np.ndarray, seed: int):
    class_counts = np.bincount(labels)
    splits = max(2, min(5, int(class_counts.min())))
    try:
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
        candidates = list(splitter.split(np.zeros(labels.size), labels, groups))
        if all(np.unique(labels[train]).size == np.unique(labels).size for train, _test in candidates):
            return candidates, "StratifiedGroupKFold", splits
    except ValueError:
        pass
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(labels.size), labels)), "StratifiedKFold", splits


def fit_binary_probe(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predictions = np.zeros(labels.size, dtype=np.int64)
    probabilities = np.zeros(labels.size, dtype=np.float64)
    splits, split_name, split_count = split_iterator(labels, groups, seed)
    for train, test in splits:
        pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        )
        pipeline.fit(matrix[train], labels[train])
        predictions[test] = pipeline.predict(matrix[test])
        probabilities[test] = pipeline.predict_proba(matrix[test])[:, 1]
    final = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
    )
    final.fit(matrix, labels)
    coefficients = final[-1].coef_[0]
    top_coefficients = sorted(
        ({"signal": name, "coefficient": float(value)} for name, value in zip(feature_names, coefficients)),
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )[:12]
    metrics = {
        "accuracy": float(sklearn_metrics.accuracy_score(labels, predictions)),
        "balanced_accuracy": float(sklearn_metrics.balanced_accuracy_score(labels, predictions)),
        "macro_F1": float(sklearn_metrics.f1_score(labels, predictions, average="macro")),
        "AUROC": float(sklearn_metrics.roc_auc_score(labels, probabilities)),
        "AUPRC": float(sklearn_metrics.average_precision_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "split": split_name,
        "folds": split_count,
        "top_coefficients_full_fit_diagnostic": top_coefficients,
    }
    return metrics, predictions, probabilities


def fit_multiclass_probe(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predictions = np.zeros(labels.size, dtype=np.int64)
    probabilities = np.zeros((labels.size, 3), dtype=np.float64)
    splits, split_name, split_count = split_iterator(labels, groups, seed)
    for train, test in splits:
        pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=seed),
        )
        pipeline.fit(matrix[train], labels[train])
        predictions[test] = pipeline.predict(matrix[test])
        probabilities[test] = pipeline.predict_proba(matrix[test])
    return (
        {
            "accuracy": float(sklearn_metrics.accuracy_score(labels, predictions)),
            "balanced_accuracy": float(sklearn_metrics.balanced_accuracy_score(labels, predictions)),
            "macro_F1": float(sklearn_metrics.f1_score(labels, predictions, average="macro")),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
            "class_counts": {str(index): int((labels == index).sum()) for index in range(3)},
            "split": split_name,
            "folds": split_count,
        },
        predictions,
        probabilities,
    )


def probe_suite(setting: str, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    groups_all = group_ids(setting, rows)
    result: dict[str, Any] = {}
    for feature_set, features in FEATURE_SETS.items():
        matrix_all = np.asarray([[float(row[key]) for key in features] for row in rows], dtype=np.float64)
        result[feature_set] = {}
        for task in ("EXPAND_vs_SHRINK", "SHRINK_vs_OTHERS", "EXPAND_vs_OTHERS"):
            subset, labels = task_labels(rows, task)
            metrics, predictions, probabilities = fit_binary_probe(
                matrix_all[subset], labels, groups_all[subset], features, seed
            )
            result[feature_set][task] = metrics
            if feature_set == "ALL":
                for local, row_index in enumerate(subset):
                    rows[row_index][f"probe_{task}_prediction"] = int(predictions[local])
                    rows[row_index][f"probe_{task}_probability"] = float(probabilities[local])

        labels_multi = np.asarray(
            [{"EXPAND": 0, "SHRINK": 1, "KEEP_AMBIGUOUS": 2}[row["error_type"]] for row in rows],
            dtype=np.int64,
        )
        metrics, predictions, probabilities = fit_multiclass_probe(matrix_all, labels_multi, groups_all, seed)
        result[feature_set]["THREE_CLASS"] = metrics
        if feature_set == "ALL":
            names = ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS")
            for index, row in enumerate(rows):
                row["probe_three_class_prediction"] = names[predictions[index]]
                for class_index, name in enumerate(names):
                    row[f"probe_three_class_probability_{name}"] = float(probabilities[index, class_index])
    return result


def error_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    multi = {
        "EXPAND_ONLY": sum(row["expand_beneficial"] and not row["shrink_beneficial"] for row in rows),
        "SHRINK_ONLY": sum(row["shrink_beneficial"] and not row["expand_beneficial"] for row in rows),
        "BOTH_BENEFICIAL": sum(row["expand_beneficial"] and row["shrink_beneficial"] for row in rows),
        "NEITHER": sum(not row["expand_beneficial"] and not row["shrink_beneficial"] for row in rows),
    }
    strict = {name: sum(row["error_type"] == name for row in rows) for name in ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS")}
    return {
        "total": total,
        "multi_label": {key: {"count": value, "fraction": value / total} for key, value in multi.items()},
        "strict_three_class": {key: {"count": value, "fraction": value / total} for key, value in strict.items()},
        "expand_gain": summarize_key(rows, "expand_gain"),
        "shrink_gain": summarize_key(rows, "shrink_gain"),
    }


def grouped_distributions(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    output = {}
    for error_type in ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS"):
        selected = [row for row in rows if row["error_type"] == error_type]
        output[error_type] = {"count": len(selected), **{key: summarize_key(selected, key) for key in keys}}
    return output


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aud = np.asarray([row["IoU_AUD"] for row in rows], dtype=np.float64)
    expand = np.maximum(aud, np.asarray([row["IoU_EXPAND_BEST"] for row in rows]))
    shrink = np.maximum(aud, np.asarray([row["IoU_SHRINK"] for row in rows]))
    combined = np.maximum(expand, shrink)
    baseline = common.summarize(aud.tolist())
    output = {"AUD": baseline}
    for name, value in (("EXPAND_ONLY_ORACLE", expand), ("SHRINK_ONLY_ORACLE", shrink), ("EXPAND_SHRINK_ORACLE", combined)):
        summary = common.summarize(value.tolist())
        summary["delta_cIoU"] = summary["cIoU"] - baseline["cIoU"]
        summary["delta_AUC"] = summary["AUC"] - baseline["AUC"]
        summary["delta_mean_sample_IoU"] = summary["mean_sample_cIoU"] - baseline["mean_sample_cIoU"]
        output[name] = summary
    return output


def relationship_51(rows: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = {
        "PROP_ONLY": lambda row: row["IoU_AUD"] < 0.5 and max(row["IoU_PROP_F34"], row["IoU_PROP_K34"]) >= 0.5,
        "IMG_ONLY_SHRINK": lambda row: row["group_IMG_ONLY_SHRINK"],
        "AUD_OVER_EXPANSION": lambda row: row["AUD_OVER_EXPANSION"],
        "PROP_HURT": lambda row: row["IoU_AUD"] >= 0.5 and row["IoU_PROP_F34"] < 0.5 and row["IoU_PROP_K34"] < 0.5,
        "OGL_RESCUE": lambda row: row["group_OGL_RESCUE"],
    }
    output = {}
    for name, predicate in definitions.items():
        selected = [row for row in rows if predicate(row)]
        counts = {error: sum(row["error_type"] == error for row in selected) for error in ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS")}
        output[name] = {
            "count": len(selected),
            "error_type_counts": counts,
            "error_type_fractions": {key: safe_ratio(value, len(selected)) for key, value in counts.items()},
            "expand_gain": summarize_key(selected, "expand_gain") if selected else {},
            "shrink_gain": summarize_key(selected, "shrink_gain") if selected else {},
        }
    return output


def failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selections: list[tuple[str, dict[str, Any]]] = []
    selections.append(("MAX_EXPAND_GAIN", max(rows, key=lambda row: (row["expand_gain"], row["sample_id"]))))
    selections.append(("MAX_SHRINK_GAIN", max(rows, key=lambda row: (row["shrink_gain"], row["sample_id"]))))
    for category, true_type, predicted in (
        ("EXPAND_MISROUTED_AS_SHRINK", "EXPAND", "SHRINK"),
        ("SHRINK_MISROUTED_AS_EXPAND", "SHRINK", "EXPAND"),
    ):
        candidates = [row for row in rows if row["error_type"] == true_type and row.get("probe_three_class_prediction") == predicted]
        if candidates:
            key = "expand_gain" if true_type == "EXPAND" else "shrink_gain"
            selections.append((category, max(candidates, key=lambda row: (row[key], row["sample_id"]))))
    output = []
    for category, row in selections:
        output.append(
            {
                "category": category,
                "sample_id": row["sample_id"],
                "error_type": row["error_type"],
                "probe_prediction": row.get("probe_three_class_prediction"),
                "IoU_AUD": row["IoU_AUD"],
                "IoU_IMG": row["IoU_IMG"],
                "IoU_PROP_F34": row["IoU_PROP_F34"],
                "IoU_PROP_K34": row["IoU_PROP_K34"],
                "expand_gain": row["expand_gain"],
                "shrink_gain": row["shrink_gain"],
                "IMG_AUD_area_ratio": row["IMG_AUD_area_ratio"],
                "AUD_only_ratio": row["AUD_only_ratio"],
            }
        )
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
    checkpoints_before = common.snapshot_files(
        {"formal_stage1": common.stage1_checkpoint_path(registry), "formal_original_1_3G": common.g_checkpoint_path(registry)}
    )
    model = common.load_original_g(registry, device)
    parameters_with_grad = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    rows: list[dict[str, Any]] = []
    sample_mismatches = 0
    no_nan_or_inf = True
    raw_errors = {"AUD_L4": 0.0, "IMG_L4": 0.0, "AUD_FINE": 0.0}
    metric_errors = {"AUD": 0.0, "IMG": 0.0, "PROP_F34": 0.0, "PROP_K34": 0.0, "OGL": 0.0}
    tensor_audit: dict[str, Any] | None = None
    global_index = 0

    for batch_index, (image, spec, bboxes, names, _labels) in enumerate(tqdm(loader)):
        if arguments.max_batches is not None and batch_index >= arguments.max_batches:
            break
        image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()
        output = extract_all(model, image, spec)
        batch = len(names)
        if tensor_audit is None:
            tensor_audit = {
                "AUD_L3_ALL": list(output["AUD_L3_ALL"].shape),
                "IMG_L3_ALL": list(output["IMG_L3_ALL"].shape),
                "AUD_L4_ALL": list(output["AUD_L4_ALL"].shape),
                "IMG_L4_ALL": list(output["IMG_L4_ALL"].shape),
                "AUD_FINE": list(output["AUD_FINE"].shape),
                "OWN_L3": list(output["OWN_L3"].shape),
                "OWN_L4": list(output["OWN_L4"].shape),
                "OWN_HR14": list(output["OWN_HR14"].shape),
                "F34": list(output["F34"].shape),
                "K34": list(output["K34"].shape),
                "f4_token_error": float(output["f4_token_error"]),
            }
        no_nan_or_inf = no_nan_or_inf and all(
            torch.isfinite(value).all().item() for value in output.values() if isinstance(value, torch.Tensor)
        )

        raw_aud_l4 = output["AUD_L4_ALL"][:, 0].reshape(batch, 7, 7).cpu().numpy()
        raw_img_l4 = output["IMG_L4_ALL"][:, 0].reshape(batch, 7, 7).cpu().numpy()
        raw_aud_fine = output["AUD_FINE"][:, 0].cpu().numpy()
        eval_aud = normalize_batch(common.resize_tensor(output["AUD_FINE"])).cpu().numpy()[:, 0]
        eval_img = normalize_batch(common.resize_tensor(output["IMG_L4_ALL"][:, 0].reshape(batch, 1, 7, 7))).cpu().numpy()[:, 0]
        prop_f_raw = torch.from_numpy(references["raw51"]["PROP_F34_RAW"][global_index : global_index + batch]).to(device)[:, None]
        prop_k_raw = torch.from_numpy(references["raw51"]["PROP_K34_RAW"][global_index : global_index + batch]).to(device)[:, None]
        eval_prop_f = normalize_batch(F.interpolate(prop_f_raw, (224, 224), mode="bicubic", align_corners=False)).cpu().numpy()[:, 0]
        eval_prop_k = normalize_batch(F.interpolate(prop_k_raw, (224, 224), mode="bicubic", align_corners=False)).cpu().numpy()[:, 0]
        img14 = normalize_batch(F.interpolate(output["IMG_L4_ALL"][:, 0].reshape(batch, 1, 7, 7), (14, 14), mode="bicubic", align_corners=False)).cpu().numpy()[:, 0]
        aud14 = normalize_batch(output["AUD_FINE"]).cpu().numpy()[:, 0]
        prop_f14 = normalize_batch(prop_f_raw).cpu().numpy()[:, 0]
        prop_k14 = normalize_batch(prop_k_raw).cpu().numpy()[:, 0]
        gt_raw = bboxes.cpu().numpy()

        for local, sample_id in enumerate(names):
            ref41 = references["rows41"][global_index]
            ref51 = references["rows51"][global_index]
            mismatch = (
                ref41["sample_id"] != sample_id
                or ref51["sample_id"] != sample_id
                or str(references["raw41"]["sample_id"][global_index]) != sample_id
                or str(references["raw51"]["sample_id"][global_index]) != sample_id
            )
            sample_mismatches += int(mismatch)
            raw_errors["AUD_L4"] = max(raw_errors["AUD_L4"], float(np.max(np.abs(raw_aud_l4[local] - references["raw41"]["AUD_L4"][global_index]))))
            raw_errors["IMG_L4"] = max(raw_errors["IMG_L4"], float(np.max(np.abs(raw_img_l4[local] - references["raw41"]["IMG_L4"][global_index]))))
            raw_errors["AUD_FINE"] = max(raw_errors["AUD_FINE"], float(np.max(np.abs(raw_aud_fine[local] - references["raw41"]["AUD_FINE"][global_index]))))

            gt = gt_raw[local]
            aud = eval_aud[local]
            img = eval_img[local]
            prop_f = eval_prop_f[local]
            prop_k = eval_prop_k[local]
            iou_aud = common.sample_iou(aud, gt)
            iou_img = common.sample_iou(img, gt)
            iou_prop_f = common.sample_iou(prop_f, gt)
            iou_prop_k = common.sample_iou(prop_k, gt)
            for key, value, reference in (
                ("AUD", iou_aud, ref51["IoU_AUD"]),
                ("IMG", iou_img, ref51["IoU_IMG"]),
                ("PROP_F34", iou_prop_f, ref51["IoU_PROP_F34"]),
                ("PROP_K34", iou_prop_k, ref51["IoU_PROP_K34"]),
                ("OGL", float(ref51["IoU_OGL"]), ref41["IoU_OGL"]),
            ):
                metric_errors[key] = max(metric_errors[key], abs(float(value) - float(reference)))

            aud_mask = aud >= 0.6
            img_mask = img >= 0.6
            expand_f = np.maximum(aud, prop_f)
            expand_k = np.maximum(aud, prop_k)
            shrink = np.logical_and(aud_mask, img_mask).astype(np.float32)
            iou_expand_f = common.sample_iou(expand_f, gt)
            iou_expand_k = common.sample_iou(expand_k, gt)
            iou_expand_best = max(iou_expand_f, iou_expand_k)
            iou_shrink = common.sample_iou(shrink, gt)
            expand_gain = max(0.0, iou_expand_best - iou_aud)
            shrink_gain = max(0.0, iou_shrink - iou_aud)
            error_type = strict_error_type(expand_gain, shrink_gain)

            intersection = aud_mask & img_mask
            aud_only = aud_mask & ~img_mask
            img_only = img_mask & ~aud_mask
            union = aud_mask | img_mask
            disagreement = np.abs(aud - img)
            row: dict[str, Any] = {
                "sample_index": global_index,
                "sample_id": sample_id,
                "dataset": arguments.experiment,
                "IoU_AUD": iou_aud,
                "IoU_IMG": iou_img,
                "IoU_PROP_F34": iou_prop_f,
                "IoU_PROP_K34": iou_prop_k,
                "IoU_OGL": float(ref51["IoU_OGL"]),
                "IoU_EXPAND_F34": iou_expand_f,
                "IoU_EXPAND_K34": iou_expand_k,
                "IoU_EXPAND_BEST": iou_expand_best,
                "IoU_SHRINK": iou_shrink,
                "expand_gain": expand_gain,
                "shrink_gain": shrink_gain,
                "expand_beneficial": expand_gain >= BENEFICIAL_GAIN,
                "shrink_beneficial": shrink_gain >= BENEFICIAL_GAIN,
                "error_type": error_type,
                "AUD_area": float(aud_mask.mean()),
                "IMG_area": float(img_mask.mean()),
                "IMG_AUD_area_ratio": safe_ratio(float(img_mask.sum()), float(aud_mask.sum())),
                "AUD_IMG_mask_IoU": binary_iou(aud_mask, img_mask),
                "AUD_only_area": float(aud_only.mean()),
                "IMG_only_area": float(img_only.mean()),
                "intersection_area": float(intersection.mean()),
                "AUD_only_ratio": safe_ratio(float(aud_only.sum()), float(aud_mask.sum())),
                "IMG_only_ratio": safe_ratio(float(img_only.sum()), float(img_mask.sum())),
                "intersection_ratio": safe_ratio(float(intersection.sum()), float(union.sum())),
                "disagreement_mean": float(disagreement.mean()),
                "disagreement_max": float(disagreement.max()),
                "disagreement_std": float(disagreement.std()),
                "AUD_IMG_Pearson": common.safe_pearson(aud, img),
                "SEED_CONF": float(ref51["SEED_CONF"]),
                "DELTA_SEMANTIC_SLOT": float(ref41["DELTA_SEMANTIC_SLOT"]),
                "DELTA_RECIPROCAL_L4": float(ref41["DELTA_RECIPROCAL_L4"]),
                "group_IMG_ONLY": ref51["group_IMG_ONLY"] == "True",
                "group_AUD_ONLY": ref51["group_AUD_ONLY"] == "True",
                "group_BOTH_SUCCESS": ref51["group_BOTH_SUCCESS"] == "True",
                "group_BOTH_FAIL": ref51["group_BOTH_FAIL"] == "True",
                "group_OGL_RESCUE": ref51["group_OGL_RESCUE"] == "True",
                "group_IMG_ONLY_SHRINK": ref51["group_IMG_ONLY_SHRINK"] == "True",
                "AUD_OVER_EXPANSION": ref51["AUD_OVER_EXPANSION"] == "True",
            }

            stats_aud = activation_stats(raw_aud_fine[local], aud)
            stats_img = activation_stats(raw_img_l4[local], img)
            for branch, stats in (("AUD", stats_aud), ("IMG", stats_img)):
                row.update({f"{branch}_{key}": value for key, value in stats.items()})
            for key in stats_aud:
                row[f"DELTA_IMG_MINUS_AUD_{key}"] = stats_img[key] - stats_aud[key]

            own_l3 = output["OWN_L3"][local].reshape(2, 7, 7).cpu().numpy()
            own_l4 = output["OWN_L4"][local].reshape(2, 7, 7).cpu().numpy()
            own_hr = output["OWN_HR14"][local].reshape(2, 14, 14).cpu().numpy()
            for level, value in (("L3", own_l3), ("L4", own_l4), ("HR14", own_hr)):
                row.update({f"{level}_{key}": result for key, result in ownership_stats(value).items()})

            aud_l3 = output["AUD_L3_ALL"][local, 0].reshape(7, 7).cpu().numpy()
            aud_l4 = raw_aud_l4[local]
            img_l3 = output["IMG_L3_ALL"][local, 0].reshape(7, 7).cpu().numpy()
            img_l4 = raw_img_l4[local]
            for branch, first, second in (("AUD", aud_l3, aud_l4), ("IMG", img_l3, img_l4)):
                row.update({f"{branch}_L3_L4_{key}": value for key, value in cross_level_stats(first, second).items()})
            row["L3_L4_target_ownership_Pearson"] = common.safe_pearson(own_l3[0], own_l4[0])

            aud_only14 = (aud14[local] >= 0.6) & ~(img14[local] >= 0.6)
            common14 = (aud14[local] >= 0.6) & (img14[local] >= 0.6)
            hr_target = own_hr[0]
            region_maps = {
                "IMG_response": img14[local],
                "PROP_F34_similarity": prop_f14[local],
                "PROP_K34_similarity": prop_k14[local],
                "HR14_target_ownership": hr_target,
            }
            for name, value in region_maps.items():
                aud_only_value = finite_mean(value, aud_only14)
                common_value = finite_mean(value, common14)
                row[f"AUD_only_{name}"] = aud_only_value
                row[f"COMMON_{name}"] = common_value
                row[f"REGION_DELTA_{name}"] = common_value - aud_only_value if math.isfinite(common_value) and math.isfinite(aud_only_value) else math.nan

            no_nan_or_inf = no_nan_or_inf and all(np.isfinite(value).all() for value in (aud, img, prop_f, prop_k, expand_f, expand_k, shrink))
            rows.append(row)
            global_index += 1

    completed_full = global_index == len(references["rows51"])
    reproduction = {
        "raw_tensor_max_errors": raw_errors,
        "per_sample_metric_max_errors": metric_errors,
        "sample_mismatches": sample_mismatches,
        "processed_samples": global_index,
        "reference_samples": len(references["rows51"]),
        "passed": max(raw_errors.values()) == 0.0 and max(metric_errors.values()) == 0.0 and sample_mismatches == 0,
    }
    if not reproduction["passed"]:
        raise RuntimeError(reproduction)

    single_variable = single_variable_analysis(rows)
    probe = probe_suite(arguments.experiment, rows, int(config.seed))
    failures = failure_cases(rows)
    checkpoint_after = common.verify_snapshots(checkpoints_before)
    zero_training = {
        "model_eval": not model.training,
        "inference_mode": True,
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": len(parameters_with_grad),
        "new_trainable_model_params": len(parameters_with_grad),
        "parameters_with_grad": parameters_with_grad,
        "analysis_only_sklearn_linear_probe": True,
        "checkpoint_hashes_and_mtimes_unchanged": checkpoint_after["all_unchanged"],
        "checkpoint_files": checkpoint_after["files"],
        "no_nan_or_inf_model_and_maps": no_nan_or_inf,
    }
    if parameters_with_grad or not checkpoint_after["all_unchanged"] or not no_nan_or_inf:
        raise RuntimeError(zero_training)

    common.write_csv(output_dir / "per_sample_diagnosis.csv", rows)
    common.write_json(output_dir / "per_sample_diagnosis.json", rows)
    common.write_csv(output_dir / "single_variable_diagnostics.csv", single_variable)
    common.write_csv(output_dir / "failure_cases.csv", failures)

    top_signals = {}
    for task in ("EXPAND_vs_SHRINK", "SHRINK_vs_OTHERS", "EXPAND_vs_OTHERS"):
        selected = [row for row in single_variable if row["task"] == task and math.isfinite(float(row["AUROC_directed"]))]
        top_signals[task] = sorted(selected, key=lambda row: (row["AUROC_directed"], abs(row["effect_size_cohen_d"])), reverse=True)[:12]

    spatial_keys = (
        "AUD_area",
        "IMG_area",
        "IMG_AUD_area_ratio",
        "AUD_IMG_mask_IoU",
        "AUD_only_area",
        "IMG_only_area",
        "intersection_area",
        "AUD_only_ratio",
        "IMG_only_ratio",
        "intersection_ratio",
    )
    internal_keys = (
        "AUD_only_IMG_response",
        "AUD_only_PROP_F34_similarity",
        "AUD_only_PROP_K34_similarity",
        "AUD_only_HR14_target_ownership",
        "REGION_DELTA_IMG_response",
        "REGION_DELTA_PROP_F34_similarity",
        "REGION_DELTA_PROP_K34_similarity",
        "REGION_DELTA_HR14_target_ownership",
    )
    summary = {
        "experiment": "5.2_expansion_shrink_error_diagnosis",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": completed_full,
        "audit": reproduction,
        "zero_training_audit": zero_training,
        "tensor_audit": tensor_audit,
        "error_definition": {
            "EXPAND_F34": "per-pixel max(AUD_norm, PROP_F34_norm); cannot delete an AUD-positive pixel",
            "EXPAND_K34": "per-pixel max(AUD_norm, PROP_K34_norm); cannot delete an AUD-positive pixel",
            "SHRINK": "binary(AUD_norm>=0.6) intersection binary(IMG_norm>=0.6); cannot add outside AUD",
            "beneficial_gain_threshold": BENEFICIAL_GAIN,
            "strict_dominance_margin": DOMINANCE_MARGIN,
            "strict_rule": "EXPAND/SHRINK requires gain>=0.01 and >= opposite gain+0.01; all other samples are KEEP_AMBIGUOUS",
        },
        "error_distribution": error_distribution(rows),
        "spatial_disagreement_by_error_type": grouped_distributions(rows, spatial_keys),
        "AUD_only_internal_signals_by_error_type": grouped_distributions(rows, internal_keys),
        "single_variable_top_signals": top_signals,
        "lightweight_probe": probe,
        "oracle_upper_bound": oracle_summary(rows),
        "relationship_with_5_1": relationship_51(rows),
        "failure_cases": failures,
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
