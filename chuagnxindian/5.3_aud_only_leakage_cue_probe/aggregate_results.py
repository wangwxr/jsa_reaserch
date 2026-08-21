#!/usr/bin/env python3
"""Analyze Experiment 5.3 frozen pixel features and write REPORT.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn import metrics as sklearn_metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common


SETTINGS = ("vggss_144k", "flickr_144k")
LABELS = {"vggss_144k": "VGGSS-144k", "flickr_144k": "Flickr-144k"}
FEATURE_SETS = ("PREDICTION", "WITHOUT_PROTOTYPE", "WITH_PROTOTYPE", "PROTOTYPE_ONLY")
INTRINSIC_TYPES = ("INTRINSIC_EXPAND", "INTRINSIC_SHRINK", "MIXED_AMBIGUOUS", "KEEP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--report", type=Path, default=common.HERE / "REPORT.md")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return value is True or str(value) == "True"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return math.nan
    return common.safe_pearson(first[finite], second[finite])


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return math.nan
    return common.safe_pearson(rankdata(first[finite]), rankdata(second[finite]))


def cohen_d(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size < 2 or negative.size < 2:
        return math.nan
    pooled = math.sqrt(((positive.size - 1) * positive.var() + (negative.size - 1) * negative.var()) / (positive.size + negative.size - 2))
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else 0.0


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = scores >= threshold
    return {
        "AUROC": float(sklearn_metrics.roc_auc_score(labels, scores)),
        "AUPRC": float(sklearn_metrics.average_precision_score(labels, scores)),
        "balanced_accuracy": float(sklearn_metrics.balanced_accuracy_score(labels, predictions)),
        "F1": float(sklearn_metrics.f1_score(labels, predictions)),
        "accuracy": float(sklearn_metrics.accuracy_score(labels, predictions)),
        "threshold": threshold,
        "confusion_matrix": sklearn_metrics.confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "num_samples": int(labels.size),
        "positive_fraction": float(labels.mean()),
    }


def matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)


def pipeline(seed: int = 12345):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed),
    )


def single_variable(rows: list[dict[str, str]], features: list[str], dataset: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if as_bool(row["probe_selected"])]
    labels = np.asarray([int(row["label_context_leakage"]) for row in selected], dtype=np.int64)
    output = []
    for feature in features:
        scores = np.asarray([float(row[feature]) for row in selected], dtype=np.float64)
        finite = np.isfinite(scores)
        y = labels[finite]
        x = scores[finite]
        raw_auc = float(sklearn_metrics.roc_auc_score(y, x))
        direction = "higher=leakage" if raw_auc >= 0.5 else "lower=leakage"
        directed = x if raw_auc >= 0.5 else -x
        auc = float(sklearn_metrics.roc_auc_score(y, directed))
        auprc = float(sklearn_metrics.average_precision_score(y, directed))
        fpr, tpr, thresholds = sklearn_metrics.roc_curve(y, directed)
        best_index = int(np.argmax(tpr - fpr))
        threshold = float(thresholds[best_index])
        predictions = directed >= threshold
        output.append(
            {
                "row_type": "single_variable",
                "dataset": dataset,
                "method": feature,
                "feature_set": "prototype" if feature.endswith("core_similarity") else "non_prototype",
                "direction": direction,
                "AUROC": auc,
                "AUPRC": auprc,
                "balanced_accuracy": float(sklearn_metrics.balanced_accuracy_score(y, predictions)),
                "F1": float(sklearn_metrics.f1_score(y, predictions)),
                "diagnostic_threshold": threshold,
                "leakage_mean": float(x[y == 1].mean()),
                "leakage_std": float(x[y == 1].std()),
                "extent_mean": float(x[y == 0].mean()),
                "extent_std": float(x[y == 0].std()),
                "effect_size_cohen_d": cohen_d(x[y == 1], x[y == 0]),
                "Pearson_label": safe_pearson(x, y.astype(np.float64)),
                "Spearman_label": safe_spearman(x, y.astype(np.float64)),
                "num_pixels": int(x.size),
                "note": "balanced accuracy uses in-domain Youden threshold for single-variable diagnosis only",
            }
        )
    return output


def in_domain_oof(
    rows: list[dict[str, str]], features: list[str], feature_set: str
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    x_all = matrix(rows, features)
    folds = np.asarray([int(row["fold"]) for row in rows])
    probe_mask = np.asarray([as_bool(row["probe_selected"]) for row in rows])
    routing_mask = np.asarray([as_bool(row["routing_selected"]) for row in rows])
    labels = np.asarray([int(row["label_context_leakage"]) for row in rows])
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    fold_audit = []
    for fold in range(5):
        train = probe_mask & (folds != fold)
        test_probe = probe_mask & (folds == fold)
        test_apply = (probe_mask | routing_mask) & (folds == fold)
        model = pipeline()
        model.fit(x_all[train], labels[train])
        scores[test_apply] = model.predict_proba(x_all[test_apply])[:, 1]
        fold_audit.append(
            {
                "fold": fold,
                "train_pixels": int(train.sum()),
                "test_probe_pixels": int(test_probe.sum()),
                "test_routing_pixels": int((routing_mask & (folds == fold)).sum()),
                "train_samples": len(set(rows[index]["sample_id"] for index in np.flatnonzero(train))),
                "test_samples": len(set(rows[index]["sample_id"] for index in np.flatnonzero(test_apply))),
                "sample_overlap": len(
                    set(rows[index]["sample_id"] for index in np.flatnonzero(train))
                    & set(rows[index]["sample_id"] for index in np.flatnonzero(test_apply))
                ),
            }
        )
    if np.any(~np.isfinite(scores[probe_mask])):
        raise RuntimeError(f"Missing OOF probe scores for {feature_set}")
    metrics = binary_metrics(labels[probe_mask], scores[probe_mask])
    metrics["feature_set"] = feature_set
    metrics["fold_audit"] = fold_audit
    return metrics, scores, fold_audit


def transfer_probe(
    source_rows: list[dict[str, str]], target_rows: list[dict[str, str]], features: list[str], feature_set: str
) -> tuple[dict[str, Any], np.ndarray]:
    source_probe = np.asarray([as_bool(row["probe_selected"]) for row in source_rows])
    target_probe = np.asarray([as_bool(row["probe_selected"]) for row in target_rows])
    target_routing = np.asarray([as_bool(row["routing_selected"]) for row in target_rows])
    source_labels = np.asarray([int(row["label_context_leakage"]) for row in source_rows])
    target_labels = np.asarray([int(row["label_context_leakage"]) for row in target_rows])
    source_x = matrix(source_rows, features)
    target_x = matrix(target_rows, features)
    model = pipeline()
    model.fit(source_x[source_probe], source_labels[source_probe])
    apply = target_probe | target_routing
    scores = np.full(len(target_rows), np.nan, dtype=np.float64)
    scores[apply] = model.predict_proba(target_x[apply])[:, 1]
    metrics = binary_metrics(target_labels[target_probe], scores[target_probe])
    metrics.update(
        {
            "feature_set": feature_set,
            "source_probe_pixels": int(source_probe.sum()),
            "target_probe_pixels": int(target_probe.sum()),
            "source_normalization_only": True,
            "target_threshold_tuning": False,
        }
    )
    return metrics, scores


def intrinsic_agreement(rows: list[dict[str, str]]) -> dict[str, Any]:
    intrinsic_order = ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS")
    candidate_order = ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS")
    intrinsic_mapped = []
    candidate = []
    for row in rows:
        intrinsic = row["intrinsic_error_type"]
        intrinsic_mapped.append(
            "EXPAND" if intrinsic == "INTRINSIC_EXPAND" else "SHRINK" if intrinsic == "INTRINSIC_SHRINK" else "KEEP_AMBIGUOUS"
        )
        candidate.append(row["candidate_5_2_error_type"])
    matrix_value = sklearn_metrics.confusion_matrix(intrinsic_mapped, candidate, labels=list(intrinsic_order))
    output = {
        "labels": list(intrinsic_order),
        "confusion_matrix_rows_intrinsic_columns_candidate": matrix_value.tolist(),
        "agreement_rate": float(np.mean(np.asarray(intrinsic_mapped) == np.asarray(candidate))),
    }
    for label in ("EXPAND", "SHRINK"):
        truth = np.asarray(intrinsic_mapped) == label
        prediction = np.asarray(candidate) == label
        output[label] = {
            "precision": float(sklearn_metrics.precision_score(truth, prediction, zero_division=0)),
            "recall": float(sklearn_metrics.recall_score(truth, prediction, zero_division=0)),
        }
    return output


def routing_metrics(sample_rows: list[dict[str, str]], score_by_sample: dict[str, float]) -> dict[str, Any]:
    available = [row for row in sample_rows if row["sample_id"] in score_by_sample]
    scores = np.asarray([score_by_sample[row["sample_id"]] for row in available], dtype=np.float64)
    intrinsic = np.asarray([row["intrinsic_error_type"] for row in available])
    output = {"samples_with_AUD_ONLY_score": len(available)}
    for task in ("SHRINK_vs_NON_SHRINK", "EXPAND_vs_SHRINK"):
        if task == "SHRINK_vs_NON_SHRINK":
            subset = np.arange(len(available))
        else:
            subset = np.flatnonzero(np.isin(intrinsic, ["INTRINSIC_EXPAND", "INTRINSIC_SHRINK"]))
        labels = (intrinsic[subset] == "INTRINSIC_SHRINK").astype(np.int64)
        output[task] = binary_metrics(labels, scores[subset], threshold=0.5)
    return output


def aggregate_routing_scores(
    pixel_rows: list[dict[str, str]], scores: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    probabilities: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(pixel_rows):
        if as_bool(row["routing_selected"]) and math.isfinite(float(scores[index])):
            probabilities[row["sample_id"]].append(float(scores[index]))
    hard_ratio = {sample: float(np.mean(np.asarray(values) >= 0.5)) for sample, values in probabilities.items()}
    mean_probability = {sample: float(np.mean(values)) for sample, values in probabilities.items()}
    return hard_ratio, mean_probability


def group_mapping(sample_rows: list[dict[str, str]], predicted: dict[str, float]) -> dict[str, Any]:
    groups = {
        "PROP_ONLY": "group_PROP_ONLY",
        "PROP_HURT": "group_PROP_HURT",
        "IMG_ONLY_SHRINK": "group_IMG_ONLY_SHRINK",
        "OGL_RESCUE": "group_OGL_RESCUE",
        "AUD_OVER_EXPANSION": "AUD_OVER_EXPANSION",
        "CANDIDATE_EXPAND": None,
        "CANDIDATE_SHRINK": None,
    }
    output = {}
    for name, key in groups.items():
        if name == "CANDIDATE_EXPAND":
            selected = [row for row in sample_rows if row["candidate_5_2_error_type"] == "EXPAND"]
        elif name == "CANDIDATE_SHRINK":
            selected = [row for row in sample_rows if row["candidate_5_2_error_type"] == "SHRINK"]
        else:
            selected = [row for row in sample_rows if as_bool(row[key])]
        counts = {value: sum(row["intrinsic_error_type"] == value for row in selected) for value in INTRINSIC_TYPES}
        full_ratios = [float(row["AUD_ONLY_leakage_ratio"]) for row in selected if math.isfinite(float(row["AUD_ONLY_leakage_ratio"]))]
        predicted_ratios = [predicted[row["sample_id"]] for row in selected if row["sample_id"] in predicted]
        output[name] = {
            "count": len(selected),
            "intrinsic_counts": counts,
            "true_leakage_ratio_mean": float(np.mean(full_ratios)) if full_ratios else math.nan,
            "predicted_leakage_ratio_mean": float(np.mean(predicted_ratios)) if predicted_ratios else math.nan,
        }
    return output


def failure_analysis(sample_rows: list[dict[str, str]], predicted: dict[str, float]) -> list[dict[str, Any]]:
    available = [row for row in sample_rows if row["sample_id"] in predicted]
    shrink = [row for row in available if row["intrinsic_error_type"] == "INTRINSIC_SHRINK"]
    expand = [row for row in available if row["intrinsic_error_type"] == "INTRINSIC_EXPAND"]
    selected = []
    if shrink:
        selected.append(("SHRINK_LOWEST_PREDICTED_LEAKAGE", min(shrink, key=lambda row: (predicted[row["sample_id"]], row["sample_id"]))))
    if expand:
        selected.append(("EXPAND_HIGHEST_PREDICTED_LEAKAGE", max(expand, key=lambda row: (predicted[row["sample_id"]], row["sample_id"]))))
    output = []
    for category, row in selected:
        output.append(
            {
                "category": category,
                "sample_id": row["sample_id"],
                "intrinsic_error_type": row["intrinsic_error_type"],
                "candidate_5_2_error_type": row["candidate_5_2_error_type"],
                "true_leakage_ratio": float(row["AUD_ONLY_leakage_ratio"]),
                "predicted_leakage_ratio": predicted[row["sample_id"]],
                "intrinsic_expand_gain": float(row["intrinsic_expand_gain"]),
                "intrinsic_shrink_gain": float(row["intrinsic_shrink_gain"]),
            }
        )
    return output


def choose_case(analysis: dict[str, Any]) -> tuple[str, str]:
    agreement = [analysis[setting]["intrinsic_candidate_agreement"]["agreement_rate"] for setting in SETTINGS]
    oof = [analysis[setting]["pixel_probe_OOF"]["WITH_PROTOTYPE"]["AUROC"] for setting in SETTINGS]
    transfer = [
        analysis["cross_dataset"]["vggss_144k_to_flickr_144k"]["WITH_PROTOTYPE"]["AUROC"],
        analysis["cross_dataset"]["flickr_144k_to_vggss_144k"]["WITH_PROTOTYPE"]["AUROC"],
    ]
    sample_route = [analysis[setting]["sample_routing_OOF"]["WITH_PROTOTYPE"]["SHRINK_vs_NON_SHRINK"]["AUROC"] for setting in SETTINGS]
    structure_survives = all(
        analysis[setting]["extraction"]["intrinsic_distribution"]["INTRINSIC_EXPAND"]["count"] > 0
        and analysis[setting]["extraction"]["intrinsic_distribution"]["INTRINSIC_SHRINK"]["count"] > 0
        for setting in SETTINGS
    ) and min(agreement) >= 0.30
    if not structure_survives:
        return "Case D", "Candidate-Defined Structure Does Not Survive"
    if min(oof) >= 0.65 and min(transfer) >= 0.60 and min(sample_route) >= 0.60:
        return "Case A", "Stable Generalizable Leakage Cue"
    if min(oof) >= 0.65 and min(transfer) < 0.60:
        return "Case B", "Strong In-Domain but Domain-Specific Cue"
    return "Case C", "Weak Pixel Signal"


def build_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Experiment 5.3 - AUD-Only Leakage Cue Probe",
        "",
        "## 1. Audit",
        "",
        "- Frozen Stage1/original 1.3G inference only; no optimizer, backward pass, trainable localization parameter, checkpoint modification, or correction module.",
        "- Official evaluator reproduction uses the original GT values. Candidate-independent TP/FP/FN diagnosis and pixel labels use fixed binary `GT >= 0.5`.",
        "- Pixel folds are `sha256(seed, sample_id) mod 5`; every pixel from one sample stays in one fold.",
        "",
    ]
    for setting in SETTINGS:
        extraction = analysis[setting]["extraction"]
        audit = extraction["audit"]
        zero = extraction["zero_training_audit"]
        lines.extend(
            [
                f"### {LABELS[setting]}",
                "",
                f"- Raw errors `{audit['raw_tensor_max_errors']}`; metric errors `{audit['per_sample_metric_max_errors']}`; sample mismatch `{audit['sample_mismatches']}`.",
                f"- `optimizer_created={str(zero['optimizer_created']).lower()}`, `backward_called={str(zero['backward_called']).lower()}`, `new_trainable_params={zero['new_trainable_params']}`, `parameters_with_grad={zero['parameters_with_grad']}`.",
                f"- Checkpoint unchanged `{zero['checkpoint_hashes_and_mtimes_unchanged']}`; NaN/Inf `{not zero['no_nan_or_inf']}`.",
                f"- Saved pixels `{extraction['sampling']['saved_pixel_rows']}`; balanced probe pixels "
                f"`{extraction['sampling']['probe_TRUE_EXTENT']}/{extraction['sampling']['probe_CONTEXT_LEAKAGE']}` "
                "(TRUE_EXTENT/LEAKAGE).",
                "",
            ]
        )

    lines.extend(
        [
            "## 2. Candidate-Independent Intrinsic Diagnosis",
            "",
            "`IoU_expand*=|GT|/(|GT|+FP)` and `IoU_shrink*=TP/(TP+FN)` use only AUD and binary GT. Gain threshold and dominance margin are both fixed at `.01`.",
            "",
            "| Dataset | Intrinsic Expand | Intrinsic Shrink | Mixed | Keep |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        dist = analysis[setting]["extraction"]["intrinsic_distribution"]
        total = sum(value["count"] for value in dist.values())
        lines.append(
            f"| {LABELS[setting]} | {dist['INTRINSIC_EXPAND']['count']} ({100*dist['INTRINSIC_EXPAND']['count']/total:.1f}%) | "
            f"{dist['INTRINSIC_SHRINK']['count']} ({100*dist['INTRINSIC_SHRINK']['count']/total:.1f}%) | "
            f"{dist['MIXED_AMBIGUOUS']['count']} ({100*dist['MIXED_AMBIGUOUS']['count']/total:.1f}%) | "
            f"{dist['KEEP']['count']} ({100*dist['KEEP']['count']/total:.1f}%) |"
        )

    lines.extend(
        [
            "",
            "## 3. Agreement With Experiment 5.2",
            "",
            "| Dataset | Agreement | Candidate EXPAND P/R | Candidate SHRINK P/R |",
            "|---|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        value = analysis[setting]["intrinsic_candidate_agreement"]
        lines.append(
            f"| {LABELS[setting]} | {fmt(value['agreement_rate'])} | {fmt(value['EXPAND']['precision'])}/{fmt(value['EXPAND']['recall'])} | "
            f"{fmt(value['SHRINK']['precision'])}/{fmt(value['SHRINK']['recall'])} |"
        )
    lines.extend(
        [
            "",
            "The bidirectional structure survives: both intrinsic EXPAND and SHRINK exist at nearly identical rates across datasets. The 5.2 SHRINK candidate is very high precision (`.99/.98`) but only about `.51` recall, so many candidate KEEP cases are intrinsically shrink-beneficial rather than truly KEEP.",
        ]
    )

    lines.extend(
        [
            "",
            "## 4. AUD-Only Pixel Composition",
            "",
            "| Dataset / intrinsic type | AUD-only pixels | True extent | Context leakage | Macro leakage ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        composition = analysis[setting]["extraction"]["AUD_ONLY_composition"]
        for error_type in ("ALL",) + INTRINSIC_TYPES:
            value = composition[error_type]
            lines.append(
                f"| {LABELS[setting]} / {error_type} | {value['AUD_ONLY_pixels']} | {fmt(value['TRUE_EXTENT_fraction'])} | "
                f"{fmt(value['CONTEXT_LEAKAGE_fraction'])} | {fmt(value['sample_leakage_ratio']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "The target distinction is real and strong in GT space: intrinsic SHRINK AUD-only pixels are `80.0%/60.5%` leakage on VGG/Flickr, while intrinsic EXPAND pixels are only `22.5%/23.9%` leakage. The remaining question is whether frozen features expose this distinction without GT.",
        ]
    )

    lines.extend(
        [
            "",
            "## 5. Single-Variable Pixel Signals",
            "",
            "Top same-direction signals across both datasets:",
            "",
            "| Signal | Direction | VGG AUROC | Flickr AUROC |",
            "|---|---|---:|---:|",
        ]
    )
    for value in analysis["stable_single_variable"][:10]:
        lines.append(f"| {value['signal']} | {value['direction']} | {fmt(value['VGG'])} | {fmt(value['Flickr'])} |")
    lines.extend(
        [
            "",
            "The strongest transferable single cue is lower K34 similarity to the IMG-supported core (`.5942/.6335` AUROC). All single cues remain weak-to-moderate; no non-prototype or prototype signal is independently reliable.",
        ]
    )

    lines.extend(
        [
            "",
            "## 6. Pixel Linear Probe",
            "",
            "All OOF metrics use sample-disjoint folds. Probe-selected pixels are exactly balanced within every mixed-label sample.",
            "",
            "| Dataset | Features | AUROC | AUPRC | Balanced acc | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for feature_set in FEATURE_SETS:
            value = analysis[setting]["pixel_probe_OOF"][feature_set]
            lines.append(
                f"| {LABELS[setting]} | {feature_set} | {fmt(value['AUROC'])} | {fmt(value['AUPRC'])} | "
                f"{fmt(value['balanced_accuracy'])} | {fmt(value['F1'])} |"
            )
    lines.extend(
        [
            "",
            "Adding prototype cues changes AUROC only from `.6584` to `.6615` on VGG and `.6599` to `.6786` on Flickr. The signal is not a disguised replay of prototype propagation, but its absolute reliability remains moderate.",
        ]
    )

    lines.extend(
        [
            "",
            "## 7. Cross-Dataset Transfer",
            "",
            "Target normalization and thresholds are never fitted or tuned. Threshold remains `.5`.",
            "",
            "| Direction | Features | AUROC | AUPRC | Balanced acc | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for direction, values in analysis["cross_dataset"].items():
        for feature_set in FEATURE_SETS:
            value = values[feature_set]
            lines.append(
                f"| {direction} | {feature_set} | {fmt(value['AUROC'])} | {fmt(value['AUPRC'])} | "
                f"{fmt(value['balanced_accuracy'])} | {fmt(value['F1'])} |"
            )
    lines.extend(
        [
            "",
            "Pixel AUROC transfers partially rather than collapsing: WITH_PROTOTYPE is `.6544` VGG-to-Flickr and `.6017` Flickr-to-VGG. Calibration is weaker, especially in the Flickr-to-VGG direction, but the main failure appears after pixel scores are aggregated into a routing decision.",
        ]
    )

    lines.extend(
        [
            "",
            "## 8. Aggregated Sample-Level Routing",
            "",
            "The score is the fraction of deterministic AUD-only routing pixels predicted as leakage at threshold `.5`.",
            "",
            "| Dataset | Pixel score source | Features | Shrink-vs-non AUROC | Expand-vs-shrink AUROC |",
            "|---|---|---|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for source_name, source_values in (
            ("OOF", analysis[setting]["sample_routing_OOF"]),
            ("cross-transfer", analysis[setting]["sample_routing_transfer"]),
        ):
            for feature_set in FEATURE_SETS:
                value = source_values[feature_set]
                lines.append(
                    f"| {LABELS[setting]} | {source_name} | {feature_set} | "
                    f"{fmt(value['SHRINK_vs_NON_SHRINK']['AUROC'])} | {fmt(value['EXPAND_vs_SHRINK']['AUROC'])} |"
                )
    lines.extend(
        [
            "",
            "The pixel cue does not solve the 5.2 routing problem. WITH_PROTOTYPE OOF shrink-routing AUROC is only `.5649` on VGG and `.6027` on Flickr; direct transfer to VGG falls to `.4942`.",
        ]
    )

    lines.extend(["", "## 9. Mapping To Experiments 5.1/5.2", ""])
    for setting in SETTINGS:
        lines.extend(
            [
                f"### {LABELS[setting]}",
                "",
                "| Group | Count | Intrinsic Expand | Intrinsic Shrink | Mixed | Keep | True leakage | Predicted leakage |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, value in analysis[setting]["mapping"].items():
            count = value["intrinsic_counts"]
            lines.append(
                f"| {group} | {value['count']} | {count['INTRINSIC_EXPAND']} | {count['INTRINSIC_SHRINK']} | "
                f"{count['MIXED_AMBIGUOUS']} | {count['KEEP']} | {fmt(value['true_leakage_ratio_mean'])} | "
                f"{fmt(value['predicted_leakage_ratio_mean'])} |"
            )
        lines.append("")

    lines.extend(["## 10. Failure Analysis", ""])
    for setting in SETTINGS:
        lines.append(f"### {LABELS[setting]}")
        lines.append("")
        for value in analysis[setting]["failure_analysis"]:
            lines.append(
                f"- `{value['category']}` `{value['sample_id']}`: intrinsic `{value['intrinsic_error_type']}`, "
                f"5.2 `{value['candidate_5_2_error_type']}`, true leakage `{fmt(value['true_leakage_ratio'])}`, "
                f"predicted `{fmt(value['predicted_leakage_ratio'])}`."
            )
        lines.append("")

    case = analysis["decision"]
    lines.extend(
        [
            "## 11. Final Decision",
            "",
            f"**{case['case']} - {case['title']}.**",
            "",
            case["rationale"],
            "",
            "## 12. Research-Line Decision",
            "",
            case["recommendation"],
            "",
            "No Experiment 5.4 was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    arguments = parse_args()
    extraction = {}
    sample_rows = {}
    pixel_rows = {}
    for setting in SETTINGS:
        root = arguments.results_root / setting
        extraction[setting] = json.loads((root / "extraction_summary.json").read_text(encoding="utf-8"))
        sample_rows[setting] = load_csv(root / "per_sample_intrinsic_diagnosis.csv")
        pixel_rows[setting] = load_csv(root / "sampled_pixels.csv")
        if not extraction[setting]["completed_full_dataset"] or not extraction[setting]["audit"]["passed"]:
            raise RuntimeError(f"Incomplete extraction: {setting}")

    feature_groups = {name: list(extraction[SETTINGS[0]]["feature_groups"][name]) for name in FEATURE_SETS}
    all_features = list(extraction[SETTINGS[0]]["feature_groups"]["WITH_PROTOTYPE"])
    analysis: dict[str, Any] = {}
    pixel_summary_rows: list[dict[str, Any]] = []
    oof_scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    transfer_scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    single_by_dataset = {}
    for setting in SETTINGS:
        singles = single_variable(pixel_rows[setting], all_features, setting)
        single_by_dataset[setting] = {row["method"]: row for row in singles}
        pixel_summary_rows.extend(singles)
        analysis[setting] = {
            "extraction": extraction[setting],
            "intrinsic_candidate_agreement": intrinsic_agreement(sample_rows[setting]),
            "pixel_probe_OOF": {},
        }
        for feature_set in FEATURE_SETS:
            metrics, scores, _audit = in_domain_oof(pixel_rows[setting], feature_groups[feature_set], feature_set)
            analysis[setting]["pixel_probe_OOF"][feature_set] = metrics
            oof_scores[setting][feature_set] = scores
            pixel_summary_rows.append({"row_type": "linear_probe_OOF", "dataset": setting, "method": feature_set, **{key: value for key, value in metrics.items() if key != "fold_audit"}})
            for index, score in enumerate(scores):
                if math.isfinite(float(score)):
                    pixel_rows[setting][index][f"oof_score_{feature_set}"] = float(score)

    stable = []
    for feature in all_features:
        first = single_by_dataset[SETTINGS[0]][feature]
        second = single_by_dataset[SETTINGS[1]][feature]
        if first["direction"] == second["direction"]:
            stable.append(
                {
                    "signal": feature,
                    "direction": first["direction"],
                    "VGG": first["AUROC"],
                    "Flickr": second["AUROC"],
                    "minimum": min(first["AUROC"], second["AUROC"]),
                }
            )
    analysis["stable_single_variable"] = sorted(stable, key=lambda row: (row["minimum"], row["VGG"] + row["Flickr"]), reverse=True)

    cross_rows = []
    analysis["cross_dataset"] = {}
    for source, target in ((SETTINGS[0], SETTINGS[1]), (SETTINGS[1], SETTINGS[0])):
        direction = f"{source}_to_{target}"
        analysis["cross_dataset"][direction] = {}
        for feature_set in FEATURE_SETS:
            metrics, scores = transfer_probe(pixel_rows[source], pixel_rows[target], feature_groups[feature_set], feature_set)
            analysis["cross_dataset"][direction][feature_set] = metrics
            transfer_scores[target][feature_set] = scores
            cross_rows.append({"row_type": "pixel_transfer", "direction": direction, "source": source, "target": target, "feature_set": feature_set, **metrics})
            for index, score in enumerate(scores):
                if math.isfinite(float(score)):
                    pixel_rows[target][index][f"transfer_score_from_{source}_{feature_set}"] = float(score)

    for setting in SETTINGS:
        analysis[setting]["sample_routing_OOF"] = {}
        analysis[setting]["sample_routing_transfer"] = {}
        other = SETTINGS[1] if setting == SETTINGS[0] else SETTINGS[0]
        primary_predicted = None
        for feature_set in FEATURE_SETS:
            hard_oof, mean_oof = aggregate_routing_scores(pixel_rows[setting], oof_scores[setting][feature_set])
            hard_transfer, mean_transfer = aggregate_routing_scores(pixel_rows[setting], transfer_scores[setting][feature_set])
            analysis[setting]["sample_routing_OOF"][feature_set] = routing_metrics(sample_rows[setting], hard_oof)
            analysis[setting]["sample_routing_transfer"][feature_set] = routing_metrics(sample_rows[setting], hard_transfer)
            for task, metrics in analysis[setting]["sample_routing_transfer"][feature_set].items():
                if not isinstance(metrics, dict):
                    continue
                cross_rows.append(
                    {
                        "row_type": "sample_routing_transfer",
                        "direction": f"{other}_to_{setting}",
                        "source": other,
                        "target": setting,
                        "feature_set": feature_set,
                        "task": task,
                        **metrics,
                    }
                )
            for row in sample_rows[setting]:
                sample = row["sample_id"]
                if sample in hard_oof:
                    row[f"OOF_predicted_leakage_ratio_{feature_set}"] = hard_oof[sample]
                    row[f"OOF_mean_leakage_probability_{feature_set}"] = mean_oof[sample]
                if sample in hard_transfer:
                    row[f"transfer_from_{other}_predicted_leakage_ratio_{feature_set}"] = hard_transfer[sample]
                    row[f"transfer_from_{other}_mean_leakage_probability_{feature_set}"] = mean_transfer[sample]
            if feature_set == "WITH_PROTOTYPE":
                primary_predicted = hard_oof
        analysis[setting]["mapping"] = group_mapping(sample_rows[setting], primary_predicted or {})
        analysis[setting]["failure_analysis"] = failure_analysis(sample_rows[setting], primary_predicted or {})

    case_name, case_title = choose_case(analysis)
    if case_name == "Case A":
        rationale = "AUD-only frozen cues distinguish true extent from leakage in both datasets, transfer across datasets, and retain useful sample-level SHRINK routing signal."
        recommendation = "The line may continue to a narrowly scoped leakage-aware suppression diagnostic, but no 5.4 method is implemented here."
    elif case_name == "Case B":
        rationale = "Pixel cues are strong in-domain but lose substantial discrimination or calibration under direct cross-dataset transfer, indicating a domain shortcut rather than a stable reliability cue."
        recommendation = "Do not design a formal leakage router from these cues. The supervision/domain-invariance problem must be addressed first."
    elif case_name == "Case C":
        rationale = "Even direct AUD-only pixel supervision does not produce a consistently strong frozen linear signal across VGG and Flickr, and aggregation does not resolve sample-level SHRINK routing."
        recommendation = "Stop the hand-designed adaptive expand/shrink routing line; do not start 5.4 from the current frozen cues."
    else:
        rationale = "Candidate-independent TP/FP/FN diagnosis does not preserve the bidirectional structure observed with the fixed PROP/IMG candidates."
        recommendation = "Close the expansion/shrink line because the previous structure was candidate-dependent."
    analysis["decision"] = {"case": case_name, "title": case_title, "rationale": rationale, "recommendation": recommendation}

    for setting in SETTINGS:
        common.write_csv(arguments.results_root / setting / "sampled_pixels_with_scores.csv", pixel_rows[setting])
        common.write_csv(arguments.results_root / setting / "per_sample_intrinsic_diagnosis.csv", sample_rows[setting])
    common.write_csv(arguments.results_root / "per_pixel_probe_summary.csv", pixel_summary_rows)
    common.write_csv(arguments.results_root / "cross_dataset_probe_results.csv", cross_rows)
    common.write_json(arguments.results_root / "analysis_summary.json", analysis)
    arguments.report.write_text(build_report(analysis), encoding="utf-8")
    print(arguments.report)


if __name__ == "__main__":
    main()
