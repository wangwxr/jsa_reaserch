#!/usr/bin/env python3
"""Analyze frozen three-model maps and write per-sample and summary artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
EXP87 = AUDIT.parent
ROOT = EXP87.parents[1]
EXP80 = ROOT / "chuagnxindian/8.0_audio_only_residual_audit"
MODELS = ("original_1.3g_final", "cram_c3_stage1", "cram_c3_stage2")
DISPLAY = {
    "original_1.3g_final": "1.3G",
    "cram_c3_stage1": "8.7-S1",
    "cram_c3_stage2": "8.7-S2",
}
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 87012345
EPS = 1e-12


def stable_seed(*parts: str) -> int:
    value = "::".join((str(BOOTSTRAP_SEED), *parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % 2**32


def stats(values: np.ndarray, *parts: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: math.nan for key in ("mean", "median", "q1", "q3", "iqr", "ci_low", "ci_high")} | {"valid": 0}
    rng = np.random.default_rng(stable_seed(*parts))
    boot = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 100):
        count = min(100, BOOTSTRAPS - start)
        indices = rng.integers(len(values), size=(count, len(values)))
        boot[start:start + count] = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "valid": int(len(values)),
    }


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(-1, keepdims=True)
    right = right - right.mean(-1, keepdims=True)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.sum(left * right, axis=-1) / np.maximum(denominator, EPS)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.sum(left * right, axis=-1) / np.maximum(denominator, EPS)


def auc_1d(positive: np.ndarray, negative: np.ndarray) -> float:
    n, m = len(positive), len(negative)
    if not n or not m:
        return math.nan
    ranks = rankdata(np.concatenate((positive, negative)), method="average")
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def conditional_auc(
    positive: np.ndarray, negative: np.ndarray,
    matched_positive: np.ndarray, matched_negative: np.ndarray,
) -> tuple[float, int]:
    """Experiment-8.1 AUD-matched within-image AUROC, using 20 fixed bins."""
    positive_bins = np.clip((matched_positive * 20).astype(int), 0, 19)
    negative_bins = np.clip((matched_negative * 20).astype(int), 0, 19)
    wins, pairs = 0.0, 0
    for bin_index in range(20):
        left = positive[positive_bins == bin_index]
        right = negative[negative_bins == bin_index]
        count = len(left) * len(right)
        if count:
            wins += auc_1d(left, right) * count
            pairs += count
    return (wins / pairs, pairs) if pairs else (math.nan, 0)


def pool_to_7(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] == 49:
        return values
    shape = values.shape[:-1]
    maps = values.reshape(*shape, 14, 14)
    pooled = maps.reshape(*shape, 7, 2, 7, 2).sum(axis=(-3, -1))
    pooled = pooled.reshape(*shape, 49)
    return pooled / np.maximum(pooled.sum(axis=-1, keepdims=True), EPS)


def load_maps(dataset: str, model: str) -> dict[str, np.ndarray]:
    if model == "cram_c3_stage1":
        path = EXP87 / f"experiment_8_1_replay/{dataset}/C3/seed12345/counterfactual_maps.npz"
        with np.load(path, allow_pickle=False) as archive:
            return {
                "ids": archive["ids"].astype(str),
                "matched": archive["H_plus"].copy(),
                "wrong": archive["wrong"].copy(),
                "wrong_indices": archive["wrong_indices"].copy(),
                "path": np.asarray(str(path.resolve())),
            }
    path = AUDIT / f"results/{dataset}/full/{model}_maps.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            "ids": archive["ids"].astype(str),
            "matched": archive["H_matched"].copy(),
            "wrong": archive["H_wrong"].copy(),
            "wrong_indices": archive["wrong_indices"].copy(),
            "path": np.asarray(str(path.resolve())),
        }


def top10_jaccard(matched: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    matched_top = np.argsort(matched, axis=-1, kind="stable")[:, -10:]
    wrong_top = np.argsort(wrong, axis=-1, kind="stable")[:, :, -10:]
    output = np.empty(wrong.shape[:2], dtype=np.float64)
    for image_index in range(len(matched)):
        reference = set(matched_top[image_index].tolist())
        for wrong_index in range(wrong.shape[1]):
            candidate = set(wrong_top[image_index, wrong_index].tolist())
            output[image_index, wrong_index] = len(reference & candidate) / len(reference | candidate)
    return output


def counterfactual_metrics(dataset: str, model: str, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    matched = pool_to_7(maps["matched"])
    wrong = pool_to_7(maps["wrong"])
    repeated = np.repeat(matched[:, None], wrong.shape[1], axis=1)
    spearman = row_correlation(
        rankdata(repeated, axis=-1, method="average"),
        rankdata(wrong, axis=-1, method="average"),
    )
    pearson = row_correlation(repeated, wrong)
    cosine = row_cosine(repeated, wrong)
    mae = np.abs(repeated - wrong).mean(-1)
    jaccard = top10_jaccard(matched, wrong)
    return pd.DataFrame({
        "dataset": dataset,
        "model": model,
        "sample_index": np.arange(len(matched)),
        "sample_id": maps["ids"],
        "audio_swap_spearman": spearman.mean(1),
        "audio_swap_spearman_min": spearman.min(1),
        "audio_swap_pearson": pearson.mean(1),
        "audio_swap_cosine": cosine.mean(1),
        "audio_map_mae": mae.mean(1),
        "audio_top10_jaccard": jaccard.mean(1),
    })


def resize_and_normalize(values: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    size = int(round(math.sqrt(values.shape[-1])))
    tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32).reshape(-1, 1, size, size)
    raw = F.interpolate(tensor, (224, 224), mode="bicubic", align_corners=False)[:, 0]
    flat = raw.flatten(1)
    low = flat.min(1).values[:, None, None]
    high = flat.max(1).values[:, None, None]
    normalized = (raw - low) / (high - low).clamp_min(EPS)
    return raw.cpu().numpy(), normalized.cpu().numpy()


def official_iou(prediction: np.ndarray, gt: np.ndarray) -> float:
    intersection = float(np.sum(prediction * gt))
    denominator = float(np.sum(gt) + np.sum(prediction * (gt == 0)))
    return intersection / denominator if denominator else 0.0


def natural_and_response_metrics(
    dataset: str,
    maps_by_model: dict[str, dict[str, np.ndarray]],
    gt_all: np.ndarray,
    fixed: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict] = []
    normalized_cache: dict[str, np.ndarray] = {
        model: np.empty((len(gt_all), 224, 224), dtype=np.float32) for model in MODELS
    }
    chunk_size = 32
    for start in range(0, len(gt_all), chunk_size):
        end = min(start + chunk_size, len(gt_all))
        for model in MODELS:
            values = maps_by_model[model]["matched"][start:end]
            wrong_median = np.median(maps_by_model[model]["wrong"][start:end], axis=1)
            raw_pair, normalized_pair = resize_and_normalize(
                np.stack((values, wrong_median), axis=1), device
            )
            raw_pair = raw_pair.reshape(end - start, 2, 224, 224)
            normalized_pair = normalized_pair.reshape(end - start, 2, 224, 224)
            normalized_cache[model][start:end] = normalized_pair[:, 0]
            for local, index in enumerate(range(start, end)):
                score = normalized_pair[local, 0]
                wrong_score = normalized_pair[local, 1]
                raw_matched = raw_pair[local, 0].ravel()
                raw_wrong = raw_pair[local, 1].ravel()
                gt_soft = gt_all[index]
                gt_binary = gt_soft >= 0.5
                prediction = score >= 0.6
                intersection = float(np.sum(prediction * gt_soft))
                gt_indices = fixed["indices"][index, 0, : int(fixed["counts"][index, 0])]
                near_indices = fixed["indices"][index, 2, : int(fixed["counts"][index, 2])]
                far_indices = fixed["indices"][index, 3, : int(fixed["counts"][index, 3])]
                flat = score.ravel()
                wrong_flat = wrong_score.ravel()
                gt_values = flat[gt_indices]
                near_values = flat[near_indices]
                cf_values = raw_matched - raw_wrong
                cf_auc = auc_1d(cf_values[gt_indices], cf_values[near_indices])
                cf_conditional_auc, cf_pairs = conditional_auc(
                    cf_values[gt_indices], cf_values[near_indices],
                    gt_values, near_values,
                )
                row = {
                    "dataset": dataset,
                    "model": model,
                    "sample_index": index,
                    "sample_id": str(maps_by_model[model]["ids"][index]),
                    "gt_response": float(gt_values.mean()) if len(gt_values) else math.nan,
                    "nearfp_response": float(near_values.mean()) if len(near_values) else math.nan,
                    "farfp_response": float(flat[far_indices].mean()) if len(far_indices) else math.nan,
                    "background_response": float(score[~gt_binary].mean()) if (~gt_binary).any() else math.nan,
                    "gt_nearfp_gap": float(gt_values.mean() - near_values.mean()) if len(gt_values) and len(near_values) else math.nan,
                    "gt_vs_nearfp_auroc": auc_1d(gt_values, near_values),
                    "wrong_gt_response": float(wrong_flat[gt_indices].mean()) if len(gt_indices) else math.nan,
                    "wrong_nearfp_response": float(wrong_flat[near_indices].mean()) if len(near_indices) else math.nan,
                    "matched_minus_wrong_gt": float((raw_matched - raw_wrong)[gt_indices].mean()) if len(gt_indices) else math.nan,
                    "matched_minus_wrong_nearfp": float((raw_matched - raw_wrong)[near_indices].mean()) if len(near_indices) else math.nan,
                    "counterfactual_gt_nearfp_gap": float(
                        cf_values[gt_indices].mean() - cf_values[near_indices].mean()
                    ) if len(gt_indices) and len(near_indices) else math.nan,
                    "counterfactual_gt_vs_nearfp_auroc": cf_auc,
                    "counterfactual_gt_vs_nearfp_conditional_auroc": cf_conditional_auc,
                    "counterfactual_gt_vs_nearfp_conditional_pairs": cf_pairs,
                    "iou": official_iou(prediction, gt_soft),
                    "precision": intersection / max(float(prediction.sum()), EPS),
                    "coverage": intersection / max(float(gt_soft.sum()), EPS),
                    "pred_area_ratio": float(prediction.mean()),
                    "fp_area_ratio": float(np.logical_and(prediction, ~gt_binary).mean()),
                }
                rows.append(row)

    baseline = normalized_cache["original_1.3g_final"] >= 0.6
    for row in rows:
        index = int(row["sample_index"])
        gt = gt_all[index] >= 0.5
        candidate = normalized_cache[row["model"]][index] >= 0.6
        add = np.logical_and(candidate, ~baseline[index])
        remove = np.logical_and(baseline[index], ~candidate)
        row.update({
            "AddTP": int(np.logical_and(add, gt).sum()),
            "AddFP": int(np.logical_and(add, ~gt).sum()),
            "RemoveTP": int(np.logical_and(remove, gt).sum()),
            "RemoveFP": int(np.logical_and(remove, ~gt).sum()),
            "addtp_rate_vs_baseline_fn": float(np.logical_and(add, gt).sum() / max(np.logical_and(~baseline[index], gt).sum(), 1)),
            "removefp_rate_vs_baseline_fp": float(np.logical_and(remove, ~gt).sum() / max(np.logical_and(baseline[index], ~gt).sum(), 1)),
        })
    return pd.DataFrame(rows), normalized_cache


def validate_formal_outputs(dataset: str, natural: pd.DataFrame, maps: dict[str, dict[str, np.ndarray]]) -> dict:
    checks: dict[str, float | bool] = {}
    baseline_path = ROOT / f"chuagnxindian/4.1_selective_fusion_capacity_evidence_probe/results/{dataset}_144k/raw_maps.npz"
    with np.load(baseline_path, allow_pickle=False) as archive:
        checks["baseline_ids_match_4_1"] = bool(np.array_equal(maps["original_1.3g_final"]["ids"], archive["sample_id"].astype(str)))
        checks["baseline_native_map_max_abs_error_vs_4_1"] = float(
            np.max(np.abs(maps["original_1.3g_final"]["matched"].reshape(-1, 14, 14) - archive["AUD_FINE"]))
        )
    stage2_reference = pd.read_csv(EXP87 / f"stage2/C3/{dataset}/seed12345/best_per_sample.csv")
    selected = natural[natural.model.eq("cram_c3_stage2")].sort_values("sample_index")
    checks["stage2_ids_match_formal"] = bool(np.array_equal(selected.sample_id.astype(str), stage2_reference.sample_id.astype(str)))
    checks["stage2_iou_max_abs_error_vs_formal"] = float(np.max(np.abs(selected.iou.to_numpy() - stage2_reference.AUD_iou.to_numpy())))
    stage1_reference = pd.read_csv(EXP87 / f"natural_localization/{dataset}/C3/seed12345/natural_per_sample.csv")
    stage1_reference = stage1_reference[stage1_reference["map"].eq("AUD")].sort_values("sample_index")
    selected = natural[natural.model.eq("cram_c3_stage1")].sort_values("sample_index")
    checks["stage1_ids_match_formal"] = bool(np.array_equal(selected.sample_id.astype(str), stage1_reference.sample_id.astype(str)))
    checks["stage1_iou_max_abs_error_vs_formal"] = float(np.max(np.abs(selected.iou.to_numpy() - stage1_reference.iou.to_numpy())))
    checks["passed"] = bool(
        checks["baseline_ids_match_4_1"] and checks["stage2_ids_match_formal"]
        and checks["stage1_ids_match_formal"]
        and checks["baseline_native_map_max_abs_error_vs_4_1"] <= 1e-6
        and checks["stage2_iou_max_abs_error_vs_formal"] <= 1e-6
        # The Stage-1 replay is bit-identical at native 7x7 (see its saved
        # replay audit). Its independently rerun 224x224 NumPy reduction can
        # differ by one threshold-boundary pixel, bounded here at 1e-4.
        and checks["stage1_iou_max_abs_error_vs_formal"] <= 1e-4
    )
    return checks


def summarize(dataset: str, counterfactual: pd.DataFrame, natural: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    cf_metrics = [
        "audio_swap_spearman", "audio_swap_spearman_min", "audio_swap_pearson",
        "audio_swap_cosine", "audio_map_mae", "audio_top10_jaccard",
    ]
    natural_metrics = [
        "gt_response", "nearfp_response", "farfp_response", "background_response",
        "gt_nearfp_gap", "gt_vs_nearfp_auroc", "wrong_gt_response",
        "wrong_nearfp_response", "matched_minus_wrong_gt",
        "matched_minus_wrong_nearfp", "counterfactual_gt_nearfp_gap", "iou",
        "counterfactual_gt_vs_nearfp_auroc",
        "counterfactual_gt_vs_nearfp_conditional_auroc",
        "precision", "coverage", "pred_area_ratio", "fp_area_ratio", "AddTP",
        "AddFP", "RemoveTP", "RemoveFP", "addtp_rate_vs_baseline_fn",
        "removefp_rate_vs_baseline_fp",
    ]
    for model in MODELS:
        for source, frame, metrics in (
            ("counterfactual", counterfactual[counterfactual.model.eq(model)], cf_metrics),
            ("matched_audio", natural[natural.model.eq(model)], natural_metrics),
        ):
            for metric in metrics:
                summary_rows.append({
                    "dataset": dataset, "model": model, "source": source,
                    "metric": metric, **stats(frame[metric].to_numpy(), dataset, model, metric),
                })
    summary = pd.DataFrame(summary_rows)

    delta_rows: list[dict] = []
    for candidate in MODELS[1:]:
        for frame, metrics in ((counterfactual, cf_metrics), (natural, natural_metrics)):
            baseline = frame[frame.model.eq(MODELS[0])].sort_values("sample_index")
            current = frame[frame.model.eq(candidate)].sort_values("sample_index")
            if not np.array_equal(baseline.sample_id.astype(str), current.sample_id.astype(str)):
                raise RuntimeError("Paired summary sample IDs differ")
            for metric in metrics:
                delta_rows.append({
                    "dataset": dataset, "model": candidate, "baseline": MODELS[0],
                    "metric": metric,
                    **stats(current[metric].to_numpy() - baseline[metric].to_numpy(),
                            dataset, candidate, metric, "paired_delta"),
                })
    deltas = pd.DataFrame(delta_rows)

    table_rows = []
    for model in MODELS:
        cf = counterfactual[counterfactual.model.eq(model)]
        nat = natural[natural.model.eq(model)]
        ious = nat.iou.to_numpy()
        thresholds = np.arange(21) * 0.05
        auc = float(np.trapezoid([(ious >= threshold).mean() for threshold in thresholds], thresholds))
        table_rows.append({
            "dataset": dataset,
            "model": model,
            "display": DISPLAY[model],
            "audio_swap_spearman": float(cf.audio_swap_spearman.mean()),
            "audio_map_difference": float(cf.audio_map_mae.mean()),
            "audio_swap_pearson": float(cf.audio_swap_pearson.mean()),
            "audio_swap_cosine": float(cf.audio_swap_cosine.mean()),
            "top10_jaccard": float(cf.audio_top10_jaccard.mean()),
            "GT_response": float(nat.gt_response.mean()),
            "NearFP_response": float(nat.nearfp_response.mean()),
            "GT_NearFP_gap": float(nat.gt_nearfp_gap.mean()),
            "GT_vs_NearFP_AUROC": float(nat.gt_vs_nearfp_auroc.mean()),
            "S_CF_GT_vs_NearFP_AUROC": float(nat.counterfactual_gt_vs_nearfp_auroc.mean()),
            "AUD_matched_S_CF_GT_vs_NearFP_AUROC": float(
                nat.counterfactual_gt_vs_nearfp_conditional_auroc.mean()
            ),
            "RemoveFP": float(nat.RemoveFP.mean()),
            "AddTP": float(nat.AddTP.mean()),
            "RemoveTP": float(nat.RemoveTP.mean()),
            "AddFP": float(nat.AddFP.mean()),
            "precision": float(nat.precision.mean()),
            "coverage": float(nat.coverage.mean()),
            "cIoU": float((ious >= 0.5).mean()),
            "AUC": auc,
        })
    return summary, deltas, pd.DataFrame(table_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    maps = {model: load_maps(args.dataset, model) for model in MODELS}
    reference_ids = maps[MODELS[0]]["ids"]
    reference_wrong = maps[MODELS[0]]["wrong_indices"]
    for model in MODELS[1:]:
        if not np.array_equal(reference_ids, maps[model]["ids"]):
            raise RuntimeError(f"{model}: sample ID mismatch")
        if not np.array_equal(reference_wrong, maps[model]["wrong_indices"]):
            raise RuntimeError(f"{model}: wrong-audio assignment mismatch")
    natural_path = EXP87 / f"natural_localization/{args.dataset}/C3/seed12345/natural_maps.npz"
    with np.load(natural_path, allow_pickle=False) as archive:
        if not np.array_equal(reference_ids, archive["ids"].astype(str)):
            raise RuntimeError("GT sample order mismatch")
        gt_all = archive["gt_masks"].copy()
    fixed_path = EXP80 / f"results/{args.dataset}_144k_sample_indices.npz"
    with np.load(fixed_path, allow_pickle=False) as archive:
        fixed = {key: archive[key].copy() for key in archive.files}
    if not np.array_equal(reference_ids, fixed["ids"].astype(str)):
        raise RuntimeError("Fixed-region sample order mismatch")

    counterfactual = pd.concat(
        [counterfactual_metrics(args.dataset, model, maps[model]) for model in MODELS],
        ignore_index=True,
    )
    natural, _normalized = natural_and_response_metrics(
        args.dataset, maps, gt_all, fixed, device
    )
    validation = validate_formal_outputs(args.dataset, natural, maps)
    if not validation["passed"]:
        raise RuntimeError(f"Formal output validation failed: {validation}")
    summary, deltas, table = summarize(args.dataset, counterfactual, natural)
    output = AUDIT / "results" / args.dataset
    counterfactual.to_csv(output / "audio_counterfactual_per_sample.csv", index=False)
    natural.to_csv(output / "object_spatial_per_sample.csv", index=False)
    summary.to_csv(output / "distribution_summary.csv", index=False)
    deltas.to_csv(output / "paired_delta_summary.csv", index=False)
    table.to_csv(output / "mechanism_table.csv", index=False)
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
