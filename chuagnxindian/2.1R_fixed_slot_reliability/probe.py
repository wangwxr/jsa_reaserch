#!/usr/bin/env python3
"""Experiment 2.1R: zero-training fixed-slot reliability recheck."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_fixed_slot_reliability_mpl")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PROBE21_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.1_slot_reliability_probe"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe21 = _load_module("slot_reliability_probe21_for_fixed", PROBE21_ROOT / "probe.py")
probe20 = probe21.probe20
viz = _load_module("fixed_slot_reliability_visualize", HERE / "visualize.py")

EXPERIMENTS = ("vggss_144k", "flickr_144k")
EXPECTED_COUNTS = {"vggss_144k": (355, 567), "flickr_144k": (11, 5)}
FEATURE_DIRECTIONS = {
    "semantic_margin": "higher",
    "ownership_confidence": "higher",
    "eval_soft_containment": "higher",
    "eval_seed_top10": "higher",
    "eval_seed_top20": "higher",
    "raw_seed_top10": "higher",
    "raw_seed_top20": "higher",
    "centroid_distance": "lower",
    "js_divergence": "lower",
    "extent_ratio": "closer_to_one",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--feature-workers", type=int, default=8)
    return parser.parse_args()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    flat = values.ravel()
    count = max(1, math.ceil(fraction * flat.size))
    return np.argpartition(flat, -count)[-count:]


def evaluator_features(aud: np.ndarray, slot: np.ndarray) -> dict[str, float]:
    if aud.shape != slot.shape:
        raise ValueError(f"Evaluator map mismatch: {aud.shape} vs {slot.shape}")
    aud_probability = probe21._spatial_probability(aud)
    slot_probability = probe21._spatial_probability(slot)
    aud_centroid = probe21._centroid(aud_probability, *aud.shape)
    slot_centroid = probe21._centroid(slot_probability, *slot.shape)
    extent_aud = probe21._extent_fraction(aud_probability)
    extent_slot = probe21._extent_fraction(slot_probability)
    return {
        "eval_soft_containment": float(np.sum(aud.ravel() * slot.ravel())),
        "eval_seed_top10": float(slot.ravel()[top_indices(aud, 0.10)].mean()),
        "eval_seed_top20": float(slot.ravel()[top_indices(aud, 0.20)].mean()),
        "centroid_distance": math.dist(aud_centroid, slot_centroid) / math.sqrt(2.0),
        "js_divergence": probe20.js_divergence(aud, slot),
        "extent_aud": extent_aud,
        "extent_slot": extent_slot,
        "extent_ratio": extent_slot / max(extent_aud, EPS),
    }


def raw_seed_features(aud14: np.ndarray, ownership_pair14: np.ndarray) -> dict[str, float]:
    if aud14.shape != (14, 14) or ownership_pair14.shape != (2, 14, 14):
        raise ValueError(f"Raw shape mismatch: {aud14.shape}, {ownership_pair14.shape}")
    slot0 = ownership_pair14[0]
    return {
        "raw_seed_top10": float(slot0.ravel()[top_indices(aud14, 0.10)].mean()),
        "raw_seed_top20": float(slot0.ravel()[top_indices(aud14, 0.20)].mean()),
    }


def outcome(aud_iou: float, fusion_iou: float) -> str:
    if aud_iou < 0.5 and fusion_iou >= 0.5:
        return "Rescue"
    if aud_iou >= 0.5 and fusion_iou < 0.5:
        return "Hurt"
    return "Neutral"


def reliability_aurocs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelled = [row for row in rows if row["outcome"] in {"Rescue", "Hurt"}]
    labels = np.asarray([row["outcome"] == "Rescue" for row in labelled], dtype=np.int64)
    if np.unique(labels).size != 2:
        raise RuntimeError("Both fixed-slot Rescue and Hurt are required")
    result = []
    for feature, direction in FEATURE_DIRECTIONS.items():
        values = np.asarray([float(row[feature]) for row in labelled], dtype=np.float64)
        if direction == "lower":
            values = -values
        elif direction == "closer_to_one":
            values = -np.abs(np.log(np.maximum(values, EPS)))
        result.append(
            {
                "feature": feature,
                "score_direction": direction,
                "AUROC": float(roc_auc_score(labels, values)),
                "rescue_samples": int(labels.sum()),
                "hurt_samples": int((labels == 0).sum()),
            }
        )
    return result


def distribution_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in ("Rescue", "Hurt", "Neutral"):
        selected = [row for row in rows if row["outcome"] == group]
        for feature in FEATURE_DIRECTIONS:
            values = np.asarray([row[feature] for row in selected], dtype=np.float64)
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


def load_refinement(arguments: argparse.Namespace):
    registry = probe20.EXPERIMENTS[arguments.experiment]
    experiment_name = registry["default_experiment"]
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    checkpoint_path = checkpoint_dir / f"{registry['dataset']}_best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
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
        raise RuntimeError(checkpoint.get("architecture"))
    refinement.student.proj3_spatial.load_state_dict(checkpoint["proj3_spatial_state_dict"], strict=True)
    refinement.student.adapter.load_state_dict(checkpoint["topdown_adapter_state_dict"], strict=True)
    for parameter in refinement.parameters():
        parameter.requires_grad = False
    refinement.eval()
    dataset = probe20.get_test_dataset(config, registry["dataset"])
    loader = probe20.build_test_loader(dataset, config, registry)
    return registry, experiment_name, checkpoint_dir, checkpoint_path, checkpoint, config, refinement, base_checkpoint, loader, device


def run(arguments: argparse.Namespace) -> None:
    (
        registry,
        experiment_name,
        checkpoint_dir,
        checkpoint_path,
        checkpoint,
        config,
        refinement,
        base_checkpoint,
        loader,
        device,
    ) = load_refinement(arguments)
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_before = sha256(checkpoint_path)
    mtime_before = checkpoint_path.stat().st_mtime_ns

    object_model = probe20.object_prior_model().to(device).eval()
    print("Stage 1/3: exact formal 1.3G reproduction", flush=True)
    baseline = probe20.verify_baseline(
        loader, refinement, object_model, config, registry, checkpoint_dir, output_dir
    )
    if not baseline["passed"]:
        raise RuntimeError("Formal 1.3G reproduction failed")

    reference20_path = PROBE21_ROOT.parent / "2.0_slot_objectness_probe" / "results" / arguments.experiment / "summary.json"
    reference20 = json.loads(reference20_path.read_text(encoding="utf-8"))
    print("Stage 2/3: exact Experiment 2.0 fixed-slot reproduction", flush=True)
    reproduction20 = probe21.run_20_reproduction(loader, refinement, reference20, device)
    if not reproduction20["passed"]:
        raise RuntimeError("Experiment 2.0 reproduction failed")

    print("Stage 3/3: fixed-slot reliability", flush=True)
    rows: list[dict[str, Any]] = []
    aud_ious: list[float] = []
    fixed_ious: list[float] = []
    raw_pair_sum_max_error = 0.0
    ownership_slot_sum_max_error = 0.0
    with ThreadPoolExecutor(max_workers=arguments.feature_workers) as executor:
        with torch.inference_mode():
            for image, spec, bboxes, names, _labels in tqdm(loader, desc=arguments.experiment, dynamic_ncols=True):
                image, spec, bboxes, names = probe20.flatten_eval_batch(image, spec, bboxes, names)
                image = image.to(device, non_blocking=True).float()
                spec = spec.to(device, non_blocking=True).float()
                aud = refinement(image, spec)["AUD_FINE"]
                internal = probe21.extract_internal(refinement, image, spec)
                ownership = internal["ownership"]
                ownership_slot_sum_max_error = max(
                    ownership_slot_sum_max_error,
                    float((ownership.sum(dim=1) - 1.0).abs().max()),
                )
                pair14 = F.interpolate(
                    ownership.reshape(-1, 2, 7, 7),
                    size=(14, 14),
                    mode="bilinear",
                    align_corners=False,
                )
                raw_pair_sum_max_error = max(
                    raw_pair_sum_max_error,
                    float((pair14.sum(dim=1) - 1.0).abs().max()),
                )
                ownership_stats = probe21.ownership_diagnostics(ownership)
                semantic = internal["semantic_similarity"].detach().cpu().numpy()
                resized = probe20.resize_maps({"AUD": aud, "SLOT0": internal["slot0_map"]})
                aud_raw = aud.detach().cpu().numpy()[:, 0]
                pair14_raw = pair14.detach().cpu().numpy()
                gt = bboxes.numpy()
                feature_args = []
                batch_rows = []
                for index, sample_id in enumerate(names):
                    maps = {key: probe20.normalize_map(value[index]) for key, value in resized.items()}
                    fixed_map = probe20.fuse_maps(maps["AUD"], maps["SLOT0"], 0.6)
                    aud_iou = probe20.sample_iou(maps["AUD"], gt[index])
                    fixed_iou = probe20.sample_iou(fixed_map, gt[index])
                    aud_ious.append(aud_iou)
                    fixed_ious.append(fixed_iou)
                    row = {
                        "sample_id": sample_id,
                        "IoU_AUD": aud_iou,
                        "IoU_AUD_FIXED_SLOT0": fixed_iou,
                        "outcome": outcome(aud_iou, fixed_iou),
                        "semantic_margin": float(abs(semantic[index, 0] - semantic[index, 1])),
                        **{key: float(value[index]) for key, value in ownership_stats.items()},
                    }
                    feature_args.append((maps["AUD"], maps["SLOT0"], aud_raw[index], pair14_raw[index]))
                    batch_rows.append(row)
                calculated = list(
                    executor.map(
                        lambda args: {**evaluator_features(args[0], args[1]), **raw_seed_features(args[2], args[3])},
                        feature_args,
                    )
                )
                for row, values in zip(batch_rows, calculated):
                    row.update(values)
                    rows.append(row)

    rescue = sum(row["outcome"] == "Rescue" for row in rows)
    hurt = sum(row["outcome"] == "Hurt" for row in rows)
    if (rescue, hurt) != EXPECTED_COUNTS[arguments.experiment]:
        raise RuntimeError(
            f"Fixed-slot labels mismatch: {(rescue, hurt)} vs {EXPECTED_COUNTS[arguments.experiment]}"
        )
    if raw_pair_sum_max_error > 1e-6 or ownership_slot_sum_max_error > 1e-6:
        raise RuntimeError("Raw ownership does not preserve slot competition")

    aurocs = reliability_aurocs(rows)
    distributions = distribution_statistics(rows)
    methods = [
        {"method": "AUD_FINE", **probe20.summarize_ious(aud_ious)},
        {"method": "AUD_FIXED_SLOT0", **probe20.summarize_ious(fixed_ious)},
    ]
    label_summary = {
        "rescue": rescue,
        "hurt": hurt,
        "neutral": len(rows) - rescue - hurt,
        "net_rescue": rescue - hurt,
    }
    write_csv(output_dir / "method_metrics.csv", methods)
    write_csv(output_dir / "per_sample_fixed_reliability.csv", rows)
    write_csv(output_dir / "fixed_reliability_auroc.csv", aurocs)
    write_csv(output_dir / "feature_group_statistics.csv", distributions)
    viz.save_auroc(aurocs, output_dir / "fig_fixed_reliability_auroc")
    viz.save_seed_distributions(rows, output_dir / "fig_fixed_feature_distributions")

    hash_after = sha256(checkpoint_path)
    zero_audit = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "sha256_before": hash_before,
        "sha256_after": hash_after,
        "mtime_ns_before": mtime_before,
        "mtime_ns_after": checkpoint_path.stat().st_mtime_ns,
        "checkpoint_unchanged": hash_before == hash_after and mtime_before == checkpoint_path.stat().st_mtime_ns,
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_parameters": 0,
        "model_trainable_parameters_during_probe": sum(p.numel() for p in refinement.parameters() if p.requires_grad),
        "base_checkpoint": str(base_checkpoint),
    }
    if not zero_audit["checkpoint_unchanged"] or zero_audit["model_trainable_parameters_during_probe"] != 0:
        raise RuntimeError("Zero-training audit failed")
    summary = {
        "experiment": "2.1R Fixed-Slot Reliability Recheck",
        "dataset": arguments.experiment,
        "formal_baseline_reproduced": baseline["passed"],
        "probe20_reproduced": reproduction20["passed"],
        "fixed_label_summary": label_summary,
        "method_metrics": methods,
        "reliability_auroc": aurocs,
        "feature_group_statistics": distributions,
        "raw_ownership_audit": {
            "ownership7_shape": [2, 49],
            "ownership14_shape": [2, 14, 14],
            "ownership7_slot_sum_max_error": ownership_slot_sum_max_error,
            "bilinear14_slot_sum_max_error": raw_pair_sum_max_error,
            "raw_seed_normalization": "no min-max; slot0 is true softmax-over-slots probability",
        },
        "zero_training_audit": zero_audit,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "baseline_reproduction.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (output_dir / "probe20_reproduction.json").write_text(json.dumps(reproduction20, indent=2), encoding="utf-8")
    (output_dir / "zero_training_audit.json").write_text(json.dumps(zero_audit, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    refinement.close()


if __name__ == "__main__":
    run(parse_args())
