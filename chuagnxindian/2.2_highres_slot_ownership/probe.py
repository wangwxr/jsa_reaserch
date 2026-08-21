#!/usr/bin/env python3
"""Experiment 2.2: true Q4 x K34 high-resolution Slot ownership probe."""

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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_highres_slot_mpl")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FIXED_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.1R_fixed_slot_reliability"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixed = _load_module("fixed_slot_reliability_for_hr", FIXED_ROOT / "probe.py")
probe20 = fixed.probe20
viz = _load_module("highres_slot_visualize", HERE / "visualize.py")

EXPERIMENTS = ("vggss_144k", "flickr_144k")
ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
METHODS = (
    "AUD_FINE",
    "SLOT_L4_7",
    "SLOT_L4_HR14",
    "AUD_SLOT_L4_7",
    "AUD_SLOT_L4_HR14",
    "OBJ_PRIOR",
    "OGL",
)
RELIABILITY_FEATURES = (
    "ownership_confidence",
    "eval_soft_containment",
    "eval_seed_top10",
    "eval_seed_top20",
    "raw_seed_top10",
    "raw_seed_top20",
    "centroid_distance",
    "js_divergence",
    "extent_ratio",
)
FEATURE_DIRECTIONS = {key: fixed.FEATURE_DIRECTIONS[key] for key in RELIABILITY_FEATURES}
EXPECTED_7_COUNTS = fixed.EXPECTED_COUNTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--qualitative-count", type=int, default=12)
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


def extract_highres(refinement, image: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
    teacher = refinement.teacher
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = refinement.feature_hooks.pop()
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    q4 = encoded["visual_queries"][-1]
    k4 = encoded["visual_keys"][-1]
    l4_branch = teacher.slot_attn.visual_branches[-1]
    ownership7 = torch.einsum("bsd,bnd->bsn", q4, k4).mul(l4_branch.scale).softmax(dim=1)

    f34, _f3_spatial, _f4_up, _delta = refinement.student(layer3_native, f4_projected)
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    ownership14 = torch.einsum("bsd,bnd->bsn", q4, k34).mul(l4_branch.scale).softmax(dim=1)
    aud_all = teacher.slot_attn._attention(
        encoded["audio_query"], k34, teacher.infer_sharpening
    )
    return {
        "Q4": q4,
        "K4": k4,
        "F34": f34,
        "K34": k34,
        "AUD_FINE": aud_all[:, 0].reshape(-1, 1, 14, 14),
        "OWNERSHIP7": ownership7,
        "OWNERSHIP14": ownership14,
        "SLOT7": ownership7[:, 0].reshape(-1, 1, 7, 7),
        "SLOTHR": ownership14[:, 0].reshape(-1, 1, 14, 14),
        "f4_token_error": (
            f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]
        ).abs().max(),
    }


def tensor_reconstruction_gate(loader, refinement, device) -> dict[str, Any]:
    batch = next(iter(loader))
    image, spec, bboxes, names, _labels = batch
    image, spec, _bboxes, _names = probe20.flatten_eval_batch(image, spec, bboxes, names)
    image = image.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()
    with torch.inference_mode():
        official_aud = refinement(image, spec)["AUD_FINE"]
        reference7, _audit = probe20.extract_slot_maps(refinement, image)
        output = extract_highres(refinement, image, spec)
    audit = {
        "Q4_shape": list(output["Q4"].shape),
        "K4_shape": list(output["K4"].shape),
        "F34_shape": list(output["F34"].shape),
        "K34_shape": list(output["K34"].shape),
        "ownership7_shape": list(output["OWNERSHIP7"].shape),
        "ownership14_shape": list(output["OWNERSHIP14"].shape),
        "ownership7_reconstruction_max_error": float(
            (output["SLOT7"] - reference7["SLOT_L4"]).abs().max()
        ),
        "aud_fine_local_vs_official_max_error": float(
            (output["AUD_FINE"] - official_aud).abs().max()
        ),
        "ownership7_slot_sum_max_error": float(
            (output["OWNERSHIP7"].sum(dim=1) - 1.0).abs().max()
        ),
        "ownership14_slot_sum_max_error": float(
            (output["OWNERSHIP14"].sum(dim=1) - 1.0).abs().max()
        ),
        "f4_token_error": float(output["f4_token_error"]),
    }
    expected_nonbatch = {
        "Q4_shape": [2, 512],
        "K4_shape": [49, 512],
        "F34_shape": [512, 14, 14],
        "K34_shape": [196, 512],
        "ownership7_shape": [2, 49],
        "ownership14_shape": [2, 196],
    }
    for key, expected in expected_nonbatch.items():
        if audit[key][1:] != expected:
            raise RuntimeError(f"{key}: {audit[key]} vs [B,{expected}]")
    for key in (
        "ownership7_reconstruction_max_error",
        "aud_fine_local_vs_official_max_error",
        "ownership7_slot_sum_max_error",
        "ownership14_slot_sum_max_error",
        "f4_token_error",
    ):
        if audit[key] > 1e-6:
            raise RuntimeError(f"Tensor reconstruction gate failed: {key}={audit[key]}")
    audit["passed"] = True
    return audit


def ownership_confidence(ownership: torch.Tensor) -> np.ndarray:
    probability = ownership.float().clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(dim=1).mean(dim=-1)
    return (1.0 - entropy / math.log(2.0)).cpu().numpy()


def candidate_features(
    aud_eval: np.ndarray,
    slot_eval: np.ndarray,
    aud_raw14: np.ndarray,
    ownership_pair14: np.ndarray,
) -> dict[str, float]:
    return {
        **fixed.evaluator_features(aud_eval, slot_eval),
        **fixed.raw_seed_features(aud_raw14, ownership_pair14),
    }


def candidate_aurocs(rows: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    labelled = [row for row in rows if row[f"outcome_{candidate}"] in {"Rescue", "Hurt"}]
    labels = np.asarray(
        [row[f"outcome_{candidate}"] == "Rescue" for row in labelled], dtype=np.int64
    )
    if np.unique(labels).size != 2:
        raise RuntimeError(f"Both Rescue/Hurt required for {candidate}")
    output = []
    for feature, direction in FEATURE_DIRECTIONS.items():
        values = np.asarray([row[f"{feature}_{candidate}"] for row in labelled], dtype=np.float64)
        if direction == "lower":
            values = -values
        elif direction == "closer_to_one":
            values = -np.abs(np.log(np.maximum(values, fixed.EPS)))
        output.append(
            {
                "candidate": "7x7" if candidate == "7" else "HR14",
                "feature": feature,
                "score_direction": direction,
                "AUROC": float(roc_auc_score(labels, values)),
                "rescue_samples": int(labels.sum()),
                "hurt_samples": int((labels == 0).sum()),
            }
        )
    return output


def select_qualitative(rows: list[dict[str, Any]], count: int) -> list[dict[str, str]]:
    categories = {
        "SLOT7_HURT_HR_FIX": [],
        "SLOT7_RESCUE_HR_RETAINED": [],
        "BOTH_HURT": [],
        "OGL_RESCUE": [],
        "HR_ONLY_RESCUE": [],
    }
    row_categories: dict[str, list[str]] = {}
    for row in rows:
        labels = []
        aud_success = row["IoU_AUD"] >= 0.5
        seven_success = row["IoU_AUD7"] >= 0.5
        hr_success = row["IoU_AUDHR"] >= 0.5
        if aud_success and not seven_success and hr_success:
            labels.append("SLOT7_HURT_HR_FIX")
        if not aud_success and seven_success and hr_success:
            labels.append("SLOT7_RESCUE_HR_RETAINED")
        if aud_success and not seven_success and not hr_success:
            labels.append("BOTH_HURT")
        if not aud_success and row["IoU_OGL"] >= 0.5:
            labels.append("OGL_RESCUE")
        if not aud_success and not seven_success and hr_success:
            labels.append("HR_ONLY_RESCUE")
        row_categories[row["sample_id"]] = labels
        for label in labels:
            categories[label].append(row["sample_id"])
    selected: list[str] = []
    for rank in range(3):
        for label in categories:
            candidates = categories[label]
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
    rule = "first-in-test-order round-robin over predefined transition categories, then test-order fill"
    return [
        {
            "sample_id": sample_id,
            "categories": "|".join(row_categories[sample_id]) or "FILL",
            "selection_rule": rule,
        }
        for sample_id in selected
    ]


def save_qualitative(loader, refinement, object_model, selected, rows, output_dir, device) -> None:
    wanted = {entry["sample_id"]: entry for entry in selected}
    row_lookup = {row["sample_id"]: row for row in rows}
    found = {}
    with torch.inference_mode():
        for image, spec, bboxes, names, _labels in tqdm(loader, desc="qualitative", dynamic_ncols=True):
            image, spec, bboxes, names = probe20.flatten_eval_batch(image, spec, bboxes, names)
            needed = [index for index, name in enumerate(names) if name in wanted]
            if not needed:
                continue
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()
            output = extract_highres(refinement, image, spec)
            obj = object_model(image)
            resized = probe20.resize_maps(
                {"AUD": output["AUD_FINE"], "SLOT7": output["SLOT7"], "SLOTHR": output["SLOTHR"], "OBJ": obj}
            )
            gt = bboxes.numpy()
            for index in needed:
                maps = {key: probe20.normalize_map(value[index]) for key, value in resized.items()}
                maps["AUD7"] = probe20.fuse_maps(maps["AUD"], maps["SLOT7"], 0.6)
                maps["AUDHR"] = probe20.fuse_maps(maps["AUD"], maps["SLOTHR"], 0.6)
                maps["OGL"] = probe20.fuse_maps(maps["AUD"], maps["OBJ"], 0.6)
                sample_id = names[index]
                rgb = probe20.inverse_normalize(image[index].cpu()).permute(1, 2, 0).numpy()
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
        raise RuntimeError(f"Missing qualitative IDs: {sorted(missing)}")
    for index, entry in enumerate(selected, start=1):
        viz.save_sample_panel(found[entry["sample_id"]], output_dir / f"{index:02d}_{entry['sample_id']}.png")
    viz.save_manifest(selected, output_dir / "selection_manifest.csv")


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
    ) = fixed.load_refinement(arguments)
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_before = sha256(checkpoint_path)
    mtime_before = checkpoint_path.stat().st_mtime_ns
    fixed_summary_path = FIXED_ROOT / "results" / arguments.experiment / "summary.json"
    fixed_summary = json.loads(fixed_summary_path.read_text(encoding="utf-8"))

    object_model = probe20.object_prior_model().to(device).eval()
    print("Stage 1/4: exact formal 1.3G reproduction", flush=True)
    baseline = probe20.verify_baseline(
        loader, refinement, object_model, config, registry, checkpoint_dir, output_dir
    )
    if not baseline["passed"]:
        raise RuntimeError("Formal baseline reproduction failed")
    print("Stage 2/4: Q4/K4 reconstruction gate", flush=True)
    tensor_audit = tensor_reconstruction_gate(loader, refinement, device)
    (output_dir / "tensor_reconstruction_audit.json").write_text(
        json.dumps(tensor_audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(tensor_audit, indent=2), flush=True)

    method_ious = {method: [] for method in METHODS}
    alpha_ious = {
        f"AUD_SLOT_{candidate}_A{alpha:.1f}": []
        for candidate in ("L4_7", "L4_HR14")
        for alpha in ALPHAS
    }
    rows: list[dict[str, Any]] = []
    audit_maxima = {
        "ownership7_slot_sum_max_error": 0.0,
        "ownership14_slot_sum_max_error": 0.0,
        "bilinear7_to14_slot_sum_max_error": 0.0,
        "f4_token_error": 0.0,
    }
    print("Stage 3/4: full high-resolution ownership evaluation", flush=True)
    with ThreadPoolExecutor(max_workers=arguments.feature_workers) as executor:
        with torch.inference_mode():
            for image, spec, bboxes, names, _labels in tqdm(loader, desc=arguments.experiment, dynamic_ncols=True):
                image, spec, bboxes, names = probe20.flatten_eval_batch(image, spec, bboxes, names)
                image = image.to(device, non_blocking=True).float()
                spec = spec.to(device, non_blocking=True).float()
                output = extract_highres(refinement, image, spec)
                obj = object_model(image)
                pair7_14 = F.interpolate(
                    output["OWNERSHIP7"].reshape(-1, 2, 7, 7),
                    size=(14, 14), mode="bilinear", align_corners=False,
                )
                pair_hr = output["OWNERSHIP14"].reshape(-1, 2, 14, 14)
                audit_maxima["ownership7_slot_sum_max_error"] = max(
                    audit_maxima["ownership7_slot_sum_max_error"],
                    float((output["OWNERSHIP7"].sum(dim=1) - 1.0).abs().max()),
                )
                audit_maxima["ownership14_slot_sum_max_error"] = max(
                    audit_maxima["ownership14_slot_sum_max_error"],
                    float((output["OWNERSHIP14"].sum(dim=1) - 1.0).abs().max()),
                )
                audit_maxima["bilinear7_to14_slot_sum_max_error"] = max(
                    audit_maxima["bilinear7_to14_slot_sum_max_error"],
                    float((pair7_14.sum(dim=1) - 1.0).abs().max()),
                )
                audit_maxima["f4_token_error"] = max(
                    audit_maxima["f4_token_error"], float(output["f4_token_error"])
                )
                confidence7 = ownership_confidence(output["OWNERSHIP7"])
                confidence_hr = ownership_confidence(output["OWNERSHIP14"])
                resized = probe20.resize_maps(
                    {"AUD": output["AUD_FINE"], "SLOT7": output["SLOT7"], "SLOTHR": output["SLOTHR"], "OBJ": obj}
                )
                aud_raw = output["AUD_FINE"].cpu().numpy()[:, 0]
                pair7_raw = pair7_14.cpu().numpy()
                pair_hr_raw = pair_hr.cpu().numpy()
                gt = bboxes.numpy()
                feature_args = []
                batch_rows = []
                for index, sample_id in enumerate(names):
                    maps = {key: probe20.normalize_map(value[index]) for key, value in resized.items()}
                    maps["AUD7"] = probe20.fuse_maps(maps["AUD"], maps["SLOT7"], 0.6)
                    maps["AUDHR"] = probe20.fuse_maps(maps["AUD"], maps["SLOTHR"], 0.6)
                    maps["OGL"] = probe20.fuse_maps(maps["AUD"], maps["OBJ"], 0.6)
                    ious = {
                        "AUD": probe20.sample_iou(maps["AUD"], gt[index]),
                        "SLOT7": probe20.sample_iou(maps["SLOT7"], gt[index]),
                        "SLOTHR": probe20.sample_iou(maps["SLOTHR"], gt[index]),
                        "AUD7": probe20.sample_iou(maps["AUD7"], gt[index]),
                        "AUDHR": probe20.sample_iou(maps["AUDHR"], gt[index]),
                        "OBJ": probe20.sample_iou(maps["OBJ"], gt[index]),
                        "OGL": probe20.sample_iou(maps["OGL"], gt[index]),
                    }
                    method_values = {
                        "AUD_FINE": ious["AUD"],
                        "SLOT_L4_7": ious["SLOT7"],
                        "SLOT_L4_HR14": ious["SLOTHR"],
                        "AUD_SLOT_L4_7": ious["AUD7"],
                        "AUD_SLOT_L4_HR14": ious["AUDHR"],
                        "OBJ_PRIOR": ious["OBJ"],
                        "OGL": ious["OGL"],
                    }
                    for method, value in method_values.items():
                        method_ious[method].append(value)
                    row = {
                        "sample_id": sample_id,
                        **{f"IoU_{key}": value for key, value in ious.items()},
                        "outcome_7": fixed.outcome(ious["AUD"], ious["AUD7"]),
                        "outcome_HR14": fixed.outcome(ious["AUD"], ious["AUDHR"]),
                        "ownership_confidence_7": float(confidence7[index]),
                        "ownership_confidence_HR14": float(confidence_hr[index]),
                    }
                    for candidate, slot_key in (("L4_7", "SLOT7"), ("L4_HR14", "SLOTHR")):
                        for alpha in ALPHAS:
                            key = f"AUD_SLOT_{candidate}_A{alpha:.1f}"
                            fusion = probe20.fuse_maps(maps["AUD"], maps[slot_key], alpha)
                            value = probe20.sample_iou(fusion, gt[index])
                            alpha_ious[key].append(value)
                            row[f"IoU_{key}"] = value
                    feature_args.append(
                        (
                            maps["AUD"], maps["SLOT7"], maps["SLOTHR"],
                            aud_raw[index], pair7_raw[index], pair_hr_raw[index],
                        )
                    )
                    batch_rows.append(row)
                calculated = list(
                    executor.map(
                        lambda args: (
                            candidate_features(args[0], args[1], args[3], args[4]),
                            candidate_features(args[0], args[2], args[3], args[5]),
                        ),
                        feature_args,
                    )
                )
                for row, (features7, features_hr) in zip(batch_rows, calculated):
                    row.update({f"{key}_7": value for key, value in features7.items()})
                    row.update({f"{key}_HR14": value for key, value in features_hr.items()})
                    rows.append(row)

    if any(value > 1e-6 for value in audit_maxima.values()):
        raise RuntimeError(f"Full-dataset tensor audit failed: {audit_maxima}")
    method_rows = [{"method": method, **probe20.summarize_ious(method_ious[method])} for method in METHODS]
    alpha_rows = []
    for candidate in ("L4_7", "L4_HR14"):
        for alpha in ALPHAS:
            key = f"AUD_SLOT_{candidate}_A{alpha:.1f}"
            alpha_rows.append(
                {"candidate": candidate, "alpha_aud": alpha, "method": key, **probe20.summarize_ious(alpha_ious[key])}
            )
    rescue_rows = []
    for candidate, outcome_key, iou_key in (
        ("VGG/Flickr 7x7", "outcome_7", "IoU_AUD7"),
        ("VGG/Flickr HR14", "outcome_HR14", "IoU_AUDHR"),
    ):
        rescue = sum(row[outcome_key] == "Rescue" for row in rows)
        hurt = sum(row[outcome_key] == "Hurt" for row in rows)
        oracle = [max(row["IoU_AUD"], row[iou_key]) for row in rows]
        rescue_rows.append(
            {
                "candidate": candidate.split()[-1],
                "rescue": rescue,
                "hurt": hurt,
                "net_rescue": rescue - hurt,
                "oracle_cIoU": probe20.summarize_ious(oracle)["cIoU"],
                "oracle_AUC": probe20.summarize_ious(oracle)["AUC"],
            }
        )
    if (rescue_rows[0]["rescue"], rescue_rows[0]["hurt"]) != EXPECTED_7_COUNTS[arguments.experiment]:
        raise RuntimeError(f"7x7 Rescue/Hurt mismatch: {rescue_rows[0]}")

    reliability_rows = candidate_aurocs(rows, "7") + candidate_aurocs(rows, "HR14")
    reference_aurocs = {row["feature"]: row["AUROC"] for row in fixed_summary["reliability_auroc"]}
    reliability7 = {row["feature"]: row["AUROC"] for row in reliability_rows if row["candidate"] == "7x7"}
    reliability_errors = {
        feature: abs(reliability7[feature] - reference_aurocs[feature])
        for feature in RELIABILITY_FEATURES
    }
    if any(error > 1e-12 for error in reliability_errors.values()):
        raise RuntimeError(f"2.1R reliability mismatch: {reliability_errors}")

    formal_lookup = baseline["observed"]
    method_lookup = {row["method"]: row for row in method_rows}
    main_baseline_errors = {
        "AUD_cIoU": abs(method_lookup["AUD_FINE"]["cIoU"] - formal_lookup["AUD"]["cIoU"]),
        "AUD_AUC": abs(method_lookup["AUD_FINE"]["AUC"] - formal_lookup["AUD"]["AUC"]),
        "OBJ_cIoU": abs(method_lookup["OBJ_PRIOR"]["cIoU"] - formal_lookup["OBJ_PRIOR"]["cIoU"]),
        "OBJ_AUC": abs(method_lookup["OBJ_PRIOR"]["AUC"] - formal_lookup["OBJ_PRIOR"]["AUC"]),
        "OGL_cIoU": abs(method_lookup["OGL"]["cIoU"] - formal_lookup["OGL"]["cIoU"]),
        "OGL_AUC": abs(method_lookup["OGL"]["AUC"] - formal_lookup["OGL"]["AUC"]),
    }
    if any(error > 1e-12 for error in main_baseline_errors.values()):
        raise RuntimeError(f"Main evaluator mismatch: {main_baseline_errors}")

    selected = select_qualitative(rows, arguments.qualitative_count)
    print("Stage 4/4: fixed qualitative panels", flush=True)
    save_qualitative(
        loader, refinement, object_model, selected, rows, output_dir / "qualitative", device
    )
    write_csv(output_dir / "method_metrics.csv", method_rows)
    write_csv(output_dir / "alpha_sweep.csv", alpha_rows)
    write_csv(output_dir / "rescue_hurt_oracle.csv", rescue_rows)
    write_csv(output_dir / "reliability_auroc.csv", reliability_rows)
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    viz.save_method_figure(method_rows, output_dir / "fig_method_comparison")
    viz.save_reliability_comparison(reliability_rows, output_dir / "fig_reliability_7_vs_hr")
    viz.save_rescue_figure(rescue_rows, output_dir / "fig_rescue_7_vs_hr")

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
        "experiment": "2.2 High-Resolution Internal Slot Ownership Probe",
        "dataset": arguments.experiment,
        "formal_baseline_reproduced": baseline["passed"],
        "fixed_2_1R_reproduced": all(error <= 1e-12 for error in reliability_errors.values()),
        "tensor_reconstruction_audit": tensor_audit,
        "full_tensor_audit": audit_maxima,
        "method_metrics": method_rows,
        "alpha_sweep": alpha_rows,
        "rescue_hurt_oracle": rescue_rows,
        "reliability_auroc": reliability_rows,
        "reliability_7_vs_2_1R_errors": reliability_errors,
        "main_baseline_errors": main_baseline_errors,
        "qualitative_ids": selected,
        "zero_training_audit": zero_audit,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "baseline_reproduction.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (output_dir / "zero_training_audit.json").write_text(json.dumps(zero_audit, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    refinement.close()


if __name__ == "__main__":
    run(parse_args())
