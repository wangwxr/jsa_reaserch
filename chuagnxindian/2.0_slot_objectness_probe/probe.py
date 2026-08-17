#!/usr/bin/env python3
"""Experiment 2.0: zero-training internal Slot ownership diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_slot_probe_mpl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn import metrics as sklearn_metrics
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
G_ROOT = (
    PROJECT_ROOT
    / "chuagnxindian"
    / "1mufasaslot"
    / "1.3G-multigeom_equivariant_l3_refine"
)
for import_path in (PROJECT_ROOT, G_ROOT):
    sys.path.insert(0, str(import_path))
# Keep this experiment's local plotting module ahead of Experiment G's module
# with the same generic filename.
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    EXPERIMENTS,
    build_model,
    build_test_loader,
    flatten_eval_batch,
    load_base_config,
    setup_seed,
)
from dataset import get_test_dataset, inverse_normalize  # noqa: E402
import test_model  # noqa: E402
from visualize import (  # noqa: E402
    save_metric_figure,
    save_rescue_figure,
    save_sample_panel,
    save_selection_manifest,
)


EXPERIMENT_KEYS = ("vggss_144k", "flickr_144k")
ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
BASELINE_METHODS = (
    "AUD",
    "IMG_QUERY",
    "IQR",
    "OBJ_PRIOR",
    "OGL",
    "EXTRA_IQR_OGL",
)
OFFICIAL_METHODS = (
    "AUD_FINE",
    "IMG_QUERY",
    "SLOT_L3",
    "SLOT_L4",
    "AUD_SLOT_L3",
    "AUD_SLOT_L4",
    "OBJ_PRIOR",
    "OGL",
)
CORRELATION_PAIRS = (
    ("AUD_FINE", "SLOT_L3"),
    ("AUD_FINE", "SLOT_L4"),
    ("AUD_FINE", "OBJ_PRIOR"),
    ("SLOT_L3", "OBJ_PRIOR"),
    ("SLOT_L4", "OBJ_PRIOR"),
)


class BaselineModel(nn.Module):
    """The exact formal 1.3G interface used by evaluate_full.py."""

    def __init__(self, refinement: nn.Module):
        super().__init__()
        self.refinement = refinement

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_l4, _aud_l4 = self.refinement.teacher.forward_eval(image, audio)
        aud_fine = self.refinement(image, audio)["AUD_FINE"]
        return img_l4, aud_fine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_KEYS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--correlation-workers", type=int, default=8)
    parser.add_argument("--qualitative-count", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_prior_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        test_model.NormReducer(dim=1),
        test_model.Unsqueeze(1),
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def normalize_map(value: np.ndarray) -> np.ndarray:
    """Exact equivalent of root utils.normalize_img, without mutating input."""
    # Preserve the evaluator's float32 arithmetic.  Promoting only this probe to
    # float64 can move values lying exactly on the fixed 0.6 threshold.
    value = np.asarray(value)
    minimum = value.min()
    maximum = value.max()
    if maximum - minimum != 0:
        return (value - minimum) / (maximum - minimum)
    return value.copy()


def fuse_maps(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    return normalize_map(alpha * first + (1.0 - alpha) * second)


def sample_iou(prediction: np.ndarray, gt_map: np.ndarray) -> float:
    inferred = prediction >= 0.6
    intersection = np.sum(inferred * gt_map)
    denominator = np.sum(gt_map) + np.sum(inferred * (gt_map == 0))
    return float(intersection / denominator)


def summarize_ious(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    thresholds = np.arange(21, dtype=np.float64) * 0.05
    curve = [float(np.mean(array >= threshold)) for threshold in thresholds]
    return {
        "cIoU": float(np.mean(array >= 0.5)),
        "AUC": float(sklearn_metrics.auc(thresholds, curve)),
        "mean_sample_cIoU": float(array.mean()),
        "num_samples": int(array.size),
    }


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64, copy=False).ravel()
    second = second.astype(np.float64, copy=False).ravel()
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return math.nan
    return float(np.dot(first, second) / denominator)


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.clip(first.astype(np.float64, copy=False).ravel(), 0.0, None)
    second = np.clip(second.astype(np.float64, copy=False).ravel(), 0.0, None)
    first = (first + 1e-12) / (first.sum() + 1e-12 * first.size)
    second = (second + 1e-12) / (second.sum() + 1e-12 * second.size)
    middle = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / middle))
        + 0.5 * np.sum(second * np.log(second / middle))
    )


def correlation_record(maps: dict[str, np.ndarray]) -> dict[str, float]:
    ranks = {
        name: rankdata(maps[name].ravel(), method="average")
        for name in ("AUD_FINE", "SLOT_L3", "SLOT_L4", "OBJ_PRIOR")
    }
    record: dict[str, float] = {}
    for first, second in CORRELATION_PAIRS:
        key = f"{first}__{second}"
        record[f"pearson__{key}"] = safe_pearson(maps[first], maps[second])
        record[f"spearman__{key}"] = safe_pearson(ranks[first], ranks[second])
        record[f"js__{key}"] = js_divergence(maps[first], maps[second])
    return record


def resize_maps(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    resized = {
        name: F.interpolate(
            tensor, size=(224, 224), mode="bicubic", align_corners=False
        )
        .detach()
        .cpu()
        .numpy()[:, 0]
        for name, tensor in tensors.items()
    }
    return resized


def extract_slot_maps(
    refinement: nn.Module, image: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, float | list[int]]]:
    """Recompute the final-iteration L3/L4 slot competition without hooks."""
    teacher = refinement.teacher
    image_levels = teacher.imgnet(image)
    # imgnet is instrumented by 1.3G feature hooks; this pass is for the probe,
    # so explicitly consume the hook outputs and leave no stale state.
    refinement.feature_hooks.pop()
    initial_slots = teacher.slot_attn.slots.expand(image.shape[0], -1, -1)

    target_maps: dict[str, torch.Tensor] = {}
    audit: dict[str, float | list[int]] = {}
    for level_name, branch, tokens in zip(
        ("L3", "L4"), teacher.slot_attn.visual_branches, image_levels
    ):
        _slots, final_query, keys = branch(tokens, initial_slots)
        logits = torch.einsum("bid,bjd->bij", final_query, keys) * branch.scale

        # This is the actual slot-competition tensor, before the later token
        # renormalization used only for weighted slot updates.
        ownership = logits.softmax(dim=1)
        update_weights = ownership + branch.eps
        update_weights = update_weights / update_weights.sum(
            dim=-1, keepdim=True
        )
        branch_reference = branch._attention(final_query, keys)

        spatial_size = math.isqrt(ownership.shape[-1])
        if spatial_size * spatial_size != ownership.shape[-1]:
            raise RuntimeError(
                f"{level_name} token count is not square: {ownership.shape[-1]}"
            )
        target_maps[f"SLOT_{level_name}"] = ownership[:, 0].reshape(
            image.shape[0], 1, spatial_size, spatial_size
        )
        # Batch size legitimately changes for the final non-full batch.  The
        # semantic axes being audited are slots and visual tokens.
        audit[f"{level_name}_logits_nonbatch_shape"] = list(logits.shape[1:])
        audit[f"{level_name}_ownership_slot_sum_max_error"] = float(
            (ownership.sum(dim=1) - 1.0).abs().max()
        )
        audit[f"{level_name}_update_token_sum_max_error"] = float(
            (update_weights.sum(dim=-1) - 1.0).abs().max()
        )
        audit[f"{level_name}_update_vs_branch_max_error"] = float(
            (update_weights - branch_reference).abs().max()
        )
        audit[f"{level_name}_ownership_min"] = float(ownership.min())
        audit[f"{level_name}_ownership_max"] = float(ownership.max())
    return target_maps, audit


def baseline_metrics_from_values(values: tuple[float, ...]) -> dict[str, dict[str, float]]:
    return {
        method: {
            "cIoU": float(values[2 * index]),
            "AUC": float(values[2 * index + 1]),
        }
        for index, method in enumerate(BASELINE_METHODS)
    }


def verify_baseline(
    loader,
    refinement,
    object_model,
    config,
    registry: dict,
    checkpoint_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    expected_path = checkpoint_dir / "best_full_six_metrics.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["metrics"]
    baseline_model = BaselineModel(refinement).to(next(refinement.parameters()).device)
    values = test_model.validate_img_aud(
        loader,
        baseline_model,
        object_model,
        str(output_dir / "baseline_protocol_tmp"),
        registry["dataset"],
        0,
        config,
    )
    observed = baseline_metrics_from_values(values)
    checks: dict[str, Any] = {
        "reference_file": str(expected_path.resolve()),
        "observed": observed,
        "expected": expected,
        "absolute_errors": {},
        "passed": True,
    }
    for method in BASELINE_METHODS:
        checks["absolute_errors"][method] = {}
        for metric in ("cIoU", "AUC"):
            error = abs(observed[method][metric] - float(expected[method][metric]))
            checks["absolute_errors"][method][metric] = error
            checks["passed"] = checks["passed"] and error <= 1e-12
    (output_dir / "baseline_reproduction.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    if not checks["passed"]:
        raise RuntimeError(
            "Formal 1.3G baseline did not reproduce exactly; slot analysis stopped"
        )
    return checks


def make_payload(
    sample_id: str,
    image: torch.Tensor,
    gt: np.ndarray,
    maps: dict[str, np.ndarray],
    ious: dict[str, float],
    categories: list[str],
) -> dict[str, Any]:
    rgb = inverse_normalize(image.detach().cpu()).permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    return {
        "sample_id": sample_id,
        "image": rgb.copy(),
        "GT": gt.copy(),
        **{name: value.copy() for name, value in maps.items()},
        "ious": dict(ious),
        "categories": list(categories),
    }


def select_qualitative(
    buckets: dict[str, list[str]],
    payloads: dict[str, dict[str, Any]],
    overall_ids: list[str],
    count: int,
) -> list[dict[str, Any]]:
    priority = (
        "OGL_RESCUE",
        "SLOT_L3_RESCUE",
        "SLOT_L4_RESCUE",
        "SLOT_HURT",
        "AUD_SUCCESS",
        "AUD_FAIL_NEITHER",
    )
    selected_ids: list[str] = []
    for rank in range(3):
        for category in priority:
            candidates = buckets.get(category, [])
            if rank < len(candidates) and candidates[rank] not in selected_ids:
                selected_ids.append(candidates[rank])
                if len(selected_ids) >= count:
                    break
        if len(selected_ids) >= count:
            break
    for sample_id in overall_ids:
        if sample_id not in selected_ids:
            selected_ids.append(sample_id)
        if len(selected_ids) >= count:
            break
    selected = []
    for sample_id in selected_ids:
        payload = payloads[sample_id]
        payload["selection_rule"] = (
            "first-in-test-order per predefined success/rescue/hurt/failure category; "
            "round-robin then first-in-test-order fill"
        )
        selected.append(payload)
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rescue_summary(rows: list[dict[str, Any]], slot_name: str) -> dict[str, Any]:
    aud_success = np.asarray([row["IoU_AUD"] >= 0.5 for row in rows])
    ogl_success = np.asarray([row["IoU_OGL"] >= 0.5 for row in rows])
    slot_success = np.asarray([row[f"IoU_{slot_name}"] >= 0.5 for row in rows])
    aud_failure = ~aud_success
    ogl_rescue = aud_failure & ogl_success
    slot_rescue = aud_failure & slot_success
    return {
        "method": slot_name,
        "aud_failure_count": int(aud_failure.sum()),
        "ogl_rescue_count": int(ogl_rescue.sum()),
        "rescue_count": int(slot_rescue.sum()),
        "overlap_with_ogl_rescue": int((slot_rescue & ogl_rescue).sum()),
        "only_slot_rescue": int((slot_rescue & ~ogl_rescue).sum()),
        "only_ogl_rescue": int((~slot_rescue & ogl_rescue).sum()),
        "neither_rescue": int((aud_failure & ~slot_rescue & ~ogl_rescue).sum()),
        "hurt_count": int((aud_success & ~slot_success).sum()),
        "net_rescue": int(slot_rescue.sum() - (aud_success & ~slot_success).sum()),
    }


def run_probe(arguments: argparse.Namespace) -> None:
    registry = EXPERIMENTS[arguments.experiment]
    experiment_name = registry["default_experiment"]
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    checkpoint_path = checkpoint_dir / f"{registry['dataset']}_best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash_before = sha256(checkpoint_path)
    checkpoint_mtime_before = checkpoint_path.stat().st_mtime_ns
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    config.alpha = 0.6
    config.model_dir = str(PROJECT_ROOT / "checkpoints")
    config.experiment_name = experiment_name
    setup_seed(config.seed)

    refinement, base_checkpoint = build_model(config, registry, device)
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
    object_model = object_prior_model().to(device).eval()

    test_dataset = get_test_dataset(config, registry["dataset"])
    test_loader = build_test_loader(test_dataset, config, registry)

    print("Stage 1/2: exact formal baseline reproduction", flush=True)
    baseline_check = verify_baseline(
        test_loader,
        refinement,
        object_model,
        config,
        registry,
        checkpoint_dir,
        output_dir,
    )
    print(json.dumps(baseline_check["observed"], indent=2), flush=True)
    print("Baseline exactly reproduced; starting Slot ownership probe.", flush=True)

    method_ious: dict[str, list[float]] = {
        method: [] for method in OFFICIAL_METHODS
    }
    alpha_ious: dict[str, list[float]] = {
        f"AUD_SLOT_{level}_A{alpha:.1f}": []
        for level in ("L3", "L4")
        for alpha in ALPHAS
    }
    per_sample: list[dict[str, Any]] = []
    normalization_audit: dict[str, Any] = {
        "ownership_definition": "softmax(logits, dim=slot_dim=1), before eps/token renormalization",
        "update_weight_definition": "(ownership + eps) / sum_over_tokens(ownership + eps)",
        "selected_tensor": "ownership[:, target_slot_0, :]",
        "reason": "ownership sums to one over slots for every visual token; update weights instead sum to one over tokens for each slot",
        "expected_slot_shape": [2, 49],
        "SLOT_L3_shape": [1, 7, 7],
        "SLOT_L4_shape": [1, 7, 7],
        "maxima": {},
    }
    source_file = Path(inspect.getsourcefile(
        refinement.teacher.slot_attn.visual_branches[0].__class__
    )).resolve()
    source_line = inspect.getsourcelines(
        refinement.teacher.slot_attn.visual_branches[0].__class__._attention
    )[1]
    normalization_audit["source_file"] = str(source_file)
    normalization_audit["attention_function_start_line"] = source_line

    category_buckets: dict[str, list[str]] = {}
    candidate_payloads: dict[str, dict[str, Any]] = {}
    overall_ids: list[str] = []

    print("Stage 2/2: zero-training ownership, fusion, rescue, and correlation", flush=True)
    with ThreadPoolExecutor(max_workers=arguments.correlation_workers) as executor:
        with torch.inference_mode():
            for image, spec, bboxes, names, _labels in tqdm(
                test_loader, desc=f"Probe {arguments.experiment}", dynamic_ncols=True
            ):
                image, spec, bboxes, names = flatten_eval_batch(
                    image, spec, bboxes, names
                )
                image = image.to(device, non_blocking=True).float()
                spec = spec.to(device, non_blocking=True).float()

                aud_fine = refinement(image, spec)["AUD_FINE"]
                slot_maps, batch_audit = extract_slot_maps(refinement, image)
                for key, value in batch_audit.items():
                    if isinstance(value, list):
                        previous = normalization_audit.get(key)
                        if previous is not None and previous != value:
                            raise RuntimeError(f"Inconsistent {key}: {previous} vs {value}")
                        normalization_audit[key] = value
                    elif key.endswith("_min"):
                        previous = normalization_audit["maxima"].get(key, math.inf)
                        normalization_audit["maxima"][key] = min(previous, value)
                    else:
                        previous = normalization_audit["maxima"].get(key, -math.inf)
                        normalization_audit["maxima"][key] = max(previous, value)

                teacher = refinement.teacher
                image_levels = teacher.imgnet(image)
                refinement.feature_hooks.pop()
                initial_slots = teacher.slot_attn.slots.expand(
                    image.shape[0], -1, -1
                )
                _l4_slots, l4_query, l4_keys = teacher.slot_attn.visual_branches[-1](
                    image_levels[-1], initial_slots
                )
                img_query_all = teacher.slot_attn._attention(
                    l4_query, l4_keys, teacher.infer_sharpening
                )
                img_query = img_query_all[:, 0].reshape(image.shape[0], 1, 7, 7)
                obj_prior = object_model(image)

                resized = resize_maps(
                    {
                        "AUD_FINE": aud_fine,
                        "IMG_QUERY": img_query,
                        "SLOT_L3": slot_maps["SLOT_L3"],
                        "SLOT_L4": slot_maps["SLOT_L4"],
                        "OBJ_PRIOR": obj_prior,
                    }
                )
                gt_values = bboxes.detach().cpu().numpy()
                batch_maps: list[dict[str, np.ndarray]] = []
                batch_rows: list[dict[str, Any]] = []

                for index, sample_id in enumerate(names):
                    maps = {
                        name: normalize_map(values[index])
                        for name, values in resized.items()
                    }
                    maps["AUD_SLOT_L3"] = fuse_maps(
                        maps["AUD_FINE"], maps["SLOT_L3"], 0.6
                    )
                    maps["AUD_SLOT_L4"] = fuse_maps(
                        maps["AUD_FINE"], maps["SLOT_L4"], 0.6
                    )
                    maps["OGL"] = fuse_maps(
                        maps["AUD_FINE"], maps["OBJ_PRIOR"], 0.6
                    )
                    ious = {
                        name: sample_iou(maps[name], gt_values[index])
                        for name in OFFICIAL_METHODS
                    }
                    row: dict[str, Any] = {
                        "sample_id": sample_id,
                        "IoU_AUD": ious["AUD_FINE"],
                        "IoU_IMG_QUERY": ious["IMG_QUERY"],
                        "IoU_SLOT_L3": ious["SLOT_L3"],
                        "IoU_SLOT_L4": ious["SLOT_L4"],
                        "IoU_AUD_SLOT_L3": ious["AUD_SLOT_L3"],
                        "IoU_AUD_SLOT_L4": ious["AUD_SLOT_L4"],
                        "IoU_OBJ_PRIOR": ious["OBJ_PRIOR"],
                        "IoU_OGL": ious["OGL"],
                    }
                    for method in OFFICIAL_METHODS:
                        method_ious[method].append(ious[method])
                    for level in ("L3", "L4"):
                        for alpha in ALPHAS:
                            key = f"AUD_SLOT_{level}_A{alpha:.1f}"
                            fusion = fuse_maps(
                                maps["AUD_FINE"], maps[f"SLOT_{level}"], alpha
                            )
                            value = sample_iou(fusion, gt_values[index])
                            alpha_ious[key].append(value)
                            row[f"IoU_{key}"] = value

                    aud_success = ious["AUD_FINE"] >= 0.5
                    ogl_rescue = (not aud_success) and ious["OGL"] >= 0.5
                    l3_rescue = (
                        not aud_success and ious["AUD_SLOT_L3"] >= 0.5
                    )
                    l4_rescue = (
                        not aud_success and ious["AUD_SLOT_L4"] >= 0.5
                    )
                    slot_hurt = aud_success and (
                        ious["AUD_SLOT_L3"] < 0.5
                        or ious["AUD_SLOT_L4"] < 0.5
                    )
                    categories: list[str] = []
                    if aud_success:
                        categories.append("AUD_SUCCESS")
                    if ogl_rescue:
                        categories.append("OGL_RESCUE")
                    if l3_rescue:
                        categories.append("SLOT_L3_RESCUE")
                    if l4_rescue:
                        categories.append("SLOT_L4_RESCUE")
                    if slot_hurt:
                        categories.append("SLOT_HURT")
                    if not aud_success and not ogl_rescue and not l3_rescue and not l4_rescue:
                        categories.append("AUD_FAIL_NEITHER")

                    should_keep = len(overall_ids) < max(arguments.qualitative_count, 20)
                    for category in categories:
                        bucket = category_buckets.setdefault(category, [])
                        if len(bucket) < 3:
                            bucket.append(sample_id)
                            should_keep = True
                    if should_keep and sample_id not in candidate_payloads:
                        payload_maps = {
                            name: maps[name]
                            for name in (
                                "AUD_FINE",
                                "IMG_QUERY",
                                "SLOT_L3",
                                "SLOT_L4",
                                "AUD_SLOT_L3",
                                "AUD_SLOT_L4",
                                "OBJ_PRIOR",
                                "OGL",
                            )
                        }
                        candidate_payloads[sample_id] = make_payload(
                            sample_id,
                            image[index],
                            gt_values[index],
                            payload_maps,
                            ious,
                            categories,
                        )
                    if len(overall_ids) < max(arguments.qualitative_count, 20):
                        overall_ids.append(sample_id)
                    batch_maps.append(
                        {
                            name: maps[name]
                            for name in (
                                "AUD_FINE", "SLOT_L3", "SLOT_L4", "OBJ_PRIOR"
                            )
                        }
                    )
                    batch_rows.append(row)

                correlations = list(executor.map(correlation_record, batch_maps))
                for row, correlation in zip(batch_rows, correlations):
                    row.update(correlation)
                    per_sample.append(row)

    summary_rows = [
        {"method": method, "alpha": 0.6 if method.startswith("AUD_SLOT") else "", **summarize_ious(method_ious[method])}
        for method in OFFICIAL_METHODS
    ]
    # The second pass must reproduce the same formal maps too.  This catches
    # accidental deviations in the local resize/min-max/threshold path.
    stage2_baseline_names = {
        "AUD_FINE": "AUD",
        "IMG_QUERY": "IMG_QUERY",
        "OBJ_PRIOR": "OBJ_PRIOR",
        "OGL": "OGL",
    }
    summary_by_method = {row["method"]: row for row in summary_rows}
    stage2_errors = {}
    for local_name, formal_name in stage2_baseline_names.items():
        stage2_errors[local_name] = {}
        for metric in ("cIoU", "AUC"):
            stage2_errors[local_name][metric] = abs(
                summary_by_method[local_name][metric]
                - baseline_check["observed"][formal_name][metric]
            )
    baseline_check["stage2_map_errors"] = stage2_errors
    baseline_check["stage2_maps_passed"] = all(
        error <= 1e-12
        for values in stage2_errors.values()
        for error in values.values()
    )
    (output_dir / "baseline_reproduction.json").write_text(
        json.dumps(baseline_check, indent=2), encoding="utf-8"
    )
    if not baseline_check["stage2_maps_passed"]:
        raise RuntimeError(
            "Local probe map processing did not exactly reproduce formal baseline maps"
        )
    alpha_rows = []
    for level in ("L3", "L4"):
        for alpha in ALPHAS:
            key = f"AUD_SLOT_{level}_A{alpha:.1f}"
            alpha_rows.append(
                {
                    "slot_level": level,
                    "alpha_aud": alpha,
                    "method": key,
                    **summarize_ious(alpha_ious[key]),
                }
            )

    rescue_rows = [
        rescue_summary(per_sample, "AUD_SLOT_L3"),
        rescue_summary(per_sample, "AUD_SLOT_L4"),
    ]
    # Transparent union control: rescued if either fixed Slot fusion succeeds;
    # hurt only if both fixed Slot fusions fail on an AUD success.
    aud_success = np.asarray([row["IoU_AUD"] >= 0.5 for row in per_sample])
    ogl_success = np.asarray([row["IoU_OGL"] >= 0.5 for row in per_sample])
    l3_success = np.asarray([row["IoU_AUD_SLOT_L3"] >= 0.5 for row in per_sample])
    l4_success = np.asarray([row["IoU_AUD_SLOT_L4"] >= 0.5 for row in per_sample])
    any_slot_success = l3_success | l4_success
    aud_failure = ~aud_success
    ogl_rescue = aud_failure & ogl_success
    any_rescue = aud_failure & any_slot_success
    any_hurt = aud_success & ~any_slot_success
    rescue_rows.append(
        {
            "method": "AUD_SLOT_ANY",
            "aud_failure_count": int(aud_failure.sum()),
            "ogl_rescue_count": int(ogl_rescue.sum()),
            "rescue_count": int(any_rescue.sum()),
            "overlap_with_ogl_rescue": int((any_rescue & ogl_rescue).sum()),
            "only_slot_rescue": int((any_rescue & ~ogl_rescue).sum()),
            "only_ogl_rescue": int((~any_rescue & ogl_rescue).sum()),
            "neither_rescue": int((aud_failure & ~any_rescue & ~ogl_rescue).sum()),
            "hurt_count": int(any_hurt.sum()),
            "net_rescue": int(any_rescue.sum() - any_hurt.sum()),
        }
    )
    ogl_hurt = aud_success & ~ogl_success
    rescue_rows.append(
        {
            "method": "OGL",
            "aud_failure_count": int(aud_failure.sum()),
            "ogl_rescue_count": int(ogl_rescue.sum()),
            "rescue_count": int(ogl_rescue.sum()),
            "overlap_with_ogl_rescue": int(ogl_rescue.sum()),
            "only_slot_rescue": "",
            "only_ogl_rescue": "",
            "neither_rescue": int((aud_failure & ~ogl_rescue).sum()),
            "hurt_count": int(ogl_hurt.sum()),
            "net_rescue": int(ogl_rescue.sum() - ogl_hurt.sum()),
        }
    )

    complementarity_rows: list[dict[str, Any]] = []
    for first, second in CORRELATION_PAIRS:
        pair = f"{first}__{second}"
        for statistic in ("pearson", "spearman", "js"):
            values = np.asarray(
                [row[f"{statistic}__{pair}"] for row in per_sample],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            complementarity_rows.append(
                {
                    "map_pair": f"{first} vs {second}",
                    "statistic": statistic,
                    "mean": float(finite.mean()),
                    "std": float(finite.std()),
                    "median": float(np.median(finite)),
                    "valid_samples": int(finite.size),
                    "total_samples": int(values.size),
                }
            )

    selected = select_qualitative(
        category_buckets,
        candidate_payloads,
        overall_ids,
        arguments.qualitative_count,
    )
    qualitative_dir = output_dir / "qualitative"
    for index, payload in enumerate(selected, start=1):
        save_sample_panel(
            payload,
            qualitative_dir / f"{index:02d}_{payload['sample_id']}.png",
        )
    save_selection_manifest(selected, qualitative_dir / "selection_manifest.csv")

    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "alpha_sweep.csv", alpha_rows)
    write_csv(output_dir / "per_sample_metrics.csv", per_sample)
    write_csv(output_dir / "rescue_hurt.csv", rescue_rows)
    write_csv(output_dir / "map_complementarity.csv", complementarity_rows)
    save_metric_figure(summary_rows, output_dir / "fig_metrics_comparison")
    save_rescue_figure(rescue_rows, output_dir / "fig_rescue_hurt")

    maxima = normalization_audit["maxima"]
    for level in ("L3", "L4"):
        if maxima[f"{level}_ownership_slot_sum_max_error"] > 1e-6:
            raise RuntimeError(f"{level} ownership does not sum to one over slots")
        if maxima[f"{level}_update_token_sum_max_error"] > 1e-6:
            raise RuntimeError(f"{level} update weights do not sum to one over tokens")
        if maxima[f"{level}_update_vs_branch_max_error"] > 1e-7:
            raise RuntimeError(f"{level} locally reconstructed update weights differ")
    (output_dir / "slot_attention_normalization_audit.json").write_text(
        json.dumps(normalization_audit, indent=2), encoding="utf-8"
    )

    checkpoint_hash_after = sha256(checkpoint_path)
    checkpoint_mtime_after = checkpoint_path.stat().st_mtime_ns
    immutability = {
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
        "base_checkpoint": str(base_checkpoint),
    }
    if not immutability["checkpoint_unchanged"]:
        raise RuntimeError("The formal 1.3G checkpoint changed during the probe")
    (output_dir / "zero_training_audit.json").write_text(
        json.dumps(immutability, indent=2), encoding="utf-8"
    )

    result = {
        "experiment": "2.0 Internal Slot Objectness Probe",
        "dataset": arguments.experiment,
        "formal_1_3g_experiment": experiment_name,
        "formal_checkpoint": str(checkpoint_path.resolve()),
        "baseline_reproduced": baseline_check["passed"],
        "slot_tensor": "final-iteration softmax(logits, dim=slot_dim=1), target slot0",
        "slot_shapes": {"SLOT_L3": [1, 7, 7], "SLOT_L4": [1, 7, 7]},
        "official_metrics": summary_rows,
        "alpha_sweep": alpha_rows,
        "rescue_hurt": rescue_rows,
        "map_complementarity": complementarity_rows,
        "qualitative_ids": [payload["sample_id"] for payload in selected],
        "qualitative_dir": str(qualitative_dir.resolve()),
        "zero_training_audit": immutability,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    refinement.close()


if __name__ == "__main__":
    run_probe(parse_args())
