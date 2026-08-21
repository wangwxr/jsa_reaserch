#!/usr/bin/env python3
"""Experiment 2.5 zero-training dual-path decision probe."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

import common

visualize = common.load_module(
    "experiment_25_visualize", common.HERE / "visualize.py"
)


ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(common.EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument("--complementarity-workers", type=int, default=8)
    return parser.parse_args()


class Stage1ProjectedL3Hook:
    """Capture the trained projected native L3 map before 7x7 pooling."""

    def __init__(self, model):
        self.output: torch.Tensor | None = None
        self.handle = model.imgnet.proj3.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: torch.Tensor) -> None:
        self.output = output

    def pop(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("Projected native L3 hook did not run")
        output = self.output
        self.output = None
        return output

    def close(self) -> None:
        self.handle.remove()


def extract_stage1(model, hook: Stage1ProjectedL3Hook, image, audio) -> dict[str, torch.Tensor]:
    """Run pooled semantic Slot Attention, then read native L3 keys without an update."""
    image_levels = model.imgnet(image)
    f3_projected_native = hook.pop()
    audio_tokens = model._audio_tokens(model.audnet(audio))
    encoded = model.slot_attn._encode(image_levels, audio_tokens)

    q3 = encoded["visual_queries"][0]
    k3_pool = encoded["visual_keys"][0]
    k4 = encoded["visual_keys"][1]
    q4 = encoded["visual_queries"][1]
    l3_branch = model.slot_attn.visual_branches[0]
    l4_branch = model.slot_attn.visual_branches[1]

    native_tokens = f3_projected_native.flatten(start_dim=2).transpose(1, 2)
    k3_native = l3_branch.img_to_k(l3_branch.img_norm_input(native_tokens))

    pooled_logits_baseline = torch.einsum("bsd,bnd->bsn", q3, k3_pool).mul(
        model.slot_attn.scale
    )
    pooled_logits_reconstructed = torch.einsum("bsd,bnd->bsn", q3, k3_pool).mul(
        l3_branch.scale
    )
    own_pool_baseline = pooled_logits_baseline.softmax(dim=1)
    own_pool_reconstructed = pooled_logits_reconstructed.softmax(dim=1)
    own_native = torch.einsum("bsd,bnd->bsn", q3, k3_native).mul(
        l3_branch.scale
    ).softmax(dim=1)
    own_l4 = torch.einsum("bsd,bnd->bsn", q4, k4).mul(l4_branch.scale).softmax(dim=1)
    aud_l4 = model.slot_attn._attention(
        encoded["audio_query"], k4, model.infer_sharpening
    )
    batch = image.shape[0]
    return {
        "F3_PROJECTED_NATIVE": f3_projected_native,
        "Q3": q3,
        "K3_POOL": k3_pool,
        "K3_NATIVE": k3_native,
        "OWN_POOL_BASELINE": own_pool_baseline,
        "OWN_POOL_RECONSTRUCTED": own_pool_reconstructed,
        "OWN_NATIVE": own_native,
        "OWN_L4": own_l4,
        "SLOT_L3_POOLED": own_pool_reconstructed[:, 0].reshape(batch, 1, 7, 7),
        "SLOT_L3_NATIVE_READOUT": own_native[:, 0].reshape(batch, 1, 14, 14),
        "SLOT_L4": own_l4[:, 0].reshape(batch, 1, 7, 7),
        "AUD_STAGE1": aud_l4[:, 0].reshape(batch, 1, 7, 7),
    }


def extract_original_g(model, image, audio) -> dict[str, torch.Tensor]:
    """Reproduce the formal 2.2 Q4 x K34 readout from the original G checkpoint."""
    teacher = model.teacher
    image_levels = teacher.imgnet(image)
    layer3_native, f4_projected = model.feature_hooks.pop()
    audio_tokens = teacher._audio_tokens(teacher.audnet(audio))
    encoded = teacher.slot_attn._encode(image_levels, audio_tokens)
    q4 = encoded["visual_queries"][-1]
    l4_branch = teacher.slot_attn.visual_branches[-1]
    f34, _f3_spatial, _f4_up, _delta = model.student(layer3_native, f4_projected)
    fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
    k34 = l4_branch.img_to_k(l4_branch.img_norm_input(fine_tokens))
    own14 = torch.einsum("bsd,bnd->bsn", q4, k34).mul(l4_branch.scale).softmax(dim=1)
    aud_all = teacher.slot_attn._attention(
        encoded["audio_query"], k34, teacher.infer_sharpening
    )
    return {
        "Q4": q4,
        "K34": k34,
        "AUD_FINE": aud_all[:, 0].reshape(-1, 1, 14, 14),
        "OWN14": own14,
        "SLOT_HR14": own14[:, 0].reshape(-1, 1, 14, 14),
        "f4_token_error": (
            f4_projected.flatten(start_dim=2).transpose(1, 2) - image_levels[-1]
        ).abs().max(),
    }


def _all_finite(output: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all().item() for value in output.values())


@torch.inference_mode()
def tensor_audit(loader, stage1, stage1_hook, original_g, model24, device) -> dict[str, Any]:
    image, spec, bboxes, names, _labels = next(iter(loader))
    image, spec, _bboxes, names = common.flatten_batch(image, spec, bboxes, names)
    image = image.to(device, non_blocking=True).float()
    spec = spec.to(device, non_blocking=True).float()

    _official_slot, official_stage1_aud = stage1.forward_eval(image, spec)
    stage1_hook.pop()
    output = extract_stage1(stage1, stage1_hook, image, spec)
    audit: dict[str, Any] = {
        "sample_ids": names[:4],
        "Q3_shape": list(output["Q3"].shape),
        "K3_POOL_shape": list(output["K3_POOL"].shape),
        "K3_NATIVE_shape": list(output["K3_NATIVE"].shape),
        "F3_PROJECTED_NATIVE_shape": list(output["F3_PROJECTED_NATIVE"].shape),
        "OWN_POOL_shape": list(output["OWN_POOL_RECONSTRUCTED"].shape),
        "OWN_NATIVE_shape": list(output["OWN_NATIVE"].shape),
        "pooled_ownership_reconstruction_max_error": float(
            (output["OWN_POOL_RECONSTRUCTED"] - output["OWN_POOL_BASELINE"])
            .abs()
            .max()
        ),
        "native_readout_slot_sum_max_error": float(
            (output["OWN_NATIVE"].sum(dim=1) - 1.0).abs().max()
        ),
        "pooled_slot_sum_max_error": float(
            (output["OWN_POOL_RECONSTRUCTED"].sum(dim=1) - 1.0).abs().max()
        ),
        "stage1_aud_local_vs_official_max_error": float(
            (output["AUD_STAGE1"] - official_stage1_aud).abs().max()
        ),
        "stage1_all_finite": _all_finite(output),
    }

    if original_g is not None:
        official_g = original_g(image, spec)["AUD_FINE"]
        local_g = extract_original_g(original_g, image, spec)
        output24 = model24(image, spec)
        audit.update(
            {
                "original_G_Q4_shape": list(local_g["Q4"].shape),
                "original_G_K34_shape": list(local_g["K34"].shape),
                "original_G_aud_local_vs_official_max_error": float(
                    (local_g["AUD_FINE"] - official_g).abs().max()
                ),
                "original_G_OWN14_slot_sum_max_error": float(
                    (local_g["OWN14"].sum(dim=1) - 1.0).abs().max()
                ),
                "experiment_2_4_OWN14_slot_sum_max_error": float(
                    (output24["OWN14"].sum(dim=1) - 1.0).abs().max()
                ),
                "original_G_f4_token_error": float(local_g["f4_token_error"]),
                "stage2_all_finite": _all_finite(local_g)
                and _all_finite(output24),
            }
        )

    expected = {
        "Q3_shape": [2, 512],
        "K3_POOL_shape": [49, 512],
        "K3_NATIVE_shape": [196, 512],
        "F3_PROJECTED_NATIVE_shape": [512, 14, 14],
        "OWN_POOL_shape": [2, 49],
        "OWN_NATIVE_shape": [2, 196],
    }
    for key, nonbatch in expected.items():
        if audit[key][1:] != nonbatch:
            raise RuntimeError(f"Unexpected {key}: {audit[key]}")
    numeric_gates = [
        "pooled_ownership_reconstruction_max_error",
        "native_readout_slot_sum_max_error",
        "pooled_slot_sum_max_error",
        "stage1_aud_local_vs_official_max_error",
    ]
    if original_g is not None:
        numeric_gates.extend(
            [
                "original_G_aud_local_vs_official_max_error",
                "original_G_OWN14_slot_sum_max_error",
                "experiment_2_4_OWN14_slot_sum_max_error",
                "original_G_f4_token_error",
            ]
        )
    for key in numeric_gates:
        if audit[key] > 1e-6:
            raise RuntimeError(f"Tensor audit failed: {key}={audit[key]}")
    if not audit["stage1_all_finite"] or not audit.get("stage2_all_finite", True):
        raise RuntimeError("Tensor audit found NaN/Inf")
    audit["passed"] = True
    return audit


def complementarity_for_sample(payload) -> dict[str, dict[str, float]]:
    audio, candidates = payload
    audio_flat = audio.ravel()
    audio_rank = rankdata(audio_flat, method="average")
    output = {}
    for name, candidate in candidates.items():
        candidate_flat = candidate.ravel()
        output[name] = {
            "pearson": common.safe_pearson(audio_flat, candidate_flat),
            "spearman": common.safe_pearson(
                audio_rank, rankdata(candidate_flat, method="average")
            ),
            "js_divergence": common.js_divergence(audio_flat, candidate_flat),
        }
    return output


def append_iou(values: dict[str, list[float]], row: dict[str, Any], method: str, value: float) -> None:
    values.setdefault(method, []).append(value)
    row[f"IoU_{method}"] = value


def candidate_summary(
    dataset: str,
    candidate: str,
    ownership_values: list[float],
    audio_values: list[float],
    fusion_values: list[float],
) -> dict[str, Any]:
    ownership = common.summarize(ownership_values)
    audio = common.summarize(audio_values)
    fusion = common.summarize(fusion_values)
    shift = common.transition(audio_values, fusion_values)
    return {
        "dataset": dataset,
        "map": candidate,
        "ownership_cIoU": ownership["cIoU"],
        "ownership_AUC": ownership["AUC"],
        "audio_cIoU": audio["cIoU"],
        "audio_AUC": audio["AUC"],
        "fusion_cIoU": fusion["cIoU"],
        "fusion_AUC": fusion["AUC"],
        "rescue": shift["rescue"],
        "hurt": shift["hurt"],
        "net": shift["net"],
        "oracle_cIoU": shift["oracle"]["cIoU"],
        "oracle_AUC": shift["oracle"]["AUC"],
    }


def method_with_transition(
    dataset: str,
    method: str,
    values: list[float],
    reference: list[float],
) -> dict[str, Any]:
    metric = common.summarize(values)
    shift = common.transition(reference, values)
    return {
        "dataset": dataset,
        "method": method,
        **metric,
        "rescue": shift["rescue"],
        "hurt": shift["hurt"],
        "net": shift["net"],
        "oracle_cIoU": shift["oracle"]["cIoU"],
        "oracle_AUC": shift["oracle"]["AUC"],
    }


def reference_metric(summary: dict[str, Any], method: str) -> dict[str, Any]:
    for row in summary["method_metrics"]:
        if row["method"] == method:
            return row
    raise KeyError(method)


def metric_error(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    return {key: abs(float(actual[key]) - float(expected[key])) for key in ("cIoU", "AUC")}


def audit_models(models: dict[str, Any]) -> dict[str, Any]:
    trainable = []
    gradients = []
    eval_modes = {}
    for model_name, model in models.items():
        if model is None:
            continue
        eval_modes[model_name] = not model.training
        for parameter_name, parameter in model.named_parameters():
            full_name = f"{model_name}.{parameter_name}"
            if parameter.requires_grad:
                trainable.append(full_name)
            if parameter.grad is not None:
                gradients.append(full_name)
    return {
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": sum(
            parameter.numel()
            for model in models.values()
            if model is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_names": trainable,
        "parameters_with_grad": gradients,
        "all_models_eval": all(eval_modes.values()),
        "model_eval_modes": eval_modes,
        "torch_inference_mode_used": True,
    }


@torch.inference_mode()
def run(arguments: argparse.Namespace) -> None:
    started = time.time()
    registry = common.EXPERIMENTS[arguments.experiment]
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    artifact_paths = {
        **common.checkpoint_paths(registry),
        **common.source_artifact_paths(registry),
    }
    snapshots_before = common.snapshot_files(artifact_paths)

    config = common.load_config(registry["stage1"])
    loader = common.build_loader(config, registry)
    native_rows = common.read_csv(
        common.PROJECT_ROOT
        / "checkpoints"
        / registry["native_update"]
        / "ownership_per_sample.csv"
    )
    native_by_id = {row["sample_id"]: row for row in native_rows}
    if len(native_by_id) != len(native_rows):
        raise RuntimeError("Duplicate sample IDs in native-update reference")

    stage1 = common.load_stage1(registry, device)
    stage1_hook = Stage1ProjectedL3Hook(stage1)
    original_g = common.load_original_g(registry, device) if registry["formal_144k"] else None
    model24 = common.load_24(registry, device) if registry["formal_144k"] else None
    object_model = (
        common.object_prior_model().to(device).eval() if registry["formal_144k"] else None
    )

    audit = tensor_audit(loader, stage1, stage1_hook, original_g, model24, device)
    common.write_json(output_dir / "tensor_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)

    selection = []
    selected_ids: set[str] = set()
    if registry["formal_144k"] and not arguments.skip_qualitative:
        selection = common.read_csv(
            common.R22_ROOT
            / registry["reference_key"]
            / "qualitative"
            / "selection_manifest.csv"
        )
        selected_ids = {row["sample_id"] for row in selection}

    values: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    qualitative: dict[str, dict[str, Any]] = {}
    slot_errors = {"pooled": 0.0, "native_readout": 0.0, "l4": 0.0}
    complementarity_values = {
        name: {metric: [] for metric in ("pearson", "spearman", "js_divergence")}
        for name in ("ORIGINAL_HR14", "OWN14_24", "L3_NATIVE_READOUT")
    }
    executor = (
        ThreadPoolExecutor(max_workers=arguments.complementarity_workers)
        if registry["formal_144k"]
        else None
    )

    try:
        for batch_index, (image, spec, bboxes, names, _labels) in enumerate(
            tqdm(loader, desc=arguments.experiment, dynamic_ncols=True)
        ):
            if arguments.max_batches is not None and batch_index >= arguments.max_batches:
                break
            image, spec, bboxes, names = common.flatten_batch(image, spec, bboxes, names)
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()
            stage1_output = extract_stage1(stage1, stage1_hook, image, spec)
            slot_errors["pooled"] = max(
                slot_errors["pooled"],
                float((stage1_output["OWN_POOL_RECONSTRUCTED"].sum(dim=1) - 1).abs().max()),
            )
            slot_errors["native_readout"] = max(
                slot_errors["native_readout"],
                float((stage1_output["OWN_NATIVE"].sum(dim=1) - 1).abs().max()),
            )
            slot_errors["l4"] = max(
                slot_errors["l4"],
                float((stage1_output["OWN_L4"].sum(dim=1) - 1).abs().max()),
            )
            resized = common.resize_maps(
                {
                    "STAGE1_AUD": stage1_output["AUD_STAGE1"],
                    "SLOT_L3_POOLED": stage1_output["SLOT_L3_POOLED"],
                    "SLOT_L3_NATIVE_READOUT": stage1_output["SLOT_L3_NATIVE_READOUT"],
                    "SLOT_L4": stage1_output["SLOT_L4"],
                }
            )

            if registry["formal_144k"]:
                original_output = extract_original_g(original_g, image, spec)
                output24 = model24(image, spec)
                object_prior = object_model(image)
                resized.update(
                    common.resize_maps(
                        {
                            "ORIGINAL_G_AUD": original_output["AUD_FINE"],
                            "ORIGINAL_HR14": original_output["SLOT_HR14"],
                            "AUD_24": output24["AUD_FINE"],
                            "OWN14_24": output24["OBJ_FINE"],
                            "OBJ_PRIOR": object_prior,
                        }
                    )
                )

            ground_truth = bboxes.numpy()
            normalized_batch: list[dict[str, np.ndarray]] = []
            for index, sample_id in enumerate(names):
                if sample_id not in native_by_id:
                    raise RuntimeError(f"Missing native-update reference sample: {sample_id}")
                if sample_id in seen_ids:
                    raise RuntimeError(f"Duplicate evaluation sample: {sample_id}")
                seen_ids.add(sample_id)
                maps = {name: common.normalize_map(value[index]) for name, value in resized.items()}
                maps["STAGE1_AUD_L3_POOLED"] = common.fuse_maps(
                    maps["STAGE1_AUD"], maps["SLOT_L3_POOLED"], 0.6
                )
                maps["STAGE1_AUD_L3_NATIVE_READOUT"] = common.fuse_maps(
                    maps["STAGE1_AUD"], maps["SLOT_L3_NATIVE_READOUT"], 0.6
                )
                maps["STAGE1_AUD_L4"] = common.fuse_maps(
                    maps["STAGE1_AUD"], maps["SLOT_L4"], 0.6
                )
                native_reference = native_by_id[sample_id]
                row: dict[str, Any] = {"sample_id": sample_id}

                for method in (
                    "STAGE1_AUD",
                    "SLOT_L3_POOLED",
                    "SLOT_L3_NATIVE_READOUT",
                    "SLOT_L4",
                    "STAGE1_AUD_L3_POOLED",
                    "STAGE1_AUD_L3_NATIVE_READOUT",
                    "STAGE1_AUD_L4",
                ):
                    append_iou(
                        values,
                        row,
                        method,
                        common.sample_iou(maps[method], ground_truth[index]),
                    )

                imported = {
                    "NATIVE_UPDATE_AUD": float(native_reference["IoU_AUD"]),
                    "SLOT_L3_NATIVE_UPDATE": float(native_reference["IoU_SLOT_L3_NATIVE"]),
                    "NATIVE_UPDATE_AUD_L3_NATIVE_UPDATE": float(
                        native_reference["IoU_AUD_SLOT_L3_NATIVE"]
                    ),
                }
                for method, iou in imported.items():
                    append_iou(values, row, method, iou)

                if registry["formal_144k"]:
                    maps["ORIGINAL_G_L3_POOLED"] = common.fuse_maps(
                        maps["ORIGINAL_G_AUD"], maps["SLOT_L3_POOLED"], 0.6
                    )
                    maps["ORIGINAL_G_L3_NATIVE_READOUT"] = common.fuse_maps(
                        maps["ORIGINAL_G_AUD"], maps["SLOT_L3_NATIVE_READOUT"], 0.6
                    )
                    maps["ORIGINAL_G_L4"] = common.fuse_maps(
                        maps["ORIGINAL_G_AUD"], maps["SLOT_L4"], 0.6
                    )
                    maps["ORIGINAL_HR_FUSION"] = common.fuse_maps(
                        maps["ORIGINAL_G_AUD"], maps["ORIGINAL_HR14"], 0.6
                    )
                    maps["SAME_24_FUSION"] = common.fuse_maps(
                        maps["AUD_24"], maps["OWN14_24"], 0.6
                    )
                    maps["CROSS_FUSION"] = common.fuse_maps(
                        maps["ORIGINAL_G_AUD"], maps["OWN14_24"], 0.6
                    )
                    maps["OGL"] = common.fuse_maps(
                        maps["ORIGINAL_G_AUD"], maps["OBJ_PRIOR"], 0.6
                    )
                    for alpha in ALPHAS:
                        maps[f"CROSS_A{alpha:.1f}"] = common.fuse_maps(
                            maps["ORIGINAL_G_AUD"], maps["OWN14_24"], alpha
                        )
                    for method in (
                        "ORIGINAL_G_AUD",
                        "ORIGINAL_HR14",
                        "AUD_24",
                        "OWN14_24",
                        "ORIGINAL_G_L3_POOLED",
                        "ORIGINAL_G_L3_NATIVE_READOUT",
                        "ORIGINAL_G_L4",
                        "ORIGINAL_HR_FUSION",
                        "SAME_24_FUSION",
                        "CROSS_FUSION",
                        "OBJ_PRIOR",
                        "OGL",
                        *(f"CROSS_A{alpha:.1f}" for alpha in ALPHAS),
                    ):
                        append_iou(
                            values,
                            row,
                            method,
                            common.sample_iou(maps[method], ground_truth[index]),
                        )

                    if sample_id in selected_ids:
                        rgb = common.inverse_normalize(image[index].cpu()).permute(1, 2, 0).numpy()
                        manifest_row = next(item for item in selection if item["sample_id"] == sample_id)
                        qualitative[sample_id] = {
                            "sample_id": sample_id,
                            "categories": manifest_row["categories"],
                            "image": np.clip(rgb, 0.0, 1.0),
                            "image_tensor": image[index].cpu(),
                            "spec_tensor": spec[index].cpu(),
                            "GT": ground_truth[index],
                            "ORIGINAL_G_AUD": maps["ORIGINAL_G_AUD"],
                            "L3_POOLED": maps["SLOT_L3_POOLED"],
                            "L3_NATIVE_READOUT": maps["SLOT_L3_NATIVE_READOUT"],
                            "ORIGINAL_HR14": maps["ORIGINAL_HR14"],
                            "OWN14_24": maps["OWN14_24"],
                            "CROSS_FUSION": maps["CROSS_FUSION"],
                            "OGL": maps["OGL"],
                            "row": row,
                        }
                    normalized_batch.append(maps)
                rows.append(row)

            if registry["formal_144k"]:
                work = [
                    (
                        maps["ORIGINAL_G_AUD"],
                        {
                            "ORIGINAL_HR14": maps["ORIGINAL_HR14"],
                            "OWN14_24": maps["OWN14_24"],
                            "L3_NATIVE_READOUT": maps["SLOT_L3_NATIVE_READOUT"],
                        },
                    )
                    for maps in normalized_batch
                ]
                for sample_result in executor.map(complementarity_for_sample, work):
                    for candidate, diagnostics in sample_result.items():
                        for metric, value in diagnostics.items():
                            complementarity_values[candidate][metric].append(value)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    stage1_hook.close()
    if arguments.max_batches is None and seen_ids != set(native_by_id):
        missing = set(native_by_id) - seen_ids
        extra = seen_ids - set(native_by_id)
        raise RuntimeError(f"Sample mismatch: missing={len(missing)}, extra={len(extra)}")
    if max(slot_errors.values()) > 1e-6:
        raise RuntimeError(f"Full ownership slot-sum audit failed: {slot_errors}")

    part_a_candidates = [
        candidate_summary(
            arguments.experiment,
            "SLOT_L3_POOLED",
            values["SLOT_L3_POOLED"],
            values["STAGE1_AUD"],
            values["STAGE1_AUD_L3_POOLED"],
        ),
        candidate_summary(
            arguments.experiment,
            "SLOT_L3_NATIVE_UPDATE",
            values["SLOT_L3_NATIVE_UPDATE"],
            values["NATIVE_UPDATE_AUD"],
            values["NATIVE_UPDATE_AUD_L3_NATIVE_UPDATE"],
        ),
        candidate_summary(
            arguments.experiment,
            "SLOT_L3_NATIVE_READOUT",
            values["SLOT_L3_NATIVE_READOUT"],
            values["STAGE1_AUD"],
            values["STAGE1_AUD_L3_NATIVE_READOUT"],
        ),
        candidate_summary(
            arguments.experiment,
            "SLOT_L4",
            values["SLOT_L4"],
            values["STAGE1_AUD"],
            values["STAGE1_AUD_L4"],
        ),
    ]

    part_a_144k: list[dict[str, Any]] = []
    part_b_methods: list[dict[str, Any]] = []
    part_b_transitions: list[dict[str, Any]] = []
    alpha_diagnostics: list[dict[str, Any]] = []
    complementarity: list[dict[str, Any]] = []
    reproduction: dict[str, Any] = {}
    qualitative_native_update_iou_max_delta = 0.0

    if registry["formal_144k"]:
        g_audio = values["ORIGINAL_G_AUD"]
        for label, method in (
            ("Original G AUD", "ORIGINAL_G_AUD"),
            ("Original G + L3 pooled", "ORIGINAL_G_L3_POOLED"),
            ("Original G + L3 native readout", "ORIGINAL_G_L3_NATIVE_READOUT"),
            ("Original G + L4", "ORIGINAL_G_L4"),
            ("OGL", "OGL"),
        ):
            part_a_144k.append(method_with_transition(arguments.experiment, label, values[method], g_audio))

        for label, method in (
            ("Original G AUD", "ORIGINAL_G_AUD"),
            ("Original G + original HR14", "ORIGINAL_HR_FUSION"),
            ("2.4 AUD", "AUD_24"),
            ("2.4 AUD + 2.4 OWN14", "SAME_24_FUSION"),
            ("Original G AUD + 2.4 OWN14", "CROSS_FUSION"),
            ("OGL", "OGL"),
        ):
            part_b_methods.append(
                {"dataset": arguments.experiment, "method": label, **common.summarize(values[method])}
            )

        transitions = (
            ("Original HR14", values["ORIGINAL_G_AUD"], values["ORIGINAL_HR_FUSION"]),
            ("2.4 same-checkpoint", values["AUD_24"], values["SAME_24_FUSION"]),
            ("Cross-checkpoint", values["ORIGINAL_G_AUD"], values["CROSS_FUSION"]),
        )
        for label, reference, candidate in transitions:
            shift = common.transition(reference, candidate)
            part_b_transitions.append(
                {
                    "dataset": arguments.experiment,
                    "method": label,
                    "rescue": shift["rescue"],
                    "hurt": shift["hurt"],
                    "net": shift["net"],
                    "oracle_cIoU": shift["oracle"]["cIoU"],
                    "oracle_AUC": shift["oracle"]["AUC"],
                }
            )
        for alpha in ALPHAS:
            method = f"CROSS_A{alpha:.1f}"
            metric = common.summarize(values[method])
            shift = common.transition(g_audio, values[method])
            alpha_diagnostics.append(
                {
                    "dataset": arguments.experiment,
                    "alpha_audio": alpha,
                    **metric,
                    "rescue": shift["rescue"],
                    "hurt": shift["hurt"],
                    "net": shift["net"],
                    "oracle_cIoU": shift["oracle"]["cIoU"],
                    "oracle_AUC": shift["oracle"]["AUC"],
                    "formal": alpha == 0.6,
                }
            )
        for candidate, metrics in complementarity_values.items():
            complementarity.append(
                {
                    "dataset": arguments.experiment,
                    "candidate": candidate,
                    **{
                        f"{metric}_{stat}": value
                        for metric, raw_values in metrics.items()
                        for stat, value in common.aggregate_distribution(raw_values).items()
                    },
                }
            )

        if arguments.max_batches is None:
            reference22 = json.loads(
                (
                    common.R22_ROOT / registry["reference_key"] / "summary.json"
                ).read_text(encoding="utf-8")
            )
            reference24 = json.loads(
                (
                    common.PROJECT_ROOT
                    / "checkpoints"
                    / registry["d24"]
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
            actual_lookup = {row["method"]: row for row in part_b_methods}
            expected24 = reference24["comparison"]["new_primary"]
            reproduction = {
                "original_G_AUD": metric_error(
                    actual_lookup["Original G AUD"],
                    reference_metric(reference22, "AUD_FINE"),
                ),
                "original_HR14_fusion": metric_error(
                    actual_lookup["Original G + original HR14"],
                    reference_metric(reference22, "AUD_SLOT_L4_HR14"),
                ),
                "OGL": metric_error(
                    actual_lookup["OGL"], reference_metric(reference22, "OGL")
                ),
                "experiment_2_4_AUD": metric_error(
                    actual_lookup["2.4 AUD"], expected24["AUD_FINE"]
                ),
                "experiment_2_4_OWN14": metric_error(
                    common.summarize(values["OWN14_24"]), expected24["OBJ_FINE"]
                ),
                "experiment_2_4_fusion": metric_error(
                    actual_lookup["2.4 AUD + 2.4 OWN14"], expected24["AUD_OBJ"]
                ),
            }
            max_reproduction_error = max(
                value for errors in reproduction.values() for value in errors.values()
            )
            reproduction["max_error"] = max_reproduction_error
            reproduction["passed"] = max_reproduction_error <= 1e-12
            if not reproduction["passed"]:
                raise RuntimeError(f"Reference reproduction failed: {reproduction}")
        else:
            reproduction = {"skipped_for_partial_run": True}

        if selection and not arguments.skip_qualitative:
            missing = selected_ids - set(qualitative)
            if missing:
                raise RuntimeError(f"Missing qualitative samples: {sorted(missing)}")
            native_model = common.load_native_update(registry, device)
            native_images = torch.stack(
                [qualitative[row["sample_id"]]["image_tensor"] for row in selection]
            ).to(device)
            native_specs = torch.stack(
                [qualitative[row["sample_id"]]["spec_tensor"] for row in selection]
            ).to(device)
            native_output = native_model.forward_eval_with_ownership(native_images, native_specs)
            native_maps = common.resize_maps({"native": native_output["SLOT_L3_NATIVE"]})["native"]
            for index, manifest_row in enumerate(selection):
                payload = qualitative[manifest_row["sample_id"]]
                payload["L3_NATIVE_UPDATE"] = common.normalize_map(native_maps[index])
                reproduced_iou = common.sample_iou(
                    payload["L3_NATIVE_UPDATE"], payload["GT"]
                )
                stored_iou = float(payload["row"]["IoU_SLOT_L3_NATIVE_UPDATE"])
                qualitative_native_update_iou_max_delta = max(
                    qualitative_native_update_iou_max_delta,
                    abs(reproduced_iou - stored_iou),
                )
                payload.pop("image_tensor")
                payload.pop("spec_tensor")
                visualize.save_sample_panel(
                    payload,
                    output_dir
                    / "qualitative"
                    / f"{index + 1:02d}_{manifest_row['sample_id']}.png",
                )
            common.write_csv(output_dir / "qualitative" / "selection_manifest.csv", selection)
        else:
            native_model = None
    else:
        native_model = None

    common.write_csv(output_dir / "per_sample_metrics.csv", rows)
    common.write_csv(output_dir / "part_a_candidate_metrics.csv", part_a_candidates)
    if part_a_144k:
        common.write_csv(output_dir / "part_a_144k_methods.csv", part_a_144k)
        common.write_csv(output_dir / "part_b_methods.csv", part_b_methods)
        common.write_csv(output_dir / "part_b_transitions.csv", part_b_transitions)
        common.write_csv(output_dir / "alpha_diagnostic.csv", alpha_diagnostics)
        common.write_csv(output_dir / "map_complementarity.csv", complementarity)

    models = {
        "stage1": stage1,
        "original_G": original_g,
        "experiment_2_4": model24,
        "object_prior": object_model,
        "native_update_qualitative": native_model,
    }
    zero_training = audit_models(models)
    snapshot_verification = common.verify_snapshots(snapshots_before)
    zero_training.update(
        {
            "checkpoint_and_source_snapshots": snapshot_verification,
            "all_checkpoint_hashes_and_mtimes_unchanged": snapshot_verification[
                "all_unchanged"
            ],
            "full_slot_sum_max_errors": slot_errors,
            "qualitative_native_update_iou_max_delta": (
                qualitative_native_update_iou_max_delta
            ),
            "no_nan_or_inf": True,
        }
    )
    if (
        zero_training["new_trainable_params"] != 0
        or zero_training["trainable_parameter_names"]
        or zero_training["parameters_with_grad"]
        or not zero_training["all_models_eval"]
        or not zero_training["all_checkpoint_hashes_and_mtimes_unchanged"]
    ):
        raise RuntimeError(f"Zero-training audit failed: {zero_training}")
    common.write_json(output_dir / "zero_training_audit.json", zero_training)

    summary = {
        "experiment": "2.5 Dual-Path Decision Probe",
        "setting": arguments.experiment,
        "dataset": registry["dataset"],
        "completed_full_dataset": arguments.max_batches is None,
        "tensor_audit": audit,
        "part_a_candidates": part_a_candidates,
        "part_a_144k_methods": part_a_144k,
        "part_b_methods": part_b_methods,
        "part_b_transitions": part_b_transitions,
        "alpha_diagnostic": alpha_diagnostics,
        "map_complementarity": complementarity,
        "reference_reproduction": reproduction,
        "zero_training_audit": zero_training,
        "qualitative_ids": [row["sample_id"] for row in selection],
        "elapsed_seconds": time.time() - started,
    }
    common.write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
