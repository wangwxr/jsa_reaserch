#!/usr/bin/env python3
"""Experiment 5.4: frozen ATT_FINE @ V34 value-aware query readout."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
EXP53_ROOT = HERE.parent / "5.3_fine_iqr_q4_k34"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp53 = load_module("fine_iqr_53_for_value_readout", EXP53_ROOT / "evaluate.py")

EXPERIMENTS = tuple(sorted(exp53.EXPERIMENTS))
REPORT_METHODS = (
    "AUD_FINE",
    "IMG_L4",
    "IMG_FINE",
    "IMG_VALUE",
    "IMG_VALUE_RES",
    "IQR_FINE",
    "IQR_VALUE",
    "IQR_VALUE_RES",
)
BASELINE_METHODS = ("AUD_FINE", "IMG_L4", "IMG_FINE", "IQR_OLD", "IQR_FINE")
COMPARISONS = (
    ("IMG_VALUE", "IMG_FINE"),
    ("IMG_VALUE_RES", "IMG_FINE"),
    ("IQR_VALUE", "IQR_FINE"),
    ("IQR_VALUE_RES", "IQR_FINE"),
    ("IQR_VALUE", "AUD_FINE"),
    ("IQR_VALUE_RES", "AUD_FINE"),
)
ALPHA = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    return parser.parse_args()


class ValueForwardCapture(exp53.ForwardCapture):
    """Add the exact pre-img_to_q slot state for the final L4 iteration."""

    def __init__(self, refinement: torch.nn.Module):
        self.l4_query_slot_inputs: list[torch.Tensor] = []
        super().__init__(refinement)
        l4_branch = refinement.teacher.slot_attn.visual_branches[-1]
        self.handles.append(
            l4_branch.img_norm_slots.register_forward_pre_hook(self._slot_input_hook)
        )

    def reset(self) -> None:
        super().reset()
        self.l4_query_slot_inputs.clear()

    def _slot_input_hook(self, _module, inputs) -> None:
        self.l4_query_slot_inputs.append(inputs[0])


def manual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    slot_attention: torch.nn.Module,
    scale_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dots = (
        torch.einsum("bsd,bnd->bsn", query, key)
        * scale_multiplier
        * slot_attention.scale
    )
    ownership_raw = dots.softmax(dim=1) + slot_attention.eps
    ownership_weight = ownership_raw / ownership_raw.sum(
        dim=-1, keepdim=True
    )
    return dots, ownership_raw, ownership_weight


def native_iqr(audio: torch.Tensor, image: torch.Tensor) -> np.ndarray:
    mixed = exp53.native_minmax(ALPHA * audio + (1.0 - ALPHA) * image)
    return exp53.evaluator_map(mixed)


def tensor_stats(values: list[np.ndarray]) -> dict[str, float | int]:
    array = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def map_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    first_flat = first.reshape(first.shape[0], -1).astype(np.float64, copy=False)
    second_flat = second.reshape(second.shape[0], -1).astype(np.float64, copy=False)
    first_centered = first_flat - first_flat.mean(axis=1, keepdims=True)
    second_centered = second_flat - second_flat.mean(axis=1, keepdims=True)
    pearson_denominator = np.linalg.norm(first_centered, axis=1) * np.linalg.norm(
        second_centered, axis=1
    )
    cosine_denominator = np.linalg.norm(first_flat, axis=1) * np.linalg.norm(
        second_flat, axis=1
    )
    pearson = np.divide(
        np.sum(first_centered * second_centered, axis=1),
        pearson_denominator,
        out=np.zeros(first.shape[0], dtype=np.float64),
        where=pearson_denominator != 0,
    )
    cosine = np.divide(
        np.sum(first_flat * second_flat, axis=1),
        cosine_denominator,
        out=np.zeros(first.shape[0], dtype=np.float64),
        where=cosine_denominator != 0,
    )
    return {
        "pixel_pearson": pearson,
        "pixel_cosine": cosine,
        "mean_absolute_difference": np.mean(
            np.abs(first_flat - second_flat), axis=1
        ),
    }


def delta_statistics(candidate: list[float], reference: list[float]) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    positive = delta[delta > 0]
    negative = delta[delta < 0]
    return {
        "improved": int(np.sum(delta >= 0.01)),
        "hurt": int(np.sum(delta <= -0.01)),
        "unchanged": int(np.sum((delta > -0.01) & (delta < 0.01))),
        "mean_positive_gain": float(positive.mean()) if positive.size else 0.0,
        "mean_negative_gain": float(negative.mean()) if negative.size else 0.0,
        "mean_net_iou_gain": float(delta.mean()),
        "total_net_iou_gain": float(delta.sum()),
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def local_eval_map(heatmap: torch.Tensor) -> np.ndarray:
    resized = (
        F.interpolate(heatmap, size=(224, 224), mode="bicubic", align_corners=False)
        .detach()
        .cpu()
        .numpy()[:, 0]
    )
    output = resized.copy()
    for index in range(output.shape[0]):
        minimum = output[index].min()
        maximum = output[index].max()
        if maximum - minimum != 0:
            output[index] = (output[index] - minimum) / (maximum - minimum)
    return output


def run(arguments: argparse.Namespace) -> None:
    (
        registry,
        experiment_name,
        checkpoint_path,
        checkpoint_state_before,
        base_path,
        base_state_before,
        _reference_path,
        _formal_reference,
        checkpoint,
        config,
        refinement,
        loader,
        device,
    ) = exp53.load_formal(arguments)
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    exp53_summary_path = EXP53_ROOT / "results" / arguments.experiment / "summary.json"
    exp53_samples_path = EXP53_ROOT / "results" / arguments.experiment / "per_sample_iou.csv"
    if not exp53_summary_path.is_file() or not exp53_samples_path.is_file():
        raise FileNotFoundError("Experiment 5.3 saved results are required for reproduction")
    exp53_summary = json.loads(exp53_summary_path.read_text(encoding="utf-8"))
    exp53_samples = load_csv(exp53_samples_path)

    capture = ValueForwardCapture(refinement)
    l4_branch = refinement.teacher.slot_attn.visual_branches[-1]
    all_ious = {
        method: []
        for method in set(REPORT_METHODS).union(BASELINE_METHODS)
    }
    per_sample: list[dict[str, Any]] = []
    distributions: dict[str, list[np.ndarray]] = {
        "norm_Q4": [],
        "norm_Q_VALUE": [],
        "norm_Q_VALUE_RES": [],
        "cos_Q4_Q_VALUE": [],
        "cos_Q4_Q_VALUE_RES": [],
        "cos_U_S4_QUERY_INPUT": [],
        "map_IMG_FINE_IMG_VALUE_pixel_pearson": [],
        "map_IMG_FINE_IMG_VALUE_pixel_cosine": [],
        "map_IMG_FINE_IMG_VALUE_mean_absolute_difference": [],
        "map_IMG_FINE_IMG_VALUE_RES_pixel_pearson": [],
        "map_IMG_FINE_IMG_VALUE_RES_pixel_cosine": [],
        "map_IMG_FINE_IMG_VALUE_RES_mean_absolute_difference": [],
    }
    shapes: dict[str, list[int]] = {}
    max_errors = {
        "AUD_FINE_raw_reconstruction": 0.0,
        "IMG_L4_raw_reconstruction": 0.0,
        "IMG_FINE_manual_vs_attention": 0.0,
        "AUD_FINE_final_map_reconstruction": 0.0,
        "IMG_L4_final_map_reconstruction": 0.0,
        "IMG_FINE_final_map_reconstruction": 0.0,
        "IQR_OLD_final_map_reconstruction": 0.0,
        "IQR_FINE_final_map_reconstruction": 0.0,
        "K34_captured_vs_recomputed": 0.0,
        "Q4_from_S4_query_input": 0.0,
        "update_attention_vs_raw_control": 0.0,
        "Q4_between_formal_calls": 0.0,
        "K4_between_formal_calls": 0.0,
    }
    nan_inf_count = 0

    print(f"Formal G checkpoint: {checkpoint_path}", flush=True)
    print(f"5.3 reference: {exp53_summary_path}", flush=True)
    print("Zero training: frozen K34/V34/Q4 and no optimizer/backward", flush=True)

    with torch.inference_mode():
        for batch_index, (image, spec, bboxes, names, _labels) in enumerate(
            tqdm(loader, desc=arguments.experiment, dynamic_ncols=True)
        ):
            image, spec, bboxes, names = exp53.flatten_eval_batch(
                image, spec, bboxes, names
            )
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()

            capture.reset()
            img_l4_formal, _aud_l4 = refinement.teacher.forward_eval(image, spec)
            if len(capture.l4_outputs) != 1:
                raise RuntimeError("Formal teacher did not produce exactly one L4 output")
            q4_formal, k4_formal = capture.l4_outputs[0]
            img_l4_all = refinement.teacher.slot_attn._attention(
                q4_formal, k4_formal, refinement.teacher.infer_sharpening
            )
            img_l4_reconstructed = exp53.target_map(img_l4_all, 7)

            capture.reset()
            aud_fine_formal = refinement(image, spec)["AUD_FINE"]
            if (
                len(capture.audio_queries) != 1
                or len(capture.l4_outputs) != 1
                or len(capture.f34_outputs) != 1
                or len(capture.l4_query_slot_inputs) != l4_branch.iters
            ):
                raise RuntimeError(
                    "Unexpected capture counts: "
                    f"audio={len(capture.audio_queries)}, "
                    f"l4={len(capture.l4_outputs)}, "
                    f"student={len(capture.f34_outputs)}, "
                    f"slot_inputs={len(capture.l4_query_slot_inputs)}"
                )
            qa = capture.audio_queries[0]
            q4, k4 = capture.l4_outputs[0]
            f34 = capture.f34_outputs[0]
            s4_query_input = capture.l4_query_slot_inputs[-1]
            k34_candidates = [value for value in capture.key_outputs if value.shape[1] == 196]
            if len(k34_candidates) != 1:
                raise RuntimeError(f"Expected one captured K34, got {len(k34_candidates)}")
            k34 = k34_candidates[0]

            fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)
            fine_norm = l4_branch.img_norm_input(fine_tokens)
            k34_recomputed = l4_branch.img_to_k(fine_norm)
            v34 = l4_branch.img_to_v(fine_norm)

            aud_fine_all = refinement.teacher.slot_attn._attention(
                qa, k34, refinement.teacher.infer_sharpening
            )
            aud_fine_reconstructed = exp53.target_map(aud_fine_all, 14)
            img_fine_all = refinement.teacher.slot_attn._attention(
                q4, k34, refinement.teacher.infer_sharpening
            )
            img_fine = exp53.target_map(img_fine_all, 14)

            dots, ownership_raw, ownership_weight = manual_attention(
                q4,
                k34,
                refinement.teacher.slot_attn,
                refinement.teacher.infer_sharpening,
            )
            update_attention = img_fine_all
            u_update = torch.einsum("bnd,bsn->bsd", v34, update_attention)
            q_value = l4_branch.img_to_q(l4_branch.img_norm_slots(u_update))
            # Q4 was produced from this exact slot state at the final iteration.
            u_res = s4_query_input + u_update
            q_value_res = l4_branch.img_to_q(l4_branch.img_norm_slots(u_res))

            img_value_all = refinement.teacher.slot_attn._attention(
                q_value, k34, refinement.teacher.infer_sharpening
            )
            img_value_res_all = refinement.teacher.slot_attn._attention(
                q_value_res, k34, refinement.teacher.infer_sharpening
            )
            img_value = exp53.target_map(img_value_all, 14)
            img_value_res = exp53.target_map(img_value_res_all, 14)

            if batch_index == 0:
                shapes = {
                    "Qa": list(qa.shape),
                    "Q4": list(q4.shape),
                    "K4": list(k4.shape),
                    "F34": list(f34.shape),
                    "fine_tokens": list(fine_tokens.shape),
                    "fine_norm": list(fine_norm.shape),
                    "K34": list(k34.shape),
                    "V34": list(v34.shape),
                    "dots": list(dots.shape),
                    "ownership_raw": list(ownership_raw.shape),
                    "update_attention": list(update_attention.shape),
                    "U_UPDATE": list(u_update.shape),
                    "S4_QUERY_INPUT": list(s4_query_input.shape),
                    "Q_VALUE": list(q_value.shape),
                    "U_RES": list(u_res.shape),
                    "Q_VALUE_RES": list(q_value_res.shape),
                    "AUD_FINE": list(aud_fine_formal.shape),
                    "IMG_L4": list(img_l4_formal.shape),
                    "IMG_FINE": list(img_fine.shape),
                    "IMG_VALUE": list(img_value.shape),
                    "IMG_VALUE_RES": list(img_value_res.shape),
                }

            q4_reconstructed = l4_branch.img_to_q(
                l4_branch.img_norm_slots(s4_query_input)
            )
            manual_img_fine = exp53.target_map(ownership_weight, 14)
            batch_errors = {
                "AUD_FINE_raw_reconstruction": float(
                    (aud_fine_formal - aud_fine_reconstructed).abs().max()
                ),
                "IMG_L4_raw_reconstruction": float(
                    (img_l4_formal - img_l4_reconstructed).abs().max()
                ),
                "IMG_FINE_manual_vs_attention": float(
                    (img_fine - manual_img_fine).abs().max()
                ),
                "K34_captured_vs_recomputed": float(
                    (k34 - k34_recomputed).abs().max()
                ),
                "Q4_from_S4_query_input": float((q4 - q4_reconstructed).abs().max()),
                "update_attention_vs_raw_control": float(
                    (update_attention - ownership_weight).abs().max()
                ),
                "Q4_between_formal_calls": float((q4_formal - q4).abs().max()),
                "K4_between_formal_calls": float((k4_formal - k4).abs().max()),
            }
            for key, value in batch_errors.items():
                max_errors[key] = max(max_errors[key], value)

            aud_eval = exp53.evaluator_map(aud_fine_formal)
            img_l4_eval = exp53.evaluator_map(img_l4_formal)
            img_fine_eval = exp53.evaluator_map(img_fine)
            img_value_eval = exp53.evaluator_map(img_value)
            img_value_res_eval = exp53.evaluator_map(img_value_res)
            iqr_old = exp53.fuse_evalspace(aud_eval, img_l4_eval)
            iqr_fine = native_iqr(aud_fine_formal, img_fine)
            iqr_value = native_iqr(aud_fine_formal, img_value)
            iqr_value_res = native_iqr(aud_fine_formal, img_value_res)

            local_aud_eval = local_eval_map(aud_fine_reconstructed)
            local_img_l4_eval = local_eval_map(img_l4_reconstructed)
            local_img_fine_eval = local_eval_map(manual_img_fine)
            local_iqr_old = exp53.normalize_batch(
                ALPHA * local_aud_eval + (1.0 - ALPHA) * local_img_l4_eval
            )
            local_iqr_fine = local_eval_map(
                exp53.native_minmax(
                    ALPHA * aud_fine_reconstructed
                    + (1.0 - ALPHA) * manual_img_fine
                )
            )
            final_errors = {
                "AUD_FINE_final_map_reconstruction": float(
                    np.max(np.abs(aud_eval - local_aud_eval))
                ),
                "IMG_L4_final_map_reconstruction": float(
                    np.max(np.abs(img_l4_eval - local_img_l4_eval))
                ),
                "IMG_FINE_final_map_reconstruction": float(
                    np.max(np.abs(img_fine_eval - local_img_fine_eval))
                ),
                "IQR_OLD_final_map_reconstruction": float(
                    np.max(np.abs(iqr_old - local_iqr_old))
                ),
                "IQR_FINE_final_map_reconstruction": float(
                    np.max(np.abs(iqr_fine - local_iqr_fine))
                ),
            }
            for key, value in final_errors.items():
                max_errors[key] = max(max_errors[key], value)

            eval_maps = {
                "AUD_FINE": aud_eval,
                "IMG_L4": img_l4_eval,
                "IMG_FINE": img_fine_eval,
                "IMG_VALUE": img_value_eval,
                "IMG_VALUE_RES": img_value_res_eval,
                "IQR_OLD": iqr_old,
                "IQR_FINE": iqr_fine,
                "IQR_VALUE": iqr_value,
                "IQR_VALUE_RES": iqr_value_res,
            }
            ground_truth = bboxes.detach().cpu().numpy()
            batch_ious = {
                method: exp53.sample_iou(value, ground_truth)
                for method, value in eval_maps.items()
            }
            for method, values in batch_ious.items():
                all_ious[method].extend(values.tolist())

            distributions["norm_Q4"].append(
                q4.norm(dim=-1).detach().cpu().numpy().reshape(-1)
            )
            distributions["norm_Q_VALUE"].append(
                q_value.norm(dim=-1).detach().cpu().numpy().reshape(-1)
            )
            distributions["norm_Q_VALUE_RES"].append(
                q_value_res.norm(dim=-1).detach().cpu().numpy().reshape(-1)
            )
            distributions["cos_Q4_Q_VALUE"].append(
                F.cosine_similarity(q4, q_value, dim=-1).detach().cpu().numpy().reshape(-1)
            )
            distributions["cos_Q4_Q_VALUE_RES"].append(
                F.cosine_similarity(q4, q_value_res, dim=-1).detach().cpu().numpy().reshape(-1)
            )
            distributions["cos_U_S4_QUERY_INPUT"].append(
                F.cosine_similarity(u_update, s4_query_input, dim=-1)
                .detach().cpu().numpy().reshape(-1)
            )
            similarity_value = map_similarity(img_fine_eval, img_value_eval)
            similarity_res = map_similarity(img_fine_eval, img_value_res_eval)
            for metric, values in similarity_value.items():
                distributions[f"map_IMG_FINE_IMG_VALUE_{metric}"].append(values)
            for metric, values in similarity_res.items():
                distributions[f"map_IMG_FINE_IMG_VALUE_RES_{metric}"].append(values)

            for tensor in (
                qa,
                q4,
                k4,
                f34,
                fine_norm,
                k34,
                v34,
                update_attention,
                u_update,
                s4_query_input,
                q_value,
                q_value_res,
                img_value_all,
                img_value_res_all,
            ):
                nan_inf_count += int((~torch.isfinite(tensor)).sum().item())
            for value in eval_maps.values():
                nan_inf_count += int((~np.isfinite(value)).sum())

            for local_index, sample_name in enumerate(names):
                row: dict[str, Any] = {
                    "sample_index": len(per_sample),
                    "sample_id": str(sample_name),
                    **{
                        f"IoU_{method}": float(batch_ious[method][local_index])
                        for method in batch_ious
                    },
                }
                for candidate, reference in COMPARISONS:
                    row[f"delta_{candidate}_vs_{reference}"] = float(
                        batch_ious[candidate][local_index]
                        - batch_ious[reference][local_index]
                    )
                for metric, values in similarity_value.items():
                    row[f"IMG_FINE_vs_IMG_VALUE_{metric}"] = float(values[local_index])
                for metric, values in similarity_res.items():
                    row[f"IMG_FINE_vs_IMG_VALUE_RES_{metric}"] = float(values[local_index])
                per_sample.append(row)

    capture.close()
    refinement.close()

    metrics = {method: exp53.summarize(all_ious[method]) for method in REPORT_METHODS}
    baseline_current = {
        method: exp53.summarize(all_ious[method]) for method in BASELINE_METHODS
    }
    baseline_metric_errors = {
        method: {
            metric: abs(
                baseline_current[method][metric]
                - exp53_summary["metrics"][method][metric]
            )
            for metric in ("cIoU", "AUC", "mean_sample_cIoU")
        }
        for method in BASELINE_METHODS
    }
    if len(exp53_samples) != len(per_sample):
        raise RuntimeError(
            f"5.3 sample count changed: {len(exp53_samples)} vs {len(per_sample)}"
        )
    baseline_per_sample_errors = {method: 0.0 for method in BASELINE_METHODS}
    sample_id_mismatches = 0
    for old, new in zip(exp53_samples, per_sample):
        sample_id_mismatches += int(old["sample_id"] != new["sample_id"])
        for method in BASELINE_METHODS:
            error = abs(float(old[f"IoU_{method}"]) - float(new[f"IoU_{method}"]))
            baseline_per_sample_errors[method] = max(
                baseline_per_sample_errors[method], error
            )

    checkpoint_state_after = exp53.file_state(checkpoint_path)
    base_state_after = exp53.file_state(base_path)
    parameters_with_grad = [
        name for name, parameter in refinement.named_parameters() if parameter.grad is not None
    ]
    requires_grad_parameters = [
        name for name, parameter in refinement.named_parameters() if parameter.requires_grad
    ]
    frozen_value_path = {
        name: not parameter.requires_grad
        for module_name, module in (
            ("img_norm_input", l4_branch.img_norm_input),
            ("img_to_k", l4_branch.img_to_k),
            ("img_to_v", l4_branch.img_to_v),
            ("img_norm_slots", l4_branch.img_norm_slots),
            ("img_to_q", l4_branch.img_to_q),
        )
        for name, parameter in (
            (f"{module_name}.{parameter_name}", parameter)
            for parameter_name, parameter in module.named_parameters()
        )
    }
    baseline_passed = bool(
        sample_id_mismatches == 0
        and all(
            value <= 1e-12
            for errors in baseline_metric_errors.values()
            for value in errors.values()
        )
        and all(value <= 1e-12 for value in baseline_per_sample_errors.values())
        and all(value <= 1e-7 for value in max_errors.values())
    )
    audit = {
        "passed": bool(
            baseline_passed
            and checkpoint_state_before == checkpoint_state_after
            and base_state_before == base_state_after
            and not parameters_with_grad
            and not requires_grad_parameters
            and all(frozen_value_path.values())
            and nan_inf_count == 0
        ),
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": 0,
        "parameters_with_grad": parameters_with_grad,
        "requires_grad_parameters": requires_grad_parameters,
        "all_model_parameters_require_grad_false": not requires_grad_parameters,
        "frozen_K34_V34_query_path": frozen_value_path,
        "nan_inf_count": int(nan_inf_count),
        "checkpoint_before": checkpoint_state_before,
        "checkpoint_after": checkpoint_state_after,
        "checkpoint_unchanged": checkpoint_state_before == checkpoint_state_after,
        "base_checkpoint_before": base_state_before,
        "base_checkpoint_after": base_state_after,
        "base_checkpoint_unchanged": base_state_before == base_state_after,
        "baseline_5_3_metric_absolute_errors": baseline_metric_errors,
        "baseline_5_3_per_sample_IoU_max_errors": baseline_per_sample_errors,
        "baseline_5_3_sample_id_mismatches": sample_id_mismatches,
        "baseline_5_3_reproduction_passed": baseline_passed,
        "max_abs_errors_all_samples": max_errors,
        "tensor_shapes_first_batch": shapes,
        "infer_sharpening": float(config.infer_sharpening),
        "alpha": ALPHA,
    }
    if not audit["passed"]:
        (output_dir / "audit_failed.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Experiment 5.4 audit failed: {audit}")

    deltas = {
        f"{candidate}_minus_{reference}": {
            metric: metrics[candidate][metric] - metrics[reference][metric]
            for metric in ("cIoU", "AUC")
        }
        for candidate, reference in (
            ("IMG_VALUE", "IMG_FINE"),
            ("IMG_VALUE_RES", "IMG_FINE"),
            ("IQR_VALUE", "IQR_FINE"),
            ("IQR_VALUE_RES", "IQR_FINE"),
            ("IQR_VALUE", "AUD_FINE"),
            ("IQR_VALUE_RES", "AUD_FINE"),
        )
    }
    per_sample_analysis = {
        f"{candidate}_vs_{reference}": delta_statistics(
            all_ious[candidate], all_ious[reference]
        )
        for candidate, reference in COMPARISONS
    }
    redundancy = {
        name: tensor_stats(values) for name, values in distributions.items()
    }
    result = {
        "experiment": "5.4_value_aware_query_readout",
        "dataset_experiment": arguments.experiment,
        "formal_G_experiment": experiment_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "zero_training": True,
        "definitions": {
            "V34": "frozen l4.img_to_v(l4.img_norm_input(fine_tokens))",
            "UPDATE_ATTENTION": "teacher.slot_attn._attention(Q4,K34,infer_sharpening)",
            "U_UPDATE": "einsum('bnd,bsn->bsd', V34, UPDATE_ATTENTION)",
            "Q_VALUE": "l4.img_to_q(l4.img_norm_slots(U_UPDATE))",
            "S4_QUERY_INPUT": "input to l4.img_norm_slots at the fifth/final Slot Attention iteration",
            "Q_VALUE_RES": "l4.img_to_q(l4.img_norm_slots(S4_QUERY_INPUT + U_UPDATE))",
            "IMG_VALUE": "target(_attention(Q_VALUE,K34,infer_sharpening))",
            "IMG_VALUE_RES": "target(_attention(Q_VALUE_RES,K34,infer_sharpening))",
            "IQR": "5.3 native-14x14 fusion: eval( native_norm(0.6*AUD+0.4*IMG) )",
        },
        "metrics": metrics,
        "metric_deltas": deltas,
        "per_sample_analysis": per_sample_analysis,
        "redundancy_statistics": redundancy,
        "audit": audit,
    }
    exp53.write_csv(output_dir / "per_sample_iou_and_similarity.csv", per_sample)
    exp53.write_csv(
        output_dir / "metrics.csv",
        [
            {"dataset": arguments.experiment, "method": method, **metrics[method]}
            for method in REPORT_METHODS
        ],
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\nMethod                    cIoU       AUC", flush=True)
    for method in REPORT_METHODS:
        print(
            f"{method:<25} {metrics[method]['cIoU']:.6f}  {metrics[method]['AUC']:.6f}",
            flush=True,
        )
    print("\nDeltas:", json.dumps(deltas, indent=2), flush=True)
    print("Per-sample:", json.dumps(per_sample_analysis, indent=2), flush=True)
    print("Redundancy:", json.dumps(redundancy, indent=2), flush=True)
    print("Audit passed:", audit["passed"], flush=True)
    print(f"Saved: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    run(parse_args())
