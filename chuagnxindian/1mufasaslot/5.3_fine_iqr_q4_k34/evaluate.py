#!/usr/bin/env python3
"""Zero-training diagnostic: replace G's Q4->K4 IQR branch with Q4->K34."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
G_ROOT = HERE.parent / "1.3G-multigeom_equivariant_l3_refine"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(G_ROOT))

from common import (  # noqa: E402
    EXPERIMENTS as G_EXPERIMENTS,
    base_checkpoint_path,
    build_model,
    build_test_loader,
    flatten_eval_batch,
    load_base_config,
    setup_seed,
)
from dataset import get_test_dataset  # noqa: E402
import utils  # noqa: E402


EXPERIMENTS = {
    "vggss_144k": {
        **G_EXPERIMENTS["vggss_144k"],
        "formal_experiment": "1.3G-multigeom_equivariant_l3_refine_vggss_144k_best",
    },
    "flickr_144k": {
        **G_EXPERIMENTS["flickr_144k"],
        "formal_experiment": "1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5_best",
    },
}
METHODS = (
    "AUD_FINE",
    "IMG_L4",
    "IMG_FINE",
    "IQR_OLD",
    "IQR_FINE",
    "IQR_FINE_EVALSPACE",
)
ALPHA = 0.6
THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "mtime_ns": int(stat.st_mtime_ns),
        "size_bytes": int(stat.st_size),
    }


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


def normalize_batch(values: np.ndarray) -> np.ndarray:
    """Apply root utils.normalize_img sample by sample with float32 arithmetic."""
    output = values.copy()
    for index in range(output.shape[0]):
        output[index] = utils.normalize_img(output[index])
    return output


def resize_raw(heatmap: torch.Tensor) -> np.ndarray:
    return (
        F.interpolate(heatmap, size=(224, 224), mode="bicubic", align_corners=False)
        .detach()
        .cpu()
        .numpy()[:, 0]
    )


def evaluator_map(heatmap: torch.Tensor) -> np.ndarray:
    return normalize_batch(resize_raw(heatmap))


def native_minmax(heatmap: torch.Tensor) -> torch.Tensor:
    flat = heatmap.flatten(start_dim=1)
    minima = flat.min(dim=1).values[:, None, None, None]
    maxima = flat.max(dim=1).values[:, None, None, None]
    spans = maxima - minima
    normalized = heatmap.clone()
    valid = spans.flatten() != 0
    if valid.any():
        normalized[valid] = (heatmap[valid] - minima[valid]) / spans[valid]
    return normalized


def fuse_evalspace(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return normalize_batch(ALPHA * first + (1.0 - ALPHA) * second)


def sample_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    inferred = prediction >= THRESHOLD
    intersection = (inferred * ground_truth).sum(axis=(1, 2))
    denominator = ground_truth.sum(axis=(1, 2)) + (
        inferred * (ground_truth == 0)
    ).sum(axis=(1, 2))
    return intersection / denominator


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    thresholds = np.arange(21, dtype=np.float64) * 0.05
    curve = [float(np.mean(array >= threshold)) for threshold in thresholds]
    return {
        "cIoU": float(np.mean(array >= 0.5)),
        "AUC": float(sklearn_metrics.auc(thresholds, curve)),
        "mean_sample_cIoU": float(array.mean()),
        "num_samples": int(array.size),
    }


def max_per_sample(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    return (
        (first - second)
        .abs()
        .flatten(start_dim=1)
        .max(dim=1).values
        .detach()
        .cpu()
        .numpy()
    )


def max_numpy_per_sample(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.max(np.abs(first - second).reshape(first.shape[0], -1), axis=1)


class ForwardCapture:
    """Read-only hooks for tensors produced inside the exact formal G calls."""

    def __init__(self, refinement: torch.nn.Module):
        self.audio_queries: list[torch.Tensor] = []
        self.l4_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.key_outputs: list[torch.Tensor] = []
        self.f34_outputs: list[torch.Tensor] = []
        teacher = refinement.teacher
        l4_branch = teacher.slot_attn.visual_branches[-1]
        self.handles = [
            teacher.slot_attn.audio_branch.register_forward_hook(self._audio_hook),
            l4_branch.register_forward_hook(self._l4_hook),
            l4_branch.img_to_k.register_forward_hook(self._key_hook),
            refinement.student.register_forward_hook(self._student_hook),
        ]

    def reset(self) -> None:
        self.audio_queries.clear()
        self.l4_outputs.clear()
        self.key_outputs.clear()
        self.f34_outputs.clear()

    def _audio_hook(self, _module, _inputs, output) -> None:
        self.audio_queries.append(output[1])

    def _l4_hook(self, _module, _inputs, output) -> None:
        self.l4_outputs.append((output[1], output[2]))

    def _key_hook(self, _module, _inputs, output) -> None:
        self.key_outputs.append(output)

    def _student_hook(self, _module, _inputs, output) -> None:
        self.f34_outputs.append(output[0])

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def target_map(attention: torch.Tensor, size: int) -> torch.Tensor:
    return attention[:, 0].reshape(attention.shape[0], 1, size, size)


def classify_delta(delta: np.ndarray) -> dict[str, int]:
    return {
        "improved": int(np.sum(delta >= 0.01)),
        "hurt": int(np.sum(delta <= -0.01)),
        "unchanged": int(np.sum((delta > -0.01) & (delta < 0.01))),
    }


def load_formal(arguments: argparse.Namespace):
    registry = dict(EXPERIMENTS[arguments.experiment])
    experiment_name = registry["formal_experiment"]
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    checkpoint_path = checkpoint_dir / f"{registry['dataset']}_best.pth"
    reference_path = checkpoint_dir / "best_full_six_metrics.json"
    if not checkpoint_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"Missing formal G files in {checkpoint_dir}")

    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config = load_base_config(registry)
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    config.alpha = ALPHA
    config.model_dir = str(PROJECT_ROOT / "checkpoints")
    config.experiment_name = experiment_name
    setup_seed(config.seed)

    checkpoint_state_before = file_state(checkpoint_path)
    base_path = base_checkpoint_path(registry)
    base_state_before = file_state(base_path)
    refinement, loaded_base_path = build_model(config, registry, device)
    if loaded_base_path.resolve() != base_path.resolve():
        raise RuntimeError("Base checkpoint path changed during model construction")
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

    dataset = get_test_dataset(config, registry["dataset"])
    loader = build_test_loader(dataset, config, registry)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    return (
        registry,
        experiment_name,
        checkpoint_path,
        checkpoint_state_before,
        base_path,
        base_state_before,
        reference_path,
        reference,
        checkpoint,
        config,
        refinement,
        loader,
        device,
    )


def run(arguments: argparse.Namespace) -> None:
    (
        registry,
        experiment_name,
        checkpoint_path,
        checkpoint_state_before,
        base_path,
        base_state_before,
        reference_path,
        reference,
        checkpoint,
        config,
        refinement,
        loader,
        device,
    ) = load_formal(arguments)
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = ForwardCapture(refinement)
    ious = {method: [] for method in METHODS}
    per_sample: list[dict[str, Any]] = []
    shapes: dict[str, list[int]] = {}
    max_errors = {
        "AUD_FINE_raw": 0.0,
        "IMG_L4_raw": 0.0,
        "IQR_OLD_final": 0.0,
        "AUD_FINE_final": 0.0,
        "IMG_L4_final": 0.0,
        "Q4_between_formal_calls": 0.0,
        "K4_between_formal_calls": 0.0,
    }
    nan_inf_count = 0

    print(f"Formal G checkpoint: {checkpoint_path}", flush=True)
    print(f"Dataset: {arguments.experiment}; alpha={ALPHA}; threshold={THRESHOLD}", flush=True)
    print("Zero training: torch.inference_mode(), all parameters frozen", flush=True)

    with torch.inference_mode():
        for batch_index, (image, spec, bboxes, names, _labels) in enumerate(
            tqdm(loader, desc=arguments.experiment, dynamic_ncols=True)
        ):
            image, spec, bboxes, names = flatten_eval_batch(
                image, spec, bboxes, names
            )
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()

            # This is the unchanged first half of G FullMetricModel.forward().
            capture.reset()
            img_l4_formal, _aud_l4 = refinement.teacher.forward_eval(image, spec)
            if len(capture.l4_outputs) != 1:
                raise RuntimeError(
                    f"Expected one formal L4 branch call, got {len(capture.l4_outputs)}"
                )
            q4_formal, k4_formal = capture.l4_outputs[0]
            img_l4_all = refinement.teacher.slot_attn._attention(
                q4_formal, k4_formal, refinement.teacher.infer_sharpening
            )
            img_l4_reconstructed = target_map(img_l4_all, 7)

            # This is the unchanged second half of G FullMetricModel.forward().
            capture.reset()
            aud_fine_formal = refinement(image, spec)["AUD_FINE"]
            if (
                len(capture.audio_queries) != 1
                or len(capture.l4_outputs) != 1
                or len(capture.f34_outputs) != 1
            ):
                raise RuntimeError(
                    "Unexpected G capture counts: "
                    f"audio={len(capture.audio_queries)}, "
                    f"l4={len(capture.l4_outputs)}, "
                    f"student={len(capture.f34_outputs)}"
                )
            q4, k4 = capture.l4_outputs[0]
            qa = capture.audio_queries[0]
            f34 = capture.f34_outputs[0]
            k34_candidates = [key for key in capture.key_outputs if key.shape[1] == 196]
            if len(k34_candidates) != 1:
                raise RuntimeError(f"Expected one K34, got {len(k34_candidates)}")
            k34 = k34_candidates[0]
            fine_tokens = f34.flatten(start_dim=2).transpose(1, 2)

            aud_fine_all = refinement.teacher.slot_attn._attention(
                qa, k34, refinement.teacher.infer_sharpening
            )
            aud_fine_reconstructed = target_map(aud_fine_all, 14)
            img_fine_all = refinement.teacher.slot_attn._attention(
                q4, k34, refinement.teacher.infer_sharpening
            )
            img_fine = target_map(img_fine_all, 14)

            if batch_index == 0:
                shapes = {
                    "Qa": list(qa.shape),
                    "Q4": list(q4.shape),
                    "K4": list(k4.shape),
                    "F34": list(f34.shape),
                    "fine_tokens": list(fine_tokens.shape),
                    "K34": list(k34.shape),
                    "AUD_FINE_ALL": list(aud_fine_all.shape),
                    "IMG_L4_ALL": list(img_l4_all.shape),
                    "IMG_FINE_ALL": list(img_fine_all.shape),
                    "AUD_FINE": list(aud_fine_formal.shape),
                    "IMG_L4": list(img_l4_formal.shape),
                    "IMG_FINE": list(img_fine.shape),
                }

            raw_aud_errors = max_per_sample(
                aud_fine_formal, aud_fine_reconstructed
            )
            raw_img_errors = max_per_sample(img_l4_formal, img_l4_reconstructed)
            max_errors["AUD_FINE_raw"] = max(
                max_errors["AUD_FINE_raw"], float(raw_aud_errors.max())
            )
            max_errors["IMG_L4_raw"] = max(
                max_errors["IMG_L4_raw"], float(raw_img_errors.max())
            )
            max_errors["Q4_between_formal_calls"] = max(
                max_errors["Q4_between_formal_calls"],
                float((q4_formal - q4).abs().max()),
            )
            max_errors["K4_between_formal_calls"] = max(
                max_errors["K4_between_formal_calls"],
                float((k4_formal - k4).abs().max()),
            )

            aud_eval = evaluator_map(aud_fine_formal)
            img_l4_eval = evaluator_map(img_l4_formal)
            img_fine_eval = evaluator_map(img_fine)
            iqr_old = fuse_evalspace(aud_eval, img_l4_eval)
            iqr_old_reconstructed = fuse_evalspace(
                evaluator_map(aud_fine_reconstructed),
                evaluator_map(img_l4_reconstructed),
            )

            # Primary new path: mix the two native 14x14 probability maps,
            # normalize at 14x14, then let the unchanged evaluator resize.
            iqr_fine_native_tensor = native_minmax(
                ALPHA * aud_fine_formal + (1.0 - ALPHA) * img_fine
            )
            iqr_fine = evaluator_map(iqr_fine_native_tensor)
            # Control: reproduce the formal evaluator order at 224x224.
            iqr_fine_evalspace = fuse_evalspace(aud_eval, img_fine_eval)

            final_aud_errors = max_numpy_per_sample(
                aud_eval, evaluator_map(aud_fine_reconstructed)
            )
            final_img_errors = max_numpy_per_sample(
                img_l4_eval, evaluator_map(img_l4_reconstructed)
            )
            final_iqr_errors = max_numpy_per_sample(
                iqr_old, iqr_old_reconstructed
            )
            max_errors["AUD_FINE_final"] = max(
                max_errors["AUD_FINE_final"], float(final_aud_errors.max())
            )
            max_errors["IMG_L4_final"] = max(
                max_errors["IMG_L4_final"], float(final_img_errors.max())
            )
            max_errors["IQR_OLD_final"] = max(
                max_errors["IQR_OLD_final"], float(final_iqr_errors.max())
            )

            eval_maps = {
                "AUD_FINE": aud_eval,
                "IMG_L4": img_l4_eval,
                "IMG_FINE": img_fine_eval,
                "IQR_OLD": iqr_old,
                "IQR_FINE": iqr_fine,
                "IQR_FINE_EVALSPACE": iqr_fine_evalspace,
            }
            for value in (
                q4,
                k4,
                qa,
                f34,
                k34,
                aud_fine_all,
                img_l4_all,
                img_fine_all,
            ):
                nan_inf_count += int((~torch.isfinite(value)).sum().item())
            for value in eval_maps.values():
                nan_inf_count += int((~np.isfinite(value)).sum())

            ground_truth = bboxes.detach().cpu().numpy()
            batch_ious = {
                method: sample_iou(value, ground_truth)
                for method, value in eval_maps.items()
            }
            for method in METHODS:
                ious[method].extend(batch_ious[method].tolist())

            for local_index, sample_name in enumerate(names):
                row = {
                    "sample_index": len(per_sample),
                    "sample_id": str(sample_name),
                    **{
                        f"IoU_{method}": float(batch_ious[method][local_index])
                        for method in METHODS
                    },
                    "delta_IQR_FINE_vs_IQR_OLD": float(
                        batch_ious["IQR_FINE"][local_index]
                        - batch_ious["IQR_OLD"][local_index]
                    ),
                    "delta_IQR_FINE_vs_AUD_FINE": float(
                        batch_ious["IQR_FINE"][local_index]
                        - batch_ious["AUD_FINE"][local_index]
                    ),
                    "delta_IQR_FINE_EVALSPACE_vs_IQR_FINE": float(
                        batch_ious["IQR_FINE_EVALSPACE"][local_index]
                        - batch_ious["IQR_FINE"][local_index]
                    ),
                    "AUD_FINE_raw_reconstruction_max_abs_error": float(
                        raw_aud_errors[local_index]
                    ),
                    "IMG_L4_raw_reconstruction_max_abs_error": float(
                        raw_img_errors[local_index]
                    ),
                    "AUD_FINE_final_map_max_abs_error": float(
                        final_aud_errors[local_index]
                    ),
                    "IMG_L4_final_map_max_abs_error": float(
                        final_img_errors[local_index]
                    ),
                    "IQR_OLD_final_map_max_abs_error": float(
                        final_iqr_errors[local_index]
                    ),
                }
                per_sample.append(row)

    capture.close()
    refinement.close()

    metrics = {method: summarize(values) for method, values in ious.items()}
    formal_expected = reference["metrics"]
    reproduction_metric_errors = {
        "AUD_FINE": {
            metric: abs(metrics["AUD_FINE"][metric] - formal_expected["AUD"][metric])
            for metric in ("cIoU", "AUC")
        },
        "IMG_L4": {
            metric: abs(metrics["IMG_L4"][metric] - formal_expected["IMG_QUERY"][metric])
            for metric in ("cIoU", "AUC")
        },
        "IQR_OLD": {
            metric: abs(metrics["IQR_OLD"][metric] - formal_expected["IQR"][metric])
            for metric in ("cIoU", "AUC")
        },
    }
    reproduction_passed = all(
        error <= 1e-12
        for values in reproduction_metric_errors.values()
        for error in values.values()
    ) and all(error <= 1e-7 for error in max_errors.values())

    delta_old = np.asarray(ious["IQR_FINE"]) - np.asarray(ious["IQR_OLD"])
    delta_aud = np.asarray(ious["IQR_FINE"]) - np.asarray(ious["AUD_FINE"])
    per_sample_analysis = {
        "IQR_FINE_vs_IQR_OLD": classify_delta(delta_old),
        "IQR_FINE_vs_AUD_FINE": classify_delta(delta_aud),
    }

    checkpoint_state_after = file_state(checkpoint_path)
    base_state_after = file_state(base_path)
    checkpoint_unchanged = checkpoint_state_before == checkpoint_state_after
    base_checkpoint_unchanged = base_state_before == base_state_after
    parameters_with_grad = [
        name for name, parameter in refinement.named_parameters() if parameter.grad is not None
    ]
    trainable_parameters = sum(
        parameter.numel() for parameter in refinement.parameters() if parameter.requires_grad
    )
    audit = {
        "passed": bool(
            reproduction_passed
            and checkpoint_unchanged
            and base_checkpoint_unchanged
            and not parameters_with_grad
            and trainable_parameters == 0
            and nan_inf_count == 0
        ),
        "optimizer_created": False,
        "backward_called": False,
        "new_trainable_params": 0,
        "model_trainable_params_during_probe": int(trainable_parameters),
        "parameters_with_grad": parameters_with_grad,
        "checkpoint_before": checkpoint_state_before,
        "checkpoint_after": checkpoint_state_after,
        "checkpoint_unchanged": checkpoint_unchanged,
        "base_checkpoint_before": base_state_before,
        "base_checkpoint_after": base_state_after,
        "base_checkpoint_unchanged": base_checkpoint_unchanged,
        "nan_inf_count": int(nan_inf_count),
        "tensor_shapes_first_batch": shapes,
        "max_abs_errors_all_samples": max_errors,
        "formal_metric_reference": str(reference_path.resolve()),
        "formal_metric_absolute_errors": reproduction_metric_errors,
        "formal_reproduction_passed": reproduction_passed,
        "infer_sharpening": float(config.infer_sharpening),
        "alpha": ALPHA,
        "threshold": THRESHOLD,
    }
    if not audit["passed"]:
        (output_dir / "audit_failed.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Read-only/reproduction audit failed: {audit}")

    delta_metrics = {
        "IMG_FINE_minus_IMG_L4": {
            metric: metrics["IMG_FINE"][metric] - metrics["IMG_L4"][metric]
            for metric in ("cIoU", "AUC")
        },
        "IQR_FINE_minus_IQR_OLD": {
            metric: metrics["IQR_FINE"][metric] - metrics["IQR_OLD"][metric]
            for metric in ("cIoU", "AUC")
        },
        "IQR_FINE_minus_AUD_FINE": {
            metric: metrics["IQR_FINE"][metric] - metrics["AUD_FINE"][metric]
            for metric in ("cIoU", "AUC")
        },
        "IQR_FINE_EVALSPACE_minus_IQR_FINE": {
            metric: metrics["IQR_FINE_EVALSPACE"][metric]
            - metrics["IQR_FINE"][metric]
            for metric in ("cIoU", "AUC")
        },
    }
    result = {
        "experiment": "5.3_fine_iqr_q4_k34",
        "dataset_experiment": arguments.experiment,
        "formal_G_experiment": experiment_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "zero_training": True,
        "definitions": {
            "AUD_FINE": "target(Qa -> K34), native 14x14; unchanged formal G path",
            "IMG_L4": "target(Q4 -> K4), native 7x7; unchanged formal G IMG_QUERY",
            "IMG_FINE": "target(Q4 -> K34), native 14x14, using the same JSA _attention",
            "IQR_OLD": "norm(0.6*norm(resize(AUD_FINE))+0.4*norm(resize(IMG_L4)))",
            "IQR_FINE": "resize/eval_norm(native_norm(0.6*AUD_FINE+0.4*IMG_FINE))",
            "IQR_FINE_EVALSPACE": "norm(0.6*norm(resize(AUD_FINE))+0.4*norm(resize(IMG_FINE)))",
        },
        "metrics": metrics,
        "metric_deltas": delta_metrics,
        "per_sample_analysis": per_sample_analysis,
        "audit": audit,
    }
    write_csv(output_dir / "per_sample_iou.csv", per_sample)
    write_csv(
        output_dir / "metrics.csv",
        [
            {
                "dataset": arguments.experiment,
                "method": method,
                **metrics[method],
            }
            for method in METHODS
        ],
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\nMethod                    cIoU       AUC", flush=True)
    for method in METHODS:
        print(
            f"{method:<25} {metrics[method]['cIoU']:.6f}  {metrics[method]['AUC']:.6f}",
            flush=True,
        )
    print("\nDeltas:", json.dumps(delta_metrics, indent=2), flush=True)
    print("Per-sample:", json.dumps(per_sample_analysis, indent=2), flush=True)
    print("Audit passed:", audit["passed"], flush=True)
    print(f"Saved: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    run(parse_args())
