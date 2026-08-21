#!/usr/bin/env python3
"""Formal six-map evaluation plus true native-L3/L4 ownership diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
V11_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, V11_ROOT, HERE):
    sys.path.insert(0, str(path))

from dataset import get_test_dataset  # noqa: E402
from model_mufasa_jsa_v1_1 import MUFASAJSA11  # noqa: E402
from model_native_l3 import MUFASAJSA11NativeL3  # noqa: E402
import test_model  # noqa: E402


REGISTRY = {
    "vggss_10k": {"dataset": "vggss", "experiment": "1.1.1_14_14_L3_vggss_10k", "checkpoint": "vggss_best.pth", "v11": "mufasa_jsa_v1_1_vggss_10k", "batch": 256, "workers": 16},
    "flickr_10k": {"dataset": "flickr", "experiment": "1.1.1_14_14_L3_flickr_10k", "checkpoint": "flickr_best.pth", "v11": "mufasa_jsa_v1_1_flickr_10k_frame8_center5", "batch": 32, "workers": 12},
    "vggss_144k": {"dataset": "vggss", "experiment": "1.1.1_14_14_L3_vggss_144k", "checkpoint": "vggss_best.pth", "v11": "mufasa_jsa_v1_1_vggss_144k", "batch": 256, "workers": 16},
    "flickr_144k": {"dataset": "flickr", "experiment": "1.1.1_14_14_L3_flickr_144k", "checkpoint": "flickr_best.pth", "v11": "mufasa_jsa_v1_1_flickr_144k_frame8_center5", "batch": 32, "workers": 12},
}
FORMAL_METHODS = ("AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL")
PROBE_METHODS = (
    "SLOT_L3_POOLED_BASELINE",
    "SLOT_L3_NATIVE",
    "SLOT_L4",
    "V11_AUD",
    "V11_AUD_SLOT_L3_POOLED",
    "AUD_SLOT_L3_NATIVE",
    "AUD_SLOT_L4",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_prior_model():
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


def flatten_batch(image, spec, bboxes, names):
    if image.ndim == 3:
        image = image.unsqueeze(0)
        spec = spec.unsqueeze(0)
        bboxes = bboxes.unsqueeze(0)
    if image.ndim == 5:
        batch, clips, channels, height, width = image.shape
        image = image.reshape(batch * clips, channels, height, width)
        _, _, channels, frequency, time = spec.shape
        spec = spec.reshape(batch * clips, channels, frequency, time)
        _, _, channels, height, width = bboxes.shape
        bboxes = bboxes.reshape(batch * clips, channels, height, width).squeeze(1)
        names = [name for name in names for _ in range(clips)]
    return image, spec, bboxes, [str(name) for name in names]


def normalize_map(value):
    value = np.asarray(value)
    minimum = value.min()
    maximum = value.max()
    if maximum - minimum != 0:
        return (value - minimum) / (maximum - minimum)
    return value.copy()


def fuse(first, second, alpha=0.6):
    return normalize_map(alpha * first + (1.0 - alpha) * second)


def sample_iou(prediction, ground_truth):
    inferred = prediction >= 0.6
    intersection = np.sum(inferred * ground_truth)
    denominator = np.sum(ground_truth) + np.sum(inferred * (ground_truth == 0))
    return float(intersection / denominator)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    thresholds = np.arange(21, dtype=np.float64) * 0.05
    curve = [float(np.mean(array >= threshold)) for threshold in thresholds]
    return {
        "cIoU": float(np.mean(array >= 0.5)),
        "AUC": float(sklearn_metrics.auc(thresholds, curve)),
        "mean_sample_cIoU": float(array.mean()),
        "num_samples": int(array.size),
    }


def write_csv(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formal_metrics(values):
    return {
        name: {"cIoU": float(values[2 * index]), "AUC": float(values[2 * index + 1])}
        for index, name in enumerate(FORMAL_METHODS)
    }


def rescue_hurt(rows, candidate, reference="AUD"):
    aud_success = np.asarray([row[f"IoU_{reference}"] >= 0.5 for row in rows])
    candidate_success = np.asarray([row[f"IoU_{candidate}"] >= 0.5 for row in rows])
    aud_failure = ~aud_success
    rescue = aud_failure & candidate_success
    hurt = aud_success & ~candidate_success
    return {
        "candidate": candidate,
        "reference": reference,
        "definition": "candidate=normalize(0.6*reference+0.4*SLOT); success iff IoU>=0.5",
        "aud_failure_count": int(aud_failure.sum()),
        "rescue": int(rescue.sum()),
        "hurt": int(hurt.sum()),
        "net_rescue": int(rescue.sum() - hurt.sum()),
    }


def main():
    arguments = parse_args()
    registry = REGISTRY[arguments.experiment]
    experiment_dir = PROJECT_ROOT / "checkpoints" / registry["experiment"]
    checkpoint_path = experiment_dir / registry["checkpoint"]
    config_path = experiment_dir / "configs.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing formal experiment files in {experiment_dir}")
    checkpoint_hash_before = sha256(checkpoint_path)
    checkpoint_mtime_before = checkpoint_path.stat().st_mtime_ns

    config = argparse.Namespace(**json.loads(config_path.read_text(encoding="utf-8")))
    if config.architecture != "1.1.1_14_14_L3":
        raise RuntimeError(f"Unexpected architecture: {config.architecture}")
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    config.workers = registry["workers"]
    config.alpha = 0.6
    config.model_dir = str(PROJECT_ROOT / "checkpoints")
    config.experiment_name = registry["experiment"]

    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    model = MUFASAJSA11NativeL3(config).to(device).eval()
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    object_model = object_prior_model().to(device).eval()
    v11_dir = PROJECT_ROOT / "checkpoints" / registry["v11"]
    v11_checkpoint_path = v11_dir / registry["checkpoint"]
    v11_config = argparse.Namespace(**json.loads((v11_dir / "configs.json").read_text(encoding="utf-8")))
    v11_checkpoint = torch.load(v11_checkpoint_path, map_location="cpu", weights_only=False)
    v11_state = {
        key.removeprefix("module."): value for key, value in v11_checkpoint["model"].items()
    }
    v11_model = MUFASAJSA11(v11_config).to(device).eval()
    v11_model.load_state_dict(v11_state, strict=True)
    for parameter in v11_model.parameters():
        parameter.requires_grad = False

    dataset = get_test_dataset(config, registry["dataset"])
    loader = DataLoader(
        dataset,
        batch_size=registry["batch"],
        shuffle=False,
        num_workers=registry["workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=registry["workers"] > 0,
    )

    print("Stage 1/2: unchanged formal JSA evaluator", flush=True)
    values = test_model.validate_img_aud(
        loader,
        model,
        object_model,
        str(experiment_dir / "viz_formal"),
        registry["dataset"],
        -1,
        config,
    )
    formal = formal_metrics(values)

    print("Stage 2/2: native ownership and rescue/hurt", flush=True)
    all_ious = {name: [] for name in FORMAL_METHODS + PROBE_METHODS}
    rows = []
    audit = {
        "ownership_definition": "softmax(einsum(query,key)*512**-0.5, dim=slot=1), before eps/token normalization",
        "SLOT_L3_NATIVE_shape": [1, 14, 14],
        "SLOT_L3_POOLED_BASELINE_shape": [1, 7, 7],
        "SLOT_L4_shape": [1, 7, 7],
        "L3_final_logits_nonbatch_shape": [2, 196],
        "L4_final_logits_nonbatch_shape": [2, 49],
        "L3_ownership_slot_sum_max_error": 0.0,
        "L4_ownership_slot_sum_max_error": 0.0,
        "L3_pooled_baseline_ownership_slot_sum_max_error": 0.0,
    }
    with torch.inference_mode():
        for image, spec, bboxes, names, _labels in tqdm(loader, desc="Ownership", dynamic_ncols=True):
            image, spec, bboxes, names = flatten_batch(image, spec, bboxes, names)
            image = image.to(device, non_blocking=True).float()
            spec = spec.to(device, non_blocking=True).float()
            output = model.forward_eval_with_ownership(image, spec)
            obj = object_model(image)
            v11_levels = v11_model.imgnet(image)
            v11_audio_tokens = v11_model._audio_tokens(v11_model.audnet(spec))
            v11_encoded = v11_model.slot_attn._encode(v11_levels, v11_audio_tokens)
            v11_attentions = v11_model.slot_attn._l4_attentions(
                v11_encoded, scale_multiplier=v11_model.infer_sharpening
            )
            v11_l3_logits = torch.einsum(
                "bsd,bnd->bsn",
                v11_encoded["visual_queries"][1],
                v11_encoded["visual_keys"][1],
            ) * v11_model.slot_attn.scale
            if v11_l3_logits.shape[1:] != (2, 49):
                raise RuntimeError(f"Unexpected pooled baseline L3 logits: {v11_l3_logits.shape}")
            v11_l3_ownership = v11_l3_logits.softmax(dim=1)
            audit["L3_pooled_baseline_ownership_slot_sum_max_error"] = max(
                audit["L3_pooled_baseline_ownership_slot_sum_max_error"],
                float((v11_l3_ownership.sum(dim=1) - 1).abs().max()),
            )
            v11_l3 = v11_l3_ownership[:, 0].reshape(-1, 1, 7, 7)
            v11_aud = v11_attentions["audq_imgk_attn"][:, 0].reshape(-1, 1, 7, 7)
            audit["L3_ownership_slot_sum_max_error"] = max(
                audit["L3_ownership_slot_sum_max_error"],
                float((output["OWNERSHIP_L3"].sum(dim=1) - 1).abs().max()),
            )
            audit["L4_ownership_slot_sum_max_error"] = max(
                audit["L4_ownership_slot_sum_max_error"],
                float((output["OWNERSHIP_L4"].sum(dim=1) - 1).abs().max()),
            )
            tensors = {
                "AUD": output["AUD"],
                "IMG_QUERY": output["IMG_QUERY"],
                "OBJ_PRIOR": obj,
                "SLOT_L3_POOLED_BASELINE": v11_l3,
                "SLOT_L3_NATIVE": output["SLOT_L3_NATIVE"],
                "SLOT_L4": output["SLOT_L4"],
                "V11_AUD": v11_aud,
            }
            resized = {
                name: F.interpolate(value, (224, 224), mode="bicubic", align_corners=False)
                .cpu().numpy()[:, 0]
                for name, value in tensors.items()
            }
            gt = bboxes.cpu().numpy()
            for index, sample_name in enumerate(names):
                maps = {name: normalize_map(value[index]) for name, value in resized.items()}
                maps["IQR"] = fuse(maps["AUD"], maps["IMG_QUERY"])
                maps["OGL"] = fuse(maps["AUD"], maps["OBJ_PRIOR"])
                maps["EXTRA_IQR_OGL"] = normalize_map(
                    0.6 * maps["AUD"] + 0.2 * maps["IMG_QUERY"] + 0.2 * maps["OBJ_PRIOR"]
                )
                maps["AUD_SLOT_L3_NATIVE"] = fuse(maps["AUD"], maps["SLOT_L3_NATIVE"])
                maps["AUD_SLOT_L4"] = fuse(maps["AUD"], maps["SLOT_L4"])
                maps["V11_AUD_SLOT_L3_POOLED"] = fuse(
                    maps["V11_AUD"], maps["SLOT_L3_POOLED_BASELINE"]
                )
                row = {"sample_id": sample_name}
                for method in FORMAL_METHODS + PROBE_METHODS:
                    iou = sample_iou(maps[method], gt[index])
                    all_ious[method].append(iou)
                    row[f"IoU_{method}"] = iou
                rows.append(row)

    metrics = {method: summarize(all_ious[method]) for method in all_ious}
    formal_reproduction_errors = {
        method: {
            metric: abs(metrics[method][metric] - formal[method][metric])
            for metric in ("cIoU", "AUC")
        }
        for method in FORMAL_METHODS
    }
    if any(error > 1e-12 for values in formal_reproduction_errors.values() for error in values.values()):
        raise RuntimeError(f"Local maps do not reproduce formal evaluator: {formal_reproduction_errors}")
    if max(audit[key] for key in audit if key.endswith("sum_max_error")) > 1e-6:
        raise RuntimeError("Ownership failed slot-sum audit")

    rescue = [
        rescue_hurt(rows, "V11_AUD_SLOT_L3_POOLED", reference="V11_AUD"),
        rescue_hurt(rows, "AUD_SLOT_L3_NATIVE"),
        rescue_hurt(rows, "AUD_SLOT_L4"),
    ]
    write_csv(experiment_dir / "ownership_per_sample.csv", rows)
    write_csv(
        experiment_dir / "ownership_metrics.csv",
        [{"method": method, **values} for method, values in metrics.items()],
    )
    write_csv(experiment_dir / "ownership_rescue_hurt.csv", rescue)
    result = {
        "architecture": "1.1.1_14_14_L3",
        "experiment": registry["experiment"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "alpha": 0.6,
        "formal_evaluator": "root test_model.validate_img_aud",
        "formal_metrics": formal,
        "ownership_metrics": {method: metrics[method] for method in PROBE_METHODS},
        "rescue_hurt": rescue,
        "formal_reproduction_errors": formal_reproduction_errors,
        "ownership_audit": audit,
        "checkpoint_unchanged": checkpoint_hash_before == sha256(checkpoint_path)
        and checkpoint_mtime_before == checkpoint_path.stat().st_mtime_ns,
    }
    if not result["checkpoint_unchanged"]:
        raise RuntimeError("Formal checkpoint changed during read-only evaluation")
    (experiment_dir / "best_full_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
