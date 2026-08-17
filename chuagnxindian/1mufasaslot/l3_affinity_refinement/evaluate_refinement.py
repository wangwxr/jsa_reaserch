#!/usr/bin/env python3
"""Zero-training native-L3 spatial refinement for existing L3+L4 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
V11_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
ABLATION_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_l3_affinity_mplconfig")
for import_path in (PROJECT_ROOT, V11_ROOT, ABLATION_ROOT):
    sys.path.insert(0, str(import_path))

from dataset import get_test_dataset  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402
from protocol import ProtocolAccumulator, metric_key  # noqa: E402
from refinement import affinity_from_seed, spatial_normalize  # noqa: E402
from figures.gen_fig_qualitative import save_qualitative_panel  # noqa: E402


EXPERIMENTS = {
    "vggss_10k": {
        "dataset": "vggss",
        "split": "10k",
        "experiment_name": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
        "expected_aud": (0.4015, 0.4074),
        "formal_batch_size": 256,
    },
    "vggss_144k": {
        "dataset": "vggss",
        "split": "144k",
        "experiment_name": "mufasa_ablation2_l3_l4_ablation_vggss_144k",
        "expected_aud": (0.4002, 0.4127),
        "formal_batch_size": 256,
    },
    "flickr_10k": {
        "dataset": "flickr",
        "split": "10k",
        "experiment_name": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
        "expected_aud": (0.7640, 0.5916),
        "formal_batch_size": 32,
    },
    "flickr_144k": {
        "dataset": "flickr",
        "split": "144k",
        "experiment_name": "mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5",
        "expected_aud": (0.8040, 0.6228),
        "formal_batch_size": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tau-aff", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--num-qualitative", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    return parser.parse_args()


def setup_seed(seed: int = 12345) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(experiment_name: str) -> argparse.Namespace:
    path = PROJECT_ROOT / "checkpoints" / experiment_name / "configs.json"
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architecture") != "mufasa_ablation2_l3_l4_ablation":
        raise RuntimeError(
            f"Expected L3+L4 ablation config, found {config.get('architecture')!r}"
        )
    return argparse.Namespace(**config)


def resolve_checkpoint(
    experiment_name: str, testset: str, requested: Path | None
) -> Path:
    if requested is not None:
        candidate = requested if requested.is_absolute() else PROJECT_ROOT / requested
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()

    experiment_dir = PROJECT_ROOT / "checkpoints" / experiment_name
    best = experiment_dir / f"{testset}_best.pth"
    final = experiment_dir / "final.pth"
    if best.is_file():
        return best.resolve()
    if final.is_file():
        return final.resolve()
    raise FileNotFoundError(f"Neither {best} nor {final} exists")


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class NativeL3Hook:
    """Read proj3 output without changing the trained model's forward path."""

    def __init__(self, model: MUFASAL3L4):
        self.output: torch.Tensor | None = None
        self.handle = model.imgnet.proj3.register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self.output = output

    def pop(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("imgnet.proj3 hook did not observe a feature")
        output = self.output
        self.output = None
        return output

    def close(self) -> None:
        self.handle.remove()


def flatten_eval_batch(image, spec, bboxes, names):
    """Preserve the root evaluator's legacy 3D/5D batch handling."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
        spec = spec.unsqueeze(0)
        bboxes = bboxes.unsqueeze(0)
    if image.ndim == 5:
        batch_size, clips, channels, height, width = image.shape
        image = image.reshape(batch_size * clips, channels, height, width)
        _, _, channels, frequency, time = spec.shape
        spec = spec.reshape(batch_size * clips, channels, frequency, time)
        _, _, channels, height, width = bboxes.shape
        bboxes = bboxes.reshape(batch_size * clips, channels, height, width).squeeze(1)
        names = [name for name in names for _ in range(clips)]
    return image, spec, bboxes, [str(name) for name in names]


def make_metric_accumulators(
    tau_values: list[float], alpha_values: list[float], baseline_only: bool
) -> dict[tuple[str, float | None, float | None], ProtocolAccumulator]:
    accumulators = {metric_key("AUD_L4", None, None): ProtocolAccumulator()}
    if baseline_only:
        return accumulators
    for tau_aff in tau_values:
        accumulators[metric_key("L3_AFFINITY", tau_aff, None)] = ProtocolAccumulator()
        for alpha in alpha_values:
            accumulators[
                metric_key("L3_NATIVE_REFINED", tau_aff, alpha)
            ] = ProtocolAccumulator()
            accumulators[
                metric_key("L3_POOLED7_REFINED", tau_aff, alpha)
            ] = ProtocolAccumulator()
    return accumulators


def one_sample_iou(heatmap: torch.Tensor, gt_map: torch.Tensor, name: str) -> float:
    evaluator = ProtocolAccumulator()
    return float(evaluator.update(heatmap[None], gt_map[None], [name])[0])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "checkpoint",
        "method",
        "native_resolution",
        "tau_aff",
        "alpha",
        "cIoU",
        "AUC",
        "mean_sample_cIoU",
        "num_samples",
        "delta_cIoU_vs_AUD",
        "delta_AUC_vs_AUD",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(arguments: argparse.Namespace) -> Path:
    registry = EXPERIMENTS[arguments.experiment]
    config = load_config(registry["experiment_name"])
    config.gpu = arguments.gpu
    config.testset = registry["dataset"]
    batch_size = arguments.batch_size or registry["formal_batch_size"]
    workers = config.workers if arguments.workers is None else arguments.workers
    checkpoint_path = resolve_checkpoint(
        registry["experiment_name"], registry["dataset"], arguments.checkpoint
    )
    output_dir = arguments.output_root / arguments.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the formal checkpoint evaluation")
    device = torch.device("cuda", arguments.gpu)
    setup_seed(12345)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
    model = MUFASAL3L4(config)
    model.load_state_dict(state, strict=True)
    projection_keys = [
        "imgnet.proj3.weight",
        "imgnet.proj3.bias",
    ]
    projection_reused = all(
        torch.equal(model.state_dict()[key].cpu(), state[key].cpu())
        for key in projection_keys
    )
    if not projection_reused:
        raise RuntimeError("Loaded proj3 parameters do not exactly match the checkpoint")
    projection_sha = tensor_sha256(model.imgnet.proj3.weight)
    model.to(device).eval()
    native_hook = NativeL3Hook(model)

    dataset = get_test_dataset(config, registry["dataset"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    tau_values = sorted(set(float(value) for value in arguments.tau_aff))
    alpha_values = sorted(set(float(value) for value in arguments.alpha))
    accumulators = make_metric_accumulators(
        tau_values, alpha_values, arguments.baseline_only
    )

    qualitative_indices = set()
    if arguments.qualitative and not arguments.baseline_only:
        qualitative_indices = set(
            np.linspace(
                0, len(dataset) - 1, num=arguments.num_qualitative, dtype=int
            ).tolist()
        )
    qualitative_rows: list[dict[str, Any]] = []
    native_shapes: set[tuple[int, ...]] = set()
    pooled_shapes: set[tuple[int, ...]] = set()
    global_offset = 0

    progress = tqdm(dataloader, desc=arguments.experiment, dynamic_ncols=True)
    for image, spec, bboxes, names, _labels in progress:
        image, spec, bboxes, names = flatten_eval_batch(
            image, spec, bboxes, names
        )
        image = image.to(device, non_blocking=True).float()
        spec = spec.to(device, non_blocking=True).float()

        _img_attention, aud_l4 = model(image, spec)
        l3_native = native_hook.pop()
        native_shapes.add(tuple(l3_native.shape[1:]))
        if l3_native.shape[1:] != (512, 14, 14):
            raise RuntimeError(
                f"Expected projected native L3 [B,512,14,14], got {tuple(l3_native.shape)}"
            )

        aud_l4_seed = spatial_normalize(aud_l4)
        l3_pooled = model.imgnet._pool_to_7x7(l3_native)
        pooled_shapes.add(tuple(l3_pooled.shape[1:]))
        accumulators[metric_key("AUD_L4", None, None)].update(
            aud_l4, bboxes, names
        )

        if arguments.baseline_only:
            global_offset += len(names)
            continue

        a4_up = F.interpolate(
            aud_l4_seed,
            size=l3_native.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        a4_up = spatial_normalize(a4_up)
        default_maps: dict[str, torch.Tensor] = {"A4_UP": a4_up}

        for tau_aff in tau_values:
            native_affinity, _ = affinity_from_seed(a4_up, l3_native, tau_aff)
            pooled_affinity, _ = affinity_from_seed(
                aud_l4_seed, l3_pooled, tau_aff
            )
            accumulators[metric_key("L3_AFFINITY", tau_aff, None)].update(
                native_affinity, bboxes, names
            )
            for alpha in alpha_values:
                native_refined = spatial_normalize(
                    alpha * a4_up + (1.0 - alpha) * native_affinity
                )
                pooled_refined = spatial_normalize(
                    alpha * aud_l4_seed + (1.0 - alpha) * pooled_affinity
                )
                accumulators[
                    metric_key("L3_NATIVE_REFINED", tau_aff, alpha)
                ].update(native_refined, bboxes, names)
                accumulators[
                    metric_key("L3_POOLED7_REFINED", tau_aff, alpha)
                ].update(pooled_refined, bboxes, names)
                if tau_aff == 0.1 and alpha == 0.5:
                    default_maps = {
                        "A4_UP": a4_up,
                        "L3_AFFINITY": native_affinity,
                        "L3_NATIVE_REFINED": native_refined,
                        "L3_POOLED7_REFINED": pooled_refined,
                    }

        if qualitative_indices:
            for local_index, name in enumerate(names):
                dataset_index = global_offset + local_index
                if dataset_index not in qualitative_indices:
                    continue
                required = {
                    "A4_UP",
                    "L3_AFFINITY",
                    "L3_NATIVE_REFINED",
                    "L3_POOLED7_REFINED",
                }
                if set(default_maps) != required:
                    raise RuntimeError(
                        "Qualitative output requires tau=0.1 and alpha=0.5"
                    )
                aud_iou = one_sample_iou(aud_l4[local_index], bboxes[local_index], name)
                refined_iou = one_sample_iou(
                    default_maps["L3_NATIVE_REFINED"][local_index],
                    bboxes[local_index],
                    name,
                )
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                qualitative_dir = output_dir / "qualitative"
                save_qualitative_panel(
                    qualitative_dir / f"{dataset_index:05d}_{safe_name}",
                    name,
                    image[local_index],
                    bboxes[local_index],
                    aud_l4[local_index],
                    default_maps["A4_UP"][local_index],
                    default_maps["L3_AFFINITY"][local_index],
                    default_maps["L3_NATIVE_REFINED"][local_index],
                    default_maps["L3_POOLED7_REFINED"][local_index],
                    aud_iou,
                    refined_iou,
                )
                qualitative_rows.append(
                    {
                        "dataset": registry["dataset"],
                        "split": registry["split"],
                        "dataset_index": dataset_index,
                        "sample_id": name,
                        "selection_rule": "10 evenly spaced indices in sorted test split",
                        "AUD_L4_sample_IoU": aud_iou,
                        "L3_NATIVE_REFINED_sample_IoU": refined_iou,
                        "delta_sample_IoU": refined_iou - aud_iou,
                    }
                )
        global_offset += len(names)

    native_hook.close()
    finalized = {key: value.finalize() for key, value in accumulators.items()}
    baseline = finalized[metric_key("AUD_L4", None, None)]
    expected_ciou, expected_auc = registry["expected_aud"]
    sanity_passed = (
        f"{baseline['cIoU']:.4f}" == f"{expected_ciou:.4f}"
        and f"{baseline['AUC']:.4f}" == f"{expected_auc:.4f}"
    )

    audit = {
        "experiment": arguments.experiment,
        "dataset": registry["dataset"],
        "split": registry["split"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "checkpoint_selection_score": checkpoint.get("selection_score"),
        "proj3_checkpoint_weight_reused_exactly": projection_reused,
        "proj3_weight_sha256": projection_sha,
        "native_l3_shapes_without_batch": sorted(native_shapes),
        "pooled_l3_shapes_without_batch": sorted(pooled_shapes),
        "aud_l4_shape_without_batch": [1, 7, 7],
        "expected_AUD_L4": {"cIoU": expected_ciou, "AUC": expected_auc},
        "observed_AUD_L4": baseline,
        "baseline_sanity_passed_at_4_decimals": sanity_passed,
        "zero_training": True,
        "uses_gt_in_refinement": False,
        "uses_ogl_or_object_prior": False,
    }
    audit_path = output_dir / "audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    print(
        f"AUD_L4 sanity {arguments.experiment}: "
        f"{baseline['cIoU']:.4f}/{baseline['AUC']:.4f}; "
        f"expected {expected_ciou:.4f}/{expected_auc:.4f}; "
        f"pass={sanity_passed}"
    )
    print(f"Native L3 shapes: {sorted(native_shapes)}; proj3 reused={projection_reused}")
    if not sanity_passed:
        raise RuntimeError(
            "Formal AUD_L4 did not reproduce at four decimals; refusing to publish "
            "refinement results. See audit.json."
        )

    if arguments.baseline_only:
        return audit_path

    rows: list[dict[str, Any]] = []
    for (method, tau_aff, alpha), metric in finalized.items():
        resolution = "14x14" if method in {"L3_AFFINITY", "L3_NATIVE_REFINED"} else "7x7"
        rows.append(
            {
                "dataset": registry["dataset"],
                "split": registry["split"],
                "checkpoint": str(checkpoint_path),
                "method": method,
                "native_resolution": resolution,
                "tau_aff": "" if tau_aff is None else tau_aff,
                "alpha": "" if alpha is None else alpha,
                "cIoU": metric["cIoU"],
                "AUC": metric["AUC"],
                "mean_sample_cIoU": metric["mean_sample_cIoU"],
                "num_samples": metric["num_samples"],
                "delta_cIoU_vs_AUD": metric["cIoU"] - baseline["cIoU"],
                "delta_AUC_vs_AUD": metric["AUC"] - baseline["AUC"],
            }
        )
    result_path = output_dir / f"l3_affinity_refinement_results_{arguments.experiment}.csv"
    write_csv(result_path, rows)

    if qualitative_rows:
        qualitative_path = output_dir / "qualitative_sample_ids.csv"
        with qualitative_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qualitative_rows[0]))
            writer.writeheader()
            writer.writerows(qualitative_rows)
        print(f"Saved {len(qualitative_rows)} fixed qualitative samples to {qualitative_path}")

    default_native = finalized[metric_key("L3_NATIVE_REFINED", 0.1, 0.5)]
    default_pooled = finalized[metric_key("L3_POOLED7_REFINED", 0.1, 0.5)]
    print(
        f"Default native tau=0.1 alpha=0.5: "
        f"{default_native['cIoU']:.4f}/{default_native['AUC']:.4f}"
    )
    print(
        f"Default pooled tau=0.1 alpha=0.5: "
        f"{default_pooled['cIoU']:.4f}/{default_pooled['AUC']:.4f}"
    )
    print(f"Wrote {result_path}")
    return result_path


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
