#!/usr/bin/env python3
"""Representation, exact-path, parameter, and gradient smoke test for 3.1."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
V11_ROOT = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
BASELINE_ROOT = PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
for path in (PROJECT_ROOT, V11_ROOT, BASELINE_ROOT, HERE):
    sys.path.insert(0, str(path))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset import get_train_dataset  # noqa: E402
from model_l3_l4 import MUFASAL3L4  # noqa: E402

import common  # noqa: E402
from model import HierarchicalAudioStage1  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=sorted(common.SETTINGS))
    parser.add_argument("--gpu", required=True, type=int)
    return parser.parse_args()


def grad_norm(model, prefixes):
    squared = 0.0
    names = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            names.append(name)
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(squared), names


def run(setting: str, gpu: int):
    args = common.load_baseline_config(setting, gpu=gpu)
    args.batch_size = 32
    args.workers = min(args.workers, 4)
    torch.cuda.set_device(gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    baseline = MUFASAL3L4(args)
    hierarchical = HierarchicalAudioStage1(args)
    parameter_audit = common.parameter_audit(baseline, hierarchical)
    if not parameter_audit["passed"]:
        raise RuntimeError(parameter_audit)

    baseline_checkpoint = torch.load(
        common.baseline_dir(setting) / common.registry(setting)["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    baseline.load_state_dict(
        {key.removeprefix("module."): value for key, value in baseline_checkpoint["model"].items()},
        strict=True,
    )
    compatible = {
        key.removeprefix("module."): value
        for key, value in baseline_checkpoint["model"].items()
    }
    incompatible = hierarchical.load_state_dict(compatible, strict=False)
    expected_missing = set(parameter_audit["added_parameter_names"])
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            {
                "missing": incompatible.missing_keys,
                "expected_missing": sorted(expected_missing),
                "unexpected": incompatible.unexpected_keys,
            }
        )

    dataset = get_train_dataset(
        args,
        hard_img=args.hard_img,
        hard_aud=args.hard_aud,
        rand_aud=args.rand_aud,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    frame, spec, *_ = next(iter(loader))
    frame = frame.cuda(gpu, non_blocking=True).float()
    spec = spec.cuda(gpu, non_blocking=True).float()
    baseline = baseline.cuda(gpu).eval()
    hierarchical = hierarchical.cuda(gpu).eval()

    with torch.inference_mode():
        original_a4 = baseline.audnet(spec)
        multilevel = hierarchical.audnet(spec)
        exact_error = float((original_a4 - multilevel["a4_feature"]).abs().max())
        image_levels, audio = hierarchical.extract_features(frame, spec)
        representations = hierarchical.slot_attn.get_representations(
            image_levels, audio["a3_tokens"], audio["a4_tokens"]
        )
        fused_init_error = float(
            (
                representations["fused_audio_slots"]
                - representations["audio_slots_a4"]
            ).abs().max()
        )
    if exact_error > 1e-7 or fused_init_error != 0.0:
        raise RuntimeError(
            {"a4_exact_error": exact_error, "fused_init_error": fused_init_error}
        )

    hierarchical.train()
    optimizer = torch.optim.Adam(
        hierarchical.parameters(),
        lr=args.init_lr,
        weight_decay=args.weight_decay,
    )
    detailed = hierarchical.forward_train_detailed(frame, spec)
    info_loss, recon_loss, div_loss, att_loss = detailed["losses"]
    total_loss = (
        info_loss
        + args.lam1 * recon_loss
        + args.lam2 * div_loss
        + args.lam3 * att_loss
    )
    total_loss.backward()
    first_gradients = {}
    prefixes = {
        "aud_proj3": ("audnet.aud_proj3.",),
        "audio_branch_a3": ("slot_attn.audio_branch_a3.",),
        "audio_hierarchical_fusion": (
            "slot_attn.audio_hierarchical_fusion.",
        ),
        "audio_branch_a4": ("slot_attn.audio_branch.",),
    }
    for name, prefix in prefixes.items():
        first_gradients[name] = grad_norm(hierarchical, prefix)[0]
    optimizer.step()
    optimizer.zero_grad()

    second = hierarchical.forward_train_detailed(frame, spec)
    second_losses = second["losses"]
    second_total = (
        second_losses[0]
        + args.lam1 * second_losses[1]
        + args.lam2 * second_losses[2]
        + args.lam3 * second_losses[3]
    )
    second_total.backward()
    second_gradients = {
        name: grad_norm(hierarchical, prefix)[0]
        for name, prefix in prefixes.items()
    }

    tensors = [
        multilevel["raw_a3"],
        multilevel["raw_a4"],
        audio["a3_tokens"],
        audio["a4_tokens"],
        representations["audio_slots_a3"],
        representations["audio_slots_a4"],
        representations["fused_audio_slots"],
        representations["audio_query_a3"],
        representations["audio_query_a4"],
        representations["fused_visual_slots"],
        *detailed["losses"],
        *second_losses,
    ]
    all_finite = all(torch.isfinite(tensor).all().item() for tensor in tensors)
    gradients_finite = all(
        math.isfinite(value)
        for value in (*first_gradients.values(), *second_gradients.values())
    )
    passed = (
        all_finite
        and gradients_finite
        and exact_error <= 1e-7
        and fused_init_error == 0.0
        and first_gradients["audio_hierarchical_fusion"] > 0
        and first_gradients["audio_branch_a4"] > 0
        and second_gradients["aud_proj3"] > 0
        and second_gradients["audio_branch_a3"] > 0
        and hierarchical.eval_audio_query_source == "A4"
    )
    report = {
        "experiment": "3.1 Hierarchical Audio Representation",
        "setting": setting,
        "raw_A3_shape": list(multilevel["raw_a3"].shape),
        "raw_A4_shape": list(multilevel["raw_a4"].shape),
        "A3_tokens_shape": list(audio["a3_tokens"].shape),
        "A4_tokens_shape": list(audio["a4_tokens"].shape),
        "T3": audio["a3_tokens"].shape[1],
        "T4": audio["a4_tokens"].shape[1],
        "T3_T4_ratio": audio["a3_tokens"].shape[1] / audio["a4_tokens"].shape[1],
        "A3_slots_shape": list(representations["audio_slots_a3"].shape),
        "A4_slots_shape": list(representations["audio_slots_a4"].shape),
        "fused_audio_slots_shape": list(
            representations["fused_audio_slots"].shape
        ),
        "A3_query_shape": list(representations["audio_query_a3"].shape),
        "A4_query_shape": list(representations["audio_query_a4"].shape),
        "visual_fused_slots_shape": list(
            representations["fused_visual_slots"].shape
        ),
        "A4_exact_reproduction_max_error": exact_error,
        "fusion_initial_A4_reproduction_max_error": fused_init_error,
        "eval_audio_query_source": hierarchical.eval_audio_query_source,
        "parameter_audit": parameter_audit,
        "first_batch_losses": {
            "info_loss": float(info_loss.detach()),
            "recon_loss": float(recon_loss.detach()),
            "div_loss": float(div_loss.detach()),
            "att_loss": float(att_loss.detach()),
            "total_loss": float(total_loss.detach()),
        },
        "first_backward_gradient_norms": first_gradients,
        "second_backward_gradient_norms": second_gradients,
        "zero_init_expected_first_step_block": (
            "aud_proj3/audio_branch_a3 receive zero gradient before the zero-initialized "
            "fusion output layer takes its first update; both are nonzero on step two"
        ),
        "temporal_diagnostics": {
            key: float(value.detach())
            for key, value in detailed["diagnostics"].items()
            if key.startswith(("a3_", "a4_"))
        },
        "no_nan_or_inf": all_finite and gradients_finite,
        "passed": passed,
    }
    output = common.result_dir(setting) / "smoke.json"
    common.write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError(report)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.setting, arguments.gpu)
