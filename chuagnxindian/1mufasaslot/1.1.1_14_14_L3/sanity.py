#!/usr/bin/env python3
"""Zero-training shape and L4 path-consistency audit for native L3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
V11_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
for path in (PROJECT_ROOT, V11_ROOT, HERE):
    sys.path.insert(0, str(path))

from model_mufasa_jsa_v1_1 import MUFASAJSA11  # noqa: E402
from model_native_l3 import MUFASAJSA11NativeL3  # noqa: E402


REGISTRY = {
    "vggss_10k": (
        "mufasa_jsa_v1_1_vggss_10k",
        "vggss_best.pth",
        "1.1.1_14_14_L3_vggss_10k",
    ),
    "flickr_10k": (
        "mufasa_jsa_v1_1_flickr_10k_frame8_center5",
        "flickr_best.pth",
        "1.1.1_14_14_L3_flickr_10k",
    ),
    "vggss_144k": (
        "mufasa_jsa_v1_1_vggss_144k",
        "vggss_best.pth",
        "1.1.1_14_14_L3_vggss_144k",
    ),
    "flickr_144k": (
        "mufasa_jsa_v1_1_flickr_144k_frame8_center5",
        "flickr_best.pth",
        "1.1.1_14_14_L3_flickr_144k",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def load_config(experiment: str):
    base_name, checkpoint_name, output_name = REGISTRY[experiment]
    config_path = PROJECT_ROOT / "checkpoints" / base_name / "configs.json"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / base_name / checkpoint_name
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing formal v1.1 source: {config_path}, {checkpoint_path}")
    config = argparse.Namespace(**json.loads(config_path.read_text(encoding="utf-8")))
    return config, checkpoint_path.resolve(), output_name


def clean_state(checkpoint):
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }


def shape_list(tensor):
    return list(tensor.shape)


def main():
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal sanity audit")
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    config, checkpoint_path, output_name = load_config(arguments.experiment)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = clean_state(checkpoint)

    torch.manual_seed(config.seed)
    old_model = MUFASAJSA11(config).to(device).eval()
    torch.manual_seed(config.seed)
    new_model = MUFASAJSA11NativeL3(config).to(device).eval()
    initialization_max_error = max(
        float((old_value - new_value).abs().max())
        for old_value, new_value in zip(
            old_model.state_dict().values(), new_model.state_dict().values()
        )
    )
    old_model.load_state_dict(state, strict=True)
    new_model.load_state_dict(state, strict=True)
    torch.manual_seed(config.seed + 1114)
    image = torch.randn(2, 3, 224, 224, device=device)
    audio = torch.randn(2, 1, 257, 501, device=device)

    with torch.inference_mode():
        old_img, old_aud = old_model.forward_eval(image, audio)
    with torch.inference_mode():
        output = new_model.forward_eval_with_ownership(image, audio)

    image_levels = output["IMAGE_LEVELS"]
    visual_slots = output["VISUAL_SLOTS"]
    new_img = output["IMG_QUERY"]
    new_aud = output["AUD"]
    l4_differences = torch.cat(
        ((new_img - old_img).flatten(), (new_aud - old_aud).flatten())
    ).abs()
    audit = {
        "experiment": arguments.experiment,
        "source_v1_1_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "zero_training": True,
        "same_seed_initialization_max_abs_diff_vs_v1_1": initialization_max_error,
        "image_token_shapes": {
            "L2": shape_list(image_levels[0]),
            "L3": shape_list(image_levels[1]),
            "L4": shape_list(image_levels[2]),
        },
        "visual_slot_shapes": {
            "L2": shape_list(visual_slots[0]),
            "L3": shape_list(visual_slots[1]),
            "L4": shape_list(visual_slots[2]),
        },
        "L3_final_slot_logits": shape_list(output["LOGITS_L3"]),
        "L3_native_ownership": shape_list(output["SLOT_L3_NATIVE"]),
        "L4_final_slot_logits": shape_list(output["LOGITS_L4"]),
        "L4_ownership": shape_list(output["SLOT_L4"]),
        "L3_ownership_slot_sum_max_error": float(
            (output["OWNERSHIP_L3"].sum(dim=1) - 1).abs().max()
        ),
        "L4_ownership_slot_sum_max_error": float(
            (output["OWNERSHIP_L4"].sum(dim=1) - 1).abs().max()
        ),
        "max_abs_diff_L4_attention": float(l4_differences.max()),
        "mean_abs_diff_L4_attention": float(l4_differences.mean()),
        "L4_comparison_tensors": ["IMG_QUERY_Q4_to_K4", "AUD_Qa_to_K4"],
        "passed": False,
    }
    expected = {
        "image_token_shapes": {"L2": [2, 49, 512], "L3": [2, 196, 512], "L4": [2, 49, 512]},
        "visual_slot_shapes": {"L2": [2, 2, 512], "L3": [2, 2, 512], "L4": [2, 2, 512]},
        "L3_final_slot_logits": [2, 2, 196],
        "L3_native_ownership": [2, 1, 14, 14],
        "L4_final_slot_logits": [2, 2, 49],
        "L4_ownership": [2, 1, 7, 7],
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(f"Shape sanity failed for {key}: {audit[key]} != {value}")
    if audit["max_abs_diff_L4_attention"] > 1e-7:
        raise RuntimeError(f"L4 path changed: {audit['max_abs_diff_L4_attention']}")
    if audit["same_seed_initialization_max_abs_diff_vs_v1_1"] != 0:
        raise RuntimeError("Formal initialization no longer matches v1.1")
    if max(
        audit["L3_ownership_slot_sum_max_error"],
        audit["L4_ownership_slot_sum_max_error"],
    ) > 1e-6:
        raise RuntimeError("Ownership does not sum to one across slots")
    audit["passed"] = True

    output_dir = PROJECT_ROOT / "checkpoints" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "sanity_zero_training.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Sanity audit saved: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
