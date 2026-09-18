#!/usr/bin/env python3
"""Run the unchanged 1.3G recipe with the user-authorized CRAM C3 teacher."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
RECIPE = ROOT / "chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_best(common, protocol, registry, output, gpu):
    config = common.load_base_config(registry)
    config.workers = registry["workers"]
    common.setup_seed(config.seed)
    device = torch.device("cuda", gpu)
    model, _ = common.build_model(config, registry, device)
    checkpoint_path = output / f"{registry['dataset']}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.student.proj3_spatial.load_state_dict(checkpoint["proj3_spatial_state_dict"], strict=True)
    model.student.adapter.load_state_dict(checkpoint["topdown_adapter_state_dict"], strict=True)
    model.eval()
    _, dataset = common.build_datasets(config, registry)
    loader = common.build_test_loader(dataset, config, registry)
    accum = {name: protocol.ProtocolAccumulator() for name in ("AUD", "IQR")}
    ids = []

    def normalize_resize(value):
        value = F.interpolate(value, (224, 224), mode="bicubic", align_corners=False)
        low = value.flatten(1).min(1).values[:, None, None, None]
        high = value.flatten(1).max(1).values[:, None, None, None]
        return (value - low) / (high - low).clamp_min(1e-12)

    for image, audio, gt, names, _ in loader:
        image, audio, gt, names = common.flatten_eval_batch(image, audio, gt, names)
        image, audio = image.to(device).float(), audio.to(device).float()
        fine = model(image, audio)["AUD_FINE"]
        img, _ = model.teacher.forward_eval(image, audio)
        iqr = 0.6 * normalize_resize(fine) + 0.4 * normalize_resize(img)
        accum["AUD"].update(fine, gt, names)
        accum["IQR"].update(iqr, gt, names)
        ids.extend(names)
    metrics = {name: value.finalize() for name, value in accum.items()}
    result = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
              "epoch": checkpoint["epoch"], "metrics": metrics,
              "teacher_sha256": sha256(Path(registry["base_checkpoint"]))}
    (output / "best_aud_iqr_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame({"sample_id": ids, **{f"{name}_iou": value.sample_ious
                 for name, value in accum.items()}}).to_csv(output / "best_per_sample.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    model.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = HERE / "C3" / args.dataset / "seed12345"
    if (output / "launch_manifest.json").exists():
        raise RuntimeError(f"Refusing duplicate launch: {output}")
    teacher = EXP / "checkpoints/C3" / args.dataset / "seed12345/selected_best.pth"
    config_path = teacher.parent / "config.json"
    expected_hash = json.loads(config_path.read_text())["selected_checkpoint"]["sha256"]
    if sha256(teacher) != expected_hash:
        raise RuntimeError("Selected C3 teacher hash changed")

    # Keep all architecture, loss, optimizer, augmentation and selection code
    # in the existing recipe. Only the teacher pointer and output path change.
    sys.path.insert(0, str(RECIPE))
    import common
    trainer = load_module("_cram_original_stage2_train", RECIPE / "train.py")
    audit_common = load_module("_cram_stage2_cache", EXP.parent / "8.6_loss_to_decision_causal_ablation/common.py")
    experiment = f"{args.dataset}_144k"
    registry = common.EXPERIMENTS[experiment]
    registry["base_checkpoint"] = str(teacher)
    per_sample = pd.read_csv(EXP / "natural_localization" / args.dataset / "C3/seed12345/natural_per_sample.csv")
    ious = per_sample.loc[per_sample["map"].eq("AUD"), "iou"].to_numpy()
    thresholds = np.arange(21) * 0.05
    registry["expected_aud"] = (float(np.mean(ious >= 0.5)),
        float(np.trapezoid([np.mean(ious >= t) for t in thresholds], thresholds)))

    def datasets(config, selected_registry):
        train = common.get_train_dataset(config, hard_img=config.hard_img,
                                        hard_aud=config.hard_aud, rand_aud=config.rand_aud)
        return train, audit_common.test_dataset(selected_registry["dataset"])

    # Byte-verified cached spectrograms retain exact evaluation sample order.
    common.build_datasets = datasets
    trainer.build_datasets = datasets
    config = common.load_base_config(registry)
    assert config.seed == 12345 and config.batch_size == 256 and config.epochs == 50
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"stage2_allowed": True, "authorization": "User: 启动stage2 来吧 你可以用两张卡",
                "stage1_mechanism_decision": "CRAM-NO-GO",
                "purpose": "Explicitly authorized performance follow-up, not a passed mechanism gate",
                "dataset": args.dataset, "gpu": args.gpu, "group": "C3", "seed": 12345,
                "teacher": str(teacher), "teacher_sha256": expected_hash,
                "epochs": 50, "batch_size": 256, "expected_teacher_aud": registry["expected_aud"],
                "source_hashes": {p.name: sha256(p) for p in RECIPE.glob("*.py")},
                "base_config_sha256": sha256(ROOT / "checkpoints" / registry["base_experiment"] / "configs.json"),
                "runner_sha256": sha256(Path(__file__)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "selection": "original AUD_FINE cIoU best; report cIoU and AUC from that same checkpoint"}
    (output / "launch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    sys.argv = [str(RECIPE / "train.py"), "--experiment", experiment,
                "--gpu", str(args.gpu), "--epochs", "50", "--model-dir", str(output.parent),
                "--experiment-name", output.name]
    trainer.main()
    if sha256(teacher) != expected_hash:
        raise RuntimeError("Teacher file changed during Stage 2")
    import protocol
    evaluate_best(common, protocol, registry, output, args.gpu)


if __name__ == "__main__":
    main()
