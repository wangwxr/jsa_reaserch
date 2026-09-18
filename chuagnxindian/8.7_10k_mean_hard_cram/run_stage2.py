#!/usr/bin/env python3
"""Run unchanged original 1.3G 10k Stage-2 from an isolated 8.7 teacher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECIPE = ROOT / "chuagnxindian" / "1mufasaslot" / "1.3G-multigeom_equivariant_l3_refine"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mean", "hard_default"), required=True)
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    teacher = HERE / args.method / "stage1" / args.dataset / "seed12345" / "selected_best.pth"
    final_teacher = teacher.with_name("final.pth")
    output = HERE / args.method / "stage2" / args.dataset / "seed12345"
    protected = ("latest.pth", "final.pth", f"{args.dataset}_best.pth", "epoch_metrics.csv")
    if not final_teacher.is_file() or not teacher.is_file():
        raise RuntimeError(f"Stage-1 is incomplete for {args.method}/{args.dataset}")
    if any((output / name).exists() for name in protected):
        raise RuntimeError(f"Refusing to overwrite {output}")
    source_checkpoint = torch.load(teacher, map_location="cpu", weights_only=False)
    metrics = source_checkpoint.get("metrics", {})
    if not {"AUD_cIoU", "AUD_AUC"}.issubset(metrics):
        raise RuntimeError("Selected Stage-1 checkpoint lacks official AUD metrics")
    sys.path.insert(0, str(RECIPE))
    import common
    trainer = load_module("_cram10k_vanilla_stage2", RECIPE / "train.py")
    experiment = f"{args.dataset}_10k"
    registry = copy.deepcopy(common.EXPERIMENTS[experiment])
    registry["expected_aud"] = (float(metrics["AUD_cIoU"]), float(metrics["AUD_AUC"]))
    common.EXPERIMENTS[experiment] = registry
    common.base_checkpoint_path = lambda _registry: teacher.resolve()
    config = common.load_base_config(registry)
    if not (config.seed == 12345 and config.batch_size == 256 and config.epochs == 100):
        raise RuntimeError("Unexpected original 1.3G 10k recipe")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": args.method, "dataset": args.dataset, "seed": 12345, "gpu": args.gpu,
        "teacher": str(teacher), "teacher_sha256": sha256(teacher),
        "stage1_epoch": int(source_checkpoint["epoch"]),
        "stage1_expected_aud": registry["expected_aud"],
        "stage2": "unchanged original 1.3G refinement", "stage2_epochs": 100,
        "batch_size": 256, "experiment_key": experiment,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (output / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    sys.argv = [str(RECIPE / "train.py"), "--experiment", experiment, "--gpu", str(args.gpu),
                "--epochs", "100", "--model-dir", str(output.parent),
                "--experiment-name", output.name]
    trainer.main()
    if sha256(teacher) != manifest["teacher_sha256"]:
        raise RuntimeError("Stage-1 teacher changed during Stage-2")


if __name__ == "__main__":
    main()
