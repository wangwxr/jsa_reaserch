#!/usr/bin/env python3
"""Extract frozen Stage-2 maps for the cross-scale CRAM audit.

This script is deliberately evaluation-only: it loads an already selected
Stage-1 teacher and already selected Stage-2 checkpoint, writes maps and
never calls an optimizer, a backward pass, or checkpoint save routine.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECIPE = ROOT / "chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine"
EXP81 = ROOT / "chuagnxindian/8.1_same_image_audio_counterfactual_audit"
EXP86 = ROOT / "chuagnxindian/8.6_loss_to_decision_causal_ablation"
EXP144 = HERE.parent
TENK = ROOT / "chuagnxindian/8.7_10k_mean_hard_cram"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attention(query: torch.Tensor, keys: torch.Tensor, sharpening: float) -> torch.Tensor:
    logits = torch.einsum("bksd,bpd->bksp", query, keys) * (query.shape[-1] ** -0.5) * sharpening
    value = logits.softmax(2) + 1e-8
    return value / value.sum(3, keepdim=True)


def paths(scale: str, dataset: str, method: str) -> tuple[Path, Path, str]:
    if scale == "144k":
        if method == "baseline":
            suffix = "vggss_144k_best" if dataset == "vggss" else "flickr_144k_frame8_center5_best"
            teacher_dir = "mufasa_ablation2_l3_l4_ablation_vggss_144k" if dataset == "vggss" else "mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5"
            teacher = ROOT / "checkpoints" / teacher_dir / f"{dataset}_best.pth"
            student = ROOT / "checkpoints" / f"1.3G-multigeom_equivariant_l3_refine_{suffix}" / f"{dataset}_best.pth"
        elif method == "mean":
            teacher = EXP144 / "checkpoints" / "C3" / dataset / "seed12345" / "selected_best.pth"
            student = EXP144 / "stage2" / "C3" / dataset / "seed12345" / f"{dataset}_best.pth"
        elif method == "hard_default":
            teacher = EXP144 / "8.7_hard_negative_cram" / "stage1" / dataset / "seed12345" / "selected_best.pth"
            student = EXP144 / "8.7_hard_negative_cram" / "stage2" / dataset / "seed12345" / f"{dataset}_best.pth"
        elif method == "hard_adjusted":
            variant = "vggss_tau_stronger_neff2" if dataset == "vggss" else "flickr_tau_weaker_neff3_08"
            root = EXP144 / "8.7_hard_negative_cram" / "tau_variants" / variant
            teacher = root / "stage1" / dataset / "seed12345" / "selected_best.pth"
            student = root / "stage2" / dataset / "seed12345" / f"{dataset}_best.pth"
        else:
            raise ValueError(method)
        experiment = f"{dataset}_144k"
    elif scale == "10k":
        if method == "baseline":
            suffix = "vggss_10k_best" if dataset == "vggss" else "flickr_10k_frame8_center5_best"
            teacher_dir = "mufasa_ablation2_l3_l4_ablation_vggss_10k" if dataset == "vggss" else "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5"
            teacher = ROOT / "checkpoints" / teacher_dir / f"{dataset}_best.pth"
            student = ROOT / "checkpoints" / f"1.3G-multigeom_equivariant_l3_refine_{suffix}" / f"{dataset}_best.pth"
        elif method in ("mean", "hard_default"):
            teacher = TENK / method / "stage1" / dataset / "seed12345" / "selected_best.pth"
            student = TENK / method / "stage2" / dataset / "seed12345" / f"{dataset}_best.pth"
        else:
            raise ValueError(f"{method} is not a 10k run")
        experiment = f"{dataset}_10k"
    else:
        raise ValueError(scale)
    if not teacher.is_file() or not student.is_file():
        raise FileNotFoundError((teacher, student))
    return teacher, student, experiment


def build(dataset: str, scale: str, method: str, device: torch.device):
    teacher, student, experiment = paths(scale, dataset, method)
    sys.path.insert(0, str(RECIPE))
    import common as C
    registry = copy.deepcopy(C.EXPERIMENTS[experiment])
    original = C.base_checkpoint_path
    C.base_checkpoint_path = lambda _registry: teacher.resolve()
    try:
        cfg = C.load_base_config(registry)
        cfg.workers = 8
        C.setup_seed(12345)
        model, loaded = C.build_model(cfg, registry, device)
    finally:
        C.base_checkpoint_path = original
    if loaded.resolve() != teacher.resolve():
        raise RuntimeError(f"wrong teacher {loaded} != {teacher}")
    checkpoint = torch.load(student, map_location="cpu", weights_only=False)
    model.student.proj3_spatial.load_state_dict(checkpoint["proj3_spatial_state_dict"], strict=True)
    model.student.adapter.load_state_dict(checkpoint["topdown_adapter_state_dict"], strict=True)
    model.eval()
    return C, model, teacher, student, checkpoint


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("10k", "144k"), required=True)
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--method", choices=("baseline", "mean", "hard_default", "hard_adjusted"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    out = HERE / "raw_maps" / args.scale / args.dataset / args.method
    out.mkdir(parents=True, exist_ok=True)
    target = out / "maps.npz"
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite {target}")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    C, model, teacher, student, checkpoint = build(args.dataset, args.scale, args.method, device)
    cache = load_module(f"cache_{args.scale}_{args.dataset}", EXP86 / "common.py")
    data = cache.test_dataset(args.dataset)
    ids = np.asarray([Path(x).stem for x in data.image_files]).astype(str)
    with np.load(EXP81 / f"results/{args.dataset}_144k_counterfactual_indices.npz") as z:
        if not np.array_equal(ids, z["anchor_ids"].astype(str)):
            raise RuntimeError("test IDs do not match frozen 8.1 assignments")
        wrong = z["set1_indices"][:, :8].astype(np.int64)
    batch_size = 128 if args.dataset == "vggss" else 32
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    n = len(ids)
    queries = np.empty((n, 2, 512), dtype=np.float32)
    offset = 0
    for _image, audio, _gt, names, _ in loader:
        q = model._extract_audio_query(audio.to(device, non_blocking=True).float())
        queries[offset:offset + len(names)] = q.cpu().numpy()
        offset += len(names)
    matched = np.empty((n, 196), dtype=np.float32)
    wrong_maps = np.empty((n, 8, 196), dtype=np.float32)
    image_maps = np.empty((n, 49), dtype=np.float32)
    replay_max_error = 0.0
    offset = 0
    for image, audio, _gt, names, _ in loader:
        batch = len(names)
        indices = np.arange(offset, offset + batch)
        image = image.to(device, non_blocking=True).float()
        q = torch.from_numpy(queries[indices]).to(device)
        view = model._extract_visual_teacher(image, q)
        fine = model._fine_from_teacher_features(view, q)
        branch = model.teacher.slot_attn.visual_branches[-1]
        tokens = model._to_tokens(fine["F34"])
        keys = branch.img_to_k(branch.img_norm_input(tokens))
        heatmap = attention(q[:, None], keys, model.teacher.infer_sharpening)[:, 0, 0]
        replay_max_error = max(replay_max_error, float((heatmap - fine["AUD_FINE"][:, 0].flatten(1)).abs().max()))
        matched[indices] = heatmap.cpu().numpy()
        q_wrong = torch.from_numpy(queries[wrong[indices]]).to(device)
        wrong_maps[indices] = attention(q_wrong, keys, model.teacher.infer_sharpening)[:, :, 0].cpu().numpy()
        image_native, _ = model.teacher.forward_eval(image, audio.to(device, non_blocking=True).float())
        image_maps[indices] = image_native[:, 0].flatten(1).cpu().numpy()
        offset += batch
        print(f"{args.scale}/{args.dataset}/{args.method}: {offset}/{n}", flush=True)
    np.savez_compressed(target, ids=ids, H_matched=matched, H_wrong=wrong_maps, wrong_indices=wrong, IMG_QUERY=image_maps)
    metadata = {
        "read_only_evaluation": True, "scale": args.scale, "dataset": args.dataset, "method": args.method,
        "seed": 12345, "teacher": str(teacher), "teacher_sha256": sha256(teacher),
        "student": str(student), "student_sha256": sha256(student), "student_epoch": int(checkpoint["epoch"]),
        "num_samples": n, "offline_replay_max_abs_error": replay_max_error,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model.close()


if __name__ == "__main__":
    main()
