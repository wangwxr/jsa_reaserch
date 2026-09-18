#!/usr/bin/env python3
"""Extract matched and fixed wrong-audio maps from a frozen Stage-2 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
EXP87 = AUDIT.parent
ROOT = EXP87.parents[1]
RECIPE = ROOT / "chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine"
EXP81 = ROOT / "chuagnxindian/8.1_same_image_audio_counterfactual_audit"
EXP86 = ROOT / "chuagnxindian/8.6_loss_to_decision_causal_ablation"
MODELS = ("original_1.3g_final", "cram_c3_stage2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path: Path) -> dict:
    path = path.resolve()
    stat = path.stat()
    return {"path": str(path), "sha256": sha256(path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_paths(dataset: str, model_name: str) -> tuple[Path, Path]:
    if dataset == "vggss":
        original_teacher = ROOT / "checkpoints/mufasa_ablation2_l3_l4_ablation_vggss_144k/vggss_best.pth"
        original_final = ROOT / "checkpoints/1.3G-multigeom_equivariant_l3_refine_vggss_144k_best/vggss_best.pth"
        final_name = "vggss_best.pth"
    else:
        original_teacher = ROOT / "checkpoints/mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5/flickr_best.pth"
        original_final = ROOT / "checkpoints/1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5_best/flickr_best.pth"
        final_name = "flickr_best.pth"
    if model_name == "original_1.3g_final":
        return original_teacher, original_final
    return (
        EXP87 / f"checkpoints/C3/{dataset}/seed12345/selected_best.pth",
        EXP87 / f"stage2/C3/{dataset}/seed12345/{final_name}",
    )


def query_inputs(dataset: str, model_name: str) -> tuple[Path, Path]:
    experiment = f"{dataset}_144k"
    if model_name == "original_1.3g_final":
        return (
            EXP81 / f"features/{experiment}_ids.npy",
            EXP81 / f"features/{experiment}_audio_queries.npy",
        )
    natural = EXP87 / f"natural_localization/{dataset}/C3/seed12345"
    return natural / "natural_maps.npz", natural / "audio_queries.npy"


def load_ids(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return archive["ids"].astype(str)
    return np.load(path, allow_pickle=False).astype(str)


def video_id(sample_id: str, dataset: str) -> str:
    return sample_id[:11] if dataset == "vggss" else sample_id


def offline_attention(queries: torch.Tensor, keys: torch.Tensor, sharpening: float) -> torch.Tensor:
    scale = queries.shape[-1] ** -0.5
    dots = torch.einsum("bksd,bpd->bksp", queries, keys) * sharpening * scale
    attention = dots.softmax(dim=2) + 1e-8
    return attention / attention.sum(dim=3, keepdim=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vggss", "flickr"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = AUDIT / "results" / args.dataset / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}_maps.npz"
    audit_path = output_dir / f"{args.model}_extraction_audit.json"
    if output_path.exists() or audit_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit output: {output_path}")

    sys.path.insert(0, str(RECIPE))
    import common as recipe_common

    audit_common = load_module("mechanism_audit_exp86_common", EXP86 / "common.py")
    registry = copy.deepcopy(recipe_common.EXPERIMENTS[f"{args.dataset}_144k"])
    teacher_path, final_path = model_paths(args.dataset, args.model)
    registry["base_checkpoint"] = str(teacher_path)
    config = recipe_common.load_base_config(registry)
    config.workers = args.workers
    recipe_common.setup_seed(12345)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, loaded_teacher = recipe_common.build_model(config, registry, device)
    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    model.student.proj3_spatial.load_state_dict(
        checkpoint["proj3_spatial_state_dict"], strict=True
    )
    model.student.adapter.load_state_dict(
        checkpoint["topdown_adapter_state_dict"], strict=True
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.teacher.parameters()):
        raise RuntimeError("Teacher is not frozen")

    dataset = audit_common.test_dataset(args.dataset)
    all_ids = np.asarray([Path(value).stem for value in dataset.image_files]).astype(str)
    query_id_path, query_path = query_inputs(args.dataset, args.model)
    query_ids = load_ids(query_id_path)
    if not np.array_equal(all_ids, query_ids):
        raise RuntimeError("Audio-query sample order differs from evaluation dataset")
    queries = np.load(query_path, mmap_mode="r")
    if len(queries) != len(dataset):
        raise RuntimeError("Audio-query count differs from evaluation dataset")

    index_path = EXP81 / f"results/{args.dataset}_144k_counterfactual_indices.npz"
    with np.load(index_path, allow_pickle=False) as archive:
        if not np.array_equal(all_ids, archive["anchor_ids"].astype(str)):
            raise RuntimeError("Counterfactual anchor order differs from evaluation dataset")
        if not np.array_equal(all_ids, archive["audio_ids"].astype(str)):
            raise RuntimeError("Counterfactual audio order differs from evaluation dataset")
        wrong = archive["set1_indices"][:, :8].astype(np.int64)
    overlap = sum(
        video_id(all_ids[i], args.dataset) == video_id(all_ids[j], args.dataset)
        for i in range(len(all_ids)) for j in wrong[i]
    )
    if overlap:
        raise RuntimeError(f"Found {overlap} same-video counterfactual pairs")

    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    selected_dataset = dataset if count == len(dataset) else Subset(dataset, range(count))
    batch_size = args.batch_size or (128 if args.dataset == "vggss" else 32)
    loader = DataLoader(
        selected_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False, persistent_workers=args.workers > 0,
    )
    matched_maps = np.empty((count, 196), dtype=np.float32)
    wrong_maps = np.empty((count, 8, 196), dtype=np.float32)
    observed_ids: list[str] = []
    replay_error = 0.0
    query_difference_min = float("inf")
    position = 0
    for image, audio, gt, names, _labels in tqdm(loader, desc=f"{args.dataset} {args.model}"):
        image, audio, gt, names = recipe_common.flatten_eval_batch(image, audio, gt, names)
        batch_count = len(names)
        if position + batch_count > count:
            keep = count - position
            image, names = image[:keep], names[:keep]
            batch_count = keep
        indices = np.arange(position, position + batch_count)
        image = image.to(device, non_blocking=True).float()
        matched_query = torch.from_numpy(np.asarray(queries[indices])).to(device)
        wrong_query = torch.from_numpy(np.asarray(queries[wrong[indices]])).to(device)
        difference = (wrong_query - matched_query[:, None]).abs().flatten(2).max(2).values
        query_difference_min = min(query_difference_min, float(difference.min()))

        teacher_view = model._extract_visual_teacher(image, matched_query)
        fine = model._fine_from_teacher_features(teacher_view, matched_query)
        branch = model.teacher.slot_attn.visual_branches[-1]
        fine_tokens = model._to_tokens(fine["F34"])
        fine_keys = branch.img_to_k(branch.img_norm_input(fine_tokens))
        replay = offline_attention(matched_query[:, None], fine_keys,
                                   model.teacher.infer_sharpening)[:, 0, 0]
        replay_error = max(
            replay_error,
            float((replay - fine["AUD_FINE"][:, 0].flatten(1)).abs().max()),
        )
        wrong_attention = offline_attention(
            wrong_query, fine_keys, model.teacher.infer_sharpening
        )[:, :, 0]
        matched_maps[indices] = replay.cpu().numpy()
        wrong_maps[indices] = wrong_attention.cpu().numpy()
        observed_ids.extend(str(value) for value in names)
        position += batch_count

    if position != count or not np.array_equal(np.asarray(observed_ids), all_ids[:count]):
        raise RuntimeError("Extracted sample order mismatch")
    if query_difference_min <= 0:
        raise RuntimeError("Matched and wrong audio queries are identical")
    if replay_error > 1e-6:
        raise RuntimeError(f"Stage-2 matched replay error: {replay_error}")
    np.savez_compressed(
        output_path, ids=all_ids[:count], H_matched=matched_maps,
        H_wrong=wrong_maps, wrong_indices=wrong[:count], wrong_ids=all_ids[wrong[:count]],
    )
    manifest = {
        "dataset": args.dataset,
        "model": args.model,
        "tag": args.tag,
        "sample_count": count,
        "seed": 12345,
        "teacher_checkpoint": snapshot(teacher_path),
        "loaded_teacher_path": str(Path(loaded_teacher).resolve()),
        "stage2_checkpoint": snapshot(final_path),
        "audio_query_file": snapshot(query_path),
        "audio_query_id_file": snapshot(query_id_path),
        "counterfactual_index_file": snapshot(index_path),
        "wrong_audio_count": 8,
        "same_video_overlap": overlap,
        "minimum_matched_wrong_query_max_abs_difference": query_difference_min,
        "matched_forward_replay_max_abs_error": replay_error,
        "matched_probability_sum_max_error": float(np.max(np.abs(matched_maps.sum(1) - 1))),
        "wrong_probability_sum_max_error": float(np.max(np.abs(wrong_maps.sum(2) - 1))),
        "all_parameters_eval": not any(module.training for module in model.modules()),
        "optimizer_created": False,
        "backward_called": False,
        "output": snapshot(output_path),
        "elapsed_seconds": time.time() - started,
    }
    audit_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    model.close()
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
