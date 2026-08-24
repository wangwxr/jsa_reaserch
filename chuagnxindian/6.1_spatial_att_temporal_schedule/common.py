"""Shared configuration and reproducibility helpers for Experiment 6.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENT_ROOT / "results"

REFERENCE_EXPERIMENTS = {
    "vggss": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
    "flickr": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
}
BEST_FILENAMES = {"vggss": "vggss_best.pth", "flickr": "flickr_best.pth"}


def setup_paths():
    import sys

    v1_root = PROJECT_ROOT / "chuagnxindian" / "1mufasaslot"
    base_root = (
        PROJECT_ROOT / "chuagnxindian" / "mufasa_ablation2_l3_l4_ablation"
    )
    for path in (PROJECT_ROOT, v1_root, base_root, EXPERIMENT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def setup_seed(seed=12345):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def reference_paths(dataset):
    exp = REFERENCE_EXPERIMENTS[dataset]
    root = PROJECT_ROOT / "checkpoints" / exp
    return root / "configs.json", root / BEST_FILENAMES[dataset]


def namespace_from_json(path, **overrides):
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    values.update(overrides)
    return argparse.Namespace(**values)


def load_checkpoint_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device), checkpoint


def require_cuda(gpu):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; Experiment 6.1 refuses to silently run the "
            "formal batch-256 audit/training on CPU."
        )
    if gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU {gpu}, but only {torch.cuda.device_count()} GPUs exist"
        )
    return torch.device(f"cuda:{gpu}")
