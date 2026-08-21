"""Reuse the formal 1.3G runtime without modifying any previous experiment."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
FIXED_ROOT = PROJECT_ROOT / "chuagnxindian" / "2.1R_fixed_slot_reliability"
G_ROOT = (
    PROJECT_ROOT
    / "chuagnxindian"
    / "1mufasaslot"
    / "1.3G-multigeom_equivariant_l3_refine"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixed = _load_module("fixed_slot_runtime_for_23", FIXED_ROOT / "probe.py")
probe20 = fixed.probe20
geometry = _load_module("geometry_for_23", G_ROOT / "geometry.py")

EXPERIMENTS = ("vggss_144k", "flickr_144k")
EXPECTED_AUD = {
    "vggss_144k": (0.4269, 0.4230),
    "flickr_144k": (0.8120, 0.6356),
}
DEFAULT_NAMES = {
    "vggss_144k": "2.3_semantic_spatial_decoupled_slot_vggss_144k",
    "flickr_144k": "2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5",
}


def load_formal_1_3g(experiment: str, gpu: int):
    arguments = argparse.Namespace(experiment=experiment, gpu=gpu)
    return fixed.load_refinement(arguments)


def build_train_dataset(config):
    from dataset import get_train_dataset

    return get_train_dataset(
        config,
        hard_img=config.hard_img,
        hard_aud=config.hard_aud,
        rand_aud=config.rand_aud,
    )


def build_train_loader(dataset, config) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.workers > 0,
        prefetch_factor=2 if config.workers > 0 else None,
    )


def build_smoke_loader(dataset, batch_size: int = 4) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )


def setup_seed(seed: int) -> None:
    probe20.setup_seed(seed)


def metric_matches(metric: dict[str, float], expected: tuple[float, float]) -> bool:
    return (
        f"{metric['cIoU']:.4f}" == f"{expected[0]:.4f}"
        and f"{metric['AUC']:.4f}" == f"{expected[1]:.4f}"
    )


def device_for(gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(gpu)
    return torch.device("cuda", gpu)

