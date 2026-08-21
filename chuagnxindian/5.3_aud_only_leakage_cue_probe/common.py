"""Shared utilities for Experiment 5.3, reusing the audited 5.2 stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE52 = PROJECT_ROOT / "chuagnxindian" / "5.2_expansion_shrink_error_diagnosis"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("experiment_52_common_for_53", BASE52 / "common.py")

EXPERIMENTS = base.EXPERIMENTS
snapshot_files = base.snapshot_files
verify_snapshots = base.verify_snapshots
load_config = base.load_config
stage1_checkpoint_path = base.stage1_checkpoint_path
g_checkpoint_path = base.g_checkpoint_path
load_original_g = base.load_original_g
build_loader = base.build_loader
flatten_batch = base.flatten_batch
normalize_map = base.normalize_map
resize_tensor = base.resize_tensor
sample_iou = base.sample_iou
summarize = base.summarize
safe_pearson = base.safe_pearson
spearman = base.spearman
distribution = base.distribution
write_csv = base.write_csv
write_json = base.write_json


def reference_41_dir(setting: str) -> Path:
    return base.reference_41_dir(setting)


def reference_51_dir(setting: str) -> Path:
    return base.reference_51_dir(setting)


def reference_52_dir(setting: str) -> Path:
    return BASE52 / "results" / setting
