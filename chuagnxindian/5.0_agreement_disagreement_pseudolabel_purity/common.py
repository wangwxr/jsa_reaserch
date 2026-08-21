"""Shared utilities for Experiment 5.0, reusing the audited 4.1 stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE41 = PROJECT_ROOT / "chuagnxindian" / "4.1_selective_fusion_capacity_evidence_probe"
BASE40 = PROJECT_ROOT / "chuagnxindian" / "4.0_aud_img_redundancy_probe"
BASE42 = PROJECT_ROOT / "chuagnxindian" / "4.2_counterfactual_crossmodal_reliability_probe"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("experiment_41_common_for_50", BASE41 / "common.py")

EXPERIMENTS = base.EXPERIMENTS
OBJECT_CHECKPOINT = base.OBJECT_CHECKPOINT
inverse_normalize = base.inverse_normalize
freeze = base.freeze
snapshot_files = base.snapshot_files
verify_snapshots = base.verify_snapshots
load_config = base.load_config
stage1_checkpoint_path = base.stage1_checkpoint_path
g_checkpoint_path = base.g_checkpoint_path
load_original_g = base.load_original_g
object_prior_model = base.object_prior_model
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


def reference_40_dir(setting: str) -> Path:
    return BASE40 / "results" / setting


def reference_41_dir(setting: str) -> Path:
    return BASE41 / "results" / setting


def reference_42_dir(setting: str) -> Path:
    return BASE42 / "results" / setting
