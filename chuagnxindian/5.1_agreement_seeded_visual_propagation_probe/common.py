"""Shared utilities for Experiment 5.1, reusing the audited 5.0 stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
BASE50 = PROJECT_ROOT / "chuagnxindian" / "5.0_agreement_disagreement_pseudolabel_purity"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("experiment_50_common_for_51", BASE50 / "common.py")

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
transition = base.base.transition
safe_pearson = base.safe_pearson
spearman = base.spearman
js_divergence = base.base.js_divergence
distribution = base.distribution
write_csv = base.write_csv
write_json = base.write_json
reference_40_dir = base.reference_40_dir
reference_41_dir = base.reference_41_dir
reference_42_dir = base.reference_42_dir


def reference_50_dir(setting: str) -> Path:
    return BASE50 / "results" / setting
