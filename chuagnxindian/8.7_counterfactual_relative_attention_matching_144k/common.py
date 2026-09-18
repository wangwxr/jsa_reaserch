"""Experiment 8.7 isolation layer over the verified 8.6 Stage-1 scaffold."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "8.6_loss_to_decision_causal_ablation"
_spec = importlib.util.spec_from_file_location("_experiment_86_common", SOURCE / "common.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load Experiment 8.6 common.py")
_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source)

# 8.6 owns model/dataset construction; its path globals are redirected only for
# functions whose artifacts belong to this experiment.
_source.HERE = HERE
PROJECT_ROOT = _source.PROJECT_ROOT
DATASETS = _source.DATASETS
EPOCHS = _source.EPOCHS
BATCH_SIZE = _source.BATCH_SIZE
LR = _source.LR
WEIGHT_DECAY = _source.WEIGHT_DECAY
SEEDS = (12345,)
GROUPS = ("C0", "C1", "C2", "C3")

setup_seed = _source.setup_seed
sha256 = _source.sha256
state_sha256 = _source.state_sha256
file_snapshot = _source.file_snapshot
write_json = _source.write_json
git_metadata = _source.git_metadata
config = _source.config
train_dataset = _source.train_dataset
uncached_test_dataset = _source.uncached_test_dataset
official_checkpoint = _source.official_checkpoint
load_state = _source.load_state
build_model = _source.build_model
EpochOrderSampler = _source.EpochOrderSampler
parameter_l2 = _source.parameter_l2


def test_dataset(dataset_name: str):
    """Reuse the byte-verified 8.6 evaluation-audio cache without copying it."""
    previous = _source.HERE
    _source.HERE = SOURCE
    try:
        return _source.test_dataset(dataset_name)
    finally:
        _source.HERE = previous


def initialization_path(dataset_name: str, seed: int) -> Path:
    return SOURCE / "checkpoints" / "initialization" / dataset_name / f"seed{seed}.pth"


def order_path(dataset_name: str, seed: int) -> Path:
    return SOURCE / "configs" / f"{dataset_name}_seed{seed}_epoch_orders.npy"


def run_dir(dataset_name: str, group: str, seed: int) -> Path:
    return HERE / "checkpoints" / group / dataset_name / f"seed{seed}"


def result_dir(dataset_name: str, group: str, seed: int) -> Path:
    return HERE / "natural_localization" / dataset_name / group / f"seed{seed}"


def checkpoint_path(dataset_name: str, group: str, seed: int, kind: str = "best") -> Path:
    if group == "C0":
        return SOURCE / "checkpoints" / "M0" / dataset_name / f"seed{seed}" / (
            "selected_best.pth" if kind == "best" else "final.pth"
        )
    name = "selected_best.pth" if kind == "best" else "final.pth"
    return run_dir(dataset_name, group, seed) / name
