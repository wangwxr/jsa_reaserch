"""10k-data isolation layer for the verified 8.7 Stage-1 trainer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
MEAN_SOURCE = HERE.parent / "8.7_counterfactual_relative_attention_matching"
AUDIT_SOURCE = HERE.parent / "8.6_loss_to_decision_causal_ablation"

spec = importlib.util.spec_from_file_location("_mean_cram_10k_source_common", MEAN_SOURCE / "common.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the verified Mean-CRAM common module")
_mean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mean)

# These are the pre-existing original-1.3G 10k recipes, with only their
# dataset aliases exposed as vggss/flickr to the verified 8.7 trainer.
DATASETS = {
    "vggss": {
        "experiment": "vggss_10k",
        "stage1_dir": "mufasa_ablation2_l3_l4_ablation_vggss_10k",
        "selected_file": "vggss_best.pth",
        "workers": 8,
        "eval_batch": 256,
    },
    "flickr": {
        "experiment": "flickr_10k",
        "stage1_dir": "mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5",
        "selected_file": "flickr_best.pth",
        "workers": 8,
        "eval_batch": 32,
    },
}
EPOCHS = 100
BATCH_SIZE = 256
LR = 5e-5
WEIGHT_DECAY = 0.01
SEEDS = (12345,)
GROUPS = ("C3",)

# _mean delegates construction to this 8.6 module.  Update its actual globals
# rather than merely changing aliases in this wrapper.
_base = _mean._source
_base.HERE = HERE
_base.DATASETS = DATASETS
_base.EPOCHS = EPOCHS
_base.BATCH_SIZE = BATCH_SIZE
_base.LR = LR
_base.WEIGHT_DECAY = WEIGHT_DECAY

PROJECT_ROOT = _base.PROJECT_ROOT
setup_seed = _base.setup_seed
sha256 = _base.sha256
state_sha256 = _base.state_sha256
file_snapshot = _base.file_snapshot
write_json = _base.write_json
git_metadata = _base.git_metadata
config = _base.config
train_dataset = _base.train_dataset
uncached_test_dataset = _base.uncached_test_dataset
official_checkpoint = _base.official_checkpoint
load_state = _base.load_state
build_model = _base.build_model
EpochOrderSampler = _base.EpochOrderSampler
parameter_l2 = _base.parameter_l2


def test_dataset(dataset_name: str):
    """Reuse the verified evaluation-audio cache without changing test order."""
    previous = _base.HERE
    _base.HERE = AUDIT_SOURCE
    try:
        return _base.test_dataset(dataset_name)
    finally:
        _base.HERE = previous


def initialization_path(dataset_name: str, seed: int) -> Path:
    return HERE / "checkpoints" / "initialization" / dataset_name / f"seed{seed}.pth"


def order_path(dataset_name: str, seed: int) -> Path:
    return HERE / "configs" / f"{dataset_name}_seed{seed}_epoch_orders.npy"


def method() -> str:
    value = os.environ.get("CRAM10K_METHOD", "mean")
    if value not in {"mean", "hard_default"}:
        raise ValueError(f"Unknown 10k method: {value}")
    return value


def run_dir(dataset_name: str, group: str, seed: int) -> Path:
    return HERE / method() / "stage1" / dataset_name / f"seed{seed}"


def result_dir(dataset_name: str, group: str, seed: int) -> Path:
    return run_dir(dataset_name, group, seed) / "natural_localization"


def checkpoint_path(dataset_name: str, group: str, seed: int, kind: str = "best") -> Path:
    name = "selected_best.pth" if kind == "best" else "final.pth"
    return run_dir(dataset_name, group, seed) / name


def curve_root() -> Path:
    return HERE / method()
