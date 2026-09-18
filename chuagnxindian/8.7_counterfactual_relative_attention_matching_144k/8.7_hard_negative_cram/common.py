"""Hard-CRAM output isolation over the verified Mean-CRAM Stage-1 scaffold."""
from __future__ import annotations
import importlib.util, os
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
spec=importlib.util.spec_from_file_location('_mean_cram_common',SOURCE/'common.py')
_source=importlib.util.module_from_spec(spec);spec.loader.exec_module(_source)
_source.HERE=HERE

PROJECT_ROOT=_source.PROJECT_ROOT; DATASETS=_source.DATASETS; EPOCHS=_source.EPOCHS; BATCH_SIZE=_source.BATCH_SIZE; LR=_source.LR; WEIGHT_DECAY=_source.WEIGHT_DECAY; SEEDS=(12345,); GROUPS=('C3',)
setup_seed=_source.setup_seed;sha256=_source.sha256;state_sha256=_source.state_sha256;file_snapshot=_source.file_snapshot;write_json=_source.write_json;git_metadata=_source.git_metadata;config=_source.config;train_dataset=_source.train_dataset;uncached_test_dataset=_source.uncached_test_dataset;official_checkpoint=_source.official_checkpoint;load_state=_source.load_state;build_model=_source.build_model;EpochOrderSampler=_source.EpochOrderSampler;parameter_l2=_source.parameter_l2
def test_dataset(dataset_name):
    previous=_source.HERE;_source.HERE=SOURCE
    try:return _source.test_dataset(dataset_name)
    finally:_source.HERE=previous
def initialization_path(dataset_name,seed):return SOURCE.parent/'8.6_loss_to_decision_causal_ablation'/'checkpoints'/'initialization'/dataset_name/f'seed{seed}.pth'
def order_path(dataset_name,seed):return SOURCE.parent/'8.6_loss_to_decision_causal_ablation'/'configs'/f'{dataset_name}_seed{seed}_epoch_orders.npy'
def active_root():return Path(os.environ.get('HARD_CRAM_RUN_ROOT',str(HERE))).resolve()
def curve_root():return active_root()
def run_dir(dataset_name,group,seed):return active_root()/'stage1'/dataset_name/f'seed{seed}'
def result_dir(dataset_name,group,seed):return active_root()/'stage1'/dataset_name/f'seed{seed}'/'natural_localization'
def checkpoint_path(dataset_name,group,seed,kind='best'):
    return run_dir(dataset_name,group,seed)/('selected_best.pth' if kind=='best' else 'final.pth')
