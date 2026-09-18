#!/usr/bin/env python3
"""Run the exact 8.1 replay implementation against 8.7 artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import common


path = Path(__file__).resolve().parent.parent / "8.6_loss_to_decision_causal_ablation" / "replay_8_1.py"
spec = importlib.util.spec_from_file_location("_experiment_86_replay", path)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load Experiment 8.6 replay")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
auc_1d = module.auc_1d
conditional_auc = module.conditional_auc


if __name__ == "__main__":
    module.main()
