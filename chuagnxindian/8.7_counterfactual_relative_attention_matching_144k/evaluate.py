#!/usr/bin/env python3
"""Run the byte-equivalent 8.6 evaluator with 8.7 paths and groups."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import common  # Ensure the delegated script binds to the 8.7 isolation layer.
import objective  # Ensure inference tensors are the 8.7-compatible implementation.


path = Path(__file__).resolve().parent.parent / "8.6_loss_to_decision_causal_ablation" / "evaluate.py"
spec = importlib.util.spec_from_file_location("_experiment_86_evaluate", path)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load Experiment 8.6 evaluator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.main()
