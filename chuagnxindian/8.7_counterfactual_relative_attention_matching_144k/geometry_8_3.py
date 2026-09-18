#!/usr/bin/env python3
"""Run the exact 8.3 geometry replay against 8.7 artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import common
import replay_8_1


path = Path(__file__).resolve().parent.parent / "8.6_loss_to_decision_causal_ablation" / "geometry_8_3.py"
spec = importlib.util.spec_from_file_location("_experiment_86_geometry", path)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load Experiment 8.6 geometry replay")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.main()
