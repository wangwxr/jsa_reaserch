#!/usr/bin/env python3
"""Cross-model preflight for the fixed mechanism-audit inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
EXP87 = AUDIT.parent
ROOT = EXP87.parents[1]


def main() -> None:
    dataset = "vggss"
    sanity_dir = AUDIT / "results" / dataset / "sanity"
    with np.load(sanity_dir / "original_1.3g_final_maps.npz", allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    with np.load(sanity_dir / "cram_c3_stage2_maps.npz", allow_pickle=False) as archive:
        stage2 = {key: archive[key].copy() for key in archive.files}
    stage1_path = EXP87 / f"experiment_8_1_replay/{dataset}/C3/seed12345/counterfactual_maps.npz"
    with np.load(stage1_path, allow_pickle=False) as archive:
        stage1_ids = archive["ids"].astype(str)[:8]
        stage1_matched = archive["H_plus"][:8]
        stage1_wrong = archive["wrong"][:8]
        stage1_wrong_indices = archive["wrong_indices"][:8]

    checks = {
        "sample_ids_equal_across_three_models": bool(
            np.array_equal(original["ids"].astype(str), stage2["ids"].astype(str))
            and np.array_equal(original["ids"].astype(str), stage1_ids)
        ),
        "wrong_indices_equal_across_three_models": bool(
            np.array_equal(original["wrong_indices"], stage2["wrong_indices"])
            and np.array_equal(original["wrong_indices"], stage1_wrong_indices)
        ),
        "original_matched_wrong_max_abs_difference": float(
            np.max(np.abs(original["H_matched"][:, None] - original["H_wrong"]))
        ),
        "stage1_matched_wrong_max_abs_difference": float(
            np.max(np.abs(stage1_matched[:, None] - stage1_wrong))
        ),
        "stage2_matched_wrong_max_abs_difference": float(
            np.max(np.abs(stage2["H_matched"][:, None] - stage2["H_wrong"]))
        ),
        "original_vs_stage2_matched_max_abs_difference": float(
            np.max(np.abs(original["H_matched"] - stage2["H_matched"]))
        ),
        "native_shapes": {
            "original_final": list(original["H_matched"].shape),
            "cram_stage1": list(stage1_matched.shape),
            "cram_stage2": list(stage2["H_matched"].shape),
        },
    }
    required = (
        checks["sample_ids_equal_across_three_models"],
        checks["wrong_indices_equal_across_three_models"],
        checks["original_matched_wrong_max_abs_difference"] > 0,
        checks["stage1_matched_wrong_max_abs_difference"] > 0,
        checks["stage2_matched_wrong_max_abs_difference"] > 0,
        checks["original_vs_stage2_matched_max_abs_difference"] > 0,
        checks["native_shapes"] == {
            "original_final": [8, 196], "cram_stage1": [8, 49],
            "cram_stage2": [8, 196],
        },
    )
    checks["passed"] = bool(all(required))
    output = sanity_dir / "three_model_sanity.json"
    output.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not checks["passed"]:
        raise RuntimeError("Three-model mechanism sanity failed")


if __name__ == "__main__":
    main()
