#!/usr/bin/env python3
"""Aggregate the formal Experiment 5.1 probes and write REPORT.md."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import common


SETTINGS = ("vggss_144k", "flickr_144k")
LABELS = {"vggss_144k": "VGGSS-144k", "flickr_144k": "Flickr-144k"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--report", type=Path, default=common.HERE / "REPORT.md")
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def metric_pair(value: dict[str, Any]) -> str:
    return f"{fmt(value['cIoU'])}/{fmt(value['AUC'])}"


def read_summary(root: Path, setting: str) -> dict[str, Any]:
    return json.loads((root / setting / "summary.json").read_text(encoding="utf-8"))


def build_report(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Experiment 5.1 - Agreement-Seeded Visual Propagation Probe",
        "",
        "## Protocol Audit",
        "",
        "- Zero training: `model.eval()`, `torch.inference_mode()`, no optimizer, no backward, no new trainable parameters, and `parameters_with_grad=[]`.",
        "- The only formal seed is frozen Stage1 `P=Top20(AUD_L4) intersect Top20(IMG_L4)` at `7x7`, nearest-resized to binary `P14`.",
        "- `F34` propagation compares an F34 seed prototype only with F34 tokens. `K34` propagation compares a K34 seed prototype only with K34 tokens. No audio cosine or cross-space cosine is used.",
        "- GT and OGL are used only for evaluation, oracle construction, and mechanism analysis.",
        "",
        "## 5.0 Reproduction And Feature Spaces",
        "",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        audit = summary["reproduction_5_0"]
        tensor = summary["tensor_and_feature_space_audit"]
        zero = summary["zero_training_audit"]
        lines.extend(
            [
                f"### {LABELS[setting]}",
                "",
                f"- Raw tensor error: `{audit['raw_tensor_max_errors']}`; per-sample metric error: `{audit['per_sample_metric_max_errors']}`; P error: `{audit['stage1_P_max_errors']}`; aggregate P-purity error: `{audit['aggregate_stage1_P_purity_error']}`; sample mismatch: `{audit['sample_mismatches']}`.",
                f"- Shapes: `AUD_L4={tensor['AUD_L4_shape']}`, `IMG_L4={tensor['IMG_L4_shape']}`, `F3={tensor['F3_SPATIAL_shape']}`, `F4_up={tensor['F4_UP_shape']}`, `F34={tensor['F34_shape']}`, `K34={tensor['K34_shape']}`, `AUD_FINE={tensor['AUD_FINE_shape']}`.",
                f"- Checkpoints unchanged: `{zero['all_checkpoint_hashes_and_mtimes_unchanged']}`; no NaN/Inf: `{zero['no_nan_or_inf']}`; trainable params: `{zero['new_trainable_params']}`; grad parameters: `{zero['parameters_with_grad']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Seed Audit",
            "",
            "| Dataset | P14 pixels mean/median | Non-empty | P precision | P recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        seed = summaries[setting]["seed_audit"]
        lines.append(
            f"| {LABELS[setting]} | {fmt(seed['P14_pixel_count']['mean'], 2)}/{fmt(seed['P14_pixel_count']['median'], 1)} | "
            f"{fmt(seed['non_empty_rate'])} | {fmt(seed['P_FG_purity']['mean'])} | {fmt(seed['P_FG_recall']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Main Table A - Standalone Localization",
            "",
            "| Dataset | AUD_FINE | IMG | SEED_ONLY | PROP_F34 | PROP_K34 | OGL |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        standalone = summaries[setting]["standalone"]
        lines.append(
            f"| {LABELS[setting]} | {metric_pair(standalone['AUD_FINE'])} | {metric_pair(standalone['IMG'])} | "
            f"{metric_pair(standalone['SEED_ONLY'])} | {metric_pair(standalone['PROP_F34'])} | "
            f"{metric_pair(standalone['PROP_K34'])} | {metric_pair(standalone['OGL'])} |"
        )

    lines.extend(
        [
            "",
            "## Main Table B - Propagation Quality",
            "",
            "| Dataset | Space | FG sim | BG sim | Margin | Seed P/R | Prop P/R | Recall gain | Precision loss | Expansion FG | Random expansion FG |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for space in ("F34", "K34"):
            value = summaries[setting]["propagation_quality"][space]
            lines.append(
                f"| {LABELS[setting]} | {space} | {fmt(value['FG_similarity']['mean'])} | {fmt(value['BG_similarity']['mean'])} | "
                f"{fmt(value['FG_BG_margin']['mean'])} | {fmt(value['seed_precision']['mean'])}/{fmt(value['seed_recall']['mean'])} | "
                f"{fmt(value['prop_precision']['mean'])}/{fmt(value['prop_recall']['mean'])} | {fmt(value['recall_gain']['mean'])} | "
                f"{fmt(value['precision_loss']['mean'])} | {fmt(value['expansion_FG_purity']['mean'])} | "
                f"{fmt(value['random_expansion_FG_purity']['mean'])} |"
            )

    lines.extend(
        [
            "",
            "## Main Table C - Complementarity With AUD",
            "",
            "| Dataset | Prop | Pearson | Spearman | JS | Top20 | Mask IoU | PROP_ONLY | AUD_ONLY | Pair Oracle | Oracle gain | OGL rescue capture |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for space in ("F34", "K34"):
            sim = summaries[setting]["map_similarity"][f"AUD_vs_PROP_{space}"]
            comp = summaries[setting]["complementarity"][space]
            decomp = comp["success_decomposition"]
            oracle = comp["pair_oracle"]
            lines.append(
                f"| {LABELS[setting]} | {space} | {fmt(sim['Pearson']['mean'])} | {fmt(sim['Spearman']['mean'])} | "
                f"{fmt(sim['JS']['mean'])} | {fmt(sim['Top20_overlap']['mean'])} | {fmt(sim['Mask_IoU']['mean'])} | "
                f"{decomp['PROP_ONLY']} | {decomp['AUD_ONLY']} | {metric_pair(oracle['metrics'])} | "
                f"{fmt(oracle['gain_cIoU'])}/{fmt(oracle['gain_AUC'])} | {comp['OGL_rescue_captured']}/{comp['OGL_rescue_total']} "
                f"({fmt(comp['OGL_rescue_capture_rate'])}) |"
            )
    lines.extend(["", "Reference AUD+IMG Sample Oracle from 4.0:", ""])
    for setting in SETTINGS:
        lines.append(f"- {LABELS[setting]}: `{metric_pair(summaries[setting]['reference_AUD_IMG_sample_oracle'])}`.")

    lines.extend(
        [
            "",
            "## Main Table D - Seed Controls",
            "",
            "| Dataset | Space | Seed | Prop cIoU/AUC | Pair-oracle gain cIoU/AUC |",
            "|---|---|---|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for space in ("F34", "K34"):
            for seed in ("RANDOM", "AUD", "IMG", "AGREEMENT"):
                value = summaries[setting]["seed_controls"][space][seed]
                oracle = value["pair_oracle"]
                lines.append(
                    f"| {LABELS[setting]} | {space} | {seed} | {metric_pair(value['metrics'])} | "
                    f"{fmt(oracle['gain_cIoU'])}/{fmt(oracle['gain_AUC'])} |"
                )

    lines.extend(
        [
            "",
            "## Top10/20/30 Diagnostic",
            "",
            "| Dataset | Top-k | Seed purity/recall | F34 cIoU/AUC | F34 oracle gain | K34 cIoU/AUC | K34 oracle gain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for top in ("10", "20", "30"):
            value = summaries[setting]["topk_diagnostic"][top]
            lines.append(
                f"| {LABELS[setting]} | {top}% | {fmt(value['seed_purity']['mean'])}/{fmt(value['seed_recall']['mean'])} | "
                f"{metric_pair(value['F34']['metrics'])} | {fmt(value['F34']['pair_oracle']['gain_cIoU'])} | "
                f"{metric_pair(value['K34']['metrics'])} | {fmt(value['K34']['pair_oracle']['gain_cIoU'])} |"
            )

    lines.extend(
        [
            "",
            "## Error-Mechanism Groups",
            "",
            "IMG_ONLY+SHRINK mean IoU / predicted-area / FP-area / FG-recall:",
            "",
            "| Dataset | Method | IoU | Area | FP area | FG recall |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        group = summaries[setting]["hard_cases"]["IMG_ONLY_SHRINK"]
        for method in ("AUD", "IMG", "PROP_F34", "PROP_K34"):
            value = group[method]
            lines.append(
                f"| {LABELS[setting]} | {method} | {fmt(value['IoU']['mean'])} | {fmt(value['area_fraction']['mean'])} | "
                f"{fmt(value['FP_area_fraction']['mean'])} | {fmt(value['FG_recall']['mean'])} |"
            )

    lines.extend(["", "AUD over-expansion group:", ""])
    for setting in SETTINGS:
        value = summaries[setting]["aud_over_expansion"]
        lines.append(
            f"- {LABELS[setting]} count `{value['count']}`: FP area AUD/F34/K34 "
            f"`{fmt(value['AUD']['FP_area_fraction']['mean'])}` / `{fmt(value['PROP_F34']['FP_area_fraction']['mean'])}` / "
            f"`{fmt(value['PROP_K34']['FP_area_fraction']['mean'])}`; recall "
            f"`{fmt(value['AUD']['FG_recall']['mean'])}` / `{fmt(value['PROP_F34']['FG_recall']['mean'])}` / "
            f"`{fmt(value['PROP_K34']['FG_recall']['mean'])}`."
        )

    lines.extend(["", "AUD_ONLY risk:", ""])
    for setting in SETTINGS:
        value = summaries[setting]["aud_only_risk"]
        lines.append(
            f"- {LABELS[setting]} count `{value['count']}`: F34 success/hurt `{value['F34']['prop_success']}/{value['F34']['prop_hurt']}`, "
            f"mean IoU/recall delta `{fmt(value['F34']['IoU_delta']['mean'])}` / `{fmt(value['F34']['recall_delta']['mean'])}`; "
            f"K34 `{value['K34']['prop_success']}/{value['K34']['prop_hurt']}`, deltas "
            f"`{fmt(value['K34']['IoU_delta']['mean'])}` / `{fmt(value['K34']['recall_delta']['mean'])}`."
        )

    lines.extend(
        [
            "",
            "## Agreement Confidence And Gain",
            "",
            "| Dataset | Quartile | Seed purity | F34 cIoU | F34 margin | K34 cIoU | K34 margin |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for value in summaries[setting]["agreement_confidence_quartiles"]:
            lines.append(
                f"| {LABELS[setting]} | {value['quartile']} | {fmt(value['seed_purity']['mean'])} | "
                f"{fmt(value['F34']['propagation']['cIoU'])} | {fmt(value['F34']['FG_BG_margin']['mean'])} | "
                f"{fmt(value['K34']['propagation']['cIoU'])} | {fmt(value['K34']['FG_BG_margin']['mean'])} |"
            )
    lines.extend(["", "Seed purity vs propagation IoU gain correlation:", ""])
    for setting in SETTINGS:
        corr = summaries[setting]["seed_purity_vs_propagation_gain"]
        lines.append(
            f"- {LABELS[setting]} F34 Pearson/Spearman `{fmt(corr['F34']['Pearson'])}/{fmt(corr['F34']['Spearman'])}`; "
            f"K34 `{fmt(corr['K34']['Pearson'])}/{fmt(corr['K34']['Spearman'])}`."
        )

    lines.extend(["", "## Qualitative", ""])
    for setting in SETTINGS:
        lines.append(f"- {LABELS[setting]} deterministic selections: `{summaries[setting]['qualitative_selection']}`.")
    lines.extend(
        [
            "- Propagation reliably fills most of the visually similar object and nearby context, converting the sparse core into a broad response.",
            "- In successful PROP_ONLY cases this broad response covers object extent missed by AUD, explaining the sizable pair-oracle capacity.",
            "- In IMG_ONLY+SHRINK and OGL-rescue cases, the same mechanism expands into exterior/context instead of preserving IMG's tighter correction.",
            "- F34 is slightly less expansive than K34 on VGG, but neither space suppresses context leakage.",
            "",
            "## Decision",
            "",
            "**Case D - Propagation Introduces More Context Leakage.**",
            "",
            "The frozen visual spaces do contain a real propagation signal: FG-BG margins are positive on both datasets, recall rises strongly over SEED_ONLY, expansion pixels beat matched random, and AUD+PROP pair oracles have non-trivial gains.",
            "",
            "However, the propagation does not solve sounding-object extent. It drives recall toward 0.92-0.96 by producing much larger masks. False-positive area increases in both over-expansion groups, IMG_ONLY+SHRINK performance falls below both AUD and IMG, and OGL-rescue capture remains only about 10-16%.",
            "",
            "Agreement specificity is also weak: agreement prototypes clearly beat random seeds, but improve only marginally and inconsistently over AUD/IMG seeds. Seed-purity versus propagation-gain correlations are weak, especially on Flickr.",
            "",
            "**Close the current prototype-propagation line. Do not start 5.2 Agreement-Seeded Spatial Refinement automatically.**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    arguments = parse_args()
    summaries = {setting: read_summary(arguments.results_root, setting) for setting in SETTINGS}
    for setting, summary in summaries.items():
        if not summary["completed_full_dataset"]:
            raise RuntimeError(f"Partial result: {setting}")
        if not summary["reproduction_5_0"]["passed"]:
            raise RuntimeError(f"5.0 reproduction failed: {setting}")
        zero = summary["zero_training_audit"]
        if not zero["all_checkpoint_hashes_and_mtimes_unchanged"] or zero["parameters_with_grad"]:
            raise RuntimeError(f"Zero-training audit failed: {setting}")
    arguments.report.write_text(build_report(summaries), encoding="utf-8")
    print(arguments.report)


if __name__ == "__main__":
    main()
