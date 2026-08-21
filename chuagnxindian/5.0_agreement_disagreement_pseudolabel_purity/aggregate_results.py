#!/usr/bin/env python3
"""Aggregate the two formal Experiment 5.0 probes and write REPORT.md."""

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


def decide(summaries: dict[str, dict[str, Any]]) -> tuple[str, str]:
    p_supported = all(
        summaries[setting]["enrichment"]["macro"]["FG_P_minus_A"] > 0
        and summaries[setting]["enrichment"]["macro"]["FG_P_minus_I"] > 0
        and summaries[setting]["stage1_primary_top20"]["P"]["non_empty_rate"] > 0.9
        for setting in SETTINGS
    )
    negative_random_fail = any(
        summaries[setting]["enrichment"]["macro"]["BG_NA_minus_random_NA"] <= 0
        for setting in SETTINGS
    )
    ranking_fail = any(
        summaries[setting]["ranking_viability"]["fraction_positive"] <= 0.5
        for setting in SETTINGS
    )
    if p_supported and (negative_random_fail or ranking_fail):
        return (
            "Case B - Positive Seeds Work, Negative Seeds Fail",
            "Agreement positive supervision supported; context-negative supervision unsupported. Next: positive-only supervision, not context suppression.",
        )
    stage1_signal = all(
        summaries[setting]["enrichment"]["macro"]["FG_P_minus_A"] > 0
        or summaries[setting]["enrichment"]["macro"]["BG_NA_minus_A"] > 0
        for setting in SETTINGS
    )
    stage2_signal = all(
        summaries[setting]["stage2_diagnostic_top20"]["P"]["macro_fg_purity"]
        > summaries[setting]["stage2_diagnostic_top20"]["A"]["macro_fg_purity"]
        for setting in SETTINGS
    )
    if not stage1_signal and stage2_signal:
        return (
            "Case C - Only Stage2 Disagreement Is Useful",
            "This is not directly usable as a Stage2 teacher under the strict two-stage constraint.",
        )
    if p_supported:
        return (
            "Case A - Stage1 Pseudo-Supervision Supported",
            "Next candidate: 5.1 Stage2 Agreement-Guided Context Ranking.",
        )
    return (
        "Case D - Agreement/Disagreement Is Not a Reliable Pseudo-Label Source",
        "Close the current dual-branch pseudo-label line.",
    )


def build_report(summaries: dict[str, dict[str, Any]]) -> str:
    decision, next_action = decide(summaries)
    lines = [
        "# Experiment 5.0 - Agreement-Disagreement Pseudo-Label Purity Probe",
        "",
        "## Protocol Audit",
        "",
        "- Zero training: `model.eval()`, `torch.inference_mode()`, no optimizer, no backward, and zero trainable parameters.",
        "- Primary teacher is frozen Stage1 `AUD_L4` / `IMG_L4` at `7x7`. Stage2 `AUD_FINE` / aligned `IMG` at `14x14` is diagnostic only.",
        "- Formal seed is fixed Top20: Stage1 `k=10`, Stage2 `k=40`; stable descending flat-index tie breaking; binary seeds are nearest-neighbor resized to the binary `224x224` GT only for analysis.",
        "- Empty seeds are excluded from purity means and retained in non-empty/empty-rate reporting. GT and OGL are never model or pseudo-label inputs.",
        "",
        "## Reproduction And Shapes",
        "",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        reproduction = summary["reproduction"]
        tensor = summary["tensor_audit"]
        stage1 = summary["formal_metrics"]["Stage1"]
        stage2 = summary["formal_metrics"]["Stage2"]
        zero = summary["zero_training_audit"]
        lines.extend(
            [
                f"### {LABELS[setting]}",
                "",
                f"- Stage1 AUD/IMG: `{metric_pair(stage1['AUD'])}` / `{metric_pair(stage1['IMG'])}`; Stage2 AUD_FINE/IMG: `{metric_pair(stage2['AUD_FINE'])}` / `{metric_pair(stage2['IMG'])}`.",
                f"- Raw map max error vs 4.1: `{reproduction['raw_map_max_errors']}`; per-sample metric max error vs 4.0/4.1: `{reproduction['per_sample_metric_max_errors']}`; vs 4.2: `{reproduction['per_sample_4_2_metric_max_errors']}`; sample mismatches: `{reproduction['sample_order_mismatches']}`.",
                f"- Shapes: `Qa={tensor['Qa_shape']}`, `Qv={tensor['Qv_shape']}`, `K4={tensor['K4_shape']}`, `K34={tensor['K34_shape']}`, `AUD_L4={tensor['Stage1_AUD_L4_shape']}`, `IMG_L4={tensor['Stage1_IMG_L4_shape']}`, `AUD_FINE={tensor['Stage2_AUD_FINE_shape']}`, aligned IMG=`{tensor['Stage2_IMG_aligned_shape']}`, GT=`{tensor['GT_shape']}`.",
                f"- Checkpoints unchanged: `{zero['all_checkpoint_hashes_and_mtimes_unchanged']}`; no NaN/Inf: `{zero['no_nan_or_inf']}`; trainable parameters: `{zero['new_trainable_params']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Stage1 Primary Purity",
            "",
            "Macro is the primary column; micro pools all seed pixels.",
            "",
            "| Dataset | Seed | Macro FG | Macro BG | Micro FG | Micro BG | FG recall | BG coverage | Non-empty | Mean area |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    seed_labels = {
        "RANDOM_A": "Random A-matched",
        "A": "AUD20",
        "I": "IMG20",
        "P": "Agreement P",
        "NA": "AUD-extra candidate",
        "NI": "IMG-extra",
    }
    for setting in SETTINGS:
        result = summaries[setting]["stage1_primary_top20"]
        for seed in ("RANDOM_A", "A", "I", "P", "NA", "NI"):
            value = result[seed]
            lines.append(
                f"| {LABELS[setting]} | {seed_labels[seed]} | {fmt(value['macro_fg_purity'])} | {fmt(value['macro_bg_purity'])} | "
                f"{fmt(value['micro_fg_purity'])} | {fmt(value['micro_bg_purity'])} | {fmt(value['macro_fg_recall'])} | "
                f"{fmt(value['macro_bg_coverage'])} | {fmt(value['non_empty_rate'])} | {fmt(value['mean_area'], 1)} |"
            )

    lines.extend(
        [
            "",
            "## Enrichment",
            "",
            "| Dataset | Scope | FG(P) | FG(A20) | FG(I20) | P-A lift | P-I lift | BG(AUD-extra) | BG(A20) | BG(Random matched) | NA-A lift | NA-Random lift |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        primary = summaries[setting]["stage1_primary_top20"]
        enrichment = summaries[setting]["enrichment"]
        for scope, prefix in (("Macro", "macro"), ("Micro", "micro")):
            lines.append(
                f"| {LABELS[setting]} | {scope} | {fmt(primary['P'][prefix + '_fg_purity'])} | "
                f"{fmt(primary['A'][prefix + '_fg_purity'])} | {fmt(primary['I'][prefix + '_fg_purity'])} | "
                f"{fmt(enrichment[prefix]['FG_P_minus_A'])} | {fmt(enrichment[prefix]['FG_P_minus_I'])} | "
                f"{fmt(primary['NA'][prefix + '_bg_purity'])} | {fmt(primary['A'][prefix + '_bg_purity'])} | "
                f"{fmt(primary['RANDOM_NA'][prefix + '_bg_purity'])} | {fmt(enrichment[prefix]['BG_NA_minus_A'])} | "
                f"{fmt(enrichment[prefix]['BG_NA_minus_random_NA'])} |"
            )

    lines.extend(
        [
            "",
            "## Purity-Coverage Tradeoff",
            "",
            "| Dataset | Top-k | P FG purity | P FG recall | P empty | AUD-extra BG purity | AUD-extra BG coverage | AUD-extra empty |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for top in ("10", "20", "30"):
            result = summaries[setting]["topk_diagnostic"]["Stage1"][top]
            lines.append(
                f"| {LABELS[setting]} | {top}% | {fmt(result['P']['macro_fg_purity'])} | {fmt(result['P']['macro_fg_recall'])} | "
                f"{fmt(result['P']['empty_rate'])} | {fmt(result['NA']['macro_bg_purity'])} | {fmt(result['NA']['macro_bg_coverage'])} | {fmt(result['NA']['empty_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Confidence Quartiles",
            "",
            "| Dataset | Score | Q1 | Q2 | Q3 | Q4 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        quartiles = summaries[setting]["confidence_quartiles"]
        for key, label in (("P_agreement_confidence", "P FG purity"), ("NA_disagreement_magnitude", "AUD-extra BG purity")):
            values = quartiles[key]
            lines.append(
                f"| {LABELS[setting]} | {label} | " + " | ".join(fmt(item["purity"]) for item in values) + " |"
            )

    lines.extend(
        [
            "",
            "## Stage1 Hard Cases",
            "",
            "| Dataset | Group | Count | P FG purity | P recall | P non-empty | AUD-extra BG purity | BG coverage | AUD-extra non-empty |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        groups = summaries[setting]["hard_cases_stage1"]
        for group in ("ALL", "IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE", "IMG_ONLY_SHRINK"):
            value = groups[group]
            lines.append(
                f"| {LABELS[setting]} | {group} | {value['count']} | {fmt(value['P']['macro_fg_purity'])} | "
                f"{fmt(value['P']['macro_fg_recall'])} | {fmt(value['P']['non_empty_rate'])} | "
                f"{fmt(value['NA']['macro_bg_purity'])} | {fmt(value['NA']['macro_bg_coverage'])} | {fmt(value['NA']['non_empty_rate'])} |"
            )

    lines.extend(["", "IMG_ONLY+SHRINK Stage1 vs Stage2 (Stage2 is diagnostic only):", ""])
    for setting in SETTINGS:
        stage1 = summaries[setting]["hard_cases_stage1"]["IMG_ONLY_SHRINK"]
        stage2 = summaries[setting]["hard_cases_stage2_diagnostic"]["IMG_ONLY_SHRINK"]
        lines.append(
            f"- {LABELS[setting]}: Stage1 P/NA purity `{fmt(stage1['P']['macro_fg_purity'])}` / `{fmt(stage1['NA']['macro_bg_purity'])}`; "
            f"Stage2 P/NA purity `{fmt(stage2['P']['macro_fg_purity'])}` / `{fmt(stage2['NA']['macro_bg_purity'])}`; "
            f"Stage1/Stage2 NA coverage `{fmt(stage1['NA']['macro_bg_coverage'])}` / `{fmt(stage2['NA']['macro_bg_coverage'])}`."
        )

    lines.extend(
        [
            "",
            "## Correctness And Ranking Viability",
            "",
        ]
    )
    for setting in SETTINGS:
        matrix = summaries[setting]["correctness_matrix"]
        compact = {f"{item['region']}->{item['GT']}": item["fraction_within_region"] for item in matrix}
        ranking = summaries[setting]["ranking_viability"]
        lines.append(f"- {LABELS[setting]} correctness fractions: `{compact}`.")
        lines.append(
            f"- {LABELS[setting]} `FG purity(P)-FG purity(AUD-extra)`: mean `{fmt(ranking['mean'])}`, median `{fmt(ranking['median'])}`, "
            f"std `{fmt(ranking['std'])}`, fraction `>0` `{fmt(ranking['fraction_positive'])}` over `{ranking['num_samples']}` valid samples."
        )

    lines.extend(
        [
            "",
            "## Stage1 Versus Stage2",
            "",
            "| Dataset | Stage1 P FG | Stage2 P FG | Stage1 AUD-extra BG | Stage2 AUD-extra BG | P Pearson/Spearman | AUD-extra Pearson/Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        summary = summaries[setting]
        s1 = summary["stage1_primary_top20"]
        s2 = summary["stage2_diagnostic_top20"]
        transfer = summary["stage1_vs_stage2_transfer"]
        lines.append(
            f"| {LABELS[setting]} | {fmt(s1['P']['macro_fg_purity'])} | {fmt(s2['P']['macro_fg_purity'])} | "
            f"{fmt(s1['NA']['macro_bg_purity'])} | {fmt(s2['NA']['macro_bg_purity'])} | "
            f"{fmt(transfer['P_FG_purity']['Pearson'])}/{fmt(transfer['P_FG_purity']['Spearman'])} | "
            f"{fmt(transfer['NA_BG_purity']['Pearson'])}/{fmt(transfer['NA_BG_purity']['Spearman'])} |"
        )

    lines.extend(["", "## Qualitative", ""])
    for setting in SETTINGS:
        lines.append(f"- {LABELS[setting]} deterministic selections: `{summaries[setting]['qualitative_selection']}`.")
    lines.extend(
        [
            "- Agreement seeds generally retain the shared response core and remove weak branch-specific fringes. Their purity increases monotonically with agreement confidence on VGG; Flickr saturates at high confidence.",
            "- AUD-extra is not a clean context mask. It is background-enriched inside IMG_ONLY/SHRINK and BOTH_FAIL subsets, but in AUD_ONLY and BOTH_SUCCESS it frequently contains real object extent.",
            "- Flickr is the decisive failure mode for negative supervision: the global Stage1 AUD-extra candidate remains foreground-dominated, despite a small background-rate lift over AUD20.",
            "",
            "## Decision",
            "",
            f"**{decision}.**",
            "",
            "Agreement P is consistently better than both single-branch Top20 seeds, stays non-empty on all samples, and has non-trivial recall. This supports sparse positive consistency.",
            "",
            "The context-negative candidate does not meet the purity requirement: although `BG(AUD-extra)-BG(AUD20)` is positive on both datasets, matched-random seeds have substantially higher background purity, Flickr AUD-extra is only about 18.5% background, and ranking viability is below 50% on Flickr. AUD_ONLY contamination is also severe.",
            "",
            f"**{next_action} Do not start 5.1 automatically.**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    arguments = parse_args()
    summaries = {setting: read_summary(arguments.results_root, setting) for setting in SETTINGS}
    for setting, summary in summaries.items():
        if not summary["completed_full_dataset"]:
            raise RuntimeError(f"Partial result: {setting}")
        if not summary["reproduction"]["passed"]:
            raise RuntimeError(f"Reproduction failed: {setting}")
        if not summary["zero_training_audit"]["all_checkpoint_hashes_and_mtimes_unchanged"]:
            raise RuntimeError(f"Checkpoint mutation: {setting}")
    arguments.report.write_text(build_report(summaries), encoding="utf-8")
    print(arguments.report)


if __name__ == "__main__":
    main()
