#!/usr/bin/env python3
"""Aggregate formal Experiment 4.2 results and write REPORT.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import common


SETTINGS = ("vggss_144k", "flickr_144k")
LABELS = {"vggss_144k": "VGGSS-144k", "flickr_144k": "Flickr-144k"}
PRIMARY = ("DELTA_CF_BLUR", "DELTA_CF_MEAN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--report", type=Path, default=common.HERE / "REPORT.md")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def pair(metrics: dict[str, Any]) -> str:
    return f"{fmt(metrics['cIoU'])}/{fmt(metrics['AUC'])}"


def evidence(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in summary["evidence_prediction"] if row["evidence"] == name)


def selector(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in summary["selector_results"] if row["method"] == name)


def transfer_metrics(rows: list[dict[str, str]], score: str, threshold: float) -> dict[str, Any]:
    aud = np.asarray([float(row["IoU_AUD"]) for row in rows])
    img = np.asarray([float(row["IoU_IMG"]) for row in rows])
    delta = np.asarray([float(row[score]) for row in rows])
    selected = np.where(delta > threshold, img, aud)
    transition = common.transition(aud.tolist(), selected.tolist())
    return {
        "threshold": threshold,
        "metrics": common.summarize(selected.tolist()),
        "rescue": transition["rescue"],
        "hurt": transition["hurt"],
        "net": transition["net"],
        "IMG_selection_rate": float((delta > threshold).mean()),
    }


def report_text(
    summaries: dict[str, dict[str, Any]], transfer: dict[str, Any]
) -> str:
    lines = [
        "# Experiment 4.2 - Counterfactual Cross-Modal Reliability Probe",
        "",
        "## Zero-Training and Reproduction Audit",
        "",
        "| Dataset | 4.1 metric error | Raw AUD error | Raw IMG error | Order mismatch | Trainable params | Checkpoints unchanged |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        reproduction = summary["reproduction_4_1"]
        zero = summary["zero_training_audit"]
        lines.append(
            f"| {LABELS[setting]} | {reproduction['per_sample_metric_max_error']} | "
            f"{reproduction['raw_tensor_max_errors']['AUD_FINE']} | "
            f"{reproduction['raw_tensor_max_errors']['IMG_L4']} | "
            f"{reproduction['sample_order_mismatches']} | {zero['new_trainable_params']} | "
            f"{zero['all_checkpoint_hashes_and_mtimes_unchanged']} |"
        )

    audit = summaries["vggss_144k"]["semantic_metric_audit"]
    lines.extend(
        [
            "",
            "## Formal InfoNCE Metric Space",
            "",
            f"- Visual: `{audit['visual_representation']}`.",
            f"- Audio: `{audit['audio_representation']}`.",
            f"- Normalization: `{audit['normalization']}`.",
            f"- Projection: `{audit['projection']}`.",
            f"- Similarity: `{audit['similarity']}`; training logit `{audit['training_logit']}` with `tau={audit['temperature']}`.",
            f"- Counterfactual scores use pre-temperature cosine. `{audit['invalid_direct_metric_spaces']}`.",
            "",
            "## Mask and Intervention Audit",
            "",
            "Primary masks use the highest 40 of 196 positions independently for AUD and IMG. Nearest resizing maps every native cell to 16x16 input pixels.",
            "",
            "| Dataset | A20 native/input | I20 native/input | Input area diff | Mean A20-I20 IoU | Mean AUD-extra/IMG-extra area |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        mask = summaries[setting]["mask_audit"]
        overlap = next(row for row in summaries[setting]["candidate_overlap_groups"] if row["group"] == "ALL")
        lines.append(
            f"| {LABELS[setting]} | {fmt(mask['A20_area_14']['mean'], 1)}/{fmt(mask['A20_area_input']['mean'], 1)} | "
            f"{fmt(mask['I20_area_14']['mean'], 1)}/{fmt(mask['I20_area_input']['mean'], 1)} | "
            f"{fmt(mask['input_area_difference']['mean'], 1)} | "
            f"{fmt(overlap['A20_I20_IoU']['mean'])} | "
            f"{fmt(mask['R_A_PLUS_area']['mean'], 2)}/{fmt(mask['R_I_PLUS_area']['mean'], 2)} |"
        )
    config = summaries["vggss_144k"]["intervention_config"]
    lines.extend(
        [
            "",
            f"- Gaussian Blur: kernel `{config['blur_kernel']}`, sigma `{config['blur_sigma']}`.",
            f"- Mean Fill: normalized value `{config['mean_fill_normalized_value']}` because ImageNet channel means map to zero.",
            f"- Random equal-area control seed: `{config['random_seed']}`.",
            "",
            "Intervention strength (A/I pairs report mean absolute perturbation, L1, and L2):",
            "",
            "| Dataset | Baseline | Keep A/I mean abs | Keep A/I L1 | Keep A/I L2 | Remove A/I mean abs | Remove A/I L1 | Remove A/I L2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        strength = summaries[setting]["intervention_strength"]
        for baseline in ("BLUR", "MEAN"):
            lines.append(
                f"| {LABELS[setting]} | {baseline} | "
                f"{fmt(strength[baseline]['KEEP_A']['mean_abs']['mean'], 6)}/"
                f"{fmt(strength[baseline]['KEEP_I']['mean_abs']['mean'], 6)} | "
                f"{fmt(strength[baseline]['KEEP_A']['L1']['mean'], 2)}/"
                f"{fmt(strength[baseline]['KEEP_I']['L1']['mean'], 2)} | "
                f"{fmt(strength[baseline]['KEEP_A']['L2']['mean'], 3)}/"
                f"{fmt(strength[baseline]['KEEP_I']['L2']['mean'], 3)} | "
                f"{fmt(strength[baseline]['REMOVE_A']['mean_abs']['mean'], 6)}/"
                f"{fmt(strength[baseline]['REMOVE_I']['mean_abs']['mean'], 6)} | "
                f"{fmt(strength[baseline]['REMOVE_A']['L1']['mean'], 2)}/"
                f"{fmt(strength[baseline]['REMOVE_I']['L1']['mean'], 2)} | "
                f"{fmt(strength[baseline]['REMOVE_A']['L2']['mean'], 3)}/"
                f"{fmt(strength[baseline]['REMOVE_I']['L2']['mean'], 3)} |"
            )

    lines.extend(
        [
            "",
            "## Semantic Score Stability",
            "",
            "| Dataset | Score | Mean | Median | Std |",
            "|---|---|---:|---:|---:|",
        ]
    )
    score_names = (
        "S_ORIGINAL",
        "S_KEEP_A_BLUR", "S_KEEP_I_BLUR", "S_REMOVE_A_BLUR", "S_REMOVE_I_BLUR",
        "S_KEEP_A_MEAN", "S_KEEP_I_MEAN", "S_REMOVE_A_MEAN", "S_REMOVE_I_MEAN",
        "CF_RANDOM_BLUR", "CF_RANDOM_MEAN",
    )
    for setting in SETTINGS:
        stability = summaries[setting]["semantic_score_stability"]
        for name in score_names:
            value = stability[name]
            lines.append(
                f"| {LABELS[setting]} | {name} | {fmt(value['mean'], 6)} | "
                f"{fmt(value['median'], 6)} | {fmt(value['std'], 6)} |"
            )

    lines.extend(
        [
            "",
            "## Counterfactual Evidence",
            "",
            "| Dataset | Evidence | AUROC IMG-better | AUPRC | AUROC/AUPRC IMG-only | BalAcc@0 | Delta>0 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["evidence_prediction"]:
            lines.append(
                f"| {LABELS[setting]} | {row['evidence']} | {fmt(row['AUROC_IMG_better'])} | "
                f"{fmt(row['AUPRC_IMG_better'])} | {fmt(row['AUROC_IMG_only'])}/"
                f"{fmt(row['AUPRC_IMG_only'])} | {fmt(row['balanced_accuracy_threshold_0'])} | "
                f"{fmt(row['fraction_delta_positive'])} |"
            )
    lines.extend(["", "Primary direction consistency:", ""])
    for name in PRIMARY:
        values = [evidence(summaries[setting], name)["AUROC_IMG_better"] for setting in SETTINGS]
        directions = ["positive" if value > 0.5 else "negative" if value < 0.5 else "neutral" for value in values]
        lines.append(
            f"- {name}: VGG `{directions[0]}` ({values[0]:.4f}), Flickr `{directions[1]}` ({values[1]:.4f}); "
            f"consistent=`{directions[0] == directions[1] and directions[0] != 'neutral'}`."
        )

    lines.extend(
        [
            "",
            "## Official Zero-Threshold Selectors",
            "",
            "| Dataset | Method | cIoU/AUC | Rescue/Hurt/Net | IMG rate | IMG-rescue retained | OGL-rescue captured | IMG-only wrong AUD | AUD-only wrong IMG |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        summary = summaries[setting]
        lines.append(
            f"| {LABELS[setting]} | AUD | {pair(summary['base_metrics']['AUD'])} | 0/0/0 | 0 | 0/N/A | 0/N/A | N/A | N/A |"
        )
        fixed = summary["fixed_IQR_transition"]
        lines.append(
            f"| {LABELS[setting]} | Fixed IQR | {pair(summary['base_metrics']['IQR'])} | "
            f"{fixed['rescue']}/{fixed['hurt']}/{fixed['net']} | N/A | N/A | N/A | N/A | N/A |"
        )
        for row in summary["selector_results"]:
            lines.append(
                f"| {LABELS[setting]} | {row['method']} | {pair(row['metrics'])} | "
                f"{row['rescue']}/{row['hurt']}/{row['net']} | {fmt(row['IMG_selection_rate'])} | "
                f"{row['IMG_rescue_retained']}/{row['IMG_rescue_total']} ({fmt(row['IMG_rescue_retention'])}) | "
                f"{row['OGL_rescue_captured']}/{row['OGL_rescue_total']} ({fmt(row['OGL_rescue_capture_rate'])}) | "
                f"{row['IMG_ONLY_wrong_AUD']} | {row['AUD_ONLY_wrong_IMG']} |"
            )
        lines.append(
            f"| {LABELS[setting]} | OGL | {pair(summary['base_metrics']['OGL'])} | N/A | N/A | N/A | N/A | N/A | N/A |"
        )
        lines.append(
            f"| {LABELS[setting]} | Sample Oracle | {pair(summary['sample_oracle'])} | N/A | N/A | N/A | N/A | N/A | N/A |"
        )

    lines.extend(
        [
            "",
            "## Mechanism by Group",
            "",
            "| Dataset | Group | Count | Delta CF Blur mean/median | Delta CF Mean mean/median | Blur/Mean IMG fraction | A20-I20 IoU | Drop A/I Blur |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["mechanism_groups"]:
            lines.append(
                f"| {LABELS[setting]} | {row['group']} | {row['count']} | "
                f"{fmt(row['DELTA_CF_BLUR']['mean'], 6)}/{fmt(row['DELTA_CF_BLUR']['median'], 6)} | "
                f"{fmt(row['DELTA_CF_MEAN']['mean'], 6)}/{fmt(row['DELTA_CF_MEAN']['median'], 6)} | "
                f"{fmt(row['fraction_BLUR_IMG'])}/{fmt(row['fraction_MEAN_IMG'])} | "
                f"{fmt(row['A20_I20_overlap']['mean'])} | "
                f"{fmt(row['drop_A_BLUR']['mean'], 6)}/{fmt(row['drop_I_BLUR']['mean'], 6)} |"
            )

    lines.extend(["", "Signed disagreement and SHRINK diagnostics:", ""])
    for setting in SETTINGS:
        correction = summaries[setting]["correction_type_analysis"]
        shrink = correction["SHRINK"]
        special = correction["IMG_ONLY_SHRINK_signed_disagreement"]
        lines.append(
            f"- {LABELS[setting]} IMG-better correction counts `{correction['counts']}`. "
            f"SHRINK Delta CF Blur/Mean mean `{fmt(shrink['DELTA_CF_BLUR']['mean'], 6)}`/"
            f"`{fmt(shrink['DELTA_CF_MEAN']['mean'], 6)}`, selected IMG fraction "
            f"`{fmt(shrink['fraction_BLUR_IMG'])}`/`{fmt(shrink['fraction_MEAN_IMG'])}`."
        )
        lines.append(
            f"- {LABELS[setting]} IMG_ONLY+SHRINK count `{special['count']}`: AUD-extra exterior fraction "
            f"`{fmt(special['AUD_extra_exterior_fraction']['mean'])}`; removal drop Blur/Mean "
            f"`{fmt(special['DROP_A_EXTRA_BLUR']['mean'], 6)}`/"
            f"`{fmt(special['DROP_A_EXTRA_MEAN']['mean'], 6)}`; density "
            f"`{fmt(special['DROP_DENSITY_A_BLUR']['mean'], 8)}`/"
            f"`{fmt(special['DROP_DENSITY_A_MEAN']['mean'], 8)}`."
        )

    lines.extend(
        [
            "",
            "## Post-hoc Threshold Transfer",
            "",
            "GT-optimal source thresholds are applied unchanged to the other dataset. These results are not official.",
            "",
            "| Evidence | Source -> Target | Threshold | Target cIoU/AUC | Rescue/Hurt/Net | IMG rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in PRIMARY:
        for key in ("vggss_144k_to_flickr_144k", "flickr_144k_to_vggss_144k"):
            row = transfer[name][key]
            lines.append(
                f"| {name} | {LABELS[row['source']]} -> {LABELS[row['target']]} | "
                f"{fmt(row['threshold'], 7)} | {pair(row['metrics'])} | "
                f"{row['rescue']}/{row['hurt']}/{row['net']} | {fmt(row['IMG_selection_rate'])} |"
            )

    lines.extend(["", "## Qualitative Selection", ""])
    for setting in SETTINGS:
        lines.append(
            f"- {LABELS[setting]} deterministic samples: `{summaries[setting]['qualitative_selection']}`."
        )
    lines.extend(
        [
            "- Correct CF selections exist, but their score margins are often only 1e-3 to 1e-2. Visually similar keep/remove interventions can therefore reverse the branch choice.",
            "- Failure examples show that equal-area masks still preserve almost the same object core. The semantic slot score reacts strongly to the intervention artifact but weakly to the subtle extent difference that determines localization IoU.",
            "- Blur and Mean Fill can choose different branches for the same hard case, matching their inconsistent full-dataset direction/calibration.",
            "",
            "## Decision",
            "",
            "**Case D - Counterfactual Evidence Fails.**",
            "",
            "- VGG is at chance: CF Blur AUROC 0.5058 and CF Mean 0.4949. Mean also has opposite direction between VGG and Flickr. Flickr's 0.5241/0.5485 does not approach the requested ~0.60 evidence level.",
            "- Threshold-zero selectors all hurt VGG: CF Blur 0.4203 with Net -34, CF Mean 0.4184 with Net -44, and consensus 0.4221 with Net -25, versus AUD 0.4269.",
            "- Flickr CF Blur gives a small 0.8120 -> 0.8240 gain with Net +3, but it is dataset-specific and is not supported by VGG. Mean and consensus do not improve Flickr cIoU.",
            "- IMG-rescue retention is insufficient and unstable: Blur retains 56/104 on VGG and 8/11 on Flickr; Mean retains 51/104 and 4/11; consensus retains 38/104 and 4/11.",
            "- IMG_ONLY+SHRINK has high AUD-extra exterior fraction, but deletion drop is not uniquely low across groups or interventions. The hypothesized context-leakage signal is not expressed reliably in the frozen semantic metric.",
            "- Cross-dataset threshold transfer fails: VGG thresholds nearly suppress all Flickr IMG choices, while Flickr thresholds reduce VGG below AUD with negative Net.",
            "",
            "**4.x next step: stop the hand-designed internal selector evidence line. Reconsider the supervision source or training objective; do not start 4.3 from this counterfactual score.**",
            "",
            "No 4.3 experiment was implemented or started.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    arguments = parse_args()
    summaries = {
        setting: load_json(arguments.results_root / setting / "summary.json")
        for setting in SETTINGS
    }
    rows = {
        setting: load_rows(arguments.results_root / setting / "per_sample_metrics.csv")
        for setting in SETTINGS
    }
    for setting, summary in summaries.items():
        if not summary["completed_full_dataset"]:
            raise RuntimeError(f"Partial result: {setting}")
        if not summary["reproduction_4_1"]["passed"]:
            raise RuntimeError(f"4.1 reproduction failed: {setting}")
        if not summary["zero_training_audit"]["all_checkpoint_hashes_and_mtimes_unchanged"]:
            raise RuntimeError(f"Checkpoint changed: {setting}")

    transfer: dict[str, Any] = {}
    for name in PRIMARY:
        transfer[name] = {}
        for source, target in (
            ("vggss_144k", "flickr_144k"),
            ("flickr_144k", "vggss_144k"),
        ):
            threshold = evidence(summaries[source], name)["optimal_threshold_diagnostic"]["threshold"]
            result = transfer_metrics(rows[target], name, float(threshold))
            transfer[name][f"{source}_to_{target}"] = {
                "source": source,
                "target": target,
                **result,
            }
    common.write_json(arguments.results_root / "transfer_threshold_diagnostic.json", transfer)
    arguments.report.write_text(report_text(summaries, transfer), encoding="utf-8")
    print(arguments.report)


if __name__ == "__main__":
    main()
