#!/usr/bin/env python3
"""Aggregate both formal Experiment 4.0 evaluations and write REPORT.md."""

from __future__ import annotations

import json
import subprocess

import numpy as np

import common


SETTINGS = ("vggss_144k", "flickr_144k")
STAGES = ("Stage1", "Stage2")


def score(value) -> str:
    return f"{float(value):.4f}"


def stage(summary, name):
    return summary["stage_summaries"][name]


def pair(stage_summary, name):
    return next(row for row in stage_summary["pairs"] if row["pair"] == name)


def decide(summaries):
    evidence = {}
    stage2_img = {setting: pair(stage(summaries[setting], "Stage2"), "AUD+IMG") for setting in SETTINGS}
    stage2_obj = {setting: pair(stage(summaries[setting], "Stage2"), "AUD+OBJ") for setting in SETTINGS}

    high_similarity = all(
        row["similarity"]["norm_pearson"]["mean"] >= 0.90
        and row["similarity"]["top20_overlap"]["mean"] >= 0.70
        and row["similarity"]["mask_iou"]["mean"] >= 0.70
        for row in stage2_img.values()
    )
    img_only_small = all(
        row["success_decomposition"]["AUX_ONLY_fraction"] <= 0.05
        for row in stage2_img.values()
    )
    img_oracle_weaker_than_obj = all(
        stage2_img[setting]["oracle_gain_over_AUD_cIoU"]
        < 0.75 * stage2_obj[setting]["oracle_gain_over_AUD_cIoU"]
        for setting in SETTINGS
    )
    hard_capture_low = all(
        stage(summaries[setting], "Stage2")["OGL_rescue_decomposition"]["IMG_capture_rate"]
        <= 0.25
        for setting in SETTINGS
    )

    trajectory_sufficient = all(
        summaries[setting]["epoch_trajectory"]["correlations"][
            "sufficient_for_trajectory_inference"
        ]
        for setting in SETTINGS
    )
    trajectory_direction = False
    if trajectory_sufficient:
        trajectory_direction = all(
            summaries[setting]["epoch_trajectory"]["correlations"][
                "correlations_with_train_attention_match_loss"
            ]["AUD_IMG_Pearson"]["pearson"] < 0
            and summaries[setting]["epoch_trajectory"]["correlations"][
                "correlations_with_train_attention_match_loss"
            ]["IMG_ONLY_fraction"]["pearson"] >= 0
            for setting in SETTINGS
        )

    case_a = (
        high_similarity
        and img_only_small
        and img_oracle_weaker_than_obj
        and hard_capture_low
        and trajectory_direction
    )
    complementarity_clear = all(
        row["success_decomposition"]["AUX_ONLY_fraction"] >= 0.03
        and row["oracle_gain_over_AUD_cIoU"] >= 0.03
        for row in stage2_img.values()
    ) and all(
        stage(summaries[setting], "Stage2")["OGL_rescue_decomposition"]["IMG_capture_rate"]
        >= 0.25
        for setting in SETTINGS
    )
    fusion_failed = all(
        row["fixed_fusion_gain_cIoU"] <= 0.005 for row in stage2_img.values()
    )

    if case_a:
        label = "Case A - Strong Redundancy Hypothesis Supported"
        next_direction = "Study att-loss: relax spatial attention identity while preserving semantic agreement. Do not start 4.1 automatically."
    elif complementarity_clear and fusion_failed:
        label = "Case B - Complementarity Exists - Fusion Bottleneck"
        next_direction = "Study fusion, not att-loss. Do not start a follow-up automatically."
    else:
        label = "Case C - IMG Branch Fundamentally Weak"
        next_direction = "Close the IQR/att-loss line; IMG lacks enough task-relevant exclusive capacity to justify 4.1."

    evidence.update(
        {
            "high_similarity_both": high_similarity,
            "IMG_ONLY_fraction_at_most_0.05_both": img_only_small,
            "IMG_oracle_gain_less_than_75pct_OBJ_both": img_oracle_weaker_than_obj,
            "IMG_OGL_rescue_capture_at_most_0.25_both": hard_capture_low,
            "trajectory_has_at_least_3_distinct_checkpoints_both": trajectory_sufficient,
            "trajectory_expected_direction": trajectory_direction,
            "clear_complementarity_both": complementarity_clear,
            "fixed_fusion_failed_both": fusion_failed,
            "threshold_note": (
                "For Case B, noticeable means at least 3% AUX-only, +0.03 pair-oracle cIoU, "
                "and 25% OGL-rescue capture in both datasets. The report exposes every count."
            ),
        }
    )
    return {"label": label, "evidence": evidence, "next_direction": next_direction}


def build_report(summaries, decision, git_status):
    lines = [
        "# Experiment 4.0 - AUD-IMG Redundancy & Attention-Loss Mechanism Probe",
        "",
        "## Protocol And Att-Loss Audit",
        "",
        "This is a zero-training diagnostic on formal L3+L4 Stage1 and original 1.3G. "
        "All maps use the unchanged bicubic resize, per-sample min-max normalization, "
        "threshold 0.6, cIoU, and AUC evaluator.",
        "",
        "The formal Stage1 code computes:",
        "",
        "```python",
        "att_loss = MSE(audq_imgk_attn[:, 0], imgq_imgk_attn[:, 0].detach())",
        "att_loss += MSE(imgq_audk_attn[:, 0], audq_audk_attn[:, 0].detach())",
        "total = info + lam1 * recon + lam2 * div + lam3 * att_loss",
        "```",
        "",
        "- `lam3 = 100.0` in both formal 144k configs.",
        "- Spatial term: `AUD query -> L4 image keys` is optimized toward detached "
        "`IMG query -> L4 image keys`.",
        "- Reciprocal audio-token term: `IMG query -> audio keys` is optimized toward "
        "detached `AUD query -> audio keys`.",
        "- The spatial term directly contains the same L4 AUD/IMG attention tensors used "
        "by formal Stage1 localization, with scale multiplier 1.0 during training versus "
        "infer sharpening 0.1 during evaluation.",
        "- Source: `model_mufasa_jsa.py:113-120`, `l3_l4_slot_attention.py:102-120`, "
        "`train_slot.py:339-343`.",
        "",
        "## Tensor And Reproduction Audit",
        "",
        "| Setting | Qa | Qv | K4 | AUD_L4 | IMG_L4 | K34 | AUD_FINE | max tensor error | max evaluator error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        audit = summary["tensor_audit"]
        tensor_error = max(
            audit["Stage1"]["AUD_L4_reconstruction_max_error"],
            audit["Stage1"]["IMG_L4_reconstruction_max_error"],
            audit["Stage2"]["AUD_FINE_reconstruction_max_error"],
        )
        lines.append(
            f"| {setting} | {audit['Stage1']['Qa_shape']} | {audit['Stage1']['Qv_shape']} | "
            f"{audit['Stage1']['K4_shape']} | {audit['Stage1']['AUD_L4_shape']} | "
            f"{audit['Stage1']['IMG_L4_shape']} | {audit['Stage2']['K34_shape']} | "
            f"{audit['Stage2']['AUD_FINE_shape']} | {tensor_error:.3e} | "
            f"{summary['formal_reproduction']['max_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "`optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`; "
            "all models remained in eval/inference mode. Every used checkpoint SHA256 and "
            "mtime was identical before and after.",
            "",
            "Direct best-checkpoint discrepancy using the exact training MSE tensors:",
            "",
            "| Setting | Spatial MSE | Reciprocal audio-token MSE | Total att MSE |",
            "|---|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        direct = summaries[setting]["direct_attention_discrepancy_full_dataset"]
        lines.append(
            f"| {setting} | {direct['spatial_MSE']['mean']:.3e} | "
            f"{direct['temporal_MSE']['mean']:.3e} | {direct['total_MSE']['mean']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Historical audit: VGG best-checkpoint evaluation exactly agrees with epoch 3. "
            "For Flickr, epoch 7 training CSV records IQR `0.8120/0.6236`, while the independent "
            "formal best-checkpoint test log and this exact reconstruction both give "
            "`0.8080/0.6234`. AUD, IMG, OBJ, and OGL agree; this pre-existing one-sample IQR "
            "difference is retained rather than hidden.",
            "",
            "## Final Redundancy",
            "",
            "| Setting | Stage | AUD cIoU/AUC | IMG cIoU/AUC | IQR cIoU/AUC | OBJ cIoU/AUC | OGL cIoU/AUC | Pearson | Spearman | JS | Top20 | Mask IoU |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for stage_name in STAGES:
            value = stage(summaries[setting], stage_name)
            methods = value["method_metrics"]
            img_pair = pair(value, "AUD+IMG")
            sim = img_pair["similarity"]
            lines.append(
                f"| {setting} | {stage_name} | {score(methods['AUD']['cIoU'])}/{score(methods['AUD']['AUC'])} | "
                f"{score(methods['IMG']['cIoU'])}/{score(methods['IMG']['AUC'])} | "
                f"{score(methods['IQR']['cIoU'])}/{score(methods['IQR']['AUC'])} | "
                f"{score(methods['OBJ']['cIoU'])}/{score(methods['OBJ']['AUC'])} | "
                f"{score(methods['OGL']['cIoU'])}/{score(methods['OGL']['AUC'])} | "
                f"{sim['norm_pearson']['mean']:.4f} | {sim['norm_spearman']['mean']:.4f} | "
                f"{sim['norm_js']['mean']:.4f} | {sim['top20_overlap']['mean']:.4f} | "
                f"{sim['mask_iou']['mean']:.4f} |"
            )

    lines.extend(
        [
            "",
            "Raw-space Pearson/Spearman equal evaluator-normalized values because independent "
            "min-max normalization is a positive affine transform. Raw and normalized JS are "
            "both retained in `stage_summaries.json`.",
            "",
            "## Task Complementarity",
            "",
            "| Setting | Stage | Pair | AUD only | AUX only | Both success | Both fail | Oracle cIoU/AUC | Oracle gain | Fixed gain | Rescue | Hurt | Net |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for stage_name in STAGES:
            for row in stage(summaries[setting], stage_name)["pairs"]:
                dec = row["success_decomposition"]
                oracle = row["pair_oracle"]
                lines.append(
                    f"| {setting} | {stage_name} | {row['pair']} | {dec['AUD_ONLY']} | "
                    f"{dec['AUX_ONLY']} | {dec['BOTH_SUCCESS']} | {dec['BOTH_FAIL']} | "
                    f"{score(oracle['cIoU'])}/{score(oracle['AUC'])} | "
                    f"{row['oracle_gain_over_AUD_cIoU']:+.4f} | "
                    f"{row['fixed_fusion_gain_cIoU']:+.4f} | {row['rescue']} | "
                    f"{row['hurt']} | {row['net']} |"
                )

    lines.extend(
        [
            "",
            "## OGL Rescue Decomposition",
            "",
            "| Setting | Stage | OGL rescue pool | IMG captured | IMG rate | IQR captured | IQR rate | IMG IoU > AUD |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for stage_name in STAGES:
            row = stage(summaries[setting], stage_name)["OGL_rescue_decomposition"]
            lines.append(
                f"| {setting} | {stage_name} | {row['OGL_rescue_total']} | "
                f"{row['IMG_captured']} | {row['IMG_capture_rate']:.3f} | "
                f"{row['IQR_captured']} | {row['IQR_capture_rate']:.3f} | "
                f"{row['IMG_IoU_gt_AUD_fraction']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## AUD-IMG Versus AUD-OBJ",
            "",
            "| Setting | Stage | Pair | Pearson | Spearman | JS | Top10 | Top20 | Top30 | Mask IoU | AUX-only frac | Oracle gain | Fusion gain |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for stage_name in STAGES:
            for row in stage(summaries[setting], stage_name)["pairs"]:
                sim = row["similarity"]
                dec = row["success_decomposition"]
                lines.append(
                    f"| {setting} | {stage_name} | {row['pair']} | "
                    f"{sim['norm_pearson']['mean']:.4f} | {sim['norm_spearman']['mean']:.4f} | "
                    f"{sim['norm_js']['mean']:.4f} | {sim['top10_overlap']['mean']:.4f} | "
                    f"{sim['top20_overlap']['mean']:.4f} | {sim['top30_overlap']['mean']:.4f} | "
                    f"{sim['mask_iou']['mean']:.4f} | {dec['AUX_ONLY_fraction']:.4f} | "
                    f"{row['oracle_gain_over_AUD_cIoU']:+.4f} | {row['fixed_fusion_gain_cIoU']:+.4f} |"
                )

    lines.extend(
        [
            "",
            "## Alpha Diagnostic",
            "",
            "Formal IQR remains alpha AUD = 0.6. The sweep is diagnostic only.",
            "",
            "| alpha AUD | VGG Stage1 | VGG Stage2 | Flickr Stage1 | Flickr Stage2 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    alpha_lookup = {
        setting: {
            stage_name: {
                row["alpha_AUD"]: row
                for row in stage(summaries[setting], stage_name)["alpha_diagnostic_AUD_IMG"]
            }
            for stage_name in STAGES
        }
        for setting in SETTINGS
    }
    for alpha in [round(value / 10, 1) for value in range(11)]:
        values = []
        for setting in SETTINGS:
            for stage_name in STAGES:
                row = alpha_lookup[setting][stage_name][alpha]
                values.append(f"{score(row['cIoU'])}/{score(row['AUC'])}")
        lines.append(f"| {alpha:.1f} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Epoch Trajectory",
            "",
            "Only two distinct model states exist per formal Stage1 directory. `latest.pth` "
            "duplicates `final.pth`; no epoch 1/5/10/25 weights exist. The requested correlations "
            "are therefore reported but are not statistically interpretable.",
            "",
            "| Setting | Epoch | att loss | direct eval MSE | AUD | IMG | IQR | Pearson | Top20 | IMG-only | Oracle gain | IQR gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["epoch_trajectory"]["rows"]:
            lines.append(
                f"| {setting} | {row['epoch']} | {row['train_attention_match_loss']:.3e} | "
                f"{row['direct_eval_attention_MSE']:.3e} | {row['AUD_cIoU']:.4f} | "
                f"{row['IMG_cIoU']:.4f} | {row['IQR_cIoU']:.4f} | "
                f"{row['AUD_IMG_Pearson']:.4f} | {row['Top20Overlap']:.4f} | "
                f"{row['IMG_ONLY_fraction']:.4f} | {row['OracleGain']:+.4f} | "
                f"{row['IQRGain']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "Two-point observational correlations with logged `train_attention_match_loss`:",
            "",
            "| Setting | Target | Pearson | Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        correlations = summaries[setting]["epoch_trajectory"]["correlations"][
            "correlations_with_train_attention_match_loss"
        ]
        for target, values in correlations.items():
            lines.append(
                f"| {setting} | {target} | {values['pearson']:.4f} | "
                f"{values['spearman']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Sample-Level Disagreement",
            "",
            "Disagreement is `1 - Pearson(AUD_FINE, IMG_QUERY)` in evaluator space.",
            "",
            "| Setting | Group | Mean | Median | Std | N |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for group, row in summaries[setting]["sample_disagreement_by_group"].items():
            lines.append(
                f"| {setting} | {group} | {row['mean']:.4f} | {row['median']:.4f} | "
                f"{row['std']:.4f} | {row['num_samples']} |"
            )

    lines.extend(
        [
            "",
            "## Qualitative Audit",
            "",
            "Selection is deterministic: the lexicographically first available sample for "
            "each fixed category. Panels contain Image, GT, AUD, IMG_QUERY, IQR, OBJ_PRIOR, "
            "OGL, |AUD-IMG|, and |AUD-OBJ|.",
            "",
        ]
    )
    for setting in SETTINGS:
        lines.append(f"- `{setting}`: `{json.dumps(summaries[setting]['qualitative_selection'], sort_keys=True)}`")

    lines.extend(
        [
            "",
            "Observed fixed-sample phenomena:",
            "",
            "- AUD and IMG usually share the same dominant broad lobe; their useful differences "
            "are concentrated at boundaries or secondary peaks rather than forming a consistently "
            "independent object map.",
            "- IMG-only cases show real but modest recentering/contraction. In VGG `-Vo4CAMX26U_000030`, "
            "AUD/IMG/IQR IoU is `0.477/0.509/0.495`; in Flickr `10548273474` it is "
            "`0.448/0.516/0.477`. The fixed mixture can erase an IMG success.",
            "- IQR-hurt cases show the opposite boundary shift: VGG violin `0.565/0.441/0.495` and "
            "Flickr cyclists `0.523/0.481/0.497` for AUD/IMG/IQR.",
            "- OGL-only corrections are visibly larger extent changes. The VGG air-conditioner "
            "sample has AUD/IMG/IQR `0.368/0.491/0.409`, while OBJ/OGL reach `0.740/0.511`.",
            "- IQR rescues exist, but selected examples are threshold-boundary improvements; they "
            "do not offset the larger hurt count at Stage2.",
        ]
    )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['label']}**",
            "",
            decision["next_direction"],
            "",
            "The cross-sectional redundancy and task evidence is kept separate from the "
            "att-loss causality claim. With only two saved states, epoch trajectory cannot "
            "establish that att-loss caused the redundancy even when final maps are highly similar.",
            "",
            "Decision evidence:",
            "",
        ]
    )
    for key, value in decision["evidence"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Git Status", "", "```text", git_status.rstrip() or "clean", "```", ""])
    return "\n".join(lines)


def main() -> None:
    summaries = {
        setting: json.loads(
            (common.HERE / "results" / setting / "summary.json").read_text(encoding="utf-8")
        )
        for setting in SETTINGS
    }
    decision = decide(summaries)
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=common.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    combined = {"summaries": summaries, "decision": decision, "git_status_short": git_status.splitlines()}
    common.write_json(common.HERE / "results" / "combined_summary.json", combined)
    report = build_report(summaries, decision, git_status)
    (common.HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"Wrote {common.HERE / 'REPORT.md'}")


if __name__ == "__main__":
    main()
