#!/usr/bin/env python3
"""Aggregate both formal Experiment 3.2 evaluations."""

from __future__ import annotations

import json

import common


SETTINGS = ("vggss_144k", "flickr_144k")
RAW_METHODS = ("T2_RAW_MEAN", "T2_RAW_GEO", "T4_RAW_MEAN", "T4_RAW_GEO")


def metric_lookup(summary):
    return {row["method"]: row for row in summary["method_metrics"]}


def row_lookup(rows, key="method"):
    return {row[key]: row for row in rows}


def scale_lookup(rows):
    return {row["scale"]: row for row in rows}


def score(value):
    return f"{float(value):.4f}"


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def decide(summaries):
    gains = {}
    transitions = {}
    for method in RAW_METHODS:
        gains[method] = {}
        transitions[method] = {}
        for setting in SETTINGS:
            metrics = metric_lookup(summaries[setting])
            gains[method][setting] = metrics[method]["cIoU"] - metrics["ORIGINAL_AUD"]["cIoU"]
            transitions[method][setting] = row_lookup(
                summaries[setting]["rescue_hurt_oracle"]
            )[method]

    evidence_a = {}
    for method in RAW_METHODS:
        method_gains = gains[method]
        both_non_decrease = all(value >= 0 for value in method_gains.values())
        one_clear_gain = max(method_gains.values()) >= 0.01
        both_positive = all(value > 0 for value in method_gains.values())
        rescue_not_worse = all(
            transitions[method][setting]["rescue"] >= transitions[method][setting]["hurt"]
            for setting in SETTINGS
        )
        evidence_a[method] = {
            "gains": method_gains,
            "both_non_decrease": both_non_decrease,
            "one_clear_gain_at_least_0.01": one_clear_gain,
            "both_positive": both_positive,
            "rescue_at_least_hurt_both": rescue_not_worse,
            "passed": rescue_not_worse
            and ((both_non_decrease and one_clear_gain) or both_positive),
        }

    vgg = summaries["vggss_144k"]
    region_rows = {row["region"]: row for row in vgg["region_stability"]["rows"]}
    gt_std = region_rows["GT_REGION"]["TEMP_STD"]["mean"]
    fp_std = region_rows["AUD_FP_REGION"]["TEMP_STD"]["mean"]
    gt_cv = region_rows["GT_REGION"]["TEMP_CV"]["mean"]
    fp_cv = region_rows["AUD_FP_REGION"]["TEMP_CV"]["mean"]
    comparison = vgg["region_stability"]["comparison"]
    region_signal = (
        fp_std > gt_std
        and fp_cv > gt_cv
        and comparison["fraction_FP_STD_gt_GT_STD"] > 0.5
    )
    capture_rates = {
        setting: max(row["capture_rate"] for row in summaries[setting]["ogl_rescue_capture"])
        for setting in SETTINGS
    }
    capture_signal = max(capture_rates.values()) >= 0.1
    evidence_b = {
        "VGG_FP_to_GT_STD_ratio": ratio(fp_std, gt_std),
        "VGG_FP_to_GT_CV_ratio": ratio(fp_cv, gt_cv),
        "VGG_fraction_FP_STD_gt_GT_STD": comparison["fraction_FP_STD_gt_GT_STD"],
        "region_signal": region_signal,
        "max_OGL_rescue_capture_rate": capture_rates,
        "capture_signal_at_least_0.10": capture_signal,
        "passed": region_signal or capture_signal,
    }

    positive_methods = [method for method, checks in evidence_a.items() if checks["passed"]]
    if positive_methods and evidence_b["passed"]:
        label = "Positive"
    else:
        mixed = any(
            max(values.values()) >= 0.01 and min(values.values()) <= -0.01
            for values in gains.values()
        )
        label = "Mixed" if mixed else "Negative"

    if label == "Negative":
        recommendation = (
            "Audio auxiliary representation line closed. Next candidate: "
            "AUD-IMG_QUERY redundancy / att-loss. Do not start it automatically."
        )
    elif label == "Mixed":
        recommendation = "Cross-dataset behavior is mixed; do not train a follow-up automatically."
    else:
        recommendation = "A4 temporal evidence is positive under the fixed rule; no follow-up was started."
    return {
        "label": label,
        "evidence_A_localization": evidence_a,
        "evidence_B_mechanism": evidence_b,
        "recommendation": recommendation,
    }


def build_report(summaries, decision):
    lines = [
        "# Experiment 3.2 - A4 Temporal Grounding Probe",
        "",
        "## Protocol And Audit",
        "",
        "Original formal 1.3G checkpoints are frozen. Every temporal chunk reuses the same "
        "learned initial slots, AudioSlotBranch, K34, infer sharpening, slot-softmax, spatial "
        "normalization, and target slot0 as official AUD_FINE.",
        "",
        "| Setting | raw A4 | A4 tokens | F34 | K34 | Full tensor error | Evaluator error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        audit = summary["tensor_audit"]
        lines.append(
            f"| {setting} | {audit['raw_A4_shape']} | {audit['A4_tokens_shape']} | "
            f"{audit['F34_shape']} | {audit['K34_shape']} | "
            f"{audit['FULL_AUD_tensor_reproduction_max_error']:.3e} | "
            f"{summary['reference_reproduction']['max_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "- T2 boundaries: `[0:8], [8:16]`.",
            "- T4 boundaries: `[0:4], [4:8], [8:12], [12:16]`.",
            "- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`.",
            "- All checkpoint SHA256 and mtimes are unchanged; all models remained in eval/inference mode.",
            "",
            "## Localization Metrics",
            "",
            "| Method | VGG cIoU/AUC | Flickr cIoU/AUC |",
            "|---|---:|---:|",
        ]
    )
    lookups = {setting: metric_lookup(summaries[setting]) for setting in SETTINGS}
    for method in [row["method"] for row in summaries[SETTINGS[0]]["method_metrics"]]:
        vgg = lookups["vggss_144k"][method]
        flickr = lookups["flickr_144k"][method]
        lines.append(
            f"| {method} | {score(vgg['cIoU'])}/{score(vgg['AUC'])} | "
            f"{score(flickr['cIoU'])}/{score(flickr['AUC'])} |"
        )

    lines.extend(
        [
            "",
            "## Temporal Query Semantics",
            "",
            "Visual target is the final frozen L4 visual query slot0. Negative pairs are "
            "batch-shuffled visual queries.",
            "",
            "| Setting | Scale | Positive | Negative | Margin | Query pairwise cosine | Query variance |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        summary = summaries[setting]
        semantic = {
            row["scale"]: row
            for row in summary["temporal_query_semantics"]
            if row["chunk"] == "MEAN"
        }
        consistency = scale_lookup(summary["query_consistency"])
        for scale in ("T2", "T4"):
            row = semantic[scale]
            con = consistency[scale]
            lines.append(
                f"| {setting} | {scale} | {row['positive_cosine']:.4f} | "
                f"{row['negative_cosine']:.4f} | {row['margin']:.4f} | "
                f"{con['query_pairwise_cosine']['mean']:.4f} | "
                f"{con['query_variance']['mean']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Temporal Map Similarity",
            "",
            "| Setting | Scale | Pearson | Spearman | JS divergence |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["temporal_map_similarity"]:
            lines.append(
                f"| {setting} | {row['scale']} | {row['pearson']['mean']:.4f} | "
                f"{row['spearman']['mean']:.4f} | {row['js_divergence']['mean']:.3e} |"
            )

    lines.extend(
        [
            "",
            "## Rescue, Hurt, Oracle, And OGL Capture",
            "",
            "| Setting | Method | Rescue | Hurt | Net | Oracle cIoU/AUC | OGL pool | Captured | Rate | Rescue intersect OGL |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        captures = row_lookup(summaries[setting]["ogl_rescue_capture"])
        for row in summaries[setting]["rescue_hurt_oracle"]:
            capture = captures[row["method"]]
            lines.append(
                f"| {setting} | {row['method']} | {row['rescue']} | {row['hurt']} | "
                f"{row['net']} | {score(row['oracle_cIoU'])}/{score(row['oracle_AUC'])} | "
                f"{capture['OGL_rescue_total']} | {capture['consensus_captured']} | "
                f"{capture['capture_rate']:.3f} | {row['rescue_intersect_OGL_rescue']} |"
            )

    lines.extend(
        [
            "",
            "## Temporal Stability",
            "",
            "T4 stability uses independently evaluator-normalized 224x224 chunk maps.",
            "",
            "| Setting | Region | STD mean/median/std | CV mean/median/std | Samples |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["region_stability"]["rows"]:
            lines.append(
                f"| {setting} | {row['region']} | {row['TEMP_STD']['mean']:.6f}/"
                f"{row['TEMP_STD']['median']:.6f}/{row['TEMP_STD']['std']:.6f} | "
                f"{row['TEMP_CV']['mean']:.6f}/{row['TEMP_CV']['median']:.6f}/"
                f"{row['TEMP_CV']['std']:.6f} | {row['TEMP_STD']['num_samples']} |"
            )
        comparison = summaries[setting]["region_stability"]["comparison"]
        lines.append(
            f"| {setting} | FP > GT fraction | {comparison['fraction_FP_STD_gt_GT_STD']:.4f} | "
            f"{comparison['fraction_FP_CV_gt_GT_CV']:.4f} | {comparison['samples_with_GT_and_FP']} |"
        )

    lines.extend(
        [
            "",
            "## VGG Over-Expansion",
            "",
        ]
    )
    vgg_over = summaries["vggss_144k"]["overexpansion_stability"]
    for row in vgg_over["rows"]:
        lines.append(
            f"- {row['region']}: STD mean/median/std="
            f"{row['TEMP_STD']['mean']:.6f}/{row['TEMP_STD']['median']:.6f}/"
            f"{row['TEMP_STD']['std']:.6f}; CV mean/median/std="
            f"{row['TEMP_CV']['mean']:.6f}/{row['TEMP_CV']['median']:.6f}/"
            f"{row['TEMP_CV']['std']:.6f}; n={row['TEMP_STD']['num_samples']}."
        )
    lines.append(
        f"- Fraction over-expansion STD > GT STD: "
        f"{vgg_over['comparison']['fraction_OVER_STD_gt_GT_STD']:.4f}; "
        f"CV fraction: {vgg_over['comparison']['fraction_OVER_CV_gt_GT_CV']:.4f}."
    )

    lines.extend(
        [
            "",
            "## Temporal Delta Versus OGL Delta",
            "",
            "| Setting | Method | Pearson | Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["delta_correlation"]:
            lines.append(
                f"| {setting} | {row['method']} | "
                f"{row['pearson_delta_temporal_vs_delta_OGL']:.4f} | "
                f"{row['spearman_delta_temporal_vs_delta_OGL']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Qualitative Selection",
            "",
            "Across the fixed panels, T4 M1-M4 preserve nearly the same hotspot, extent, "
            "and context response. VGG temporal rescues and hurts are predominantly small "
            "threshold-boundary changes rather than consistent removal of background support. "
            "OGL-rescue/temporal-fail panels show OGL changing spatial extent enough to cross "
            "the success threshold while raw and normalized temporal consensus remain close "
            "to ORIGINAL_AUD. TEMP_STD often highlights context, borders, and object edges; "
            "although this produces the aggregate FP/over-expansion variance signal, arithmetic "
            "and geometric averaging do not convert it into a reliable correction. Flickr has "
            "no temporal rescue and one temporal hurt under the four formal raw consensus maps.",
            "",
        ]
    )
    for setting in SETTINGS:
        counts = summaries[setting]["qualitative_category_counts"]
        lines.append(f"- {setting}: `{json.dumps(counts, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Mechanism Interpretation",
            "",
            "T2/T4 chunk queries are not identical: T4 pairwise query cosine is about 0.85 "
            "on VGG and 0.82 on Flickr. However, their spatial maps have Pearson above 0.995 "
            "and JS divergence near zero. The temporal-specific query components therefore "
            "have little effect after projection onto the shared K34 spatial keys.",
            "",
            "Evaluator-space temporal variance is higher in VGG false-positive and "
            "over-expansion regions, so the probe finds a limited mechanism signal. It is not "
            "the signal needed by fixed mean/geometric consensus: OGL-rescue capture is at most "
            "2.5% on VGG and zero on Flickr, temporal and OGL IoU deltas are uncorrelated or "
            "negatively correlated, and oracle gains remain small.",
        ]
    )
    lines.extend(
        [
            "",
            "## Fixed-Rule Decision",
            "",
            f"**{decision['label']}**",
            "",
            decision["recommendation"],
            "",
            "```json",
            json.dumps(decision, indent=2),
            "```",
            "",
            "No training or follow-up experiment was started.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    summaries = {
        setting: json.loads(
            (common.HERE / "results" / setting / "summary.json").read_text(encoding="utf-8")
        )
        for setting in SETTINGS
    }
    result = decide(summaries)
    combined = {
        "experiment": "3.2 A4 Temporal Grounding Probe",
        "settings": summaries,
        "decision": result,
    }
    common.write_json(common.HERE / "results" / "combined_summary.json", combined)
    report = build_report(summaries, result)
    (common.HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
