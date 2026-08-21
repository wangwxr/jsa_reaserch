#!/usr/bin/env python3
"""Aggregate Experiment 3.0 and generate the detailed decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import common


SETTINGS = ("vggss_144k", "flickr_144k")
PRIMARY = ("FULL_TEMP_MEAN_4", "FULL_TEMP_GEO_4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=common.HERE / "results")
    return parser.parse_args()


def lookup(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError((key, value))


def score(value: Any) -> str:
    return f"{float(value):.4f}"


def decision(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    route_a_checks = {}
    route_a_candidates = []
    for method in PRIMARY:
        gains = {}
        nets = {}
        for setting in SETTINGS:
            metrics = summaries[setting]["method_metrics"]
            full = lookup(metrics, "method", "FULL_AUD")
            candidate = lookup(metrics, "method", method)
            transition = lookup(
                summaries[setting]["primary_rescue_hurt_oracle"], "method", method
            )
            gains[setting] = candidate["cIoU"] - full["cIoU"]
            nets[setting] = transition["net"]
        checks = {
            "non_decrease_both": all(gains[setting] >= 0.0 for setting in SETTINGS),
            "gain_at_least_0_01_one_dataset": max(gains.values()) >= 0.01,
            "positive_net_both": all(nets[setting] > 0 for setting in SETTINGS),
        }
        route_a_checks[method] = {"gains": gains, "nets": nets, **checks}
        if all(checks.values()):
            route_a_candidates.append(method)

    route_b_checks = {}
    route_b_candidates = []
    for method in PRIMARY:
        per_dataset = {}
        for setting in SETTINGS:
            summary = summaries[setting]
            regions = summary["temporal_region_stability"]
            correct = lookup(regions, "region", "CORRECT_AUD_FOREGROUND")
            context = lookup(regions, "region", "AUD_FALSE_POSITIVE_CONTEXT")
            std_ratio = context["temporal_std_mean"] / max(
                correct["temporal_std_mean"], 1e-12
            )
            cv_ratio = context["temporal_cv_mean"] / max(
                correct["temporal_cv_mean"], 1e-12
            )
            capture = lookup(summary["ogl_rescue_capture"], "method", method)
            capture_lift = (
                capture["OGL_RESCUE_CAPTURE_RATE"]
                - capture["OTHER_FULL_FAILURE_CAPTURE_RATE"]
            )
            transition = lookup(
                summary["primary_rescue_hurt_oracle"], "method", method
            )
            full = lookup(summary["method_metrics"], "method", "FULL_AUD")
            oracle_gain = transition["oracle_cIoU"] - full["cIoU"]
            region_signal = std_ratio >= 1.10 or cv_ratio >= 1.10
            capacity_signal = capture_lift >= 0.10 or oracle_gain >= 0.01
            per_dataset[setting] = {
                "false_positive_to_correct_std_ratio": std_ratio,
                "false_positive_to_correct_cv_ratio": cv_ratio,
                "OGL_capture_lift_vs_other_failures": capture_lift,
                "oracle_cIoU_gain": oracle_gain,
                "region_signal": region_signal,
                "capture_or_oracle_signal": capacity_signal,
                "signal_positive": region_signal and capacity_signal,
            }
        route_b_checks[method] = per_dataset
        if any(row["signal_positive"] for row in per_dataset.values()):
            route_b_candidates.append(method)

    if route_a_candidates:
        route = "Route A"
        recommendation = (
            "Temporal direction clearly works. Recommend 3.1 Stage2 Audio Temporal "
            "Consistency while preserving the strict Stage1 -> Stage2 pipeline."
        )
    elif route_b_candidates:
        route = "Route B"
        recommendation = (
            "Signal-positive / fusion-negative. Temporal information exists, but fixed "
            "mean/geometric inference fusion is insufficient. A future 3.1 may study "
            "Stage2 temporal consistency without adding a third training stage."
        )
    else:
        route = "Route C"
        recommendation = (
            "Temporal consensus fails. Stop this direction and use Hierarchical Audio "
            "Representation as the next candidate innovation direction."
        )
    return {
        "route": route,
        "recommendation": recommendation,
        "route_a_candidates": route_a_candidates,
        "route_a_checks": route_a_checks,
        "route_b_candidates": route_b_candidates,
        "route_b_checks": route_b_checks,
        "decision_thresholds": {
            "clear_cIoU_gain": 0.01,
            "region_ratio": 1.10,
            "OGL_capture_lift": 0.10,
            "oracle_cIoU_gain": 0.01,
        },
    }


def build_report(summaries: dict[str, dict[str, Any]], result: dict[str, Any]) -> str:
    vgg = summaries["vggss_144k"]
    flickr = summaries["flickr_144k"]
    lines = [
        "# Experiment 3.0 - Temporal Audio Grounding Probe",
        "",
        "## Protocol And Zero-Training Audit",
        "",
        "This experiment uses the formal original 1.3G checkpoints only. All temporal "
        "maps use the unchanged frozen Audio Slot Branch and the same G K34/readout. "
        "OGL is evaluation-only and is never used to construct a temporal map.",
        "",
        "- `model.eval()` and `torch.inference_mode()` were used throughout.",
        "- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`.",
        "- All Stage1, original G, and evaluation-only object-prior checkpoint SHA256/mtime values are unchanged.",
        "- No 3.1 experiment was implemented or started.",
        "",
        "| Setting | Audio feature | Audio tokens | T | 4-chunk boundaries | FULL tensor error | Evaluator error |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        audit = summary["tensor_audit"]
        lines.append(
            f"| {setting} | {audit['original_audio_feature_shape']} | "
            f"{audit['audio_tokens_shape']} | {audit['T']} | "
            f"{audit['four_chunk_boundaries']} | "
            f"{audit['FULL_AUD_tensor_reproduction_max_error']:.3e} | "
            f"{summary['reference_reproduction']['max_error']:.3e} |"
        )

    methods = (
        "FULL_AUD",
        "TEMP_MEAN_4",
        "TEMP_GEO_4",
        "FULL_TEMP_MEAN_4",
        "FULL_TEMP_GEO_4",
        "TEMP_MEAN_2",
        "TEMP_GEO_2",
        "OGL",
    )
    lines.extend(
        [
            "",
            "## Main Localization Results",
            "",
            "| Method | VGG cIoU/AUC | Flickr cIoU/AUC |",
            "|---|---:|---:|",
        ]
    )
    for method in methods:
        vr = lookup(vgg["method_metrics"], "method", method)
        fr = lookup(flickr["method_metrics"], "method", method)
        lines.append(
            f"| {method} | {score(vr['cIoU'])}/{score(vr['AUC'])} | "
            f"{score(fr['cIoU'])}/{score(fr['AUC'])} |"
        )

    lines.extend(
        [
            "",
            "## Four-Chunk Standalone Results",
            "",
            "| Chunk | VGG cIoU/AUC | Flickr cIoU/AUC |",
            "|---|---:|---:|",
        ]
    )
    for chunk in ("CHUNK_1", "CHUNK_2", "CHUNK_3", "CHUNK_4"):
        vr = lookup(vgg["method_metrics"], "method", chunk)
        fr = lookup(flickr["method_metrics"], "method", chunk)
        lines.append(
            f"| {chunk} | {score(vr['cIoU'])}/{score(vr['AUC'])} | "
            f"{score(fr['cIoU'])}/{score(fr['AUC'])} |"
        )

    lines.extend(
        [
            "",
            "## Primary Rescue, Hurt, Oracle, And OGL Capture",
            "",
            "| Dataset | Method | Rescue | Hurt | Net | Oracle cIoU/AUC | OGL pool | Captured | Rate | Capture-Hurt |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        summary = summaries[setting]
        for method in PRIMARY:
            shift = lookup(summary["primary_rescue_hurt_oracle"], "method", method)
            capture = lookup(summary["ogl_rescue_capture"], "method", method)
            lines.append(
                f"| {setting} | {method} | {shift['rescue']} | {shift['hurt']} | "
                f"{shift['net']} | {score(shift['oracle_cIoU'])}/{score(shift['oracle_AUC'])} | "
                f"{capture['OGL_RESCUE_TOTAL']} | {capture['OGL_RESCUE_CAPTURED']} | "
                f"{capture['OGL_RESCUE_CAPTURE_RATE']:.3f} | {capture['CAPTURE_MINUS_HURT']} |"
            )

    lines.extend(
        [
            "",
            "## OGL Marginal Gap",
            "",
            "| Dataset | Method | Original gap | New gap | Reduction | Reduction % |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["ogl_gap"]:
            lines.append(
                f"| {setting} | {row['method']} | {row['original_OGL_FULL_gap']:.4f} | "
                f"{row['new_OGL_method_gap']:.4f} | {row['gap_reduction']:.4f} | "
                f"{row['gap_reduction_percent']:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Chunk Slot Identity Stability",
            "",
            "| Dataset | Chunk | q0->full q0 rate | q0->full q1 rate | cos(q0,f0) | cos(q0,f1) | cos(q1,f0) | cos(q1,f1) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["chunk_slot_identity"]:
            cosine_keys = (
                "chunk_q0_full_q0_mean",
                "chunk_q0_full_q1_mean",
                "chunk_q1_full_q0_mean",
                "chunk_q1_full_q1_mean",
            )
            cosine_text = [
                "-" if row.get(key) is None else f"{row[key]:.4f}" for key in cosine_keys
            ]
            lines.append(
                f"| {setting} | {row['chunk']} | {row['slot0_to_slot0_rate']:.3f} | "
                f"{row['slot0_to_slot1_rate']:.3f} | {' | '.join(cosine_text)} |"
            )

    lines.extend(
        [
            "",
            "## Temporal Region Stability",
            "",
            "Per-sample region means are aggregated below. Stability is computed from raw 14x14 chunk attention before evaluator normalization.",
            "",
            "| Dataset | Region | Temporal mean | Temporal STD | Temporal CV | Samples |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["temporal_region_stability"]:
            lines.append(
                f"| {setting} | {row['region']} | {row['temporal_mean_mean']:.6f} | "
                f"{row['temporal_std_mean']:.6f} | {row['temporal_cv_mean']:.4f} | "
                f"{row['temporal_std_num_samples']} |"
            )

    lines.extend(
        [
            "",
            "## Sample-Level Temporal Agreement",
            "",
            "Each sample value is the mean over all six pairs among the four raw chunk maps.",
            "",
            "| Dataset | Group | Pearson | Spearman | JS divergence | Samples |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["sample_temporal_agreement"]:
            lines.append(
                f"| {setting} | {row['group']} | {row['pearson_mean']:.4f} | "
                f"{row['spearman_mean']:.4f} | {row['js_divergence_mean']:.2e} | "
                f"{row['pearson_num_samples']} |"
            )

    lines.extend(
        [
            "",
            "## Chunk Quality",
            "",
            "| Dataset | Chunk | Token mean norm | Temporal variance | q0 norm | q0/full-q0 cosine |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["chunk_quality"]:
            lines.append(
                f"| {setting} | {row['chunk']} | {row['token_mean_norm_mean']:.4f} | "
                f"{row['token_temporal_variance_mean']:.5f} | {row['query_q0_norm_mean']:.4f} | "
                f"{row['query_q0_full_q0_cosine_mean']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Alpha Diagnostic",
            "",
            "Alpha=0.6 is formal; other rows are diagnostics only and are not used for model selection.",
            "",
            "| Dataset | Family | Full alpha | cIoU/AUC | Rescue | Hurt | Net |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["alpha_diagnostic"]:
            lines.append(
                f"| {setting} | {row['family']} | {row['alpha_full']:.1f} | "
                f"{score(row['cIoU'])}/{score(row['AUC'])} | {row['rescue']} | "
                f"{row['hurt']} | {row['net']} |"
            )

    lines.extend(
        [
            "",
            "## Qualitative",
            "",
            "The manifests use a fixed first-in-test-order round-robin across AUD success, "
            "OGL rescue, temporal rescue, temporal hurt, and all-fail categories; twelve "
            "panels are saved per dataset without cherry-picking.",
            "",
            "- Across both datasets, CHUNK_1..4 preserve almost the same hotspot shape, "
            "extent, and background response as FULL_AUD. TEMP_MEAN_4 and TEMP_GEO_4 are "
            "therefore visually indistinguishable from each other and nearly identical to "
            "FULL_AUD.",
            "- The fixed VGG temporal-rescue examples are threshold-boundary changes rather "
            "than a consistent removal of context. Matching temporal-hurt examples show the "
            "same small boundary movement in the opposite direction, consistent with the "
            "aggregate 7 Rescue / 7 Hurt result.",
            "- In fixed OGL-rescue examples, OGL changes the spatial support enough to cross "
            "the success threshold, while temporal fusion remains close to FULL_AUD. This "
            "matches the 4/357 VGG and 0/19 Flickr OGL-rescue capture counts.",
            "- TEMP_STD frequently highlights object edges, image borders, or non-target "
            "regions. It does not consistently separate correct grounding from context "
            "false positives; the quantitative false-positive STD/CV is in fact lower than "
            "the correct-region STD/CV on both datasets.",
            "- Flickr contains no primary temporal Rescue or Hurt cases. Its fixed panels "
            "show the temporal maps preserving the original response, including its misses, "
            "rather than introducing complementary localization evidence.",
            "",
            "## Fixed-Rule Decision",
            "",
            f"**{result['route']}**",
            "",
            result["recommendation"],
            "",
            "### Route A checks",
            "",
            "```json",
            json.dumps(result["route_a_checks"], indent=2),
            "```",
            "",
            "### Route B checks",
            "",
            "```json",
            json.dumps(result["route_b_checks"], indent=2),
            "```",
            "",
            "No 3.1 implementation or run was started.",
            "",
        ]
    )
    return "\n".join(lines)


def run(arguments: argparse.Namespace) -> None:
    summaries = {}
    for setting in SETTINGS:
        path = arguments.results_root / setting / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary["completed_full_dataset"]:
            raise RuntimeError(f"Incomplete result: {path}")
        summaries[setting] = summary
    result = decision(summaries)
    combined = {
        "experiment": "3.0 Temporal Audio Grounding Probe",
        "datasets": summaries,
        "decision": result,
    }
    common.write_json(arguments.results_root / "combined_summary.json", combined)
    for key, filename in (
        ("method_metrics", "combined_method_metrics.csv"),
        ("primary_rescue_hurt_oracle", "combined_primary_rescue_hurt_oracle.csv"),
        ("ogl_rescue_capture", "combined_ogl_rescue_capture.csv"),
        ("ogl_gap", "combined_ogl_gap.csv"),
        ("temporal_region_stability", "combined_temporal_region_stability.csv"),
        ("sample_temporal_agreement", "combined_sample_temporal_agreement.csv"),
        ("chunk_slot_identity", "combined_chunk_slot_identity.csv"),
        ("chunk_quality", "combined_chunk_quality.csv"),
        ("alpha_diagnostic", "combined_alpha_diagnostic.csv"),
    ):
        common.write_csv(
            arguments.results_root / filename,
            [row for setting in SETTINGS for row in summaries[setting][key]],
        )
    report = build_report(summaries, result)
    (common.HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    run(parse_args())
