#!/usr/bin/env python3
"""Aggregate all four Experiment 2.5 settings and apply the fixed route rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import common


SETTINGS = ("vggss_10k", "flickr_10k", "vggss_144k", "flickr_144k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=common.HERE / "results")
    return parser.parse_args()


def lookup(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError((key, value))


def score(value: Any) -> str:
    return f"{float(value):.4f}"


def route_decision(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vgg = summaries["vggss_144k"]
    flickr = summaries["flickr_144k"]
    vgg_pooled = lookup(vgg["part_a_candidates"], "map", "SLOT_L3_POOLED")
    vgg_native = lookup(vgg["part_a_candidates"], "map", "SLOT_L3_NATIVE_READOUT")
    vgg_g = lookup(vgg["part_a_144k_methods"], "method", "Original G AUD")
    vgg_g_native = lookup(
        vgg["part_a_144k_methods"], "method", "Original G + L3 native readout"
    )
    flickr_g = lookup(flickr["part_a_144k_methods"], "method", "Original G AUD")
    flickr_g_native = lookup(
        flickr["part_a_144k_methods"], "method", "Original G + L3 native readout"
    )

    flickr_drop_tolerance = 0.01
    cross_close_tolerance = 0.02
    route_a_checks = {
        "vgg_native_rescue_gt_pooled": vgg_native["rescue"] > vgg_pooled["rescue"],
        "vgg_native_hurt_lt_pooled": vgg_native["hurt"] < vgg_pooled["hurt"],
        "vgg_original_G_plus_native_gt_original_G": vgg_g_native["cIoU"] > vgg_g["cIoU"],
        "flickr_original_G_plus_native_no_material_drop": (
            flickr_g_native["cIoU"] >= flickr_g["cIoU"] - flickr_drop_tolerance
        ),
    }
    route_a = all(route_a_checks.values())

    route_b_checks = {}
    for setting, summary in (("vggss_144k", vgg), ("flickr_144k", flickr)):
        methods = summary["part_b_methods"]
        original = lookup(methods, "method", "Original G AUD")
        same = lookup(methods, "method", "2.4 AUD + 2.4 OWN14")
        cross = lookup(methods, "method", "Original G AUD + 2.4 OWN14")
        route_b_checks[f"{setting}_cross_gt_same_checkpoint"] = (
            cross["cIoU"] > same["cIoU"]
        )
        route_b_checks[f"{setting}_cross_near_or_above_original_G"] = (
            cross["cIoU"] >= original["cIoU"] - cross_close_tolerance
        )
    route_b = (not route_a) and all(route_b_checks.values())

    if route_a:
        route = "Route A"
        recommendation = (
            "Innovation 2 -> Stage1: Dual-Resolution Semantic/Spatial Slot; "
            "semantic update remains pooled 7x7 and spatial readout uses native 14x14."
        )
    elif route_b:
        route = "Route B"
        recommendation = (
            "Innovation 2 -> Stage2: parallel Audio/Object spatial paths trained "
            "simultaneously inside the same Stage2."
        )
    else:
        route = "Route C"
        recommendation = (
            "Do not continue the current object-ownership line: internal Slot ownership "
            "has oracle capacity, but the current self-supervised extent signal is not "
            "stable across datasets."
        )
    return {
        "route": route,
        "recommendation": recommendation,
        "route_a_checks": route_a_checks,
        "route_b_checks": route_b_checks,
        "decision_tolerances": {
            "flickr_no_material_drop_cIoU": flickr_drop_tolerance,
            "cross_near_original_G_cIoU": cross_close_tolerance,
        },
    }


def build_report(summaries: dict[str, dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# Experiment 2.5 - Dual-Path Decision Probe",
        "",
        "## Zero-Training Audit",
        "",
        "All four settings used `model.eval()` and `torch.inference_mode()`. No optimizer "
        "or backward call was created; all loaded parameters had `requires_grad=False`; "
        "checkpoint SHA256 and mtime values were unchanged.",
        "",
        "| Setting | Q3 | K3_POOL | K3_NATIVE | Pooled reconstruction max error | Native slot-sum max error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        audit = summaries[setting]["tensor_audit"]
        lines.append(
            f"| {setting} | {audit['Q3_shape']} | {audit['K3_POOL_shape']} | "
            f"{audit['K3_NATIVE_shape']} | "
            f"{audit['pooled_ownership_reconstruction_max_error']:.3e} | "
            f"{audit['native_readout_slot_sum_max_error']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Part A - Ownership and Corresponding-Audio Fusion",
            "",
            "`cIoU/AUC` below are standalone ownership metrics. Rescue/Hurt/Net and Oracle "
            "use the corresponding audio checkpoint and its alpha=0.6 fusion.",
            "",
            "| Dataset | Map | Ownership cIoU/AUC | Fusion cIoU/AUC | Rescue | Hurt | Net | Oracle cIoU/AUC |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for row in summaries[setting]["part_a_candidates"]:
            lines.append(
                f"| {setting} | {row['map']} | {score(row['ownership_cIoU'])}/{score(row['ownership_AUC'])} | "
                f"{score(row['fusion_cIoU'])}/{score(row['fusion_AUC'])} | {row['rescue']} | "
                f"{row['hurt']} | {row['net']} | {score(row['oracle_cIoU'])}/{score(row['oracle_AUC'])} |"
            )

    lines.extend(
        [
            "",
            "## Part A - Original G Combinations (144k)",
            "",
            "| Dataset | Method | cIoU/AUC | Rescue | Hurt | Net | Oracle cIoU/AUC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in ("vggss_144k", "flickr_144k"):
        for row in summaries[setting]["part_a_144k_methods"]:
            lines.append(
                f"| {setting} | {row['method']} | {score(row['cIoU'])}/{score(row['AUC'])} | "
                f"{row['rescue']} | {row['hurt']} | {row['net']} | "
                f"{score(row['oracle_cIoU'])}/{score(row['oracle_AUC'])} |"
            )

    lines.extend(
        [
            "",
            "## Part B - Cross-Checkpoint Fusion",
            "",
            "| Method | VGG cIoU/AUC | Flickr cIoU/AUC |",
            "|---|---:|---:|",
        ]
    )
    vgg_methods = {row["method"]: row for row in summaries["vggss_144k"]["part_b_methods"]}
    flickr_methods = {
        row["method"]: row for row in summaries["flickr_144k"]["part_b_methods"]
    }
    for method in vgg_methods:
        vgg = vgg_methods[method]
        flickr = flickr_methods[method]
        lines.append(
            f"| {method} | {score(vgg['cIoU'])}/{score(vgg['AUC'])} | "
            f"{score(flickr['cIoU'])}/{score(flickr['AUC'])} |"
        )

    lines.extend(
        [
            "",
            "| Method | VGG Rescue/Hurt/Net, Oracle | Flickr Rescue/Hurt/Net, Oracle |",
            "|---|---:|---:|",
        ]
    )
    vgg_transitions = {
        row["method"]: row for row in summaries["vggss_144k"]["part_b_transitions"]
    }
    flickr_transitions = {
        row["method"]: row for row in summaries["flickr_144k"]["part_b_transitions"]
    }
    for method in vgg_transitions:
        vgg = vgg_transitions[method]
        flickr = flickr_transitions[method]
        lines.append(
            f"| {method} | {vgg['rescue']}/{vgg['hurt']}/{vgg['net']}, "
            f"{score(vgg['oracle_cIoU'])}/{score(vgg['oracle_AUC'])} | "
            f"{flickr['rescue']}/{flickr['hurt']}/{flickr['net']}, "
            f"{score(flickr['oracle_cIoU'])}/{score(flickr['oracle_AUC'])} |"
        )

    lines.extend(
        [
            "",
            "## Map Complementarity",
            "",
            "Statistics compare each normalized 224x224 candidate map against original G AUD.",
            "",
            "| Dataset | Candidate | Pearson mean | Spearman mean | JS mean |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for setting in ("vggss_144k", "flickr_144k"):
        for row in summaries[setting]["map_complementarity"]:
            lines.append(
                f"| {setting} | {row['candidate']} | {score(row['pearson_mean'])} | "
                f"{score(row['spearman_mean'])} | {score(row['js_divergence_mean'])} |"
            )

    lines.extend(
        [
            "",
            "## Alpha Diagnostic",
            "",
            "Alpha=0.6 is the formal result; other values are diagnostics only.",
            "",
            "| Dataset | Audio alpha | cIoU/AUC | Rescue | Hurt | Net |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in ("vggss_144k", "flickr_144k"):
        for row in summaries[setting]["alpha_diagnostic"]:
            lines.append(
                f"| {setting} | {row['alpha_audio']:.1f} | {score(row['cIoU'])}/{score(row['AUC'])} | "
                f"{row['rescue']} | {row['hurt']} | {row['net']} |"
            )

    lines.extend(
        [
            "",
            "## Fixed-Selection Qualitative Findings",
            "",
            "The panels use the unchanged deterministic Experiment 2.2 manifests; no "
            "2.5 outcome was used for selection.",
            "",
            "- L3 native-readout is finer than the pooled map, but its new detail is mostly "
            "fragmented high-frequency activation over foreground and background. It avoids "
            "the severe Flickr-144k native-update collapse, yet does not produce a coherent "
            "object boundary or a useful correction map.",
            "- Original HR14 and 2.4 OWN14 are visually close to original G AUD. On Flickr, "
            "the shared broad region often overlaps the large/centered sounding object, so "
            "fusion helps. On VGG, the same behavior commonly over-expands object extent into "
            "surrounding scene context; occasional misplaced or undersized peaks occur, but "
            "over-expansion is the dominant fixed-sample failure.",
            "- Cross fusion preserves the original G audio response, but 2.4 OWN14 usually "
            "changes intensity inside the same region instead of supplying an independent "
            "spatial correction. This matches its high Pearson/Spearman correlation with AUD.",
            "",
            "",
            "## Decision",
            "",
            f"**{decision['route']}**",
            "",
            decision["recommendation"],
            "",
            "Decision checks:",
            "",
        ]
    )
    for name, passed in decision["route_a_checks"].items():
        lines.append(f"- Route A `{name}`: {passed}")
    for name, passed in decision["route_b_checks"].items():
        lines.append(f"- Route B `{name}`: {passed}")
    lines.extend(
        [
            "",
            "No subsequent experiment was implemented or started.",
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
            raise RuntimeError(f"Incomplete result cannot be aggregated: {path}")
        summaries[setting] = summary

    decision = route_decision(summaries)
    output = {
        "experiment": "2.5 Dual-Path Decision Probe",
        "settings": summaries,
        "decision": decision,
    }
    common.write_json(arguments.results_root / "combined_summary.json", output)
    common.write_csv(
        arguments.results_root / "combined_part_a_candidates.csv",
        [row for setting in SETTINGS for row in summaries[setting]["part_a_candidates"]],
    )
    common.write_csv(
        arguments.results_root / "combined_part_a_144k.csv",
        [
            row
            for setting in ("vggss_144k", "flickr_144k")
            for row in summaries[setting]["part_a_144k_methods"]
        ],
    )
    common.write_csv(
        arguments.results_root / "combined_part_b_methods.csv",
        [
            row
            for setting in ("vggss_144k", "flickr_144k")
            for row in summaries[setting]["part_b_methods"]
        ],
    )
    common.write_csv(
        arguments.results_root / "combined_part_b_transitions.csv",
        [
            row
            for setting in ("vggss_144k", "flickr_144k")
            for row in summaries[setting]["part_b_transitions"]
        ],
    )
    common.write_csv(
        arguments.results_root / "combined_map_complementarity.csv",
        [
            row
            for setting in ("vggss_144k", "flickr_144k")
            for row in summaries[setting]["map_complementarity"]
        ],
    )
    common.write_csv(
        arguments.results_root / "combined_alpha_diagnostic.csv",
        [
            row
            for setting in ("vggss_144k", "flickr_144k")
            for row in summaries[setting]["alpha_diagnostic"]
        ],
    )
    report = build_report(summaries, decision)
    (common.HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    run(parse_args())
