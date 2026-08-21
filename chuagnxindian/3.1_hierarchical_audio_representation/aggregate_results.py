#!/usr/bin/env python3
"""Aggregate the four formal Experiment 3.1 settings."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jsa_31_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common


FORMAL_144K = ("vggss_144k", "flickr_144k")
METHODS = ("AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL")


def score(value):
    return f"{value:.4f}"


def diagnostic_curve(setting, summary):
    rows = common.read_csv(common.experiment_dir(setting) / "epoch_metrics.csv")
    numeric = [
        {key: float(value) for key, value in row.items() if value not in {"", None}}
        for row in rows
    ]
    keys = (
        "cos_a3_a4_slot0",
        "cos_a3_a4_slot1",
        "cos_fused_a4_slot0",
        "cos_fused_a4_slot1",
        "delta_norm_over_a4_norm",
    )
    epochs = [row["epoch"] for row in numeric]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for key in keys[:-1]:
        axes[0].plot(epochs, [row[key] for row in numeric], label=key)
    axes[0].set_title("Audio slot cosine diagnostics")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].plot(
        epochs,
        [row["delta_norm_over_a4_norm"] for row in numeric],
        label="delta/A4",
    )
    axes[1].set_title("Residual utilization")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    best_epoch = summary["best_epoch"]
    for axis in axes:
        axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    output = common.result_dir(setting) / "fusion_health_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    best_row = next(row for row in numeric if int(row["epoch"]) == best_epoch)
    return {
        "first_epoch": {key: numeric[0][key] for key in keys},
        "best_epoch": {key: best_row[key] for key in keys},
        "last_epoch": {key: numeric[-1][key] for key in keys},
        "delta_min": min(row["delta_norm_over_a4_norm"] for row in numeric),
        "delta_max": max(row["delta_norm_over_a4_norm"] for row in numeric),
        "hierarchy_degenerated_to_a4": (
            best_row["delta_norm_over_a4_norm"] < 0.01
            and best_row["cos_fused_a4_slot0"] > 0.9999
            and best_row["cos_fused_a4_slot1"] > 0.9999
        ),
    }


def decision(summaries):
    gains = {
        setting: max(
            summaries[setting]["deltas"]["AUD"]["cIoU"],
            summaries[setting]["deltas"]["IQR"]["cIoU"],
        )
        for setting in FORMAL_144K
    }
    gap_reductions = {
        setting: max(
            summaries[setting]["ogl_gaps"]["reduction"]["OGL_minus_AUD"],
            summaries[setting]["ogl_gaps"]["reduction"]["OGL_minus_IQR"],
        )
        for setting in FORMAL_144K
    }
    one_large = max(gains.values()) >= 0.01
    other_tolerable = min(gains.values()) >= -0.004
    mixed = max(gains.values()) >= 0.01 and min(gains.values()) < -0.01
    both_improve = all(value > 0 for value in gains.values())
    gaps_shrink_clearly = all(value >= 0.005 for value in gap_reductions.values())

    if mixed:
        label = "Mixed"
        recommendation = "Do not attach G; report the cross-dataset mechanism difference."
    elif one_large and other_tolerable and (both_improve or gaps_shrink_clearly):
        label = "Very Strong Positive"
        recommendation = (
            "Strong candidate for Innovation 2; only the next experiment may attach the "
            "unchanged original G to this Stage1."
        )
    elif one_large and other_tolerable:
        label = "Strong Positive"
        recommendation = (
            "Hierarchical Audio is worth a later Stage2 validation with the original G."
        )
    else:
        label = "Negative"
        recommendation = (
            "Stop the A3+A4 hierarchical-audio mainline; do not expand to A2/A3/A4 or "
            "automatically train G."
        )
    return {
        "label": label,
        "recommendation": recommendation,
        "formal_144k_best_AUD_or_IQR_gains": gains,
        "formal_144k_best_gap_reductions": gap_reductions,
        "checks": {
            "one_dataset_gain_at_least_0.01": one_large,
            "other_dataset_at_least_baseline_minus_0.004": other_tolerable,
            "mixed_condition": mixed,
            "both_datasets_improve": both_improve,
            "both_gaps_shrink_at_least_0.005": gaps_shrink_clearly,
        },
    }


def build_report(summaries, curves, result):
    lines = [
        "# Experiment 3.1 - Hierarchical Audio Representation",
        "",
        "## Protocol",
        "",
        "Visual L3/L4, visual fusion, A4 localization, A4 attention loss, A4 "
        "reconstruction, loss weights, optimizer, scheduler, data configuration, and "
        "IQR-cIoU checkpoint selection are inherited from the formal L3+L4 Stage1.",
        "",
        "The only added trainable modules are aud_proj3, AudioSlotBranch_A3, and the "
        "zero-initialized A4-residual AudioHierarchicalFusion. No G or Stage2 training "
        "is part of this experiment.",
        "",
        "## Best Epochs",
        "",
        "| Setting | Baseline epoch | 3.1 epoch | Selection |",
        "|---|---:|---:|---|",
    ]
    for setting in common.SETTINGS:
        summary = summaries[setting]
        lines.append(
            f"| {setting} | {summary['baseline_best_epoch']} | {summary['best_epoch']} | "
            f"{summary['selection_metric']}={summary['selection_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Six Metrics",
            "",
            "| Setting | Method | Baseline cIoU/AUC | 3.1 cIoU/AUC | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for setting in common.SETTINGS:
        summary = summaries[setting]
        for method in METHODS:
            base = summary["baseline_metrics_from_epoch_csv"][method]
            new = summary["hierarchical_metrics"][method]
            lines.append(
                f"| {setting} | {method} | {score(base['cIoU'])}/{score(base['AUC'])} | "
                f"{score(new['cIoU'])}/{score(new['AUC'])} | "
                f"{new['cIoU'] - base['cIoU']:+.4f}/{new['AUC'] - base['AUC']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## OGL Gaps",
            "",
            "| Setting | Baseline OGL-AUD | 3.1 OGL-AUD | Reduction | Baseline OGL-IQR | 3.1 OGL-IQR | Reduction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in common.SETTINGS:
        gaps = summaries[setting]["ogl_gaps"]
        lines.append(
            f"| {setting} | {gaps['baseline']['OGL_minus_AUD']:.4f} | "
            f"{gaps['hierarchical']['OGL_minus_AUD']:.4f} | "
            f"{gaps['reduction']['OGL_minus_AUD']:+.4f} | "
            f"{gaps['baseline']['OGL_minus_IQR']:.4f} | "
            f"{gaps['hierarchical']['OGL_minus_IQR']:.4f} | "
            f"{gaps['reduction']['OGL_minus_IQR']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## A3 Query Diagnostic",
            "",
            "| Setting | A3_QUERY_AUD cIoU/AUC |",
            "|---|---:|",
        ]
    )
    for setting in common.SETTINGS:
        metric = summaries[setting]["A3_QUERY_AUD"]
        lines.append(f"| {setting} | {score(metric['cIoU'])}/{score(metric['AUC'])} |")

    lines.extend(
        [
            "",
            "## Semantic Alignment",
            "",
            "| Setting | Audio representation | Positive | Shuffled negative | Margin |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for setting in common.SETTINGS:
        for level, values in summaries[setting]["semantic_alignment"].items():
            lines.append(
                f"| {setting} | {level} | {values['positive_cosine']:.4f} | "
                f"{values['shuffled_negative_cosine']:.4f} | {values['margin']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Temporal Diversity At Best Checkpoint",
            "",
            "| Setting | Level | Adjacent cosine | Pairwise cosine | Temporal variance |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for setting in common.SETTINGS:
        values = summaries[setting]["representation_diagnostics"]
        for level in ("a3", "a4"):
            lines.append(
                f"| {setting} | {level.upper()} | {values[f'{level}_adjacent_cosine']:.4f} | "
                f"{values[f'{level}_pairwise_cosine']:.4f} | "
                f"{values[f'{level}_temporal_feature_variance']:.5f} |"
            )

    lines.extend(
        [
            "",
            "## Fusion Utilization",
            "",
            "| Setting | Best delta/A4 | cos(fused,A4) slot0/slot1 | Degenerated to A4 |",
            "|---|---:|---:|---:|",
        ]
    )
    for setting in common.SETTINGS:
        best = curves[setting]["best_epoch"]
        lines.append(
            f"| {setting} | {best['delta_norm_over_a4_norm']:.4f} | "
            f"{best['cos_fused_a4_slot0']:.4f}/{best['cos_fused_a4_slot1']:.4f} | "
            f"{curves[setting]['hierarchy_degenerated_to_a4']} |"
        )

    lines.extend(
        [
            "",
            "## Qualitative",
            "",
            "Twelve deterministic panels per setting compare the formal baseline and 3.1 "
            "AUD/IMG/IQR maps, A3 query diagnostic, OGL, and absolute map changes.",
            "",
            "On isolated improvements, 3.1 sometimes suppresses broad background response and "
            "moves the A4 peak onto the sounding person or object. The failure cases are more "
            "common in both 144k evaluations: response is displaced toward roads, room structure, "
            "or another contextual region, or becomes too narrow to cover the annotated source. "
            "A3_QUERY_AUD is usually diffuse or context-focused and does not provide a stable "
            "localization cue. AUD and IMG changes frequently move together, consistent with the "
            "semantic objective perturbing the shared cross-modal solution rather than adding a "
            "reliably complementary acoustic representation.",
            "",
            "| Setting | AUD improve | AUD hurt | IQR improve | IQR hurt |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for setting in common.SETTINGS:
        rows = common.read_csv(common.result_dir(setting) / "per_sample_metrics.csv")
        counts = Counter(row["category"].strip() for row in rows)
        lines.append(
            f"| {setting} | {counts['AUD_IMPROVE']} | {counts['AUD_HURT']} | "
            f"{counts['IQR_IMPROVE']} | {counts['IQR_HURT']} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-Rule Decision",
            "",
            f"**{result['label']}**",
            "",
            result["recommendation"],
            "",
            "```json",
            json.dumps(result, indent=2),
            "```",
            "",
            "No G, Stage2, or follow-up experiment was trained automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    summaries = {}
    curves = {}
    for setting in common.SETTINGS:
        path = common.result_dir(setting) / "summary.json"
        summaries[setting] = json.loads(path.read_text(encoding="utf-8"))
        curves[setting] = diagnostic_curve(setting, summaries[setting])
    result = decision(summaries)
    combined = {
        "experiment": "3.1 Hierarchical Audio Representation",
        "settings": summaries,
        "fusion_health_curves": curves,
        "decision": result,
    }
    common.write_json(common.RESULTS_ROOT / "combined_summary.json", combined)
    report = build_report(summaries, curves, result)
    (common.HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
