#!/usr/bin/env python3
"""Aggregate Experiment 5.2 results and write REPORT.md."""

from __future__ import annotations

import argparse
import csv
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


def metric(value: dict[str, Any]) -> str:
    return f"{fmt(value['cIoU'])}/{fmt(value['AUC'])}"


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%" if total else "N/A"


def load_summaries(root: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    for setting in SETTINGS:
        summary = json.loads((root / setting / "summary.json").read_text(encoding="utf-8"))
        with (root / setting / "single_variable_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            summary["_single_variable_all"] = list(csv.DictReader(handle))
        summaries[setting] = summary
    return summaries


def stable_signal_rows(summaries: dict[str, dict[str, Any]], task: str) -> list[dict[str, Any]]:
    lookup = {}
    for setting in SETTINGS:
        rows = [row for row in summaries[setting]["_single_variable_all"] if row["task"] == task]
        lookup[setting] = {row["signal"]: row for row in rows}
    common_names = set(lookup[SETTINGS[0]]) & set(lookup[SETTINGS[1]])
    rows = []
    for name in common_names:
        first = lookup[SETTINGS[0]][name]
        second = lookup[SETTINGS[1]][name]
        if first["direction"] != second["direction"]:
            continue
        rows.append(
            {
                "signal": name,
                "direction": first["direction"],
                "VGG": float(first["AUROC_directed"]),
                "Flickr": float(second["AUROC_directed"]),
                "minimum": min(float(first["AUROC_directed"]), float(second["AUROC_directed"])),
            }
        )
    return sorted(rows, key=lambda row: (row["minimum"], row["VGG"] + row["Flickr"]), reverse=True)


def build_report(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Experiment 5.2 - Expansion-Shrink Error Diagnosis",
        "",
        "## 1. Audit",
        "",
        "- Formal Stage1 and original 1.3G checkpoints, loaders, preprocessing, evaluator, sample order, and cached 5.1 propagation maps are reused unchanged.",
        "- Frozen model inference uses `model.eval()` and `torch.inference_mode()`; no optimizer, backward pass, localization training, or new torch parameter is created.",
        "- The scikit-learn logistic regressions are analysis-only probes evaluated out of fold. They never modify or feed back into AUD/IMG/PROP predictions.",
        "",
    ]
    for setting in SETTINGS:
        summary = summaries[setting]
        audit = summary["audit"]
        zero = summary["zero_training_audit"]
        tensor = summary["tensor_audit"]
        lines.extend(
            [
                f"### {LABELS[setting]}",
                "",
                f"- Raw reconstruction errors: `{audit['raw_tensor_max_errors']}`; metric errors: `{audit['per_sample_metric_max_errors']}`; sample mismatch: `{audit['sample_mismatches']}`.",
                f"- Shapes: `AUD/IMG L3={tensor['AUD_L3_ALL']}/{tensor['IMG_L3_ALL']}`, `AUD/IMG L4={tensor['AUD_L4_ALL']}/{tensor['IMG_L4_ALL']}`, `AUD_FINE={tensor['AUD_FINE']}`, `F34={tensor['F34']}`, `K34={tensor['K34']}`.",
                f"- `optimizer_created={str(zero['optimizer_created']).lower()}`, `backward_called={str(zero['backward_called']).lower()}`, `new_trainable_params={zero['new_trainable_params']}`, `parameters_with_grad={zero['parameters_with_grad']}`.",
                f"- Checkpoint SHA256/mtime unchanged: `{zero['checkpoint_hashes_and_mtimes_unchanged']}`; NaN/Inf: `{not zero['no_nan_or_inf_model_and_maps']}`.",
                "",
            ]
        )

    definition = summaries[SETTINGS[0]]["error_definition"]
    lines.extend(
        [
            "## 2. Error-Type Definition",
            "",
            f"- EXPAND candidates: `{definition['EXPAND_F34']}` and `{definition['EXPAND_K34']}`; GT selects the better fixed candidate only for oracle labeling.",
            f"- SHRINK candidate: `{definition['SHRINK']}`.",
            f"- Beneficial threshold: absolute IoU gain `>={definition['beneficial_gain_threshold']:.2f}`.",
            f"- Strict three-class rule: `{definition['strict_rule']}`.",
            "- The threshold and dominance margin were fixed before the formal run and were not tuned per dataset.",
            "",
            "## 3. Dataset Distribution",
            "",
            "| Dataset | Expand only | Shrink only | Both beneficial | Neither | Strict Expand | Strict Shrink | Keep/Ambiguous |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        dist = summaries[setting]["error_distribution"]
        total = dist["total"]
        multi = dist["multi_label"]
        strict = dist["strict_three_class"]
        lines.append(
            f"| {LABELS[setting]} | {multi['EXPAND_ONLY']['count']} ({pct(multi['EXPAND_ONLY']['count'], total)}) | "
            f"{multi['SHRINK_ONLY']['count']} ({pct(multi['SHRINK_ONLY']['count'], total)}) | "
            f"{multi['BOTH_BENEFICIAL']['count']} ({pct(multi['BOTH_BENEFICIAL']['count'], total)}) | "
            f"{multi['NEITHER']['count']} ({pct(multi['NEITHER']['count'], total)}) | "
            f"{strict['EXPAND']['count']} ({pct(strict['EXPAND']['count'], total)}) | "
            f"{strict['SHRINK']['count']} ({pct(strict['SHRINK']['count'], total)}) | "
            f"{strict['KEEP_AMBIGUOUS']['count']} ({pct(strict['KEEP_AMBIGUOUS']['count'], total)}) |"
        )

    lines.extend(
        [
            "",
            "The strict distributions are almost identical across datasets: roughly 22% EXPAND, 39% SHRINK, and 38-39% KEEP/AMBIGUOUS. This is strong evidence that the bidirectional error structure is not a VGG-only artifact.",
            "",
            "## 4. Spatial Disagreement Statistics",
            "",
            "| Dataset | Type | IMG/AUD area | AUD-only/AUD | IMG-only/IMG | AUD-IMG mask IoU |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        grouped = summaries[setting]["spatial_disagreement_by_error_type"]
        for error in ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS"):
            value = grouped[error]
            lines.append(
                f"| {LABELS[setting]} | {error} | {fmt(value['IMG_AUD_area_ratio']['mean'])} | "
                f"{fmt(value['AUD_only_ratio']['mean'])} | {fmt(value['IMG_only_ratio']['mean'])} | "
                f"{fmt(value['AUD_IMG_mask_IoU']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "- VGG SHRINK samples show the expected geometry: smaller IMG/AUD area (`0.8504`) and larger AUD-only fraction (`0.1754`) than EXPAND (`0.9051`, `0.1444`).",
            "- Flickr does not preserve that ordering: EXPAND has a smaller IMG/AUD ratio and larger AUD-only fraction than SHRINK. Area disagreement alone therefore cannot identify which removed pixels are context rather than true extent.",
            "- Across both datasets, SHRINK tends to have very little IMG-only addition. The most stable geometry signal is low `IMG_only_area`, but its EXPAND-vs-SHRINK AUROC is only about `0.60`.",
            "",
            "## 5. Internal Signals",
            "",
            "| Dataset | Type | AUD-only F34 support | AUD-only K34 support | Common-minus-extra F34 | Common-minus-extra K34 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        grouped = summaries[setting]["AUD_only_internal_signals_by_error_type"]
        for error in ("EXPAND", "SHRINK", "KEEP_AMBIGUOUS"):
            value = grouped[error]
            lines.append(
                f"| {LABELS[setting]} | {error} | {fmt(value['AUD_only_PROP_F34_similarity']['mean'])} | "
                f"{fmt(value['AUD_only_PROP_K34_similarity']['mean'])} | {fmt(value['REGION_DELTA_PROP_F34_similarity']['mean'])} | "
                f"{fmt(value['REGION_DELTA_PROP_K34_similarity']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "AUD-only regions in SHRINK samples receive slightly weaker agreement-prototype support in both F34 and K34. The direction transfers, but the separation is modest on VGG and does not form a standalone reliable router.",
            "",
            "## 6. Single-Variable Diagnostics",
            "",
            "Best same-direction signals that remain near the top on both datasets:",
            "",
            "| Task | Signal | Positive direction | VGG AUROC | Flickr AUROC |",
            "|---|---|---|---:|---:|",
        ]
    )
    for task in ("EXPAND_vs_SHRINK", "SHRINK_vs_OTHERS", "EXPAND_vs_OTHERS"):
        stable = stable_signal_rows(summaries, task)
        for value in stable[:3]:
            lines.append(
                f"| {task} | {value['signal']} | {value['direction']} | {fmt(value['VGG'])} | {fmt(value['Flickr'])} |"
            )
    lines.extend(
        [
            "",
            "No single signal is strong on both datasets. The best stable EXPAND-vs-SHRINK diagnostics remain around `0.60 AUROC`; dataset-specific peaks reach about `0.65-0.66` but use different signals.",
            "",
            "## 7. Lightweight Probe",
            "",
            "All results are out-of-fold. VGG uses video-id grouped folds; Flickr uses sample-id groups. Imputation and standardization are fitted only inside each training fold.",
            "",
            "| Dataset | Features | Task | AUROC | Balanced acc | Macro F1 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        probe = summaries[setting]["lightweight_probe"]
        for feature_set in ("SPATIAL", "INTERNAL", "ALL"):
            for task in ("EXPAND_vs_SHRINK", "SHRINK_vs_OTHERS", "EXPAND_vs_OTHERS"):
                value = probe[feature_set][task]
                lines.append(
                    f"| {LABELS[setting]} | {feature_set} | {task} | {fmt(value['AUROC'])} | "
                    f"{fmt(value['balanced_accuracy'])} | {fmt(value['macro_F1'])} |"
                )
        three = probe["ALL"]["THREE_CLASS"]
        lines.append(
            f"| {LABELS[setting]} | ALL | THREE_CLASS | N/A | {fmt(three['balanced_accuracy'])} | {fmt(three['macro_F1'])} |"
        )
    lines.extend(
        [
            "",
            "- VGG has a real combined signal: ALL EXPAND-vs-SHRINK AUROC `0.7664`, balanced accuracy `0.6996`.",
            "- Flickr is only moderate: its best EXPAND-vs-SHRINK result is INTERNAL AUROC `0.6211`, balanced accuracy `0.6063`; adding all spatial features lowers it.",
            "- Three-class routing is weak: balanced accuracy `0.5179` on VGG and `0.3811` on Flickr.",
            "- The signal combination is therefore diagnostic, not reliable enough for a general adaptive router.",
            "",
            "## 8. Oracle Routing Upper Bound",
            "",
            "| Dataset | AUD | Expand-only oracle | Shrink-only oracle | Combined oracle | Combined gain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        oracle = summaries[setting]["oracle_upper_bound"]
        combined = oracle["EXPAND_SHRINK_ORACLE"]
        lines.append(
            f"| {LABELS[setting]} | {metric(oracle['AUD'])} | {metric(oracle['EXPAND_ONLY_ORACLE'])} | "
            f"{metric(oracle['SHRINK_ONLY_ORACLE'])} | {metric(combined)} | "
            f"+{fmt(combined['delta_cIoU'])}/+{fmt(combined['delta_AUC'])} |"
        )
    lines.extend(
        [
            "",
            "- VGG: expansion and shrink independently contribute about `+.0335` and `+.0322 cIoU`; combined capacity reaches `.4913/.4689`.",
            "- Flickr: shrink is dominant (`+.0520 cIoU`) but expansion still adds `+.0120`; combined capacity reaches `.8760/.6724`.",
            "- Bidirectional correction capacity is substantial and consistently larger than either one-direction oracle.",
            "",
            "## 9. Relationship With Experiment 5.1",
            "",
            "| Dataset | 5.1 group | Count | Expand | Shrink | Keep/Ambiguous |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        relation = summaries[setting]["relationship_with_5_1"]
        for group in ("PROP_ONLY", "IMG_ONLY_SHRINK", "AUD_OVER_EXPANSION", "PROP_HURT", "OGL_RESCUE"):
            value = relation[group]
            counts = value["error_type_counts"]
            lines.append(
                f"| {LABELS[setting]} | {group} | {value['count']} | {counts['EXPAND']} | {counts['SHRINK']} | {counts['KEEP_AMBIGUOUS']} |"
            )
    lines.extend(
        [
            "",
            "- PROP_ONLY maps primarily to EXPAND: `177/210` VGG and `3/4` Flickr.",
            "- IMG_ONLY+SHRINK maps almost perfectly to SHRINK: `123/125` VGG and `10/10` Flickr.",
            "- OGL rescues are mostly SHRINK: `235/357` VGG and `16/19` Flickr.",
            "- PROP hurt is overwhelmingly SHRINK or KEEP: only `4/480` VGG and `0/19` Flickr are EXPAND.",
            "- This directly explains 5.1: unconditional propagation helps the EXPAND minority but damages samples that require deletion or no change.",
            "",
            "## 10. Failure-Case Analysis",
            "",
        ]
    )
    for setting in SETTINGS:
        lines.append(f"### {LABELS[setting]}")
        lines.append("")
        for value in summaries[setting]["failure_cases"]:
            lines.append(
                f"- `{value['category']}` `{value['sample_id']}`: true `{value['error_type']}`, probe `{value['probe_prediction']}`, "
                f"AUD `{fmt(value['IoU_AUD'])}`, expand gain `{fmt(value['expand_gain'])}`, shrink gain `{fmt(value['shrink_gain'])}`, "
                f"IMG/AUD area `{fmt(value['IMG_AUD_area_ratio'])}`."
            )
        lines.append("")
    lines.extend(
        [
            "The misrouted examples show the core ambiguity: a large AUD-only region can be missing extent in one sample and removable context in another. IMG suppression is a valid operation, but the frozen model does not consistently encode the semantic status of the suppressed pixels.",
            "",
            "## 11. Final Decision",
            "",
            "**Case B - Error Structure Exists but Routing Signal Is Weak.**",
            "",
            "EXPAND and SHRINK are stable, complementary failure modes with a large combined oracle upper bound. IMG is a highly effective SHRINK candidate in the known IMG_ONLY+SHRINK and OGL-rescue groups. However, IMG/AUD area disagreement is not directionally stable across datasets, the strongest transferable single signals are only around 0.60 AUROC, and the three-class probe is weak, especially on Flickr.",
            "",
            "The evidence does not support implementing a complex adaptive expand/shrink localization module yet.",
            "",
            "## 12. Recommended Next Experiment",
            "",
            "Continue only with a narrowly scoped routing-cue study that targets the semantic status of AUD-only pixels: true object extent versus context leakage. Do not resume unconditional prototype propagation, and do not treat IMG area or disagreement magnitude alone as a SHRINK gate.",
            "",
            "No Experiment 5.3 was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    arguments = parse_args()
    summaries = load_summaries(arguments.results_root)
    for setting, summary in summaries.items():
        if not summary["completed_full_dataset"] or not summary["audit"]["passed"]:
            raise RuntimeError(f"Incomplete or failed result: {setting}")
        zero = summary["zero_training_audit"]
        if zero["parameters_with_grad"] or not zero["checkpoint_hashes_and_mtimes_unchanged"]:
            raise RuntimeError(f"Zero-training audit failed: {setting}")
    arguments.report.write_text(build_report(summaries), encoding="utf-8")
    print(arguments.report)


if __name__ == "__main__":
    main()
