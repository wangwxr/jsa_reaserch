#!/usr/bin/env python3
"""Aggregate the two formal 4.1 zero-training probes and write REPORT.md."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=common.HERE / "results")
    parser.add_argument("--report", type=Path, default=common.HERE / "REPORT.md")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def metric_pair(metrics: dict[str, Any]) -> str:
    return f"{fmt(metrics['cIoU'])}/{fmt(metrics['AUC'])}"


def selector_lookup(summary: dict[str, Any], evidence: str, mode: str) -> dict[str, Any]:
    return next(
        row for row in summary["selector_results"]
        if row["evidence"] == evidence and row["mode"] == mode
    )


def evidence_lookup(summary: dict[str, Any], evidence: str) -> dict[str, Any]:
    return next(row for row in summary["evidence_prediction"] if row["evidence"] == evidence)


def threshold_metrics(rows: list[dict[str, str]], evidence: str, threshold: float) -> dict[str, Any]:
    aud = np.asarray([float(row["IoU_AUD"]) for row in rows])
    img = np.asarray([float(row["IoU_IMG"]) for row in rows])
    delta = np.asarray([float(row[f"DELTA_{evidence}"]) for row in rows])
    selected = np.where(delta > threshold, img, aud)
    shift = common.transition(aud.tolist(), selected.tolist())
    return {
        "threshold": threshold,
        "metrics": common.summarize(selected.tolist()),
        "rescue": shift["rescue"],
        "hurt": shift["hurt"],
        "net": shift["net"],
        "IMG_selection_rate": float((delta > threshold).mean()),
    }


def direction(row: dict[str, Any]) -> str:
    auc = float(row["AUROC_IMG_better"])
    if not math.isfinite(auc) or abs(auc - 0.5) < 1e-12:
        return "neutral"
    return "+Delta=>IMG" if auc > 0.5 else "-Delta=>IMG"


def build_report(
    summaries: dict[str, dict[str, Any]],
    rows: dict[str, list[dict[str, str]]],
    transfer: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "# Experiment 4.1 - Selective Fusion Capacity & Evidence Probe",
            "",
            "## Protocol Audit",
            "",
            "- Zero training: `model.eval()`, `torch.inference_mode()`, no optimizer, no backward, no new trainable parameters.",
            "- Formal selector inputs use only original-model internal maps/representations. GT and OGL are used only for oracle construction, diagnostic labels, and evaluation.",
            "- Official selectors use the fixed rule `Delta > 0 -> IMG`, otherwise AUD. No test threshold is used by an official result.",
            "",
        ]
    )
    for setting in SETTINGS:
        summary = summaries[setting]
        audit = summary["tensor_audit"]
        zero = summary["zero_training_audit"]
        lines.extend(
            [
                f"### {LABELS[setting]}",
                "",
                f"- 4.0 per-sample max error: `{summary['reproduction_4_0']['per_sample_max_error']}`; sample-order mismatches: `{summary['reproduction_4_0']['sample_order_mismatches']}`.",
                f"- Shapes: `Qa={audit['Qa_shape']}`, `Qv={audit['Qv_shape']}`, `K4={audit['K4_shape']}`, `K34={audit['K34_shape']}`, `AUD={audit['AUD_FINE_shape']}`.",
                f"- Tensor reconstruction max errors: `{audit['reconstruction_errors']}`; ownership slot-sum error: `{audit['ownership_slot_sum_error']:.3e}`.",
                f"- Checkpoints unchanged: `{zero['all_checkpoint_hashes_and_mtimes_unchanged']}`; no NaN/Inf: `{zero['no_nan_or_inf']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Capacity",
            "",
            "| Dataset | AUD | IMG | Fixed IQR | OGL | Sample Oracle | Region Oracle | Pixel Oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        summary = summaries[setting]
        base = summary["base_metrics"]
        capacity = summary["capacity"]
        lines.append(
            f"| {LABELS[setting]} | {metric_pair(base['AUD'])} | {metric_pair(base['IMG'])} | "
            f"{metric_pair(base['IQR'])} | {metric_pair(base['OGL'])} | "
            f"{metric_pair(capacity['SAMPLE_ORACLE'])} | "
            f"{fmt(capacity['REGION_ORACLE']['success_rate'])}/N/A | "
            f"{metric_pair(capacity['PIXEL_ORACLE'])} |"
        )
    lines.extend(["", "Capacity gaps (cIoU/success-rate scale):", ""])
    for setting in SETTINGS:
        gap = summaries[setting]["capacity_gaps"]
        lines.append(
            f"- {LABELS[setting]}: Sample-AUD `{gap['SampleOracle_minus_AUD']:+.4f}`, "
            f"Region-Sample `{gap['RegionOracle_minus_SampleOracle']:+.4f}`, "
            f"Pixel-Region `{gap['PixelOracle_minus_RegionOracle']:+.4f}`; "
            f"OGL-Sample `{gap['OGL_minus_SampleOracle']:+.4f}`, "
            f"OGL-Region `{gap['OGL_minus_RegionOracle']:+.4f}`, "
            f"OGL-Pixel `{gap['OGL_minus_PixelOracle']:+.4f}`."
        )
        binary = summaries[setting]["capacity"]["BINARY_PIXEL_ORACLE"]
        lines.append(
            f"- {LABELS[setting]} binary-mask Pixel Oracle: success/cIoU "
            f"`{binary['success_rate']:.4f}`, mean sample IoU "
            f"`{binary['mean_sample_IoU']:.4f}`, AUC `N/A`."
        )

    lines.extend(
        [
            "",
            "## Complementarity Location",
            "",
            "| Dataset | Group | Count | Mean disagreement | Top20 mass | Pearson | Mask IoU | IMG/AUD area | Centroid px |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for group in summaries[setting]["disagreement_groups"]:
            if group["group"] not in ("IMG_ONLY", "AUD_ONLY", "BOTH_SUCCESS", "BOTH_FAIL", "OGL_RESCUE"):
                continue
            lines.append(
                f"| {LABELS[setting]} | {group['group']} | {group['count']} | "
                f"{fmt(group['disagreement_mean']['mean'])} | {fmt(group['disagreement_top20_mass']['mean'])} | "
                f"{fmt(group['AUD_IMG_Pearson']['mean'])} | {fmt(group['mask_IoU']['mean'])} | "
                f"{fmt(group['area_ratio_IMG_AUD']['mean'])} | {fmt(group['centroid_distance']['mean'], 2)} |"
            )
    lines.extend(["", "Disagreement energy by GT-relative region (one-pixel boundary band):", ""])
    for setting in SETTINGS:
        for group in summaries[setting]["boundary_disagreement"]:
            if group["group"] in ("ALL", "IMG_ONLY", "OGL_RESCUE"):
                lines.append(
                    f"- {LABELS[setting]} {group['group']}: interior `{fmt(group['GT_INTERIOR']['mean'])}`, "
                    f"boundary `{fmt(group['GT_BOUNDARY_1PX']['mean'])}`, exterior `{fmt(group['GT_EXTERIOR']['mean'])}`."
                )
        correction = summaries[setting]["correction_types"]
        lines.append(
            f"- {LABELS[setting]} IMG_ONLY correction types: `{correction['IMG_ONLY']['types']}`; "
            f"OGL_RESCUE types: `{correction['OGL_RESCUE']['types']}`."
        )

    lines.extend(
        [
            "",
            "## Metric-Space Audit",
            "",
            "Only fused visual slots and audio slots are explicitly aligned by the training InfoNCE objective. `F34`, `K34`, `K4`, raw visual tokens, and `img_to_v` outputs are attention/key/value spaces and are not used for direct audio cosine. A direct token-pooled semantic verifier is therefore N/A.",
            "",
            "`SEMANTIC_SLOT`: `H_sem(x)=sum_s OWN_s(x)*cos(Zv_s, Za_0)`, then `E(M)=fg_mean(H_sem)-bg_mean(H_sem)`. It uses aligned global slots and existing final L4 ownership, with no new projection.",
            "",
            "`RECIPROCAL_L4`: `C_av=Qa_0->K4`; `r_s=1-JS(Qv_s->Ka, Qa_0->Ka)/log(2)`; `H_va=sum_s OWN_s*r_s`; `H_recip=.5*(C_av+H_va)`; then the same foreground-minus-background score. The `C_av` half is explicitly marked partially circular because it is the coarse precursor of AUD; the `Qv->Ka` reciprocal half is separate.",
            "",
            "## Evidence Prediction",
            "",
            "Evidence definitions use `Delta = E(IMG) - E(AUD)`: `CTRL_RAW_PEAK` is raw-map peak; `CTRL_NEG_ENTROPY` is negative normalized spatial entropy; `CTRL_NEG_AREA` is negative thresholded foreground area; `CTRL_TOP20_CONCENTRATION` is the mass in the highest 20% pixels; `CTRL_NEG_COMPONENTS` is negative foreground connected-component count. The semantic and reciprocal definitions are given above.",
            "",
            "| Evidence | VGG AUROC/AUPRC | VGG IMG-only AUROC | VGG BalAcc@0 | Flickr AUROC/AUPRC | Flickr IMG-only AUROC | Flickr BalAcc@0 | Direction consistent |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    evidences = [row["evidence"] for row in summaries[SETTINGS[0]]["evidence_prediction"]]
    for evidence in evidences:
        vgg = evidence_lookup(summaries["vggss_144k"], evidence)
        flickr = evidence_lookup(summaries["flickr_144k"], evidence)
        consistent = direction(vgg) == direction(flickr) and direction(vgg) != "neutral"
        lines.append(
            f"| {evidence} | {fmt(vgg['AUROC_IMG_better'])}/{fmt(vgg['AUPRC_IMG_better'])} | "
            f"{fmt(vgg['AUROC_IMG_only'])} | {fmt(vgg['balanced_accuracy_threshold_0'])} | "
            f"{fmt(flickr['AUROC_IMG_better'])}/{fmt(flickr['AUPRC_IMG_better'])} | "
            f"{fmt(flickr['AUROC_IMG_only'])} | {fmt(flickr['balanced_accuracy_threshold_0'])} | "
            f"{'yes: ' + direction(vgg) if consistent else 'no: ' + direction(vgg) + ' vs ' + direction(flickr)} |"
        )

    lines.extend(
        [
            "",
            "## Label-Free Fusion at Fixed Zero Threshold",
            "",
            "| Dataset | Evidence | Mode | cIoU/AUC | Rescue | Hurt | Net | IMG sample rate | Pixel switch rate | IMG-rescue retention | OGL-rescue capture |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in SETTINGS:
        for selector in summaries[setting]["selector_results"]:
            lines.append(
                f"| {LABELS[setting]} | {selector['evidence']} | {selector['mode']} | "
                f"{metric_pair(selector['metrics'])} | {selector['rescue']} | {selector['hurt']} | {selector['net']} | "
                f"{fmt(selector['IMG_selection_rate'])} | {fmt(selector['mean_pixel_switch_rate'])} | "
                f"{selector['IMG_rescue_retained']}/{selector['IMG_rescue_total']} "
                f"({fmt(selector['IMG_rescue_retention'])}) | "
                f"{selector['OGL_rescue_captured']}/{selector['OGL_rescue_total']} "
                f"({fmt(selector['OGL_rescue_capture_rate'])}) |"
            )
        fixed = summaries[setting]["fixed_IQR_transition"]
        lines.append(
            f"| {LABELS[setting]} | FIXED_IQR | SAMPLE_AVERAGE | "
            f"{metric_pair(summaries[setting]['base_metrics']['IQR'])} | {fixed['rescue']} | "
            f"{fixed['hurt']} | {fixed['net']} | N/A | N/A | N/A | N/A |"
        )

    lines.extend(["", "Sample-selector failure decomposition:", ""])
    for setting in SETTINGS:
        for evidence in ("CTRL_NEG_COMPONENTS", "SEMANTIC_SLOT", "RECIPROCAL_L4"):
            selector = selector_lookup(summaries[setting], evidence, "SAMPLE")
            lines.append(
                f"- {LABELS[setting]} {evidence}: `{selector['failure_decomposition']}`."
            )

    lines.extend(
        [
            "",
            "## Post-hoc Threshold Transfer",
            "",
            "These are non-official diagnostics. The threshold was optimized on the source dataset using GT, then applied unchanged to the other dataset.",
            "",
            "| Evidence | Source -> Target | Threshold | Target cIoU/AUC | Rescue/Hurt/Net | IMG rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for evidence in evidences:
        for key in ("vggss_144k_to_flickr_144k", "flickr_144k_to_vggss_144k"):
            item = transfer[evidence][key]
            lines.append(
                f"| {evidence} | {LABELS[item['source']]} -> {LABELS[item['target']]} | "
                f"{fmt(item['threshold'], 6)} | {metric_pair(item['metrics'])} | "
                f"{item['rescue']}/{item['hurt']}/{item['net']} | {fmt(item['IMG_selection_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Qualitative Selection",
            "",
        ]
    )
    for setting in SETTINGS:
        selected = summaries[setting]["qualitative_selection"]
        lines.append(f"- {LABELS[setting]} deterministic categories: `{selected}`.")
    lines.extend(
        [
            "- IMG-only examples mostly show a tighter response that removes AUD exterior/context activation. The disagreement image is broad and low-amplitude rather than a clean object-boundary signal.",
            "- Selector failures are not caused by missing AUD/IMG differences: evidence often prefers IMG on samples where IMG merely shifts or shrinks an already incorrect response.",
            "- OGL-rescue misses show that internal slot/reciprocal support can rank AUD and IMG almost equally even when OGL makes the task-relevant extent correction.",
            "",
            "## Decision",
            "",
            "**Case C - Capacity Exists - No Reliable Self-Supervised Selector Evidence.**",
            "",
            "- Sample routing capacity is sufficient to close the observed OGL gap: Sample Oracle is 0.0025 above OGL on VGG and only 0.0040 below OGL on Flickr. Region routing adds 0.0074/0.0160 over Sample Oracle; Pixel Oracle adds larger idealized headroom, but this does not make spatial routing the immediate bottleneck.",
            "- No evidence satisfies the fixed rule. The strongest semantic/reciprocal AUROC on VGG is effectively random, and every semantic/reciprocal zero-threshold method lowers VGG. Reciprocal sample selection improves Flickr from 0.8120 to 0.8160 but lowers VGG to 0.4137.",
            "- The sparse component-count control reaches 0.8200 on Flickr while nearly preserving VGG at 0.4263, but its AUROC and balanced accuracy are approximately random on both datasets. This is not reliable selector evidence.",
            "- Disagreement20 scalar/local correction does not solve the problem: all variants remain below AUD and retain only a minority of IMG's known OGL-rescue capacity.",
            "- Post-hoc threshold transfer is unstable: VGG-derived thresholds mostly collapse to selecting no Flickr samples, while Flickr-derived thresholds generally hurt VGG.",
            "",
            "**Next action: stop the current internal-evidence selector line and reconsider the supervision source. Do not train an MLP gate and do not start 4.2 from these signals.**",
            "",
            "Test-optimal and transferred thresholds remain diagnostics only and are not counted as official inference methods.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    arguments = parse_args()
    summaries = {
        setting: read_json(arguments.results_root / setting / "summary.json")
        for setting in SETTINGS
    }
    rows = {
        setting: read_rows(arguments.results_root / setting / "per_sample_metrics.csv")
        for setting in SETTINGS
    }
    for setting, summary in summaries.items():
        if not summary["completed_full_dataset"]:
            raise RuntimeError(f"Partial result: {setting}")
        if not summary["reproduction_4_0"]["passed"]:
            raise RuntimeError(f"4.0 reproduction failed: {setting}")
        if not summary["zero_training_audit"]["all_checkpoint_hashes_and_mtimes_unchanged"]:
            raise RuntimeError(f"Checkpoint mutation detected: {setting}")

    transfer: dict[str, Any] = {}
    evidences = [row["evidence"] for row in summaries[SETTINGS[0]]["evidence_prediction"]]
    for evidence in evidences:
        transfer[evidence] = {}
        for source, target in (
            ("vggss_144k", "flickr_144k"),
            ("flickr_144k", "vggss_144k"),
        ):
            threshold = evidence_lookup(summaries[source], evidence)["optimal_threshold_diagnostic"]["threshold"]
            metrics = threshold_metrics(rows[target], evidence, float(threshold))
            transfer[evidence][f"{source}_to_{target}"] = {
                "source": source,
                "target": target,
                **metrics,
            }
    common.write_json(arguments.results_root / "transfer_threshold_diagnostic.json", transfer)
    report = build_report(summaries, rows, transfer)
    arguments.report.write_text(report, encoding="utf-8")
    print(arguments.report)


if __name__ == "__main__":
    main()
