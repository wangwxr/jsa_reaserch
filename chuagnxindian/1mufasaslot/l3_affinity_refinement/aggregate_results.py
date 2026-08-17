#!/usr/bin/env python3
"""Merge four refinement sweeps and print default/diagnostic reference tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
EXPERIMENT_ORDER = ["vggss_10k", "vggss_144k", "flickr_10k", "flickr_144k"]
DISPLAY_NAMES = {
    "vggss_10k": "VGG10",
    "vggss_144k": "VGG144",
    "flickr_10k": "Flickr10",
    "flickr_144k": "Flickr144",
}

# Reference numbers only. No reference map/model is loaded by this experiment.
OGL_REFERENCES = {
    "vggss_10k": {
        "JSA_OGL": (0.4190, 0.4176),
        "L3L4_OGL": (0.4432, 0.4292),
        "JSA_source": "checkpoints/jsa_vggss_10k/jsa_vggss_10k_clean_test_vggss_final_20260810_002316.log",
        "L3L4_source": "checkpoints/mufasa_ablation2_l3_l4_ablation_vggss_10k/logs/mufasa_ablation2_l3_l4_ablation_vggss_10k_test_vggss_vggss_best_20260815_181700.log",
    },
    "vggss_144k": {
        "JSA_OGL": (0.4294, 0.4258),
        "L3L4_OGL": (0.4343, 0.4307),
        "JSA_source": "checkpoints/jsa_vggss_144k/logs/jsa_vggss_144k_test_vggss_vggss_best_20260814_003536.log",
        "L3L4_source": "checkpoints/mufasa_ablation2_l3_l4_ablation_vggss_144k/logs/mufasa_ablation2_l3_l4_ablation_vggss_144k_test_vggss_vggss_best_20260815_181724.log",
    },
    "flickr_10k": {
        "JSA_OGL": (0.8160, 0.6242),
        "L3L4_OGL": (0.8400, 0.6154),
        "JSA_source": "checkpoints/jsa_flickr_10k_frame8_center5/logs/jsa_flickr_10k_frame8_center5_test_flickr_flickr_best_20260812_065541.log",
        "L3L4_source": "checkpoints/mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5/logs/mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5_test_flickr_flickr_best_20260815_181605.log",
    },
    "flickr_144k": {
        "JSA_OGL": (0.8440, 0.6250),
        "L3L4_OGL": (0.8440, 0.6392),
        "JSA_source": "checkpoints/jsa_flickr_144k_frame8_center5/logs/jsa_flickr_144k_frame8_center5_test_flickr_flickr_best_20260813_233656.log",
        "L3L4_source": "checkpoints/mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5/logs/mufasa_ablation2_l3_l4_ablation_flickr_144k_frame8_center5_test_flickr_flickr_best_20260815_181612.log",
    },
}


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        path = (
            OUTPUT_ROOT
            / experiment
            / f"l3_affinity_refinement_results_{experiment}.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["experiment"] = experiment
                for key in (
                    "cIoU",
                    "AUC",
                    "mean_sample_cIoU",
                    "delta_cIoU_vs_AUD",
                    "delta_AUC_vs_AUD",
                ):
                    row[key] = float(row[key])
                row["tau_aff"] = None if row["tau_aff"] == "" else float(row["tau_aff"])
                row["alpha"] = None if row["alpha"] == "" else float(row["alpha"])
                row["num_samples"] = int(row["num_samples"])
                rows.append(row)
    expected_rows = len(EXPERIMENT_ORDER) * 22
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} sweep rows, found {len(rows)}")
    return rows


def write_rows(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select(
    rows: list[dict[str, Any]],
    experiment: str,
    method: str,
    tau_aff: float | None,
    alpha: float | None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["experiment"] == experiment
        and row["method"] == method
        and row["tau_aff"] == tau_aff
        and row["alpha"] == alpha
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one row for {experiment}/{method}/{tau_aff}/{alpha}, "
            f"found {len(matches)}"
        )
    return matches[0]


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def main() -> None:
    rows = read_rows()
    combined_path = HERE / "l3_affinity_refinement_results.csv"
    csv_fields = [
        "dataset",
        "split",
        "checkpoint",
        "method",
        "native_resolution",
        "tau_aff",
        "alpha",
        "cIoU",
        "AUC",
        "mean_sample_cIoU",
        "num_samples",
        "delta_cIoU_vs_AUD",
        "delta_AUC_vs_AUD",
    ]
    write_rows(combined_path, rows, csv_fields)

    default_specs = [
        ("AUD_L4", None, None, "AUD_L4"),
        ("L3_AFFINITY", 0.1, None, "L3_AFFINITY"),
        ("L3_NATIVE_REFINED", 0.1, 0.5, "L3_NATIVE_REFINE"),
        ("L3_POOLED7_REFINED", 0.1, 0.5, "L3_POOLED_REFINE"),
    ]
    default_body: list[list[str]] = []
    for method, tau_aff, alpha, label in default_specs:
        output = [label]
        for experiment in EXPERIMENT_ORDER:
            row = select(rows, experiment, method, tau_aff, alpha)
            output.append(f"{row['cIoU']:.4f} / {row['AUC']:.4f}")
        default_body.append(output)
    default_table = markdown_table(
        ["Default τ=0.1, α=0.5"]
        + [DISPLAY_NAMES[key] for key in EXPERIMENT_ORDER],
        default_body,
    )

    diagnostic_rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        for method in (
            "L3_AFFINITY",
            "L3_NATIVE_REFINED",
            "L3_POOLED7_REFINED",
        ):
            candidates = [
                row
                for row in rows
                if row["experiment"] == experiment and row["method"] == method
            ]
            best = max(candidates, key=lambda row: (row["cIoU"], row["AUC"]))
            diagnostic_rows.append(best)
    diagnostic_path = HERE / "diagnostic_best.csv"
    write_rows(diagnostic_path, diagnostic_rows, ["experiment"] + csv_fields)

    resolution_rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        default_native = select(
            rows, experiment, "L3_NATIVE_REFINED", 0.1, 0.5
        )
        default_pooled = select(
            rows, experiment, "L3_POOLED7_REFINED", 0.1, 0.5
        )
        best_native = max(
            (
                row
                for row in rows
                if row["experiment"] == experiment
                and row["method"] == "L3_NATIVE_REFINED"
            ),
            key=lambda row: (row["cIoU"], row["AUC"]),
        )
        best_pooled = max(
            (
                row
                for row in rows
                if row["experiment"] == experiment
                and row["method"] == "L3_POOLED7_REFINED"
            ),
            key=lambda row: (row["cIoU"], row["AUC"]),
        )
        resolution_rows.append(
            {
                "experiment": experiment,
                "default_native_cIoU": default_native["cIoU"],
                "default_native_AUC": default_native["AUC"],
                "default_pooled_cIoU": default_pooled["cIoU"],
                "default_pooled_AUC": default_pooled["AUC"],
                "default_native_minus_pooled_cIoU": default_native["cIoU"]
                - default_pooled["cIoU"],
                "default_native_minus_pooled_AUC": default_native["AUC"]
                - default_pooled["AUC"],
                "best_native_tau_aff": best_native["tau_aff"],
                "best_native_alpha": best_native["alpha"],
                "best_native_cIoU": best_native["cIoU"],
                "best_native_AUC": best_native["AUC"],
                "best_pooled_tau_aff": best_pooled["tau_aff"],
                "best_pooled_alpha": best_pooled["alpha"],
                "best_pooled_cIoU": best_pooled["cIoU"],
                "best_pooled_AUC": best_pooled["AUC"],
                "best_native_minus_pooled_cIoU": best_native["cIoU"]
                - best_pooled["cIoU"],
                "best_native_minus_pooled_AUC": best_native["AUC"]
                - best_pooled["AUC"],
            }
        )
    resolution_path = HERE / "native_vs_pooled_comparison.csv"
    write_rows(resolution_path, resolution_rows, list(resolution_rows[0]))

    ogl_rows: list[dict[str, Any]] = []
    ogl_body: list[list[str]] = []
    for experiment in EXPERIMENT_ORDER:
        native_candidates = [
            row
            for row in rows
            if row["experiment"] == experiment
            and row["method"] == "L3_NATIVE_REFINED"
        ]
        all_refined_candidates = [
            row
            for row in rows
            if row["experiment"] == experiment
            and row["method"] in {"L3_NATIVE_REFINED", "L3_POOLED7_REFINED"}
        ]
        best_native = max(
            native_candidates, key=lambda row: (row["cIoU"], row["AUC"])
        )
        best_overall = max(
            all_refined_candidates, key=lambda row: (row["cIoU"], row["AUC"])
        )
        references = OGL_REFERENCES[experiment]
        jsa_ciou, jsa_auc = references["JSA_OGL"]
        l3l4_ciou, l3l4_auc = references["L3L4_OGL"]
        ogl_rows.append(
            {
                "experiment": experiment,
                "best_native_tau_aff": best_native["tau_aff"],
                "best_native_alpha": best_native["alpha"],
                "best_native_cIoU": best_native["cIoU"],
                "best_native_AUC": best_native["AUC"],
                "best_overall_no_OGL_method": best_overall["method"],
                "best_overall_tau_aff": best_overall["tau_aff"],
                "best_overall_alpha": best_overall["alpha"],
                "best_overall_no_OGL_cIoU": best_overall["cIoU"],
                "best_overall_no_OGL_AUC": best_overall["AUC"],
                "JSA_OGL_cIoU": jsa_ciou,
                "JSA_OGL_AUC": jsa_auc,
                "native_gap_cIoU_vs_JSA_OGL": best_native["cIoU"] - jsa_ciou,
                "native_gap_AUC_vs_JSA_OGL": best_native["AUC"] - jsa_auc,
                "overall_gap_cIoU_vs_JSA_OGL": best_overall["cIoU"] - jsa_ciou,
                "overall_gap_AUC_vs_JSA_OGL": best_overall["AUC"] - jsa_auc,
                "L3L4_OGL_cIoU": l3l4_ciou,
                "L3L4_OGL_AUC": l3l4_auc,
                "native_gap_cIoU_vs_L3L4_OGL": best_native["cIoU"] - l3l4_ciou,
                "native_gap_AUC_vs_L3L4_OGL": best_native["AUC"] - l3l4_auc,
                "overall_gap_cIoU_vs_L3L4_OGL": best_overall["cIoU"] - l3l4_ciou,
                "overall_gap_AUC_vs_L3L4_OGL": best_overall["AUC"] - l3l4_auc,
                "JSA_reference_source": references["JSA_source"],
                "L3L4_reference_source": references["L3L4_source"],
            }
        )
        ogl_body.append(
            [
                DISPLAY_NAMES[experiment],
                f"τ={best_native['tau_aff']} α={best_native['alpha']}: "
                f"{best_native['cIoU']:.4f} / {best_native['AUC']:.4f}",
                f"{best_overall['method']} τ={best_overall['tau_aff']} "
                f"α={best_overall['alpha']}: "
                f"{best_overall['cIoU']:.4f} / {best_overall['AUC']:.4f}",
                f"{jsa_ciou:.4f} / {jsa_auc:.4f}",
                f"{l3l4_ciou:.4f} / {l3l4_auc:.4f}",
            ]
        )
    ogl_path = HERE / "ogl_reference_comparison.csv"
    write_rows(ogl_path, ogl_rows, list(ogl_rows[0]))
    ogl_table = markdown_table(
        [
            "Dataset",
            "Best native no-OGL",
            "Best any refinement",
            "JSA OGL ref",
            "L3+L4 OGL ref",
        ],
        ogl_body,
    )

    summary = (
        "DEFAULT RESULTS (all maps evaluated with the unchanged JSA protocol)\n\n"
        + default_table
        + "\n\nPOST-HOC DIAGNOSTIC BEST VS OGL NUMERIC REFERENCES\n\n"
        + ogl_table
        + "\n\nOGL/object-prior maps were not loaded or used; the table contains reference numbers only.\n"
    )
    summary_path = HERE / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Combined CSV: {combined_path}")
    print(f"Diagnostic best CSV: {diagnostic_path}")
    print(f"Native-vs-pooled CSV: {resolution_path}")
    print(f"OGL numeric-reference CSV: {ogl_path}")


if __name__ == "__main__":
    main()
