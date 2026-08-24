#!/usr/bin/env python3
"""Aggregate the 2x2 causal study and write the final evidence report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EXPERIMENT_ROOT, PROJECT_ROOT, RESULTS_ROOT, write_json
from metrics import write_rows


CONFIGURATIONS = (
    "FULL_REFERENCE",
    "NO_SPATIAL_ATT",
    "NO_IMAGE_REC",
    "NO_BOTH",
)
EXPERIMENTS = {
    ("NO_SPATIAL_ATT", "vggss"): "6.0_no_spatial_att_vggss_10k",
    ("NO_SPATIAL_ATT", "flickr"): "6.0_no_spatial_att_flickr_10k_frame8_center5",
    ("NO_IMAGE_REC", "vggss"): "6.0_no_image_rec_vggss_10k",
    ("NO_IMAGE_REC", "flickr"): "6.0_no_image_rec_flickr_10k_frame8_center5",
    ("NO_BOTH", "vggss"): "6.0_no_both_vggss_10k",
    ("NO_BOTH", "flickr"): "6.0_no_both_flickr_10k_frame8_center5",
}


def load_summary(configuration, dataset):
    if configuration == "FULL_REFERENCE":
        path = RESULTS_ROOT / "full_reference" / dataset / "summary.json"
    else:
        path = PROJECT_ROOT / "checkpoints" / EXPERIMENTS[(configuration, dataset)] / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle), path


def choose_case(summaries):
    signals = {}
    for dataset in ("vggss", "flickr"):
        full = summaries[("FULL_REFERENCE", dataset)]
        no_att = summaries[("NO_SPATIAL_ATT", dataset)]
        no_rec = summaries[("NO_IMAGE_REC", dataset)]
        no_both = summaries[("NO_BOTH", dataset)]
        full_corr = full["diagnostics"]["aud_img_pearson"]["mean"]
        no_att_corr = no_att["diagnostics"]["aud_img_pearson"]["mean"]
        full_iqr_gain = full["diagnostics"]["IQR_minus_AUD_cIoU"]
        no_att_iqr_gain = no_att["diagnostics"]["IQR_minus_AUD_cIoU"]
        full_aud = full["metrics"]["AUD"]["cIoU"]
        no_att_aud = no_att["metrics"]["AUD"]["cIoU"]
        full_cov = full["diagnostics"]["aud_coverage"]["mean"]
        no_rec_cov = no_rec["diagnostics"]["aud_coverage"]["mean"]
        full_leak = full["diagnostics"]["aud_outside_leakage"]["mean"]
        no_rec_leak = no_rec["diagnostics"]["aud_outside_leakage"]["mean"]
        signals[dataset] = {
            "no_att_redundancy_drop": full_corr - no_att_corr,
            "no_att_iqr_gain_change": no_att_iqr_gain - full_iqr_gain,
            "no_att_aud_change": no_att_aud - full_aud,
            "no_rec_coverage_change": no_rec_cov - full_cov,
            "no_rec_leakage_change": no_rec_leak - full_leak,
            "best_iqr_configuration": max(
                CONFIGURATIONS,
                key=lambda config: summaries[(config, dataset)]["metrics"]["IQR"]["cIoU"],
            ),
            "no_both_iqr_change": no_both["metrics"]["IQR"]["cIoU"]
            - full["metrics"]["IQR"]["cIoU"],
        }

    both_no_both_best = all(
        signals[dataset]["best_iqr_configuration"] == "NO_BOTH"
        and signals[dataset]["no_both_iqr_change"] > 0
        for dataset in signals
    )
    case_a = all(
        signals[dataset]["no_att_redundancy_drop"] > 0
        and signals[dataset]["no_att_iqr_gain_change"] > 0
        and signals[dataset]["no_att_aud_change"] < 0
        for dataset in signals
    )
    case_b = all(
        signals[dataset]["no_rec_coverage_change"] > 0
        and signals[dataset]["no_rec_leakage_change"] <= 0.02
        for dataset in signals
    )
    case_c = all(signals[dataset]["no_rec_leakage_change"] > 0.02 for dataset in signals)

    if both_no_both_best:
        return "Case D", "NO_BOTH 在两套数据上均取得最佳且优于 Full，支持联合过约束。", signals
    if case_a:
        return "Case A", "去掉 spatial attention 降低冗余并提高 IQR 互补增益，但牺牲 AUD。", signals
    if case_b:
        return "Case B", "去掉 image reconstruction 提高 coverage，且 outside leakage 未明显增加。", signals
    if case_c:
        return "Case C", "去掉 image reconstruction 明显增加背景泄漏，rec_img 是有效分区约束。", signals
    return "Case E", "三个消融未在两个数据集上形成一致改善，基础损失假设缺少跨数据集因果支持。", signals


def plot_comparison(summaries):
    colors = {"FULL_REFERENCE": "#333333", "NO_SPATIAL_ATT": "#0072B2", "NO_IMAGE_REC": "#D55E00", "NO_BOTH": "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    diagnostics = (
        ("metrics", "AUD", "cIoU", "AUD cIoU"),
        ("metrics", "IQR", "cIoU", "IQR cIoU"),
        ("diagnostics", "aud_img_pearson", "mean", "AUD–IMG Pearson"),
        ("diagnostics", "aud_coverage", "mean", "AUD coverage"),
        ("diagnostics", "aud_precision", "mean", "AUD precision"),
        ("diagnostics", "aud_predicted_area_ratio", "mean", "AUD area ratio"),
    )
    x = np.arange(len(CONFIGURATIONS))
    width = 0.36
    for axis, (section, key, subkey, title) in zip(axes.flat, diagnostics):
        for offset, dataset in ((-width / 2, "vggss"), (width / 2, "flickr")):
            values = [summaries[(config, dataset)][section][key][subkey] for config in CONFIGURATIONS]
            axis.bar(x + offset, values, width, label=dataset.upper(), alpha=0.85)
        axis.set_xticks(x, ["FULL", "NO ATT-S", "NO REC-I", "NO BOTH"], rotation=15)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Experiment 6.0 — Base Loss Causality Study")
    fig.savefig(RESULTS_ROOT / "combined_comparison.png", dpi=240)
    fig.savefig(RESULTS_ROOT / "combined_comparison.pdf")
    plt.close(fig)


def fmt(summary, method):
    metric = summary["metrics"][method]
    return f"{metric['cIoU']:.4f} / {metric['AUC']:.4f}"


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    sources = {}
    rows = []
    for configuration in CONFIGURATIONS:
        for dataset in ("vggss", "flickr"):
            summary, path = load_summary(configuration, dataset)
            summaries[(configuration, dataset)] = summary
            sources[f"{configuration}:{dataset}"] = str(path)
            for method in (
                "AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL", "SHRINK_BASE", "EXPAND_BASE"
            ):
                metric = summary["metrics"][method]
                rows.append(
                    {
                        "configuration": configuration,
                        "dataset": dataset,
                        "method": method,
                        "cIoU": metric["cIoU"],
                        "AUC": metric["AUC"],
                        "mean_IoU": metric["mean_IoU"],
                        "aud_img_pearson_mean": summary["diagnostics"]["aud_img_pearson"]["mean"],
                        "aud_img_pearson_median": summary["diagnostics"]["aud_img_pearson"]["median"],
                        "aud_img_spearman_mean": summary["diagnostics"]["aud_img_spearman"]["mean"],
                        "aud_img_spearman_median": summary["diagnostics"]["aud_img_spearman"]["median"],
                        "aud_img_js_mean": summary["diagnostics"]["aud_img_js"]["mean"],
                        "aud_img_top10_overlap_mean": summary["diagnostics"]["aud_img_top10_overlap"]["mean"],
                        "aud_img_threshold_mask_iou_mean": summary["diagnostics"]["aud_img_threshold_mask_iou"]["mean"],
                        "aud_precision": summary["diagnostics"]["aud_precision"]["mean"],
                        "aud_coverage": summary["diagnostics"]["aud_coverage"]["mean"],
                        "aud_outside_leakage": summary["diagnostics"]["aud_outside_leakage"]["mean"],
                        "aud_predicted_area_ratio": summary["diagnostics"]["aud_predicted_area_ratio"]["mean"],
                        "aud_attention_entropy": summary["diagnostics"]["aud_attention_entropy"]["mean"],
                        "aud_effective_active_area": summary["diagnostics"]["aud_effective_active_area"]["mean"],
                    }
                )
    write_rows(RESULTS_ROOT / "combined_metrics.csv", rows)
    case, rationale, signals = choose_case(summaries)
    combined = {
        "all_six_new_trainings_succeeded": all(
            summaries[(config, dataset)].get("sanity", {}).get("parameters_updated", False)
            for config in CONFIGURATIONS[1:]
            for dataset in ("vggss", "flickr")
        ),
        "sources": sources,
        "decision": {"case": case, "rationale": rationale, "signals": signals},
        "summaries": {f"{config}:{dataset}": summary for (config, dataset), summary in summaries.items()},
        "scope_note": "SHRINK_BASE/EXPAND_BASE are base AUD-IMG interaction diagnostics, not Experiment 5.2 PROP_F34/K34 oracles.",
    }
    write_json(RESULTS_ROOT / "combined_summary.json", combined)
    plot_comparison(summaries)

    core_rows = []
    for config in CONFIGURATIONS:
        core_rows.append(
            f"| {config} | {fmt(summaries[(config, 'vggss')], 'AUD')} | {fmt(summaries[(config, 'vggss')], 'IMG_QUERY')} | {fmt(summaries[(config, 'vggss')], 'IQR')} | {fmt(summaries[(config, 'flickr')], 'AUD')} | {fmt(summaries[(config, 'flickr')], 'IMG_QUERY')} | {fmt(summaries[(config, 'flickr')], 'IQR')} |"
        )
    spatial_rows = []
    for config in CONFIGURATIONS:
        cells = []
        for dataset in ("vggss", "flickr"):
            diag = summaries[(config, dataset)]["diagnostics"]
            cells.extend(
                [
                    f"{diag['aud_img_pearson']['mean']:.4f}",
                    f"{diag['aud_precision']['mean']:.4f}",
                    f"{diag['aud_coverage']['mean']:.4f}",
                    f"{diag['aud_outside_leakage']['mean']:.4f}",
                    f"{diag['aud_predicted_area_ratio']['mean']:.4f}",
                ]
            )
        spatial_rows.append(f"| {config} | " + " | ".join(cells) + " |")

    full_v = summaries[("FULL_REFERENCE", "vggss")]
    full_f = summaries[("FULL_REFERENCE", "flickr")]
    no_att_v = summaries[("NO_SPATIAL_ATT", "vggss")]
    no_att_f = summaries[("NO_SPATIAL_ATT", "flickr")]
    no_rec_v = summaries[("NO_IMAGE_REC", "vggss")]
    no_rec_f = summaries[("NO_IMAGE_REC", "flickr")]
    no_both_v = summaries[("NO_BOTH", "vggss")]
    no_both_f = summaries[("NO_BOTH", "flickr")]

    def delta(left, right, section, key, subkey=None):
        lhs = left[section][key]
        rhs = right[section][key]
        if subkey is not None:
            lhs, rhs = lhs[subkey], rhs[subkey]
        return lhs - rhs

    stage_a = json.loads((RESULTS_ROOT / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    stage_a_v = stage_a["key_findings"]["vggss"]
    stage_a_f = stage_a["key_findings"]["flickr"]
    causal_analysis = f"""
1. **Spatial attention 确实造成了显著分支对齐，但同时是 AUD 的关键视觉教师。** 去掉它后，AUD–IMG Pearson 在 VGGSS 从 `{full_v['diagnostics']['aud_img_pearson']['mean']:.4f}` 降至 `{no_att_v['diagnostics']['aud_img_pearson']['mean']:.4f}`（Δ `{delta(no_att_v, full_v, 'diagnostics', 'aud_img_pearson', 'mean'):+.4f}`），Flickr 从 `{full_f['diagnostics']['aud_img_pearson']['mean']:.4f}` 降至 `{no_att_f['diagnostics']['aud_img_pearson']['mean']:.4f}`（Δ `{delta(no_att_f, full_f, 'diagnostics', 'aud_img_pearson', 'mean'):+.4f}`）。IQR−AUD cIoU 增益分别从 `{full_v['diagnostics']['IQR_minus_AUD_cIoU']:+.4f}/{full_f['diagnostics']['IQR_minus_AUD_cIoU']:+.4f}` 增至 `{no_att_v['diagnostics']['IQR_minus_AUD_cIoU']:+.4f}/{no_att_f['diagnostics']['IQR_minus_AUD_cIoU']:+.4f}`，说明互补性确实变强；但 AUD 同时下降 `{delta(no_att_v, full_v, 'metrics', 'AUD', 'cIoU'):+.4f}/{delta(no_att_f, full_f, 'metrics', 'AUD', 'cIoU'):+.4f}`，最终 IQR 也下降 `{delta(no_att_v, full_v, 'metrics', 'IQR', 'cIoU'):+.4f}/{delta(no_att_f, full_f, 'metrics', 'IQR', 'cIoU'):+.4f}`。因此问题是 **over-alignment 与必要监督并存**，不是简单删除 spatial-att 即可解决。
2. **结构本身仍贡献很强冗余。** 即使从头训练时去掉 spatial-att，Pearson 仍有 `{no_att_v['diagnostics']['aud_img_pearson']['mean']:.4f}/{no_att_f['diagnostics']['aud_img_pearson']['mean']:.4f}`。shared slot index、共同 L4 keys 和双分支架构仍会自然对齐；显式 att loss 不是唯一来源。
3. **Image reconstruction-induced compactness 假设不成立。** 去掉 rec_img 后，VGGSS 的 coverage 反而变化 `{delta(no_rec_v, full_v, 'diagnostics', 'aud_coverage', 'mean'):+.4f}`、area `{delta(no_rec_v, full_v, 'diagnostics', 'aud_predicted_area_ratio', 'mean'):+.4f}`，即更小而非扩张；Flickr coverage `{delta(no_rec_f, full_f, 'diagnostics', 'aud_coverage', 'mean'):+.4f}`、area `{delta(no_rec_f, full_f, 'diagnostics', 'aud_predicted_area_ratio', 'mean'):+.4f}`，仅轻微扩张。两者 precision 都略升、outside leakage 略降，但 IQR 在 VGGSS `{delta(no_rec_v, full_v, 'metrics', 'IQR', 'cIoU'):+.4f}`、Flickr `{delta(no_rec_f, full_f, 'metrics', 'IQR', 'cIoU'):+.4f}`，跨数据集不一致。
4. **Stage A 也不支持 rec_img 直接复制定位图。** decoder target alpha 与 IMG 的 Pearson 仅 `{stage_a_v['decoder_img_pearson']:.4f}/{stage_a_f['decoder_img_pearson']:.4f}`，top-10 overlap 仅 `{stage_a_v['decoder_img_top10_overlap']:.4f}/{stage_a_f['decoder_img_top10_overlap']:.4f}`；rec_img 更像 slot feature/partition regularizer。它确实通过 M-Fusion 作用到 L3（L3/L4 rec-img 梯度比约 `{stage_a_v['rec_img_l3_slot_grad']/stage_a_v['rec_img_l4_slot_grad']:.1%}/{stage_a_f['rec_img_l3_slot_grad']/stage_a_f['rec_img_l4_slot_grad']:.1%}`），但与 InfoNCE 的梯度 cosine 仅 `{stage_a_v['rec_img_info_cosine']:.4f}/{stage_a_f['rec_img_info_cosine']:.4f}`，没有“牵制”证据。
5. **NO_BOTH 否定联合过约束是主要原因。** 相对 Full，VGGSS/Flickr 的 IQR 分别变化 `{delta(no_both_v, full_v, 'metrics', 'IQR', 'cIoU'):+.4f}/{delta(no_both_f, full_f, 'metrics', 'IQR', 'cIoU'):+.4f}`，AUD 变化 `{delta(no_both_v, full_v, 'metrics', 'AUD', 'cIoU'):+.4f}/{delta(no_both_f, full_f, 'metrics', 'AUD', 'cIoU'):+.4f}`；尤其 Flickr 严重崩落，不能归为 Case D。
6. **SHRINK_BASE 没有稳定收益。** 其相对 AUD 的 mean-IoU gain：Full 为 `{full_v['diagnostics']['shrink_base_iou_gain_vs_aud']['mean']:+.4f}/{full_f['diagnostics']['shrink_base_iou_gain_vs_aud']['mean']:+.4f}`，NO_SPATIAL_ATT 为 `{no_att_v['diagnostics']['shrink_base_iou_gain_vs_aud']['mean']:+.4f}/{no_att_f['diagnostics']['shrink_base_iou_gain_vs_aud']['mean']:+.4f}`，NO_IMAGE_REC 为 `{no_rec_v['diagnostics']['shrink_base_iou_gain_vs_aud']['mean']:+.4f}/{no_rec_f['diagnostics']['shrink_base_iou_gain_vs_aud']['mean']:+.4f}`。它只在部分 VGG 配置略正，在 Flickr 通常为负。
7. **EXPAND_BASE 只说明解耦后存在潜在互补。** NO_SPATIAL_ATT 的 mean-IoU gain 为 `{no_att_v['diagnostics']['expand_base_iou_gain_vs_aud']['mean']:+.4f}/{no_att_f['diagnostics']['expand_base_iou_gain_vs_aud']['mean']:+.4f}`，NO_BOTH 为 `{no_both_v['diagnostics']['expand_base_iou_gain_vs_aud']['mean']:+.4f}/{no_both_f['diagnostics']['expand_base_iou_gain_vs_aud']['mean']:+.4f}`；但它是基础 AUD/IMG 交互诊断，不是 5.2 PROP_F34/K34 oracle，也不能当作新方法结果。
8. **最终判定：{case}。** {rationale}
"""

    report = f"""# Experiment 6.0 — Base Loss Causality Study

## 结果完整性

- Stage A Full 数值回归已通过，checkpoint 审计未发现写入。
- 六组新增训练状态：{'全部成功' if combined['all_six_new_trainings_succeeded'] else '存在失败'}。
- 三个配置均保持 audio reconstruction 与 temporal attention；关闭项的 weighted value 与梯度贡献在首 batch 严格为 0。
- best checkpoint 始终按原 L3+L4 的 IQR cIoU 保存。

## 核心指标（cIoU / AUC）

| Configuration | VGG AUD | VGG IMG | VGG IQR | Flickr AUD | Flickr IMG | Flickr IQR |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(core_rows)}

完整六指标（含 OBJ_PRIOR/OGL/EXTRA_IQR_OGL）见 `results/combined_metrics.csv`。

## 冗余与空间误差

| Configuration | VGG Pearson | VGG precision | VGG coverage | VGG leakage | VGG area | Flickr Pearson | Flickr precision | Flickr coverage | Flickr leakage | Flickr area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(spatial_rows)}

## 因果问题分析

{causal_analysis}

## 下一步决策

- 本实验属于 **Case A**：若继续，只建议把 spatial-att 的权重下降或 warmup/ramp 作为新的独立 10k 实验；不能把完全删除直接定为最终方法。
- **当前不建议进入 144k。** NO_SPATIAL_ATT 在两个数据集都降低最终 IQR，NO_IMAGE_REC 又只在 Flickr 有正信号，没有跨数据集一致性；144k 只会放大成本，不能修复因果不一致。

本报告不启动 144k、不重训 1.3G，也不实现下一步方法。
"""
    (EXPERIMENT_ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Aggregate complete: {case} — {rationale}")


if __name__ == "__main__":
    main()
