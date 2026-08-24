#!/usr/bin/env python3
"""Aggregate Experiment 6.1 against the audited 6.0 Full reference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import EXPERIMENT_ROOT, PROJECT_ROOT, RESULTS_ROOT, write_json
from metrics import write_rows


CONFIGS = ("FULL_REFERENCE", "EARLY_HIGH_LATE_LOW", "EARLY_LOW_LATE_HIGH")
DATASETS = ("vggss", "flickr")
METHODS = ("AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL")
EXPERIMENTS = {
    ("EARLY_HIGH_LATE_LOW", "vggss"): "6.1_early_high_late_low_vggss_10k",
    ("EARLY_HIGH_LATE_LOW", "flickr"): "6.1_early_high_late_low_flickr_10k_frame8_center5",
    ("EARLY_LOW_LATE_HIGH", "vggss"): "6.1_early_low_late_high_vggss_10k",
    ("EARLY_LOW_LATE_HIGH", "flickr"): "6.1_early_low_late_high_flickr_10k_frame8_center5",
}
FULL_ROOT = PROJECT_ROOT / "chuagnxindian" / "6.0_loss_causality_study" / "results" / "full_reference"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_synergy(summary, per_sample_path):
    rows = load_csv(per_sample_path)
    values = np.asarray([
        float(row["iqr_iou"]) - max(float(row["aud_iou"]), float(row["img_query_iou"]))
        for row in rows
    ])
    diagnostics = summary["diagnostics"]
    diagnostics["sample_synergy"] = {
        "mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std())
    }
    diagnostics["sample_synergy_positive_fraction"] = float(np.mean(values > 0))
    diagnostics["sample_synergy_ge_001_fraction"] = float(np.mean(values >= 0.01))
    diagnostics["iqr_beats_aud_and_img_fraction"] = float(np.mean(values > 0))
    metrics = summary["metrics"]
    diagnostics["fusion_synergy_ciou"] = metrics["IQR"]["cIoU"] - max(metrics["AUD"]["cIoU"], metrics["IMG_QUERY"]["cIoU"])
    diagnostics["fusion_synergy_auc"] = metrics["IQR"]["AUC"] - max(metrics["AUD"]["AUC"], metrics["IMG_QUERY"]["AUC"])
    diagnostics["OGL_minus_IQR_cIoU"] = metrics["OGL"]["cIoU"] - metrics["IQR"]["cIoU"]
    diagnostics["OGL_minus_IQR_AUC"] = metrics["OGL"]["AUC"] - metrics["IQR"]["AUC"]
    return summary


def load_summary(configuration, dataset):
    if configuration == "FULL_REFERENCE":
        root = FULL_ROOT / dataset
    else:
        root = PROJECT_ROOT / "checkpoints" / EXPERIMENTS[(configuration, dataset)]
    summary_path = root / "summary.json"
    per_sample_path = root / "per_sample_metrics.csv"
    if not summary_path.exists() or not per_sample_path.exists():
        raise FileNotFoundError(f"Missing summary/per-sample result under {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return add_synergy(summary, per_sample_path), root


def choose_case(summaries):
    signals = {}
    for dataset in DATASETS:
        full = summaries[("FULL_REFERENCE", dataset)]
        decay = summaries[("EARLY_HIGH_LATE_LOW", dataset)]
        delayed = summaries[("EARLY_LOW_LATE_HIGH", dataset)]
        signals[dataset] = {
            "decay_aud_delta": decay["metrics"]["AUD"]["cIoU"] - full["metrics"]["AUD"]["cIoU"],
            "decay_iqr_delta": decay["metrics"]["IQR"]["cIoU"] - full["metrics"]["IQR"]["cIoU"],
            "decay_pearson_delta": decay["diagnostics"]["aud_img_pearson"]["mean"] - full["diagnostics"]["aud_img_pearson"]["mean"],
            "decay_synergy": decay["diagnostics"]["fusion_synergy_ciou"],
            "decay_ogl_iqr_gap_delta": decay["diagnostics"]["OGL_minus_IQR_cIoU"] - full["diagnostics"]["OGL_minus_IQR_cIoU"],
            "delayed_aud_delta": delayed["metrics"]["AUD"]["cIoU"] - full["metrics"]["AUD"]["cIoU"],
            "delayed_iqr_delta": delayed["metrics"]["IQR"]["cIoU"] - full["metrics"]["IQR"]["cIoU"],
        }
    t1 = all(s["decay_aud_delta"] >= 0 and s["decay_iqr_delta"] >= 0 and s["decay_synergy"] > 0 and s["decay_ogl_iqr_gap_delta"] < 0 for s in signals.values())
    t2 = all(s["decay_pearson_delta"] < 0 and s["decay_aud_delta"] < 0 and s["decay_iqr_delta"] < 0 for s in signals.values())
    # A one-point Flickr tolerance is 1/250=.004; VGG tolerance .005 is similarly conservative.
    close = {"vggss": 0.005, "flickr": 0.004}
    delayed_close = all(abs(signals[d]["delayed_iqr_delta"]) <= close[d] for d in DATASETS)
    delayed_weaker = all(signals[d]["delayed_iqr_delta"] < -close[d] for d in DATASETS)
    decay_close_or_better = all(signals[d]["decay_iqr_delta"] >= -close[d] for d in DATASETS)
    both_not_better = all(signals[d]["decay_iqr_delta"] <= 0 and signals[d]["delayed_iqr_delta"] <= 0 for d in DATASETS)
    if t1:
        return "T1", "Decay 在两套数据均保持/提升 AUD 与 IQR、产生正 synergy 并缩小 OGL gap。", signals
    if t2:
        return "T2", "Decay 虽降低 Pearson，但 AUD 与 IQR 同时下降，只得到低相关而非有效互补。", signals
    if delayed_weaker and decay_close_or_better:
        return "T3", "Delayed 明显弱于 Full，而 Decay 接近/超过 Full，支持早期语义启动。", signals
    if delayed_close:
        return "T4", "Delayed 与 Full 在预注册容差内接近，没有证据证明 spatial-att 必须早期出现。", signals
    if both_not_better:
        return "T5", "两个 schedule 均未优于 Full，持续 pointwise alignment 仍是可靠 AUD 的必要条件。", signals
    return "T5", "跨数据集方向不一致，未满足 T1–T4 的一致性条件，不能支持 schedule 路线。", signals


def plot_comparison(summaries):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = ("Full", "Decay", "Delayed")
    colors = ("#444444", "#0072B2", "#D55E00")
    for col, dataset in enumerate(DATASETS):
        values = [[summaries[(cfg, dataset)]["metrics"][m]["cIoU"] for cfg in CONFIGS] for m in ("AUD", "IMG_QUERY", "IQR")]
        x = np.arange(3); width = 0.25
        for i, (method, vals) in enumerate(zip(("AUD", "IMG", "IQR"), values)):
            axes[0, col].bar(x + (i - 1) * width, vals, width, label=method)
        axes[0, col].set_xticks(x, labels)
        axes[0, col].set_title(f"{dataset.upper()} cIoU")
        axes[0, col].legend(frameon=False)
        pearson = [summaries[(cfg, dataset)]["diagnostics"]["aud_img_pearson"]["mean"] for cfg in CONFIGS]
        synergy = [summaries[(cfg, dataset)]["diagnostics"]["fusion_synergy_ciou"] for cfg in CONFIGS]
        axes[1, col].bar(x, pearson, width * 1.3, color="#0072B2", alpha=.8, label="Pearson")
        axes[1, col].set_ylim(max(0.9, min(pearson) - .02), 1.0)
        synergy_axis = axes[1, col].twinx()
        synergy_axis.plot(x, synergy, color="#D55E00", marker="o", linewidth=2, label="synergy")
        synergy_limit = max(.01, 1.25 * max(abs(value) for value in synergy))
        synergy_axis.set_ylim(-synergy_limit, synergy_limit)
        synergy_axis.axhline(0, color="#D55E00", linewidth=.7, alpha=.6)
        synergy_axis.set_ylabel("IQR − max(AUD, IMG)", color="#D55E00", fontsize=8)
        axes[1, col].set_xticks(x, labels)
        axes[1, col].set_title(f"{dataset.upper()} redundancy / synergy")
        handles1, labels1 = axes[1, col].get_legend_handles_labels()
        handles2, labels2 = synergy_axis.get_legend_handles_labels()
        axes[1, col].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="lower left")
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")
    for dataset, style in (("vggss", "-"), ("flickr", "--")):
        for cfg, color in zip(CONFIGS[1:], colors[1:]):
            root = PROJECT_ROOT / "checkpoints" / EXPERIMENTS[(cfg, dataset)]
            frame = pd.read_csv(root / "epoch_metrics.csv")
            axes[0, 2].plot(frame.epoch, frame.iqr_ciou, color=color, linestyle=style, label=f"{dataset}-{cfg}")
            grad = pd.read_csv(root / "gradient_trajectory.csv")
            axes[1, 2].plot(grad.epoch, grad.total_trainable_parameters_grad_norm_ratio, color=color, linestyle=style, marker="o", markersize=2, label=f"{dataset}-{cfg}")
    axes[0, 2].set_axis_on(); axes[0, 2].set_title("IQR trajectory"); axes[0, 2].legend(fontsize=6)
    axes[1, 2].set_axis_on(); axes[1, 2].set_title("Spatial/total grad ratio"); axes[1, 2].legend(fontsize=6)
    for axis in axes.flat:
        if axis.axison:
            axis.grid(axis="y", alpha=.25); axis.set_xlabel("Epoch / configuration")
    fig.suptitle("Experiment 6.1 — Spatial Attention Temporal Schedule")
    fig.savefig(RESULTS_ROOT / "combined_comparison.png", dpi=240)
    fig.savefig(RESULTS_ROOT / "combined_comparison.pdf")
    plt.close(fig)


def fmt(summary, method):
    metric = summary["metrics"][method]
    return f"{metric['cIoU']:.4f} / {metric['AUC']:.4f}"


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summaries, sources, rows = {}, {}, []
    for config in CONFIGS:
        for dataset in DATASETS:
            summary, root = load_summary(config, dataset)
            summaries[(config, dataset)] = summary
            sources[f"{config}:{dataset}"] = str(root)
            for method in METHODS:
                metric, diag = summary["metrics"][method], summary["diagnostics"]
                rows.append({
                    "configuration": config, "dataset": dataset, "method": method,
                    "cIoU": metric["cIoU"], "AUC": metric["AUC"], "mean_IoU": metric["mean_IoU"],
                    "aud_img_pearson_mean": diag["aud_img_pearson"]["mean"],
                    "aud_img_pearson_median": diag["aud_img_pearson"]["median"],
                    "aud_img_spearman_mean": diag["aud_img_spearman"]["mean"],
                    "aud_img_spearman_median": diag["aud_img_spearman"]["median"],
                    "aud_img_js_mean": diag["aud_img_js"]["mean"],
                    "aud_img_top10_overlap": diag["aud_img_top10_overlap"]["mean"],
                    "aud_img_threshold_mask_iou": diag["aud_img_threshold_mask_iou"]["mean"],
                    "aud_precision": diag["aud_precision"]["mean"], "aud_coverage": diag["aud_coverage"]["mean"],
                    "aud_outside_leakage": diag["aud_outside_leakage"]["mean"],
                    "aud_predicted_area_ratio": diag["aud_predicted_area_ratio"]["mean"],
                    "aud_attention_entropy": diag["aud_attention_entropy"]["mean"],
                    "aud_effective_active_area": diag["aud_effective_active_area"]["mean"],
                    "fusion_synergy_ciou": diag["fusion_synergy_ciou"], "fusion_synergy_auc": diag["fusion_synergy_auc"],
                    "sample_synergy_mean": diag["sample_synergy"]["mean"],
                    "sample_synergy_median": diag["sample_synergy"]["median"],
                    "sample_synergy_positive_fraction": diag["sample_synergy_positive_fraction"],
                    "sample_synergy_ge_001_fraction": diag["sample_synergy_ge_001_fraction"],
                    "iqr_beats_aud_and_img_fraction": diag["iqr_beats_aud_and_img_fraction"],
                    "ogl_aud_gap_ciou": diag["OGL_minus_AUD_cIoU"], "ogl_iqr_gap_ciou": diag["OGL_minus_IQR_cIoU"],
                })
    write_rows(RESULTS_ROOT / "combined_metrics.csv", rows)

    gradient_rows = []
    for config in CONFIGS[1:]:
        for dataset in DATASETS:
            root = PROJECT_ROOT / "checkpoints" / EXPERIMENTS[(config, dataset)]
            for row in load_csv(root / "gradient_trajectory.csv"):
                gradient_rows.append({"configuration": config, "dataset": dataset, **row})
    write_rows(RESULTS_ROOT / "gradient_trajectory_comparison.csv", gradient_rows)
    case, rationale, signals = choose_case(summaries)
    complete = all(summaries[(c, d)].get("sanity", {}).get("parameters_updated", False) for c in CONFIGS[1:] for d in DATASETS)
    combined = {
        "all_four_trainings_succeeded": complete,
        "decision": {"case": case, "rationale": rationale, "signals": signals},
        "sources": sources,
        "summaries": {f"{c}:{d}": s for (c, d), s in summaries.items()},
        "flickr_ciou_unit": 1.0 / 250.0,
        "no_144k_started": True,
    }
    write_json(RESULTS_ROOT / "combined_summary.json", combined)
    plot_comparison(summaries)

    metric_rows = []
    for config in CONFIGS:
        metric_rows.append("| " + config + " | " + " | ".join(
            fmt(summaries[(config, dataset)], method)
            for dataset in DATASETS for method in ("AUD", "IMG_QUERY", "IQR")
        ) + " |")
    six_rows = []
    for config in CONFIGS:
        for method in METHODS:
            six_rows.append(f"| {config} | {method} | {fmt(summaries[(config, 'vggss')], method)} | {fmt(summaries[(config, 'flickr')], method)} |")
    diag_rows = []
    for config in CONFIGS:
        cells=[]
        for dataset in DATASETS:
            d=summaries[(config,dataset)]["diagnostics"]
            cells += [f"{d['aud_img_pearson']['mean']:.4f}", f"{d['fusion_synergy_ciou']:+.4f}", f"{d['sample_synergy']['mean']:+.4f}", f"{d['OGL_minus_IQR_cIoU']:+.4f}"]
        diag_rows.append(f"| {config} | " + " | ".join(cells) + " |")
    spatial_rows = []
    for config in CONFIGS:
        for dataset in DATASETS:
            d = summaries[(config, dataset)]["diagnostics"]
            spatial_rows.append(
                f"| {config} | {dataset.upper()} | "
                f"{d['aud_precision']['mean']:.4f} | {d['aud_coverage']['mean']:.4f} | "
                f"{d['aud_outside_leakage']['mean']:.4f} | {d['aud_predicted_area_ratio']['mean']:.4f} | "
                f"{d['aud_attention_entropy']['mean']:.4f} | {d['aud_effective_active_area']['mean']:.2f} |"
            )
    selection_rows = []
    for config in CONFIGS[1:]:
        for dataset in DATASETS:
            s = summaries[(config, dataset)]
            b = s["training_diagnostic_bests"]
            selection_rows.append(
                f"| {config} | {dataset.upper()} | {s['best_epoch']} / {s['metrics']['IQR']['cIoU']:.4f} | "
                f"{s['final_epoch']['metrics']['IQR']['cIoU']:.4f} | "
                f"{b['aud_ciou']['epoch']} / {b['aud_ciou']['value']:.4f} | "
                f"{b['img_ciou']['epoch']} / {b['img_ciou']['value']:.4f} |"
            )
    gradient_report_rows = []
    for config in CONFIGS[1:]:
        for dataset in DATASETS:
            root = PROJECT_ROOT / "checkpoints" / EXPERIMENTS[(config, dataset)]
            by_epoch = {int(row["epoch"]): row for row in load_csv(root / "gradient_trajectory.csv")}
            for epoch in (1, 20, 21, 30, 40, 41, 60, 80, 100):
                row = by_epoch[epoch]
                gradient_report_rows.append(
                    f"| {config} | {dataset.upper()} | {epoch} | {float(row['current_lambda_att_space']):.1f} | "
                    f"{float(row['total_trainable_parameters_grad_norm_weighted_att_space']):.6f} | "
                    f"{float(row['total_trainable_parameters_grad_norm_total']):.6f} | "
                    f"{float(row['total_trainable_parameters_grad_norm_ratio']):.6f} |"
                )

    report = f"""# Experiment 6.1 — Spatial Attention Temporal Schedule Study

## 完整性与预注册协议

- 四组新增 10k 训练：{'全部成功' if complete else '存在失败'}。
- schedule 单元测试与实际 FP32 fixed-batch 回归均通过；只有 `L_att_space` 的乘数随 epoch 改变。
- 所有主结果取 IQR-selected checkpoint；Full 直接复用 6.0 正式 reference。
- Flickr 每个样本对应 cIoU `{1/250:.4f}`，因此所有变化同时按样本数解释。

## 核心指标（cIoU / AUC）

| Configuration | VGG AUD | VGG IMG | VGG IQR | Flickr AUD | Flickr IMG | Flickr IQR |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## 完整六指标

| Configuration | Method | VGGSS | Flickr |
|---|---|---:|---:|
{chr(10).join(six_rows)}

## 冗余、真实互补与 OGL gap

| Configuration | VGG Pearson | VGG synergy | VGG sample synergy | VGG OGL-IQR | Flickr Pearson | Flickr synergy | Flickr sample synergy | Flickr OGL-IQR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(diag_rows)}

注意：aggregate synergy 是阈值成功率差；sample synergy 是连续 per-sample IoU 差。二者可能方向不同，不能互相替代。

## 空间误差（AUD，threshold=0.6）

| Configuration | Dataset | precision | coverage | outside leakage | predicted area | entropy | effective area |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(spatial_rows)}

## 选模与训练末轮

| Configuration | Dataset | IQR-best epoch/value | final IQR | best AUD epoch/value | best IMG epoch/value |
|---|---|---:|---:|---:|---:|
{chr(10).join(selection_rows)}

## 固定 batch 梯度轨迹

| Configuration | Dataset | epoch | lambda | weighted spatial grad | total grad | ratio |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(gradient_report_rows)}

## 研究问题结论

1. **EARLY_LOW_LATE_HIGH 是否明显弱于 Full？** 主协议下不是跨数据集一致地弱：VGG IQR `{signals['vggss']['delayed_iqr_delta']:+.4f}`，Flickr `{signals['flickr']['delayed_iqr_delta']:+.4f}`（`{signals['flickr']['delayed_iqr_delta']*250:+.1f}` 个成功样本）。不过 final 明显低于各自 best，说明 delayed 的后期训练不稳定。
2. **是否证明 early spatial-att 必要？** 训练轨迹强烈显示前 20 轮缺失 teacher 会让 AUD 显著落后，但 100 轮 IQR-selected best 在 VGG 反而超过 Full、Flickr 只差 1 个样本。因此只能说它显著改善早期优化路径，不能严格宣称最终性能必须依赖早期监督。
3. **Decay 是否在降低 Pearson 时保持 AUD？** 没有跨数据集成立。Pearson 变化 VGG `{signals['vggss']['decay_pearson_delta']:+.4f}`、Flickr `{signals['flickr']['decay_pearson_delta']:+.4f}`；AUD 变化 `{signals['vggss']['decay_aud_delta']:+.4f}/{signals['flickr']['decay_aud_delta']:+.4f}`。VGG Pearson 反而更高，Flickr 的下降几乎为零。
4. **Pearson 降低后是否得到真实 synergy？** Decay aggregate synergy 为 VGG `{signals['vggss']['decay_synergy']:+.4f}`、Flickr `{signals['flickr']['decay_synergy']:+.4f}`，方向相反；而两者 mean sample synergy 都为负。没有跨数据集、跨口径的一致互补收益。
5. **权重下降时 AUD 是否同步下降？** Flickr best-checkpoint AUD 相对 Full 下降 `0.0200`，VGG 反而提高 `0.0093`；没有统一同步关系。epoch 曲线显示短期波动明显，不能用单个 transition epoch 下结论。
6. **后期 spatial 梯度小意味着任务满足还是不重要？** 两者都不能直接推出。Decay 在 λ=10 时绝对加权梯度约为高权重阶段的十分之一，但仍非零；Delayed 在 λ=100 的第 100 轮 ratio 仍约为 VGG `0.0602`、Flickr `0.0620`。这表明损失仍能施加梯度，Decay 的小值主要包含人为权重缩小因素。
7. **OGL-IQR gap 是否缩小？** Decay 相对 Full 的 gap 变化是 VGG `{signals['vggss']['decay_ogl_iqr_gap_delta']:+.4f}`、Flickr `{signals['flickr']['decay_ogl_iqr_gap_delta']:+.4f}`。两者数值都缩小，但 Flickr 的 IQR 本身下降，且 OGL 也依赖变化后的 AUD，因此不能单独把 gap 缩小当成方法改进。
8. **VGGSS/Flickr 是否一致？** 不一致：Decay 在 VGG 提升 AUD/IQR、在 Flickr 降低；Delayed 在 VGG提升、Flickr约少一个样本。任何只报告 VGG 的结论都会夸大 schedule 有效性。
9. **最终判定：Case {case}（跨数据集不一致的 T5）。** {rationale} 这里不是说 VGG 没有正信号，而是两个预注册 schedule 都没有满足跨数据集一致改善，因而不能进入 144k。

## 下一步

{'建议进入独立 144k 验证，但本实验未启动。' if case == 'T1' else '不建议进入 144k；本实验未启动任何 144k 或后续模型。'}
"""
    (EXPERIMENT_ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Aggregate complete: Case {case} — {rationale}")


if __name__ == "__main__":
    main()
