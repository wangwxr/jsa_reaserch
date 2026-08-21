#!/usr/bin/env python3
"""Generate the auditable native-L3 comparison without hand-entered metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
METHODS = ("AUD", "IMG_QUERY", "IQR", "OBJ_PRIOR", "OGL", "EXTRA_IQR_OGL")
DATASETS = ("vggss_10k", "flickr_10k", "vggss_144k", "flickr_144k")
DISPLAY = {
    "vggss_10k": "VGGSS-10k",
    "flickr_10k": "Flickr-10k",
    "vggss_144k": "VGGSS-144k",
    "flickr_144k": "Flickr-144k",
}


def names(key):
    dataset, split = key.split("_")
    suffix = "_frame8_center5" if dataset == "flickr" else ""
    return {
        "v1.1 baseline": f"mufasa_jsa_v1_1_{dataset}_{split}{suffix}",
        "1.1.1_14_14_L3": f"1.1.1_14_14_L3_{dataset}_{split}",
        "original 1.3G": f"1.3G-multigeom_equivariant_l3_refine_{dataset}_{split}{suffix}",
        "1.3G_14_14_L3": f"1.3G_14_14_L3_{dataset}_{split}",
    }


def v11_metrics(experiment):
    path = CHECKPOINTS / experiment / "epoch_metrics.csv"
    if not path.is_file():
        return None, str(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, str(path)
    row = max(rows, key=lambda value: float(value["iqr_ciou"]))
    metrics = {}
    for method in METHODS:
        prefix = method.lower()
        metrics[method] = {
            "cIoU": float(row[f"{prefix}_ciou"]),
            "AUC": float(row[f"{prefix}_auc"]),
        }
    return metrics, f"{path} (best IQR epoch {row['epoch']})"


def json_metrics(experiment, filename, field):
    path = CHECKPOINTS / experiment / filename
    if not path.is_file():
        return None, str(path), None
    result = json.loads(path.read_text(encoding="utf-8"))
    return result.get(field), str(path), result


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def delta(first, second, method, metric="cIoU"):
    if first is None or second is None:
        return None
    return second[method][metric] - first[method][metric]


def sentence_delta(label, value):
    if value is None:
        return f"{label}：N/A（实验尚未完成或正式结果文件不存在）。"
    direction = "提升" if value > 0 else "下降" if value < 0 else "持平"
    return f"{label}：{direction} {abs(value):.4f}。"


def main():
    records = {}
    sources = []
    raw_results = {}
    for key in DATASETS:
        records[key] = {}
        experiments = names(key)
        metrics, source = v11_metrics(experiments["v1.1 baseline"])
        records[key]["v1.1 baseline"] = metrics
        sources.append((key, "v1.1 baseline", source))

        metrics, source, result = json_metrics(
            experiments["1.1.1_14_14_L3"], "best_full_metrics.json", "formal_metrics"
        )
        records[key]["1.1.1_14_14_L3"] = metrics
        raw_results[(key, "1.1.1_14_14_L3")] = result
        sources.append((key, "1.1.1_14_14_L3", source))

        for label in ("original 1.3G", "1.3G_14_14_L3"):
            metrics, source, result = json_metrics(
                experiments[label], "best_full_six_metrics.json", "metrics"
            )
            records[key][label] = metrics
            raw_results[(key, label)] = result
            sources.append((key, label, source))

    lines = [
        "# Native 14×14 L3 + G 两阶段结果",
        "",
        "所有数字均由脚本读取正式 epoch CSV 或 best-checkpoint evaluator JSON；缺失项标记为 N/A。",
        "",
    ]
    for key in DATASETS:
        lines.extend([
            f"## {DISPLAY[key]}",
            "",
            "| Experiment | AUD cIoU/AUC | IMG_QUERY cIoU/AUC | IQR cIoU/AUC | OGL cIoU/AUC | EXTRA cIoU/AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for label in ("v1.1 baseline", "1.1.1_14_14_L3", "original 1.3G", "1.3G_14_14_L3"):
            metrics = records[key][label]
            def pair(method):
                return "N/A" if metrics is None else f"{fmt(metrics[method]['cIoU'])}/{fmt(metrics[method]['AUC'])}"
            lines.append(
                f"| {label} | {pair('AUD')} | {pair('IMG_QUERY')} | {pair('IQR')} | {pair('OGL')} | {pair('EXTRA_IQR_OGL')} |"
            )
        lines.append("")

        native_result = raw_results.get((key, "1.1.1_14_14_L3"))
        lines.extend([
            "### Ownership",
            "",
            "| Map | cIoU | AUC | Rescue | Hurt | Net |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        if native_result is None:
            lines.append("| SLOT_L3_POOLED_BASELINE / SLOT_L3_NATIVE / SLOT_L4 | N/A | N/A | N/A | N/A | N/A |")
        else:
            ownership = native_result["ownership_metrics"]
            rescue = {row["candidate"]: row for row in native_result["rescue_hurt"]}
            rows = (
                ("SLOT_L3_POOLED_BASELINE", "V11_AUD_SLOT_L3_POOLED"),
                ("SLOT_L3_NATIVE", "AUD_SLOT_L3_NATIVE"),
                ("SLOT_L4", "AUD_SLOT_L4"),
            )
            for map_name, candidate in rows:
                value = ownership[map_name]
                outcome = rescue.get(candidate, {})
                lines.append(
                    f"| {map_name} | {value['cIoU']:.4f} | {value['AUC']:.4f} | "
                    f"{outcome.get('rescue', 'N/A')} | {outcome.get('hurt', 'N/A')} | {outcome.get('net_rescue', 'N/A')} |"
                )
        lines.append("")

    lines.extend(["## 科研问题", ""])
    for key in DATASETS:
        base = records[key]["v1.1 baseline"]
        native = records[key]["1.1.1_14_14_L3"]
        old_g = records[key]["original 1.3G"]
        new_g = records[key]["1.3G_14_14_L3"]
        aud_change = delta(base, native, "AUD")
        iqr_change = delta(base, native, "IQR")
        g_change = delta(old_g, new_g, "AUD")
        native_result = raw_results.get((key, "1.1.1_14_14_L3"))
        ownership_statement = "Native/pooled ownership：N/A。"
        if native_result is not None:
            own = native_result["ownership_metrics"]
            own_delta = own["SLOT_L3_NATIVE"]["cIoU"] - own["SLOT_L3_POOLED_BASELINE"]["cIoU"]
            ownership_statement = sentence_delta("SLOT_L3_NATIVE 相对 pooled L3 cIoU", own_delta)

        if g_change is None:
            relation = "互补/冗余/冲突：N/A。"
        elif g_change > 0:
            relation = "Native fine slot 与 G refinement 在 AUD cIoU 上表现为互补。"
        elif abs(g_change) < 1e-12:
            relation = "Native fine slot 与 G refinement 在 AUD cIoU 上表现为冗余。"
        else:
            relation = "Native fine slot teacher 使 G 的 AUD cIoU 下降，当前表现为冲突。"
        q3 = "Q3 条件未满足或数据不足。"
        if native_result is not None and base is not None and native is not None:
            own = native_result["ownership_metrics"]
            own_gain = own["SLOT_L3_NATIVE"]["cIoU"] - own["SLOT_L3_POOLED_BASELINE"]["cIoU"]
            if own_gain > 0 and (aud_change is None or aud_change <= 0):
                q3 = "Q3：fine object information 已经存在于 L3 Slot 内部，但当前 L4-only inference path 没有利用它。"
        lines.extend([
            f"### {DISPLAY[key]}",
            "",
            sentence_delta("Q1 AUD cIoU", aud_change) + " " + sentence_delta("IQR cIoU", iqr_change),
            "",
            "Q2：" + ownership_statement,
            "",
            q3,
            "",
            "Q4：" + sentence_delta("新 G 相对原 G 的 AUD cIoU", g_change) + " " + relation,
            "",
        ])

    lines.extend([
        "## 数据来源",
        "",
        "| Dataset | Experiment | Source |",
        "|---|---|---|",
    ])
    for key, label, source in sources:
        lines.append(f"| {DISPLAY[key]} | {label} | `{source}` |")
    lines.append("")
    content = "\n".join(lines)
    (HERE / "RESULTS.md").write_text(content, encoding="utf-8")
    (HERE.parent / "14_14_L3_COMPARISON.md").write_text(content, encoding="utf-8")
    print(f"Saved {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
