# Experiment 2.1R：Fixed-Slot Reliability Recheck

本实验为严格 zero-training diagnostic。所有 Rescue/Hurt 标签均由固定候选
`0.6 * AUD_FINE + 0.4 * SLOT_L4(slot0)` 定义，不再使用 audio-selected slot。

## 复现检查

| 数据集 | Rescue | Hurt | Neutral | Net |
|---|---:|---:|---:|---:|
| VGGSoundSS-144k | 355 | 567 | 4236 | -212 |
| Flickr-144k | 11 | 5 | 234 | +6 |

两组结果精确等于预期的 `355/567` 和 `11/5`。AUD、fixed-slot fusion、2.0
ownership 也逐项复现。

## Fixed-slot Rescue-vs-Hurt AUROC

分数方向已统一为“数值越大越可靠”；centroid distance 和 JS 使用负方向，
extent ratio 使用接近 1 的程度。

| Feature | VGG AUROC | Flickr AUROC |
|---|---:|---:|
| semantic margin（control） | 0.4704 | 0.2182 |
| ownership confidence | 0.4857 | 0.7455 |
| evaluator soft containment | 0.5548 | 0.6364 |
| evaluator seed top10 | **0.5936** | 0.6909 |
| evaluator seed top20 | 0.5931 | 0.7455 |
| raw ownership seed top10 | 0.5022 | **0.8000** |
| raw ownership seed top20 | 0.5077 | 0.7818 |
| centroid distance | 0.5610 | 0.6909 |
| JS divergence | 0.5078 | 0.5091 |
| extent ratio | 0.4955 | 0.4364 |

## 关键解释

- `raw_seed_top20` 在 VGG 上几乎等于随机（0.5077）。Rescue/Hurt 的均值分别为
  0.3341/0.3317，二者几乎不可分，因此不能把它当作跨数据集可靠门控信号。
- Flickr 上 `raw_seed_top20=0.7818`，Rescue/Hurt 均值为 0.7238/0.6340；但只有
  11 个 Rescue 和 5 个 Hurt，样本太少，结论只能看作候选信号。
- VGG 最好的单特征仅是 evaluator-space top10（0.5936），仍不足以支撑可靠的
  per-sample hard gate。
- semantic margin 对固定 slot0 candidate 没有帮助。这也验证了 2.1 中基于
  dynamic-selected candidate 得出的 reliability 结论不能直接迁移到最终固定候选。

完整数据见：

- `results/combined_fixed_reliability_auroc.csv`
- `results/combined_feature_group_statistics.csv`
- `results/*/per_sample_fixed_reliability.csv`
- `results/*/zero_training_audit.json`

