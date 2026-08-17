# 2.1 Audio-Guided Slot Reliability Probe

这是完全零训练、只读的诊断实验。它加载正式 1.3G-144k best checkpoint，先精确
复现正式 1.3G 和 Experiment 2.0，再分析 audio-guided slot selection 与无需
GT/OGL 的内部 reliability signals。

## 语义选择空间

L3+L4 原训练的 InfoNCE 使用：

```text
F.normalize(fused_img_slots) <-> F.normalize(aud_slots)
target pair = visual slot0 <-> audio slot0
```

本实验的字面目标是比较两个 L4 visual slots，因此主选择使用 raw S4。S4 通过已有
MFusion 和 base losses 端到端训练，但直接 InfoNCE 位于 MFusion 之后；这是解释结果
时必须保留的限制。选择过程不增加 projection：

```text
A  = normalize(audio_slots[:,0])
Sj = normalize(raw_L4_visual_slots[:,j])
sj = cosine(A,Sj), j in {0,1}
selected_slot = argmax(s0,s1)
```

L3/L4 共享 slot initialization 且没有 Hungarian swapping，因此选出的 index 再映射
到对应 L4 ownership。直接接受 InfoNCE 的 fused-slot cosine 同时作为 control 保存；
它不参与主选择，可用于判断 target-index 语义是否已经让 Sf 永远偏向 slot0。

## Reliability features

所有 feature 只依赖内部 AUD、selected ownership 和 slot representation：

- semantic margin；
- ownership entropy/confidence 及竞争概率阈值比例；
- soft containment、AUD top10/top20 seed containment；
- soft centroid distance、JS divergence；
- 覆盖 80% probability mass 的 extent ratio；
- 无训练参数的 R1/R2/R3。

GT/IoU/Rescue/Hurt/OGL/OBJ_PRIOR 均不参与 feature 或 selection 构造。GT 只用于正式
evaluation、事后 Rescue/Hurt、AUROC 和 oracle upper bound。extent ratio 的 AUROC
方向在运行前固定为 `-abs(log(ratio))`，即越接近 1 越可靠，不根据结果翻转方向。

定性图中的范围过扩张案例以 `extent_ratio >= 1.25` 为透明阈值；若某个数据集没有
任何样本满足阈值，则展示该数据集真实的最大 extent 样本，并在 manifest 中标为
`MAX_EXTENT_FALLBACK_NO_GE_1P25_CASE`，不把它伪称为过扩张案例。

## 运行

从仓库根目录：

```bash
bash chuagnxindian/2.1_slot_reliability_probe/run_144k_two_gpus.sh
```

输出包括逐样本 reliability CSV、分组统计、AUROC、selection summary、oracle、固定
规则 qualitative panels，以及 PNG/PDF 数值图。checkpoint 哈希与 mtime 在运行前后
都会审计。完整结果与结论见 `REPORT.md`。
