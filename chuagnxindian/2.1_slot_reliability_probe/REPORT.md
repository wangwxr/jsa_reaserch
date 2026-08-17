# Experiment 2.1：Audio-Guided Slot Reliability Probe 报告

## 结论先行

1. **不要用 raw L4 slot cosine 动态替换固定 slot0。** 它在 VGGSoundSS-144k
   改变 34.72% 的样本，在 Flickr-144k 改变 12.80%，但改变后分别有 72.42% 和
   81.25% 的样本 IoU 变差。正式 selected fusion 也从 fixed fusion 的
   `0.3858/0.4157`、`0.8360/0.6396` 降至 `0.3414/0.3900`、`0.7880/0.6210`。
2. **固定 slot0 仍然是更可靠的内部 object-completion candidate。** 直接受 InfoNCE
   约束的 fused-slot control 在两个测试集均 100% 选择 slot0，符合 JSA 的 target-slot
   语义。
3. **内部 reliability signal 确实存在。** AUD top-region 的 Slot containment 是最稳定
   的特征：top10 AUROC 为 VGG `0.7198`、Flickr `0.9011`；top20 为 `0.7171`、
   `0.9231`。Flickr 只有 7 个 Rescue 和 13 个 Hurt，数值应视为支持性证据；VGG 的
   1013 个标注事件提供了更可靠的主证据。
4. **主要问题不是选错 slot，而是 Slot map 是否适合参与 completion。** semantic
   margin 接近随机，dynamic selection 总体恶化；而 fixed-slot oracle 的上限很高：
   VGG `0.4957/0.4617`，Flickr `0.8560/0.6614`。
5. 因此 2.2 应走 **Reliability-Aware Internal Object Completion**：保留 fixed slot0
   作为候选，用 seed containment（首选）和 JS/centroid 等内部一致性做保守 gate；
   不应把 raw L4 audio-guided argmax 作为主路径。

## 实际语义空间与 ownership

主诊断严格比较用户指定的两个 raw L4 visual slots：

```text
A  = normalize(aud_slots[:, 0])
Sj = normalize(raw_L4_visual_slots[:, j]), j ∈ {0, 1}
sj = dot(A, Sj) = cosine(A, Sj)
selected_slot = argmax(s0, s1)
```

没有新 projection、参数、optimizer 或训练。需要明确的限制是：raw `S4` 通过
M-Fusion/base losses 端到端参与训练，但直接 InfoNCE 施加在 fused visual slots `Sf`
上，而不是 raw `S4` 上。因此同时保存了严格 InfoNCE-space control：
`cos(Sa[:,0], Sf[:,j])`。该 control 在 VGG 和 Flickr 都是 0% 改选，始终选择 slot0。

L4 ownership 使用最终一次 Slot Attention 中、token-renormalization 之前的：

```text
ownership = softmax(final_logits, dim=slot_dim)
shape = [B, 2, 49] -> [B, 2, 7, 7]
```

每个 token 的两个 slot 概率和为 1；两套数据观测到的最大数值误差均为
`1.1921e-7`。后续按 token dimension 归一化、用于 slot weighted update 的权重没有
被误当成 ownership。

## 强制复现检查

| 数据集 | 1.3G AUD 参考 | 本次 AUD | 2.0 L4 Rescue/Hurt 参考 | 本次 |
|---|---:|---:|---:|---:|
| VGGSoundSS-144k | 0.4269 / 0.4230 | 0.4269 / 0.4230 | 355 / 567 | 355 / 567 |
| Flickr-144k | 0.8120 / 0.6356 | 0.8120 / 0.6356 | 11 / 5 | 11 / 5 |

所有误差均为 0；两个 gate 均通过后才继续 2.1。

## 完整方法结果

| Method | VGG cIoU / AUC | Flickr cIoU / AUC |
|---|---:|---:|
| AUD | 0.4269 / 0.4230 | 0.8120 / 0.6356 |
| SLOT_L4_FIXED_SLOT0 | 0.2144 / 0.3472 | 0.7320 / 0.5952 |
| SLOT_L4_AUDIO_SELECTED | 0.1444 / 0.2689 | 0.6560 / 0.5312 |
| AUD + FIXED_SLOT0 | 0.3858 / 0.4157 | 0.8360 / 0.6396 |
| AUD + AUDIO_SELECTED_SLOT | 0.3414 / 0.3900 | 0.7880 / 0.6210 |
| OGL reference | 0.4570 / 0.4401 | 0.8680 / 0.6596 |
| **ORACLE: AUD vs AUD+fixed slot0** | **0.4957 / 0.4617** | **0.8560 / 0.6614** |
| **ORACLE: AUD vs AUD+selected slot** | **0.4824 / 0.4549** | **0.8400 / 0.6574** |

Oracle 使用 GT 后验逐样本选较高 IoU，**不可部署、不是正式结果**。它只衡量一个完美
reliability gate 的理论潜力。

## Selection、Rescue 与 Hurt

| 统计 | VGG | Flickr |
|---|---:|---:|
| 测试样本 | 5158 | 250 |
| raw L4 audio selection 改选 | 1791 (34.72%) | 32 (12.80%) |
| direct-InfoNCE fused control 改选 | 0 (0%) | 0 (0%) |
| 改选后 IoU 提高 | 494 (27.58% of changed) | 6 (18.75%) |
| 改选后 IoU 降低 | 1297 (72.42% of changed) | 26 (81.25%) |
| Fixed fusion Rescue / Hurt / Net | 355 / 567 / -212 | 11 / 5 / +6 |
| Selected fusion Rescue / Hurt / Net | 286 / 727 / -441 | 7 / 13 / -6 |
| OGL Rescue / Hurt / Net | 357 / 202 / +155 | 19 / 5 / +14 |

状态转移进一步说明 dynamic selection 不是解决方案：

- VGG：fixed Hurt 中 133 个被改选修复，但又新造 293 个 Hurt；fixed Rescue 中保留
  245、丢失 110，只新造 41 个 Rescue。
- Flickr：fixed Hurt 一个也没修复，却新造 8 个 Hurt；fixed Rescue 保留 7、丢失 4，
  没有新造 Rescue。

## Rescue-vs-Hurt AUROC

所有 score 均只由内部 representation、AUD 和 selected Slot map 构造；GT 只负责生成
Rescue/Hurt 离线标签。方向均在运行前固定。

| Reliability feature | VGG AUROC (286 R / 727 H) | Flickr AUROC (7 R / 13 H) |
|---|---:|---:|
| semantic margin ↑ | 0.4934 | 0.5275 |
| ownership confidence ↑ | 0.5406 | 0.5275 |
| soft containment ↑ | 0.4046 | 0.7912 |
| seed containment top10 ↑ | **0.7198** | **0.9011** |
| seed containment top20 ↑ | **0.7171** | **0.9231** |
| centroid distance ↓ | 0.5981 | 0.8242 |
| JS divergence ↓ | 0.6694 | 0.8352 |
| extent ratio 接近 1 | 0.3449 | 0.5165 |
| R1 = margin × seed20 | 0.5950 | 0.7473 |
| R2 = ownership confidence × seed20 | 0.6875 | 0.8571 |
| R3 = margin × ownership × seed20 | 0.5946 | 0.7582 |

结论：简单乘法组合没有超过 seed containment 本身。semantic margin 和 ownership
confidence 单独几乎无区分力；top-region seed containment 是唯一在大样本 VGG 上超过
0.70、且在 Flickr 上方向一致的信号。JS 是次优且跨数据集一致。

## Rescue / Hurt / Neutral 分布

格式为 `mean / median / std`。soft containment 使用 evaluator 的 min-max map，绝对值
受像素规模影响，只应在同一数据集内比较。

### VGGSoundSS-144k

| Feature | Rescue (n=286) | Hurt (n=727) | Neutral (n=4145) |
|---|---:|---:|---:|
| semantic margin | .0285/.0197/.0271 | .0272/.0217/.0229 | .0274/.0210/.0240 |
| ownership confidence | .5519/.5602/.1269 | .5352/.5365/.1195 | .5028/.5053/.1381 |
| soft containment | 12561.7/12083.2/3645.0 | 13685.8/13920.8/3697.5 | 14288.1/14387.0/3836.6 |
| seed top10 | .7440/.8311/.2342 | .5037/.6949/.3377 | .6086/.7942/.3243 |
| seed top20 | .6741/.7283/.1755 | .4896/.5636/.2513 | .5785/.6873/.2535 |
| centroid distance | .0458/.0390/.0306 | .0564/.0503/.0332 | .0475/.0404/.0318 |
| JS divergence | .0484/.0439/.0246 | .0683/.0654/.0351 | .0541/.0466/.0336 |
| extent ratio | .7087/.6722/.1603 | .8033/.8191/.1845 | .7836/.7745/.1605 |

### Flickr-144k

| Feature | Rescue (n=7) | Hurt (n=13) | Neutral (n=230) |
|---|---:|---:|---:|
| semantic margin | .0288/.0198/.0211 | .0254/.0207/.0200 | .0285/.0248/.0200 |
| ownership confidence | .2689/.2694/.0464 | .2718/.2632/.0873 | .2604/.2495/.0933 |
| soft containment | 20015.7/17450.5/4770.4 | 14554.7/13288.5/3451.6 | 20398.2/20573.9/4643.3 |
| seed top10 | .9032/.9023/.0282 | .3887/.1871/.3670 | .8279/.9189/.2578 |
| seed top20 | .8606/.8536/.0350 | .3906/.2692/.3221 | .8050/.8913/.2383 |
| centroid distance | .0271/.0274/.0079 | .0725/.0527/.0532 | .0320/.0243/.0307 |
| JS divergence | .0193/.0157/.0090 | .0867/.0670/.0637 | .0277/.0128/.0409 |
| extent ratio | .8018/.7965/.0708 | .8104/.8110/.0949 | .8387/.8491/.0908 |

这里有一个反直觉但重要的现象：VGG Rescue 的 extent ratio 比 Hurt 更小，而预设的
“越接近 1 越可靠”AUROC 只有 0.3449。说明有效 Slot 融合并不总是在扩大物体范围；
在固定 0.6 threshold 下，它经常更像是帮助 AUD 收缩/选择正确 extent。因此不能把
“object completion”简单实现为无条件扩张。

## Selection error 还是 extent/reliability error

透明后验 heuristic（只分析、不训练、不用于部署）给出：

| 分类 | VGG selected Hurt (n=727) | Flickr selected Hurt (n=13) |
|---|---:|---:|
| possible selection-related | 8 (1.10%) | 0 (0%) |
| possible extent/reliability-related | 91 (12.52%) | 0 (0%) |
| 未被规则严谨分类 | 628 | 13 |

规则覆盖率有限，不能把未分类样本强行归因。但结合 dynamic selection 的整体恶化、
InfoNCE control 始终选择 slot0、semantic-margin AUROC 约 0.5，可以比较有把握地说：
**wrong slot selection 不是主要瓶颈；关键是何时、以多大强度允许 slot0 改写 AUD。**

## 定性结果

每套数据固定 12 个、按预先定义类别与测试顺序选择，不 cherry-pick。每张图包含 Image、
GT、AUD、slot0、slot1、audio-selected slot、fusion、OGL，并标出 `s0/s1`、margin、
ownership confidence、seed20、JS 和各 IoU。

- VGG：`results/vggss_144k/qualitative/`
- Flickr：`results/flickr_144k/qualitative/`

VGG 存在真实 `extent_ratio >= 1.25` 样本；Flickr 最大值仅为 `1.0282`，因此其 manifest
明确标注 `MAX_EXTENT_FALLBACK_NO_GE_1P25_CASE`，没有伪造“过扩张”案例。

## 零训练与文件审计

两套 checkpoint 的 SHA256 与 mtime 在诊断前后完全一致；`optimizer_created=false`、
`backward_called=false`、新增和运行时 trainable parameters 都是 0。GT/OGL/OBJ_PRIOR
没有进入 selection 或 reliability feature。完整审计见每个数据集的
`zero_training_audit.json` 与 `semantic_space_audit.json`。

数值总表位于：

- `results/combined_method_metrics.csv`
- `results/combined_reliability_auroc.csv`
- `results/combined_selection_summary.csv`
- `results/combined_feature_group_statistics.csv`
- `results/combined_summary.json`

## 2.2 路线建议

选择 **Reliability-Aware Internal Object Completion**，但 candidate 应使用固定 slot0，
不是 raw-S4 dynamic slot。第一优先级 reliability signal 是 AUD top20/top10 seed
containment，JS divergence 和 centroid distance 可作为次级一致性约束。当前 oracle
表明 gate 的潜力存在，尤其 VGG fixed-slot oracle 已明显超过 OGL；真正需要解决的是
conservative gating，而不是继续设计 slot-selection 规则。
