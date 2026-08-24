# Experiment 6.0 — Base Loss Causality Study

## 结果完整性

- Stage A Full 数值回归已通过，checkpoint 审计未发现写入。
- 六组新增训练状态：全部成功。
- 三个配置均保持 audio reconstruction 与 temporal attention；关闭项的 weighted value 与梯度贡献在首 batch 严格为 0。
- best checkpoint 始终按原 L3+L4 的 IQR cIoU 保存。

## 核心指标（cIoU / AUC）

| Configuration | VGG AUD | VGG IMG | VGG IQR | Flickr AUD | Flickr IMG | Flickr IQR |
|---|---:|---:|---:|---:|---:|---:|
| FULL_REFERENCE | 0.4015 / 0.4074 | 0.4064 / 0.4094 | 0.4073 / 0.4092 | 0.7640 / 0.5916 | 0.7520 / 0.5890 | 0.7720 / 0.5922 |
| NO_SPATIAL_ATT | 0.3484 / 0.3847 | 0.4120 / 0.4108 | 0.3947 / 0.4037 | 0.6640 / 0.5460 | 0.7240 / 0.5818 | 0.7240 / 0.5772 |
| NO_IMAGE_REC | 0.4015 / 0.4083 | 0.4073 / 0.4100 | 0.4050 / 0.4096 | 0.7840 / 0.6026 | 0.7640 / 0.6004 | 0.7880 / 0.6030 |
| NO_BOTH | 0.3480 / 0.3873 | 0.4085 / 0.4104 | 0.3876 / 0.4016 | 0.5480 / 0.5126 | 0.7440 / 0.5910 | 0.6760 / 0.5650 |

完整六指标（含 OBJ_PRIOR/OGL/EXTRA_IQR_OGL）见 `results/combined_metrics.csv`。

## 冗余与空间误差

| Configuration | VGG Pearson | VGG precision | VGG coverage | VGG leakage | VGG area | Flickr Pearson | Flickr precision | Flickr coverage | Flickr leakage | Flickr area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_REFERENCE | 0.9859 | 0.4932 | 0.8279 | 0.5068 | 0.4848 | 0.9873 | 0.7423 | 0.7645 | 0.2577 | 0.5189 |
| NO_SPATIAL_ATT | 0.8796 | 0.4728 | 0.7970 | 0.5272 | 0.4840 | 0.8545 | 0.7209 | 0.7141 | 0.2791 | 0.4966 |
| NO_IMAGE_REC | 0.9894 | 0.5043 | 0.8001 | 0.4957 | 0.4589 | 0.9885 | 0.7479 | 0.7779 | 0.2521 | 0.5202 |
| NO_BOTH | 0.9158 | 0.4743 | 0.8068 | 0.5257 | 0.4872 | 0.7960 | 0.7081 | 0.6712 | 0.2919 | 0.4784 |

## 因果问题分析


1. **Spatial attention 确实造成了显著分支对齐，但同时是 AUD 的关键视觉教师。** 去掉它后，AUD–IMG Pearson 在 VGGSS 从 `0.9859` 降至 `0.8796`（Δ `-0.1063`），Flickr 从 `0.9873` 降至 `0.8545`（Δ `-0.1327`）。IQR−AUD cIoU 增益分别从 `+0.0058/+0.0080` 增至 `+0.0463/+0.0600`，说明互补性确实变强；但 AUD 同时下降 `-0.0531/-0.1000`，最终 IQR 也下降 `-0.0126/-0.0480`。因此问题是 **over-alignment 与必要监督并存**，不是简单删除 spatial-att 即可解决。
2. **结构本身仍贡献很强冗余。** 即使从头训练时去掉 spatial-att，Pearson 仍有 `0.8796/0.8545`。shared slot index、共同 L4 keys 和双分支架构仍会自然对齐；显式 att loss 不是唯一来源。
3. **Image reconstruction-induced compactness 假设不成立。** 去掉 rec_img 后，VGGSS 的 coverage 反而变化 `-0.0277`、area `-0.0260`，即更小而非扩张；Flickr coverage `+0.0133`、area `+0.0013`，仅轻微扩张。两者 precision 都略升、outside leakage 略降，但 IQR 在 VGGSS `-0.0023`、Flickr `+0.0160`，跨数据集不一致。
4. **Stage A 也不支持 rec_img 直接复制定位图。** decoder target alpha 与 IMG 的 Pearson 仅 `0.5508/0.4395`，top-10 overlap 仅 `0.2301/0.1312`；rec_img 更像 slot feature/partition regularizer。它确实通过 M-Fusion 作用到 L3（L3/L4 rec-img 梯度比约 `75.3%/75.8%`），但与 InfoNCE 的梯度 cosine 仅 `-0.0022/0.0073`，没有“牵制”证据。
5. **NO_BOTH 否定联合过约束是主要原因。** 相对 Full，VGGSS/Flickr 的 IQR 分别变化 `-0.0198/-0.0960`，AUD 变化 `-0.0535/-0.2160`；尤其 Flickr 严重崩落，不能归为 Case D。
6. **SHRINK_BASE 没有稳定收益。** 其相对 AUD 的 mean-IoU gain：Full 为 `+0.0010/-0.0074`，NO_SPATIAL_ATT 为 `+0.0101/-0.0059`，NO_IMAGE_REC 为 `+0.0003/-0.0059`。它只在部分 VGG 配置略正，在 Flickr 通常为负。
7. **EXPAND_BASE 只说明解耦后存在潜在互补。** NO_SPATIAL_ATT 的 mean-IoU gain 为 `+0.0164/+0.0384`，NO_BOTH 为 `+0.0136/+0.0715`；但它是基础 AUD/IMG 交互诊断，不是 5.2 PROP_F34/K34 oracle，也不能当作新方法结果。
8. **最终判定：Case A。** 去掉 spatial attention 降低冗余并提高 IQR 互补增益，但牺牲 AUD。


## 下一步决策

- 本实验属于 **Case A**：若继续，只建议把 spatial-att 的权重下降或 warmup/ramp 作为新的独立 10k 实验；不能把完全删除直接定为最终方法。
- **当前不建议进入 144k。** NO_SPATIAL_ATT 在两个数据集都降低最终 IQR，NO_IMAGE_REC 又只在 Flickr 有正信号，没有跨数据集一致性；144k 只会放大成本，不能修复因果不一致。

本报告不启动 144k、不重训 1.3G，也不实现下一步方法。
