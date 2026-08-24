# Experiment 6.1 — Spatial Attention Temporal Schedule Study

## 完整性与预注册协议

- 四组新增 10k 训练：全部成功。
- schedule 单元测试与实际 FP32 fixed-batch 回归均通过；只有 `L_att_space` 的乘数随 epoch 改变。
- 所有主结果取 IQR-selected checkpoint；Full 直接复用 6.0 正式 reference。
- Flickr 每个样本对应 cIoU `0.0040`，因此所有变化同时按样本数解释。

## 核心指标（cIoU / AUC）

| Configuration | VGG AUD | VGG IMG | VGG IQR | Flickr AUD | Flickr IMG | Flickr IQR |
|---|---:|---:|---:|---:|---:|---:|
| FULL_REFERENCE | 0.4015 / 0.4074 | 0.4064 / 0.4094 | 0.4073 / 0.4092 | 0.7640 / 0.5916 | 0.7520 / 0.5890 | 0.7720 / 0.5922 |
| EARLY_HIGH_LATE_LOW | 0.4108 / 0.4118 | 0.4213 / 0.4152 | 0.4174 / 0.4139 | 0.7440 / 0.5988 | 0.7400 / 0.5966 | 0.7600 / 0.6010 |
| EARLY_LOW_LATE_HIGH | 0.4110 / 0.4126 | 0.4124 / 0.4135 | 0.4143 / 0.4137 | 0.7760 / 0.6002 | 0.7400 / 0.5920 | 0.7680 / 0.6014 |

## 完整六指标

| Configuration | Method | VGGSS | Flickr |
|---|---|---:|---:|
| FULL_REFERENCE | AUD | 0.4015 / 0.4074 | 0.7640 / 0.5916 |
| FULL_REFERENCE | IMG_QUERY | 0.4064 / 0.4094 | 0.7520 / 0.5890 |
| FULL_REFERENCE | IQR | 0.4073 / 0.4092 | 0.7720 / 0.5922 |
| FULL_REFERENCE | OBJ_PRIOR | 0.3478 / 0.3924 | 0.4480 / 0.4668 |
| FULL_REFERENCE | OGL | 0.4432 / 0.4292 | 0.8400 / 0.6154 |
| FULL_REFERENCE | EXTRA_IQR_OGL | 0.4314 / 0.4215 | 0.8120 / 0.6124 |
| EARLY_HIGH_LATE_LOW | AUD | 0.4108 / 0.4118 | 0.7440 / 0.5988 |
| EARLY_HIGH_LATE_LOW | IMG_QUERY | 0.4213 / 0.4152 | 0.7400 / 0.5966 |
| EARLY_HIGH_LATE_LOW | IQR | 0.4174 / 0.4139 | 0.7600 / 0.6010 |
| EARLY_HIGH_LATE_LOW | OBJ_PRIOR | 0.3478 / 0.3924 | 0.4480 / 0.4668 |
| EARLY_HIGH_LATE_LOW | OGL | 0.4490 / 0.4304 | 0.8120 / 0.6196 |
| EARLY_HIGH_LATE_LOW | EXTRA_IQR_OGL | 0.4376 / 0.4244 | 0.7720 / 0.6200 |
| EARLY_LOW_LATE_HIGH | AUD | 0.4110 / 0.4126 | 0.7760 / 0.6002 |
| EARLY_LOW_LATE_HIGH | IMG_QUERY | 0.4124 / 0.4135 | 0.7400 / 0.5920 |
| EARLY_LOW_LATE_HIGH | IQR | 0.4143 / 0.4137 | 0.7680 / 0.6014 |
| EARLY_LOW_LATE_HIGH | OBJ_PRIOR | 0.3478 / 0.3924 | 0.4480 / 0.4668 |
| EARLY_LOW_LATE_HIGH | OGL | 0.4463 / 0.4294 | 0.8240 / 0.6244 |
| EARLY_LOW_LATE_HIGH | EXTRA_IQR_OGL | 0.4352 / 0.4239 | 0.8040 / 0.6188 |

## 冗余、真实互补与 OGL gap

| Configuration | VGG Pearson | VGG synergy | VGG sample synergy | VGG OGL-IQR | Flickr Pearson | Flickr synergy | Flickr sample synergy | Flickr OGL-IQR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_REFERENCE | 0.9859 | +0.0010 | -0.0106 | +0.0359 | 0.9873 | +0.0080 | -0.0134 | +0.0680 |
| EARLY_HIGH_LATE_LOW | 0.9872 | -0.0039 | -0.0107 | +0.0316 | 0.9871 | +0.0160 | -0.0110 | +0.0520 |
| EARLY_LOW_LATE_HIGH | 0.9889 | +0.0019 | -0.0092 | +0.0320 | 0.9642 | -0.0080 | -0.0156 | +0.0560 |

注意：aggregate synergy 是阈值成功率差；sample synergy 是连续 per-sample IoU 差。二者可能方向不同，不能互相替代。

## 空间误差（AUD，threshold=0.6）

| Configuration | Dataset | precision | coverage | outside leakage | predicted area | entropy | effective area |
|---|---|---:|---:|---:|---:|---:|---:|
| FULL_REFERENCE | VGGSS | 0.4932 | 0.8279 | 0.5068 | 0.4848 | 3.8875 | 48.79 |
| FULL_REFERENCE | FLICKR | 0.7423 | 0.7645 | 0.2577 | 0.5189 | 3.8851 | 48.67 |
| EARLY_HIGH_LATE_LOW | VGGSS | 0.4896 | 0.8415 | 0.5104 | 0.4976 | 3.8860 | 48.72 |
| EARLY_HIGH_LATE_LOW | FLICKR | 0.7174 | 0.8136 | 0.2826 | 0.5692 | 3.8862 | 48.73 |
| EARLY_LOW_LATE_HIGH | VGGSS | 0.4947 | 0.8341 | 0.5053 | 0.4861 | 3.8863 | 48.73 |
| EARLY_LOW_LATE_HIGH | FLICKR | 0.7232 | 0.8134 | 0.2768 | 0.5639 | 3.8891 | 48.87 |

## 选模与训练末轮

| Configuration | Dataset | IQR-best epoch/value | final IQR | best AUD epoch/value | best IMG epoch/value |
|---|---|---:|---:|---:|---:|
| EARLY_HIGH_LATE_LOW | VGGSS | 89 / 0.4174 | 0.4067 | 76 / 0.4112 | 89 / 0.4213 |
| EARLY_HIGH_LATE_LOW | FLICKR | 99 / 0.7600 | 0.7440 | 81 / 0.7600 | 75 / 0.7520 |
| EARLY_LOW_LATE_HIGH | VGGSS | 92 / 0.4143 | 0.3934 | 92 / 0.4110 | 92 / 0.4124 |
| EARLY_LOW_LATE_HIGH | FLICKR | 25 / 0.7680 | 0.7240 | 25 / 0.7760 | 58 / 0.7600 |

## 固定 batch 梯度轨迹

| Configuration | Dataset | epoch | lambda | weighted spatial grad | total grad | ratio |
|---|---|---:|---:|---:|---:|---:|
| EARLY_HIGH_LATE_LOW | VGGSS | 1 | 100.0 | 0.011679 | 84.618881 | 0.000138 |
| EARLY_HIGH_LATE_LOW | VGGSS | 20 | 100.0 | 0.203225 | 100.775295 | 0.002017 |
| EARLY_HIGH_LATE_LOW | VGGSS | 21 | 95.5 | 0.180895 | 15.061220 | 0.012011 |
| EARLY_HIGH_LATE_LOW | VGGSS | 30 | 55.0 | 0.183447 | 8.259895 | 0.022209 |
| EARLY_HIGH_LATE_LOW | VGGSS | 40 | 10.0 | 0.018530 | 9.861744 | 0.001879 |
| EARLY_HIGH_LATE_LOW | VGGSS | 41 | 10.0 | 0.017659 | 4.286943 | 0.004119 |
| EARLY_HIGH_LATE_LOW | VGGSS | 60 | 10.0 | 0.019314 | 3.771004 | 0.005122 |
| EARLY_HIGH_LATE_LOW | VGGSS | 80 | 10.0 | 0.016945 | 2.705632 | 0.006263 |
| EARLY_HIGH_LATE_LOW | VGGSS | 100 | 10.0 | 0.019612 | 3.487735 | 0.005623 |
| EARLY_HIGH_LATE_LOW | FLICKR | 1 | 100.0 | 0.005437 | 29.444096 | 0.000185 |
| EARLY_HIGH_LATE_LOW | FLICKR | 20 | 100.0 | 0.098572 | 96.873387 | 0.001018 |
| EARLY_HIGH_LATE_LOW | FLICKR | 21 | 95.5 | 0.106210 | 113.560453 | 0.000935 |
| EARLY_HIGH_LATE_LOW | FLICKR | 30 | 55.0 | 0.068169 | 1.907655 | 0.035735 |
| EARLY_HIGH_LATE_LOW | FLICKR | 40 | 10.0 | 0.018194 | 2.226794 | 0.008171 |
| EARLY_HIGH_LATE_LOW | FLICKR | 41 | 10.0 | 0.022518 | 4.102655 | 0.005489 |
| EARLY_HIGH_LATE_LOW | FLICKR | 60 | 10.0 | 0.016485 | 40.440562 | 0.000408 |
| EARLY_HIGH_LATE_LOW | FLICKR | 80 | 10.0 | 0.017668 | 1.259451 | 0.014029 |
| EARLY_HIGH_LATE_LOW | FLICKR | 100 | 10.0 | 0.011016 | 1.536042 | 0.007172 |
| EARLY_LOW_LATE_HIGH | VGGSS | 1 | 0.0 | 0.000000 | 82.902287 | 0.000000 |
| EARLY_LOW_LATE_HIGH | VGGSS | 20 | 0.0 | 0.000000 | 8.311531 | 0.000000 |
| EARLY_LOW_LATE_HIGH | VGGSS | 21 | 5.0 | 0.020421 | 7.242500 | 0.002820 |
| EARLY_LOW_LATE_HIGH | VGGSS | 30 | 50.0 | 0.108585 | 4.591566 | 0.023649 |
| EARLY_LOW_LATE_HIGH | VGGSS | 40 | 100.0 | 0.183209 | 3.387638 | 0.054082 |
| EARLY_LOW_LATE_HIGH | VGGSS | 41 | 100.0 | 0.180425 | 4.256059 | 0.042393 |
| EARLY_LOW_LATE_HIGH | VGGSS | 60 | 100.0 | 0.174808 | 3.995144 | 0.043755 |
| EARLY_LOW_LATE_HIGH | VGGSS | 80 | 100.0 | 0.196972 | 4.082440 | 0.048249 |
| EARLY_LOW_LATE_HIGH | VGGSS | 100 | 100.0 | 0.138905 | 2.306980 | 0.060211 |
| EARLY_LOW_LATE_HIGH | FLICKR | 1 | 0.0 | 0.000000 | 26.094826 | 0.000000 |
| EARLY_LOW_LATE_HIGH | FLICKR | 20 | 0.0 | 0.000000 | 120.761329 | 0.000000 |
| EARLY_LOW_LATE_HIGH | FLICKR | 21 | 5.0 | 0.017064 | 8.069228 | 0.002115 |
| EARLY_LOW_LATE_HIGH | FLICKR | 30 | 50.0 | 0.062546 | 165.990924 | 0.000377 |
| EARLY_LOW_LATE_HIGH | FLICKR | 40 | 100.0 | 0.126065 | 3.785154 | 0.033305 |
| EARLY_LOW_LATE_HIGH | FLICKR | 41 | 100.0 | 0.109029 | 282.926151 | 0.000385 |
| EARLY_LOW_LATE_HIGH | FLICKR | 60 | 100.0 | 0.069132 | 5.444313 | 0.012698 |
| EARLY_LOW_LATE_HIGH | FLICKR | 80 | 100.0 | 0.077218 | 1.748133 | 0.044172 |
| EARLY_LOW_LATE_HIGH | FLICKR | 100 | 100.0 | 0.095895 | 1.547537 | 0.061966 |

## 研究问题结论

1. **EARLY_LOW_LATE_HIGH 是否明显弱于 Full？** 主协议下不是跨数据集一致地弱：VGG IQR `+0.0070`，Flickr `-0.0040`（`-1.0` 个成功样本）。不过 final 明显低于各自 best，说明 delayed 的后期训练不稳定。
2. **是否证明 early spatial-att 必要？** 训练轨迹强烈显示前 20 轮缺失 teacher 会让 AUD 显著落后，但 100 轮 IQR-selected best 在 VGG 反而超过 Full、Flickr 只差 1 个样本。因此只能说它显著改善早期优化路径，不能严格宣称最终性能必须依赖早期监督。
3. **Decay 是否在降低 Pearson 时保持 AUD？** 没有跨数据集成立。Pearson 变化 VGG `+0.0013`、Flickr `-0.0002`；AUD 变化 `+0.0093/-0.0200`。VGG Pearson 反而更高，Flickr 的下降几乎为零。
4. **Pearson 降低后是否得到真实 synergy？** Decay aggregate synergy 为 VGG `-0.0039`、Flickr `+0.0160`，方向相反；而两者 mean sample synergy 都为负。没有跨数据集、跨口径的一致互补收益。
5. **权重下降时 AUD 是否同步下降？** Flickr best-checkpoint AUD 相对 Full 下降 `0.0200`，VGG 反而提高 `0.0093`；没有统一同步关系。epoch 曲线显示短期波动明显，不能用单个 transition epoch 下结论。
6. **后期 spatial 梯度小意味着任务满足还是不重要？** 两者都不能直接推出。Decay 在 λ=10 时绝对加权梯度约为高权重阶段的十分之一，但仍非零；Delayed 在 λ=100 的第 100 轮 ratio 仍约为 VGG `0.0602`、Flickr `0.0620`。这表明损失仍能施加梯度，Decay 的小值主要包含人为权重缩小因素。
7. **OGL-IQR gap 是否缩小？** Decay 相对 Full 的 gap 变化是 VGG `-0.0043`、Flickr `-0.0160`。两者数值都缩小，但 Flickr 的 IQR 本身下降，且 OGL 也依赖变化后的 AUD，因此不能单独把 gap 缩小当成方法改进。
8. **VGGSS/Flickr 是否一致？** 不一致：Decay 在 VGG 提升 AUD/IQR、在 Flickr 降低；Delayed 在 VGG提升、Flickr约少一个样本。任何只报告 VGG 的结论都会夸大 schedule 有效性。
9. **最终判定：Case T5（跨数据集不一致的 T5）。** 跨数据集方向不一致，未满足 T1–T4 的一致性条件，不能支持 schedule 路线。 这里不是说 VGG 没有正信号，而是两个预注册 schedule 都没有满足跨数据集一致改善，因而不能进入 144k。

## 下一步

不建议进入 144k；本实验未启动任何 144k 或后续模型。
