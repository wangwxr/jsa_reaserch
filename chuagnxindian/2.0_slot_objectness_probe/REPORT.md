# Experiment 2.0 结果报告

## 结论摘要

当前模型内部已经存在可利用的 object-level ownership，主要位于 L4 target slot；但它
不是一个可以对所有样本固定加权的稳定 object prior。Flickr-144k 上固定 0.6/0.4
融合有明确正增益，VGGSoundSS-144k 上则因 hurt 多于 rescue 而下降。L3 target
ownership 在两个数据集都不适合直接融合。

因此创新点 2.1 应走 **Audio-Guided Slot Selection / Internal Object Completion**，重点
解决“何时使用 L4 ownership”，而不是直接进入 Semantic-Spatial Decoupled Slot
Learning。后者更适合 slot assignment 基本没有有效信号的情形；本实验并非如此。

## 正式基线复现

正式 evaluator 的第一遍结果与 checkpoint 内已有结果逐项完全一致，第二遍 probe
局部 map 处理中的 AUD、IMG_QUERY、OBJ_PRIOR、OGL 也逐项误差为 0。

| Dataset | AUD_FINE | IMG_QUERY | IQR | OBJ_PRIOR | OGL | EXTRA_IQR_OGL |
|---|---:|---:|---:|---:|---:|---:|
| VGGSoundSS-144k | .4269/.4230 | .4069/.4166 | .4230/.4234 | .3478/.3924 | .4570/.4401 | .4447/.4343 |
| Flickr-144k | .8120/.6356 | .8040/.6166 | .8040/.6366 | .4480/.4668 | .8680/.6596 | .8440/.6534 |

每格均为 `cIoU/AUC`。测试样本数分别为 VGG 5158、Flickr 250。

## 固定 alpha=0.6

| Method | VGGSoundSS-144k | Flickr-144k |
|---|---:|---:|
| AUD_FINE | .4269/.4230 | .8120/.6356 |
| IMG_QUERY | .4069/.4166 | .8040/.6166 |
| SLOT_L3 | .1247/.2560 | .3520/.4524 |
| SLOT_L4 | .2144/.3472 | .7320/.5952 |
| AUD + SLOT_L3 | .2780/.3673 | .7240/.5982 |
| AUD + SLOT_L4 | .3858/.4157 | **.8360/.6396** |
| OBJ_PRIOR | .3478/.3924 | .4480/.4668 |
| OGL | **.4570/.4401** | **.8680/.6596** |

相对 AUD：

- VGG `AUD+SLOT_L4`：cIoU -0.0411，AUC -0.0072；
- Flickr `AUD+SLOT_L4`：cIoU +0.0240，AUC +0.0040；
- OGL 的增益分别为 VGG +0.0301/+0.0171、Flickr +0.0560/+0.0240。

## Alpha diagnostic（全部组合）

| Dataset | Level | alpha=.5 | alpha=.6 | alpha=.7 | alpha=.8 | alpha=.9 |
|---|---|---:|---:|---:|---:|---:|
| VGG | L3 | .2055/.3325 | .2780/.3673 | .3645/.3998 | .4052/.4157 | .4188/.4215 |
| VGG | L4 | .3548/.4058 | .3858/.4157 | .4064/.4224 | .4184/.4257 | .4246/.4260 |
| Flickr | L3 | .6480/.5648 | .7240/.5982 | .7680/.6258 | .8080/.6418 | .8120/.6422 |
| Flickr | L4 | .8280/.6360 | .8360/.6396 | .8320/.6422 | .8200/.6424 | .8120/.6398 |

该表仅用于诊断潜力，未据此训练或选择参数。VGG 即使 alpha=.9，L4 的 AUC 有
+0.0031，但 cIoU 仍比 AUD 低 0.0023；说明其主要问题不是固定权重偏大这么简单。

## Rescue / hurt（固定 alpha=0.6，成功阈值 IoU>=0.5）

| Dataset | Fusion | Rescue | 与 OGL rescue 重叠 | Only Slot | Only OGL | Neither | Hurt | Net |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VGG | AUD+SLOT_L3 | 85 | 35 | 50 | 322 | 2549 | 853 | -768 |
| VGG | AUD+SLOT_L4 | 355 | 220 | 135 | 137 | 2464 | 567 | -212 |
| Flickr | AUD+SLOT_L3 | 9 | 7 | 2 | 12 | 26 | 31 | -22 |
| Flickr | AUD+SLOT_L4 | 11 | 9 | 2 | 10 | 26 | 5 | +6 |

OGL 在 VGG rescue/hurt/net 为 357/202/+155，在 Flickr 为 19/5/+14。L4 在 VGG
救回数量 355 几乎等于 OGL 的 357，且 220 个救回重叠；失败来自它同时伤害了 567
个 AUD 成功样本，而非完全没有 object-completion 能力。Flickr 中 11 个 L4 rescue
有 9 个与 OGL 重叠，同时只伤害 5 个，所以得到 +0.024 cIoU。

`AUD_SLOT_ANY` 是“L3/L4 任一固定融合成功”的 **GT 后验 oracle union**，不是可部署
map，只用于显示内部上限，不作为正式方法结果。其 net 为 VGG +142、Flickr +15。

## Map complementarity

| Dataset | Pair | Pearson | Spearman | JS |
|---|---|---:|---:|---:|
| VGG | AUD vs SLOT_L3 | -0.106 | -0.125 | .0382 |
| VGG | AUD vs SLOT_L4 | .908 | .950 | .0357 |
| VGG | AUD vs OBJ_PRIOR | .805 | .814 | .0231 |
| VGG | SLOT_L3 vs OBJ_PRIOR | -.142 | -.149 | .0771 |
| VGG | SLOT_L4 vs OBJ_PRIOR | .754 | .777 | .0374 |
| Flickr | AUD vs SLOT_L3 | .177 | .162 | .0385 |
| Flickr | AUD vs SLOT_L4 | .934 | .938 | .0152 |
| Flickr | AUD vs OBJ_PRIOR | .735 | .764 | .0279 |
| Flickr | SLOT_L3 vs OBJ_PRIOR | .171 | .155 | .0701 |
| Flickr | SLOT_L4 vs OBJ_PRIOR | .695 | .711 | .0291 |

L4 ownership 与 AUD 高度相关，说明它不是像外部 object prior 那样完全独立的分支；
但它与真实 OBJ_PRIOR 仍有 .69--.78 的 rank/linear correlation，并能复现一部分 OGL
rescue。L3 虽更“不同”，但与 OBJ_PRIOR 同样负相关或弱相关，差异主要是噪声/错误
ownership，而不是有效互补。

## 定性观察

- VGG `-3Kv4fdm7Uk_000030`：AUD IoU .213，L4 ownership .798，融合后 .625，属于
  Slot 独立救回；L4 提供了 AUD coarse peak 缺失的物体范围。
- VGG `-2-wdcN5vOw_000017`：AUD .618，而 L4 融合降为 .467；L4 的空间范围偏移，
  说明固定注入会破坏已有正确定位。
- Flickr `10548273474`：AUD .448，L4 融合 .558，OGL .663；内部 L4 与外部 prior
  都能扩展到正确目标。
- Flickr `10061269855`：AUD .696、L4 单独 .681，但融合后反而 .483；在固定 0.6
  阈值和 min-max 口径下，两张各自成功但 extent 不同的 map 仍可能产生 hurt。

可视化使用预定义类别与测试顺序确定，不人工 cherry-pick。完整样本 ID 和选择规则见
各数据集 `qualitative/selection_manifest.csv`。

## 零训练审计

两个正式 checkpoint 的 SHA256 和纳秒时间戳在评测前后完全一致；没有 optimizer、
没有 backward、新增可训练参数为 0。ownership 的 slot-sum 最大误差为
`1.19e-7`，update-weight 的 token-sum 最大误差不超过 `2.38e-7`，本地重建与原
`VisualSlotBranch._attention` 最大误差为 0。
