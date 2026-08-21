# Experiment 2.3 正式实验报告

## 结论

本实验严格完成了 VGGSoundSS-144k 与 Flickr-144k 各 50 epoch 的正式训练。结果不支持当前版本的 Semantic-Spatial Decoupled Slot Learning。

新 Spatial Slot 分支在第一个 epoch 内迅速塌缩为几乎所有 token 都属于 slot0：slot0 mass 接近 1，slot1 mass 接近 0，ownership entropy 接近 0。这个坏解同时使 sound-seed loss 和 equivariance loss 接近 0，但没有形成可靠的对象边界。后期指标逐渐靠近冻结 AUD，主要是饱和 ownership 中的极小数值差经 evaluator 的 min-max normalization 放大，并非学到了互补 objectness。

正式 alpha=0.6 的结果：

| Dataset | Best epoch | AUD_FINE | SPATIAL_SLOT0 | AUD_SPATIAL | OGL |
|---|---:|---:|---:|---:|---:|
| VGGSoundSS-144k | 49 | 0.4269 / 0.4230 | 0.2197 / 0.3126 | 0.4199 / 0.4195 | 0.4570 / 0.4401 |
| Flickr-144k | 49 | 0.8120 / 0.6356 | 0.4800 / 0.4898 | 0.8120 / 0.6298 | 0.8680 / 0.6596 |

每格为 `cIoU / AUC`。VGG 的融合相对 AUD 为 `-0.0070 / -0.0035`；Flickr cIoU 追平 AUD，但 AUC 下降 0.0058。两者均未产生正增益。

## 与 2.2 HR14 逐项比较

| Dataset | Candidate | Fusion cIoU/AUC | Rescue | Hurt | Net | Oracle cIoU/AUC |
|---|---|---:|---:|---:|---:|---:|
| VGG | 2.2 OLD HR14 | 0.3777 / 0.4175 | 430 | 684 | -254 | 0.5103 / 0.4694 |
| VGG | 2.3 NEW Spatial | 0.4199 / 0.4195 | 3 | 39 | -36 | 0.4275 / 0.4237 |
| Flickr | 2.2 OLD HR14 | 0.8440 / 0.6502 | 12 | 4 | +8 | 0.8600 / 0.6664 |
| Flickr | 2.3 NEW Spatial | 0.8120 / 0.6298 | 1 | 1 | 0 | 0.8160 / 0.6368 |

2.3 在 VGG 上比旧 HR14 fusion 少伤害，但同时几乎完全丢掉 Rescue，oracle 从 0.5103 降至 0.4275，说明 candidate capacity 并没有改善。在 Flickr 上，2.2 的明确正增益和 oracle 上限都被破坏。

## Alpha diagnostic

| AUD alpha | VGG cIoU/AUC | Flickr cIoU/AUC |
|---:|---:|---:|
| 0.5 | 0.4166 / 0.4177 | 0.8040 / 0.6254 |
| 0.6 | 0.4199 / 0.4195 | 0.8120 / 0.6298 |
| 0.7 | 0.4232 / 0.4212 | 0.8120 / 0.6336 |
| 0.8 | 0.4248 / 0.4222 | 0.8120 / 0.6352 |
| 0.9 | 0.4259 / 0.4228 | 0.8120 / 0.6352 |

AUD 权重越高，结果越接近冻结 AUD；没有任何透明 alpha 组合超过 baseline。这进一步表明新 Spatial Slot 没有带来互补信息。

## 塌缩证据

训练前 smoke test 中，外部 semantic slots 初始化是健康的：

| Dataset | slot0 mass | slot1 mass | entropy | seed grad | equiv grad | visual grad | mass grad |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG | 0.5164 | 0.4836 | 0.9980 | 4.5588 | 0.0772 | 0.0097 | 2.0888 |
| Flickr | 0.5152 | 0.4848 | 0.9986 | 4.0395 | 0.0371 | 0.0051 | 1.2254 |

因此模块、反向传播和四项 loss 都不是一开始就失效。

但正式训练第 1 epoch 的均值已经变为：

| Dataset | slot0 mass | entropy | seed loss | equiv loss | mass loss |
|---|---:|---:|---:|---:|---:|
| VGG | 0.9959 | 0.0121 | 0.00516 | 0.000044 | 0.9104 |
| Flickr | 0.9962 | 0.0119 | 0.00411 | 0.000184 | 0.6247 |

best epoch 49：

| Dataset | slot0 mass | slot1 mass | entropy | seed loss | equiv loss | visual loss | mass loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG | 1.0000000000 | 8.67e-11 | 2.67e-7 | 9.11e-12 | 3.06e-12 | 0.1454 | 0.9145 |
| Flickr | 0.9999999994 | 1.23e-9 | 2.81e-7 | 1.42e-10 | 1.51e-10 | 0.1619 | 0.6287 |

在饱和以后，diagnostic interval 上四项梯度范数都下降到数值 0。`0.1 * L_mass` 虽然保持较大的 loss 值，但 softmax 已经饱和，无法把 slot 分配拉回。visual coherence 也允许“所有 token 属于同一个 prototype”的退化解，不能独立阻止 collapse。

塌缩的因果路径为：

1. `L_seed` 只奖励 AUD top10% 上的 slot0，天然推动 slot0 增大。
2. 全图 slot0 是完美等变的，因此 `L_equiv` 同时趋近 0。
3. 两 slot visual prototype 在极端分配下仍可得到有限的 `L_visual`，没有强制前景/背景分离。
4. `L_mass` 只约束全局均值且权重为 0.1；进入 softmax 饱和区后梯度消失。
5. 最终 ownership 是近常数图。evaluator 的 min-max normalization 会放大其极小残差，造成后期融合指标看似逐渐回升，但定性图与 mass/entropy 明确表明这不是对象 ownership。

## 冻结与参数审计

- 总参数：41,277,442
- 冻结参数：37,860,354
- 可训练参数：3,417,088
- Spatial Slot 参数：3,417,088
- 所有可训练参数名均以 `spatial_slot.` 开头
- 1.3G checkpoint 哈希/路径检查未发生变化
- `F34.requires_grad=False`
- `semantic_initial_slots.requires_grad=False`
- `AUD_FINE.requires_grad=False`
- ownership shape：`[B,2,196]`
- ownership slot-sum 最大误差：约 `5.96e-8`
- smoke test 中 frozen model 无梯度，Spatial Slot 四类 loss 均有非零梯度
- 无 NaN/Inf

## 训练时间与输出

- VGGSoundSS：11,068.4 秒，约 3:04:28
- Flickr：6,990.7 秒，约 1:56:31

VGG 输出：

- `checkpoints/2.3_semantic_spatial_decoupled_slot_vggss_144k/summary.json`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_vggss_144k/epoch_metrics.csv`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_vggss_144k/training_curves.png`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_vggss_144k/training_curves.pdf`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_vggss_144k/results/qualitative/`

Flickr 输出：

- `checkpoints/2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5/summary.json`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5/epoch_metrics.csv`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5/training_curves.png`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5/training_curves.pdf`
- `checkpoints/2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5/results/qualitative/`

定性样本按 test order 和预定义类别轮询选择，没有 cherry-pick。VGG 与 Flickr 的 NEW SPATIAL_SLOT0 面板都呈现近乎整幅均匀前景，和 mass/entropy 诊断一致。

## 最终判断

2.3 的“语义 slot 初始化 + sound seed + 几何等变 + visual coherence + weak mass”在当前固定权重下没有学到可靠 object ownership。它找到的是一个更简单的全 slot0 解。该结果否定的是本次具体自监督约束组合，而不是“semantic/spatial decoupling”概念本身；但按本轮实验标准，方向没有成立，不应把 2.3 作为有效创新结果。
