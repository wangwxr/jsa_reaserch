# Experiment 6.0 — Stage A Loss Audit

## 审计结论状态

- Full 数值回归：通过。
- VGGSS 最大误差：`0.000e+00`。
- Flickr 最大误差：`0.000e+00`。
- 两个正式 checkpoint 的 SHA256 / size / mtime 在审计前后保持一致。
- 本阶段未创建 optimizer、未调用 optimizer.step，也未写入模型权重。

## Decoder alpha 与定位分支

| Dataset | decoder target–IMG Pearson | decoder target–AUD Pearson | IMG–AUD Pearson |
|---|---:|---:|---:|
| VGGSS | 0.5508 | 0.5504 | 0.9859 |
| Flickr | 0.4395 | 0.4414 | 0.9873 |

## 加权梯度证据（20 batches 均值）

| Dataset | info global | att-space global | rec-img global | rec-img L3-SA | rec-img L4-SA | cos(rec-img, info) | cos(rec-img, div-img) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS | 68.8471 | 0.4112 | 0.1113 | 0.0311 | 0.0412 | -0.0022 | 0.0221 |
| Flickr | 62.3850 | 0.1828 | 0.0561 | 0.0152 | 0.0201 | 0.0073 | 0.0617 |

## 五个因果问题

1. **spatial attention 是否在梯度上主导 AUD 空间定位**：在已收敛的正式 checkpoint 上不主导。VGGSS/Flickr 的加权 `att_space` global norm 仅为 `0.4112/0.1828`，而 InfoNCE 为 `68.8471/62.3850`。这不否定早期训练的因果作用，因此需由 Stage B 验证。
2. **image reconstruction 是否直接塑造 localization attention**：decoder target alpha 与 IMG/AUD 只有中等 Pearson（VGG约 0.55，Flickr约 0.44），而 top-10 overlap 仅 `0.230/0.131`；相比之下 IMG–AUD Pearson 约 0.986/0.987。因此 rec_img 更像 slot-feature/partition regularizer，不能仅凭 alpha 直接解释 SHRINK。
3. **rec_img、div_img、info_loss 是否冲突**：没有明显冲突。`cos(rec_img, info)` 为 `-0.0022/0.0073`，`cos(rec_img, div_img)` 为 `0.0221/0.0617`，均接近零或轻微同向。
4. **L3 是否被 L4 reconstruction 牵制**：存在明确耦合但没有冲突证据。rec_img 对 L3-SA 的梯度约为 L4-SA 的 `75.3%`（VGG）和 `75.8%`（Flickr），说明 L4 target 经 M-Fusion 实质性约束 L3；但方向性与 InfoNCE 近正交，不能称为已证实的“牵制”。
5. **AUD–IMG 高相关来自何处**：收敛时显式 att-space 梯度很小，但 IMG–AUD 已高度相关，Stage A 更支持“共享 slot index/相同 L4 keys 与既有训练共同形成的结构性对齐”，不能仅归因当前 att loss。NO_SPATIAL_ATT 的从头训练才是区分 architecture 与 loss 的因果证据。

完整数值位于本目录 CSV/JSON；固定测试样本可视化位于 `visualizations/`。
