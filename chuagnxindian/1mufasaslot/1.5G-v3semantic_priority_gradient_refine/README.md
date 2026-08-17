# Experiment G-v3：Semantic-Priority Gradient Projection

本目录是独立的 1.5 实验。它完整复用 G-v2 的 frozen teacher、student、数据、Semantic-Preserving Crop、Single Semantic Anchor、loss、优化器和评测；唯一变化是训练时如何合并 `L_coarse` 与 `L_equiv` 的梯度。不会修改根目录 JSA 或 1.3G/1.4G-v2。

## 保持不变

- Frozen L3+L4 teacher，始终 `eval()`，不进入 optimizer；
- trainable `proj3_spatial + TopDownL3Adapter`，参数量与 G/G-v2 相同；
- `F34 = Up(F4) + Adapter(proj3_spatial(L3_native))`；
- frozen `Qa → K34 → AUD_FINE [B,1,14,14]`；
- RandomResizedCrop `scale=(0.6,1.0)`、`ratio=(0.9,1.1)`；
- crop teacher mass 阈值 0.60，最多10次，失败回退 identity；
- horizontal flip 概率 0.5；
- `T_B = Transform_A_to_B(T_A)`，不使用 `teacher(View B)` 作为 coarse target；
- `L_total = L_coarse + L_equiv`，两项权重均为1；
- AdamW，`lr=5e-5`，`weight_decay=0.01`，无 scheduler；
- 原 batch size、数据增强、seed、10 epochs、best checkpoint 规则均不变；
- 不使用 GT localization、OGL、OBJ_PRIOR 或新网络/新 loss。

## 唯一算法修改：全局 semantic-priority projection

对全部可训练参数分别计算未缩放 FP32 梯度：

```text
gc = grad(L_coarse)
ge = grad(L_equiv)
dot     = sum_i <gc_i, ge_i>
norm_c2 = sum_i ||gc_i||^2
```

若 `dot >= 0`，保持 `ge' = ge`。若 `dot < 0`，使用所有 trainable parameters 共同计算出的唯一系数：

```text
alpha = dot / (norm_c2 + eps)
ge'_i = ge_i - alpha * gc_i
g_i   = gc_i + ge'_i
```

只投影 equivariance gradient，绝不修改 coarse semantic gradient。实现不会按单个 parameter 分别求 alpha。若某项 loss 对某参数的 gradient 为 `None`，该项按零安全处理；若两项均为 `None`，该参数保持 `grad=None`。

AMP 下，两项 loss 使用同一 GradScaler scale 分别调用 `torch.autograd.grad`，随后统一除以 scale，在同一个未缩放 FP32 空间投影。最终梯度按当前 scale 写入 `.grad`，由原 `GradScaler.step()` 正常检查、unscale 和更新。`loss_total` 只记录数值，代码不调用 `loss_total.backward()`。

投影在每个 batch 执行。`--gradient-diagnostic-interval` 只控制详细 JSONL 审计的打印/保存频率，不会降低投影频率。

## 审计输出

每个 batch 都参与 epoch 均值，记录：

- `grad_norm_coarse`
- `grad_norm_equiv_before / after`
- `grad_norm_final`
- `projection_applied / projection_alpha`
- `grad_cosine_before / after`

每 epoch 额外记录 `projection_rate`、`mean_cosine_before/after` 和冲突 batch 的 `max_abs_conflict_cosine_after`。按间隔保存的逐 batch 详情位于：

```text
checkpoints/<experiment>/gradient_projection_audit.jsonl
```

冲突 batch 会运行时断言 `abs(cosine_after) <= 1e-4`；非冲突 batch 的 equivariance gradient 保持原样。

## Sanity check

```bash
bash chuagnxindian/1mufasaslot/1.5G-v3semantic_priority_gradient_refine/sanity_check_10k.sh
```

除 G-v2 原有 geometry、single-anchor、probability、teacher freeze 和 baseline AUD 检查外，还验证：全局 alpha、冲突正交性、非冲突不变、None-gradient 路径、真实模型 projected gradient、AMP 手动梯度路径和 teacher 无梯度。

## 双卡运行两个10k实验

```bash
bash chuagnxindian/1mufasaslot/1.5G-v3semantic_priority_gradient_refine/run_10k_two_gpus.sh
```

它并行运行：

- GPU 0：VGGSoundSS-10k，10 epochs；
- GPU 1：Flickr-10k，10 epochs。

单独运行：

```bash
bash chuagnxindian/1mufasaslot/1.5G-v3semantic_priority_gradient_refine/train_vggss_10k.sh 0
bash chuagnxindian/1mufasaslot/1.5G-v3semantic_priority_gradient_refine/train_flickr_10k.sh 1
```

默认 checkpoint 目录：

```text
checkpoints/1.5G-v3semantic_priority_gradient_refine_vggss_10k
checkpoints/1.5G-v3semantic_priority_gradient_refine_flickr_10k_frame8_center5
```

训练结束后脚本自动用 best checkpoint 输出六指标：`AUD / IMG_QUERY / IQR / OBJ_PRIOR / OGL / EXTRA_IQR_OGL`，并报告 `DELTA_OGL = OGL - AUD`。OGL 和 OBJ_PRIOR 只在训练结束后的 evaluator 中使用，不参与 loss、训练或选模。

若只需对已有 best checkpoint 重跑六指标：

```bash
bash chuagnxindian/1mufasaslot/1.5G-v3semantic_priority_gradient_refine/test_full_10k_two_gpus.sh
```
