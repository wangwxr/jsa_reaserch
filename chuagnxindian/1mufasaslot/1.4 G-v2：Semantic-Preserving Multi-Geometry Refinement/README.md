# Experiment G-v2：Semantic-Preserving Multi-Geometry Refinement

本目录是独立的 1.4 实验，不修改根目录 JSA、L3+L4、Experiment F 或 1.3G。Base checkpoint 仍为现有 L3+L4 Two-Level best checkpoint。

## 保持不变的模型

- Frozen L3+L4 teacher；teacher 始终 `eval()` 且不进入 optimizer；
- trainable `proj3_spatial`，由 frozen teacher `proj3` 精确复制初始化；
- trainable `TopDownL3Adapter`：`Conv1x1(512→256) → GELU → Conv3x3(256→512)`；
- `F34 = Up(F4) + Adapter(proj3_spatial(L3_native))`；
- frozen `Qa` 和 frozen L4 key transformation；
- `Qa → K34 → AUD_FINE [B,1,14,14]`；
- 总可训练参数应为 1,443,072，和 1.3G 一致；
- AdamW、`lr=5e-5`、`weight_decay=0.01`、batch size、数据和 evaluator 均不变。

## Semantic-Preserving Crop

先在 View A 上计算并归一化 frozen `AUD_L4_A = T_A [B,1,7,7]`。将它 bilinear upsample 到模型输入大小并再次归一化。对每个样本最多采样10个：

```text
RandomResizedCrop scale=(0.6,1.0), ratio=(0.9,1.1)
```

使用输入像素空间的积分图精确计算候选矩形中的 teacher probability mass。第一个满足 `mass >= 0.60` 的 crop 被接受；10次均失败则使用 identity crop，但仍独立按 `p=0.5` 做 horizontal flip。不会丢弃样本。

## Single Semantic Anchor

View B 正常通过 frozen image encoder 取得 student 所需 `F3_B/F4_B`，但不计算、也不使用 `teacher(View B)` 的 coarse prediction。唯一语义锚点是：

```text
T_A = AUD_L4_A
T_B_target = crop_resize_flip(T_A)
```

`T_B_target` 使用与 View B 图像完全相同的坐标变换并重新归一化。

## Loss

`AUD_FINE_B` inverse warp 到 A 后，在有效重叠区重归一化，继续使用1.3G的 symmetric KL：

```text
L_equiv = 0.5 * [KL(Fine_A || Align(Fine_B)) + KL(Align(Fine_B) || Fine_A)]
```

14×14 fine map 使用 mass-preserving 2×2 sum pooling：

```text
L_coarse_A = KL(T_A || Pool(AUD_FINE_A))
L_coarse_B = KL(Transform(T_A) || Pool(AUD_FINE_B))
L_coarse   = 0.5 * (L_coarse_A + L_coarse_B)
L_total    = L_coarse + L_equiv
```

两项权重固定为1，不 sweep。不使用 GT localization、OGL、OBJ_PRIOR、InfoNCE、reconstruction 或新 head。

## Gradient diagnostic

每20个 iteration 用 `torch.autograd.grad` 对全部 trainable student 参数分别读取 `L_coarse` 与 `L_equiv` 的梯度，记录：

- `grad_norm_coarse`
- `grad_norm_equiv`
- `grad_norm_ratio`
- `grad_cosine`
- `negative_grad_cosine_ratio`

诊断不会写入 `.grad`、不会额外 `optimizer.step()`。sanity 会比较启用诊断前后的正式 total gradient，确认诊断没有改变训练梯度。

## Sanity

```bash
bash "chuagnxindian/1mufasaslot/1.4 G-v2：Semantic-Preserving Multi-Geometry Refinement/sanity_check_10k.sh"
```

检查包括 full crop mass、所有 accepted crop mass、强制 fallback、single anchor 来源、identity/flip/crop warp、概率和、teacher/student gradient、gradient diagnostic 无副作用以及 L3+L4 AUD 基准精确复现。

## 10轮双卡训练与自动六指标测试

```bash
bash "chuagnxindian/1mufasaslot/1.4 G-v2：Semantic-Preserving Multi-Geometry Refinement/run_10k_two_gpus.sh"
```

它并行执行：

- GPU 0：VGGSoundSS-10k，10 epochs；
- GPU 1：Flickr-10k，10 epochs。

每个训练完成后自动读取按 `AUD_FINE cIoU` 保存的 best checkpoint，并用原 evaluator 输出：

```text
AUD
IMG_QUERY
IQR
OBJ_PRIOR
OGL
EXTRA_IQR_OGL
DELTA_OGL = OGL - AUD
```

OGL/OBJ_PRIOR 只在训练完成后的测试脚本加载，绝不参与训练、loss、选 epoch 或 checkpoint 保存。

单独运行：

```bash
bash "chuagnxindian/1mufasaslot/1.4 G-v2：Semantic-Preserving Multi-Geometry Refinement/train_vggss_10k.sh" 0
bash "chuagnxindian/1mufasaslot/1.4 G-v2：Semantic-Preserving Multi-Geometry Refinement/train_flickr_10k.sh" 1
```

正式输出目录默认是：

```text
checkpoints/1.4G-v2-semantic_preserving_multigeom_vggss_10k
checkpoints/1.4G-v2-semantic_preserving_multigeom_flickr_10k_frame8_center5
```
