# Experiment G：Multi-Geometry Equivariant Spatial Refinement

本目录包含 Experiment G 的完整本地实现。Experiment F 中不变的冻结教师、student、single-view evaluation、optimizer 和 checkpoint 逻辑已经复制到这里，再将 horizontal-flip-only 几何替换为可审计的 crop/scale/flip 几何。原 Experiment F 目录与 checkpoint 保留不动，避免破坏复现。

## 不变部分

- Frozen L3+L4 semantic teacher，始终 `eval()` 且不进 optimizer。
- Trainable `proj3_spatial`，从 frozen teacher `proj3` 精确复制初始化。
- Trainable `Conv1x1(512→256) → GELU → Conv3x3(256→512)` adapter，末层 zero-init。
- `F34 = Up(F4) + Adapter(F3_spatial)`。
- Frozen `Qa` 和 frozen L4 `LayerNorm → img_to_k` 生成 14×14 `AUD_FINE`。
- AdamW、lr `5e-5`、weight decay `0.01`、batch size 256、无 scheduler；epochs沿用对应 base 配置（10k为100，144k为50）。
- Test 为单 view，无 crop/flip averaging，best 只按 `AUD_FINE cIoU` 保存。

## View B 和显式几何

View A 是现有 DataLoader 的 normal training view。View B 从同一个 View A tensor 构造：

```text
RandomResizedCrop area scale=(0.6,1.0)
RandomResizedCrop aspect ratio=(0.9,1.1)
resize to 224×224
horizontal flip with p=0.5
```

`geometry.py` 为每个样本保存：`crop_top/crop_left/crop_height/crop_width/flipped/original_height/original_width`。View B 图像生成和 B heatmap 反变换共享同一套 `align_corners=False` 网格公式，不调用会丢弃参数的黑盒 transform。

对 A 的任意空间点，先将它映射到 crop 内的相对坐标，再根据 `flipped` 反转 x，作为 `grid_sample` 在 B map 上的采样位置，得到 `AUD_FINE_B_TO_A`。`VALID_MASK` 采用保守的 full-bilinear-support overlap：只有采样 footprint 完全落在 B heatmap 内的位置才参与 KL，避免 crop 边界 padding zero 污染。

## Loss

在 14×14 overlap 内分别重新归一化：

```text
P = normalize(AUD_FINE_A * VALID_MASK)
Q = normalize(AUD_FINE_B_TO_A * VALID_MASK)

L_equiv = 0.5 * [KL(P||Q) + KL(Q||P)]
```

`valid_ratio < 0.2` 的样本跳过 equivariance loss并计数。当前 crop scale 下通常不会触发，但保留数值保护。

Coarse anchor：

```text
L_coarse_A = KL(detach(AUD_L4_A) || SumPool(AUD_FINE_A))

AUD_L4_B 和 AUD_FINE_B 均按同一 crop/flip warp 回 A；
在 7×7 overlap 内重新归一化后：
L_coarse_B = KL(detach(AUD_L4_B_TO_A) || SumPool(AUD_FINE_B_TO_A))

L_coarse = 0.5 * (L_coarse_A + L_coarse_B)
L_total = L_coarse + 1.0 * L_equiv
```

不使用 GT localization、OGL、OBJ_PRIOR、InfoNCE、reconstruction、divergence、IMG_FINE matching、entropy loss 或 pseudo mask。

## 自动 sanity

正式训练前必须通过：

1. 人工非对称 14×14 affine heatmap 的 crop+resize+flip→inverse-warp 恢复误差检查。
2. identity crop + no flip 的最大误差接近 0。
3. pure flip 与 Experiment F 的 `torch.flip(..., dims=[-1])` 一致。
4. 保存 synthetic geometry 和至少四个真实训练样本的 View A/View B/Fine A/Fine B/B→A/VALID_MASK 面板。
5. 完整 test split 精确复现 frozen `AUD_L4`。
6. KL finite、teacher 无 gradient、adapter 第一轮 gradient 非零。
7. 临时执行一次并恢复的 audit update 后，验证 `proj3_spatial` 和 adapter 都有非零 gradient；正式训练前恢复原始 zero-init 权重。

实际几何参数保存在 `augmentation_geometry_samples.json`，面板为 `training_augmentation_audit.png` 和 `synthetic_geometry_sanity.png`。

## 每轮输出

```text
loss_coarse / loss_equiv / loss_total
mean_valid_ratio
skipped_small_overlap_samples
actual_flip_ratio
mean_crop_scale
AUD_L4 cIoU/AUC
AUD_FINE cIoU/AUC
```

同时写入 `epoch_metrics.csv` 与 `training_curves.png/pdf`。

## 从 repo root 运行

Sanity：

```bash
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/sanity_check_10k.sh
```

两个 10k 在 GPU 0/1 并行：

```bash
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/run_10k_two_gpus.sh
```

分别运行：

```bash
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/train_vggss_10k.sh 0
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/train_flickr_10k.sh 1
```

两个144k在 GPU 0/1 并行，完成后自动使用 best checkpoint 输出六指标：

```bash
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/run_144k_two_gpus.sh
```

分别运行：

```bash
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/train_vggss_144k.sh 0
bash chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine/train_flickr_144k.sh 1
```

输出目录：

```text
checkpoints/1.3G-multigeom_equivariant_l3_refine_vggss_144k
checkpoints/1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5
```
