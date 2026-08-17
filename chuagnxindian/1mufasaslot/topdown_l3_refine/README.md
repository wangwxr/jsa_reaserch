# Top-down native-L3 refinement（Stage-2）

本目录实现第二创新点的第一版验证：冻结已经训练完成的 **L3+L4 Two-Level Slot Aggregation + L4-only Grounding**，只训练一个很小的 top-down residual adapter，测试深层 L4 语义能否把 native L3 变成可用的 14×14 sound-localization feature。代码不修改根目录 JSA、v1/v1.1、L4×3、L3+L4 ablation 或上一个 zero-shot affinity 实验。

## 冻结的 base 与 feature 位置

四个 base checkpoint 由 `common.py` 固定映射到已有实验目录中的 `vggss_best.pth` / `flickr_best.pth`。加载后执行：

```python
for p in base_model.parameters():
    p.requires_grad = False
base_model.eval()
```

`model.py::_ProjectedFeatureHooks` 观察 frozen backbone 的 `imgnet.proj3` 和 `imgnet.proj4` 输出：

- `F3_native`：`two_level_resnet.py` 中 `proj3(layer3)` 的输出，发生在 `_pool_to_7x7` 之前，实际要求为 `[B,512,14,14]`。
- `F4_projected`：同一次正式 backbone forward 中的 `proj4(layer4)` 输出，实际要求为 `[B,512,7,7]`；其 flatten 结果会与正式 `image_levels[-1]` 做逐元素一致性检查。

因此 proj3/proj4 都来自原 checkpoint，不新增、不重新初始化，也不会更新。

## Top-down head

唯一可训练模块为：

```text
F3_native
  └─ Conv1×1(512→256) ─ GELU ─ Conv3×3(256→512) ─ ΔF3

F34 = bilinear_up(F4_projected) + ΔF3
```

最后一个 3×3 convolution 的 weight 和 bias 均为 zero initialization。初始化时严格满足：

```text
ΔF3 = 0
F34 = bilinear_up(F4_projected)
```

没有 FPN、Transformer、L2、新 Slot Attention 或外部模型。

## 复用正式 L4 query-key 空间

`F34` flatten 为 `[B,196,512]` 后，直接经过 frozen L4 visual branch 的同一套：

```text
img_norm_input → img_to_k
```

得到 `K34`。这里没有 `fine_img_to_k`。音频 query `Qa` 与 L4 image query `Q4` 也来自 frozen base 的正式 `_encode()`。细粒度 attention 调用 base 原有 `_attention()`，因而沿用相同的 `512^-0.5` scale、`infer_sharpening=0.1`、slot normalization 和 spatial probability normalization：

```text
AUD_FINE = Attention(Qa, K34)[:, target] → [B,1,14,14]
IMG_FINE = Attention(Q4, K34)[:, target] → [B,1,14,14]
```

coarse teacher 完全沿用正式 L4-only 路径：

```text
AUD_L4 = Attention(Qa, K4)[:, target] → [B,1,7,7]
IMG_L4 = Attention(Q4, K4)[:, target] → [B,1,7,7]
```

两张 teacher map 与全部 base feature/query 都 detach。

## Stage-2 loss

14×14 probability map 使用保概率质量的 pooling：

```python
y = avg_pool2d(x, 2, 2) * 4.0
y = y / (y.sum((-2, -1), keepdim=True) + 1e-8)
```

仅优化三个无 GT loss：

```text
L_fine_match = MSE(AUD_FINE, detach(IMG_FINE))
L_coarse_aud = MSE(sum_pool_2x2(AUD_FINE), detach(AUD_L4))
L_coarse_img = MSE(sum_pool_2x2(IMG_FINE), detach(IMG_L4))

L_refine = 1.0 * L_fine_match
         + 1.0 * (L_coarse_aud + L_coarse_img)
```

不再计算/优化 InfoNCE、reconstruction、divergence 或原始 attention loss。optimizer 只有 head parameters：AdamW，lr `5e-5`，weight decay `0.01`，无 scheduler。batch size、epoch、seed、augmentation 和数据路径继承对应 base config；当前 launcher 的训练 workers 为 VGGSoundSS 16、Flickr 12。10k 为 100 epochs，144k 为 50 epochs。

## Evaluation 与 checkpoint

本地 evaluator 只把现有 JSA 协议泛化到任意 heatmap resolution：bicubic resize 到 224×224、逐图 min-max normalization、threshold 0.6、同一 cIoU/AP50 与 AUC 定义。14×14 map 直接评测，不先降采样。

输出：`AUD_L4`、`AUD_FINE`、`IMG_L4`、`IMG_FINE`、`IQR_FINE`。IQR 延续现有 evaluator：AUD/IMG 分别 resize 和 min-max normalize 后，以 `0.6/0.4` 混合并再次 normalize。主结果与 best checkpoint 的唯一选择指标是 `AUD_FINE cIoU`。没有加载或计算 OGL、OBJ_PRIOR、ImageNet objectness，GT 只用于 evaluation/可视化，绝不进入 head 或 loss。

checkpoint 仅保存：base checkpoint 路径、refinement head state、optimizer state、epoch 与 metrics，不复制或覆盖 base model。

每轮生成：

- `epoch_metrics.csv`
- `training_curves.png` 与 `training_curves.pdf`
- `latest.pth`，按 AUD_FINE cIoU 选择的 `<testset>_best.pth`，以及 `final.pth`
- best 模型固定等间距选取 10 个 test sample 的 qualitative PNG/PDF 和 `qualitative_sample_ids.csv`

## 强制 sanity checks

每次正式训练前自动完成：

1. 在完整 test split 上精确复现既有 AUD_L4 与 IMG_L4（比较到四位小数）。
2. 一次 backward 后全部 base parameter 的 `.grad is None`。
3. refinement head 至少有一个非零 gradient。
4. zero-init 时 `max|F34-Up(F4)| <= 1e-7`，且 hook F4 与正式 L4 tokens 一致。
5. AUD_FINE、IMG_FINE、AUD_L4、IMG_L4 的空间概率和约等于 1。
6. `sum_pool_2x2(AUD_FINE)` 的空间概率和仍约等于 1。

`--sanity-only` 只执行完整 baseline evaluation 和一个 batch backward，不创建 optimizer、不执行 optimizer step、不写训练 checkpoint。

## 从 repo root 运行

先只做 10k sanity check（GPU 0/1 并行）：

```bash
bash chuagnxindian/1mufasaslot/topdown_l3_refine/sanity_check_10k.sh
```

第一轮正式实验只跑两个 10k：

```bash
bash chuagnxindian/1mufasaslot/topdown_l3_refine/run_10k_two_gpus.sh
```

也可以分别运行并自定义 GPU/实验名：

```bash
bash chuagnxindian/1mufasaslot/topdown_l3_refine/train_vggss.sh 10k 0 topdown_l3_refine_vggss_10k
bash chuagnxindian/1mufasaslot/topdown_l3_refine/train_flickr.sh 10k 1 topdown_l3_refine_flickr_10k_frame8_center5
```

只有 10k 出现正信号后才运行 144k：

```bash
bash chuagnxindian/1mufasaslot/topdown_l3_refine/run_144k_two_gpus.sh
```

测试 best head：

```bash
bash chuagnxindian/1mufasaslot/topdown_l3_refine/test_vggss.sh 10k 0
bash chuagnxindian/1mufasaslot/topdown_l3_refine/test_flickr.sh 10k 1
```

如当前 shell 未激活正确环境，可显式指定：

```bash
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python bash chuagnxindian/1mufasaslot/topdown_l3_refine/sanity_check_10k.sh
```
