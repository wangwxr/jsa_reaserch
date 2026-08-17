# Experiment D：Joint Fine-tuning

本目录在不修改根目录 JSA、现有 L3+L4 ablation 和 Frozen Top-down 实验的前提下，实现 **L3+L4 Two-Level + Top-down L3 refinement 的联合微调**。第一轮仅提供 VGGSoundSS-10k 与 Flickr-10k，不会自动运行 144k。

## 初始化与冻结规则

主模型从已有 L3+L4 best checkpoint 初始化：

- VGGSS-10k：`mufasa_ablation2_l3_l4_ablation_vggss_10k/vggss_best.pth`
- Flickr-10k：`mufasa_ablation2_l3_l4_ablation_flickr_10k_frame8_center5/flickr_best.pth`

代码先实例化原 `MUFASAL3L4`，逐参数记录其原始 `requires_grad`，加载 checkpoint 后再逐参数恢复该状态。因此没有使用无条件的 `requires_grad_(True)`。

当前本地原 L3+L4 训练代码没有设置任何 base parameter 为 frozen；`train_slot.py` 将所有 `requires_grad=True` 参数交给 AdamW。因此 Experiment D 实际恢复为：

- `imgnet`：可训练，包括视觉 ResNet、`proj3`、`proj4`；
- `audnet`：可训练；
- `slot_attn`：可训练，包括 L3/L4 visual SA、single audio SA、shared slots、mask tokens、M-Fusion；
- `img_decoder`、`aud_decoder`：可训练；
- 新 `TopDownL3RefinementHead`：可训练；
- 原模型中本来 frozen 的参数：0；若以后原实现出现 frozen 参数，逐参数模板会继续保持其 frozen 状态。

启动时打印 total/trainable/frozen base、refinement head 和 optimizer 参数量。优化器参数数必须严格等于“原 base trainable + head”，否则立即报错。

## 数据流

Top-down 结构与 Frozen Top-down 完全相同：

```text
F3_native = proj3(layer3), pooling 前                    [B,512,14,14]
F4         = proj4(layer4)                               [B,512,7,7]
DeltaF3    = Conv1x1(512->256)-GELU-Conv3x3(256->512)(F3)
F34        = bilinear_up(F4) + DeltaF3                   [B,512,14,14]
```

最后一层 Conv3×3 的 weight/bias 仍为全零，所以初始严格满足 `F34=Up(F4)`。不加载已经训练过的 Frozen Top-down head。

`F34` 继续复用现有 L4 branch 的 `img_norm_input -> img_to_k`，没有新建 fine key projection。`Qa`、`Q4`、attention scale 和 `infer_sharpening=0.1` 均来自同一次 L3+L4 forward：

```text
AUD_FINE = Attention(Qa, K34)[:,slot0] -> 14x14
IMG_FINE = Attention(Q4, K34)[:,slot0] -> 14x14
```

训练时只执行一次 imgnet、audnet 和 Slot Attention，保证原 masking 采样逻辑没有被复制或改变。

## Loss

完整 base loss 与原 L3+L4 相同：

```text
L_base = L_info + lam1*L_recon + lam2*L_div + lam3*L_att
lam1=0.1, lam2=0.1, lam3=100.0
```

refinement loss 与 Frozen Top-down 相同：

```text
L_fine_match = MSE(AUD_FINE, detach(IMG_FINE))
L_coarse_aud = MSE(sum_pool_2x2(AUD_FINE), detach(AUD_L4))
L_coarse_img = MSE(sum_pool_2x2(IMG_FINE), detach(IMG_L4))
L_refine = L_fine_match + L_coarse_aud + L_coarse_img

L_total = L_base + 1.0*L_refine
```

`sum_pool_2x2` 保持 probability mass。coarse maps 的 target 端始终 detach，但 Qa/Q4/F3/F4 在 Fine 计算路径中可以收到梯度。当前 checkpoint 的 `warmup=-1`，所以四项 base loss 从第一个 epoch 就全部启用。

优化器为 AdamW，lr `5e-5`、weight decay `0.01`、无 scheduler；batch size 256、100 epochs、seed 12345、数据增强和采样逻辑均继承原 base config。VGGSS workers=16，Flickr workers=12。

## Evaluation 与 checkpoint

每个 epoch 使用与 Frozen Top-down 相同的 evaluator，记录：

- `AUD_L4`、`AUD_FINE`；
- `IMG_L4`、`IMG_FINE`；
- `IQR_FINE = 0.6*AUD_FINE + 0.4*IMG_FINE`（沿用 evaluator 的 resize/normalize/fusion 顺序）；
- InfoNCE、reconstruction、divergence、attention、refinement 分项和 total loss。

best checkpoint 只按 `AUD_FINE cIoU` 选择。checkpoint 保存完整 joint model、optimizer、epoch、metrics 和 base checkpoint 路径，不覆盖任何已有实验。训练不加载 OGL/OBJ_PRIOR，也不使用 GT localization supervision。

## Sanity / gradient audit

正式训练前自动执行完整 baseline evaluation 和一个小 batch backward：

1. `AUD_L4/IMG_L4` 精确复现原 L3+L4 best checkpoint；
2. 原 base trainable 参数具有非零 gradient；
3. refinement head 具有非零 gradient；
4. 原本应 frozen 的参数没有 gradient；
5. zero-init 满足 `F34=Up(F4)`；
6. F4 hook 与正式 L4 tokens 一致；
7. 四张 map 和 sum-pool 后的 map 均保持 spatial probability sum=1。

正式训练会在 audit 后重新从 base checkpoint 构建一份干净模型，避免 audit 的 train-mode forward 改变 BatchNorm buffer。

## 从 repo root 运行

只做两个 10k sanity/gradient audit，不执行 optimizer step：

```bash
bash chuagnxindian/1.2topdown/sanity_check_10k.sh
```

分别启动两个 10k：

```bash
bash chuagnxindian/1.2topdown/train_vggss_10k.sh 0
bash chuagnxindian/1.2topdown/train_flickr_10k.sh 1
```

两卡同时启动两个 10k：

```bash
bash chuagnxindian/1.2topdown/run_10k_two_gpus.sh
```

可以用第二个 positional argument 自定义实验名。若没有激活环境：

```bash
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python bash chuagnxindian/1.2topdown/run_10k_two_gpus.sh
```
