# Experiment B：L4×3 Parameter-Control

## 唯一变量

v1.1 输入三套独立 visual Slot Attention 的特征为 `L2/L3/L4`。本实验改为：

```text
same projected L4 -> Visual SA2 -> S2
same projected L4 -> Visual SA3 -> S3
same projected L4 -> Visual SA4 -> S4
S2,S3,S4 -> unchanged three-level M-Fusion -> Sf
```

三套 SA 是三个不同模块，参数不共享；initial slots、audio encoder、single audio SA、
masking、M-Fusion、decoder、四个 loss、loss 权重和 v1.1 完全一致。

Sf 继续用于 InfoNCE、image reconstruction 和 visual divergence。reconstruction target
仍是 L4。`att_loss`、`AUD`、`IMG_QUERY` 仍只使用第三套 visual SA 对 L4 得到的 attention，
checkpoint 仍按 L4-only IQR cIoU 保存。

## 参数量

运行时自动打印：

```text
L4x3 control total/trainable: 41,014,018
MUFASA-JSA v1.1 total/trainable: 41,211,650
difference: -197,632 (-0.480%)
```

差异只来自 control 不再需要 L2/L3 projection，共 197,632 个参数。保留这些 projection
只能形成永远没有梯度的死参数，因此没有为了表面上的完全相等而保留它们。三套 visual
Slot Attention 和 M-Fusion 的参数量与 v1.1 完全相同，这是本消融要控制的主要容量。

## 命令

```bash
bash chuagnxindian/mufasa_ablation1_l4x3_control/train_vggss.sh \
10k 0 mufasa_ablation1_l4x3_control_vggss_10k

bash chuagnxindian/mufasa_ablation1_l4x3_control/train_flickr.sh \
10k 1 mufasa_ablation1_l4x3_control_flickr_10k_frame8_center5
```
