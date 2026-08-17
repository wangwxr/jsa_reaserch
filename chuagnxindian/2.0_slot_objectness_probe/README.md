# 2.0 Internal Slot Objectness Probe

这是创新点 2 的第一个实验。它只读取正式 1.3G-144k best checkpoint，探测 L3/L4
Slot Attention 内部的空间 ownership；不训练、不反向传播、不创建 optimizer、不修改
checkpoint，也不增加模型参数。

## Slot attention 的真实归一化

实际实现位于 `chuagnxindian/1mufasaslot/multi_layer_slot_attention.py` 的
`VisualSlotBranch`：

1. 最后一次迭代的 logits 为
   `einsum(query, key) * 512**-0.5`，实际 shape 为 `[B, 2, 49]`。
2. `softmax(logits, dim=1)` 在 slot 维竞争，因此每个 visual token 对两个 slot 的
   概率和为 1。该张量表示 token ownership，本实验取 target `slot0`。
3. 原实现随后加 `eps`，再除以沿 49 个 token 的总和。这一步只为每个 slot 的
   weighted update 构造聚合权重；它不再表示每个 token 属于哪个 slot。

因此正式 probe 定义为：

```text
SLOT_L3 = softmax(L3 final logits, slot_dim)[:, 0] -> [B,1,7,7]
SLOT_L4 = softmax(L4 final logits, slot_dim)[:, 0] -> [B,1,7,7]
```

L3 在进入 Slot Attention 前已被训练代码 pool 到 7×7，所以这里不会伪造 native
14×14 slot mask。

## 评测口径

脚本首先调用根目录当前正式 evaluator，严格复现 `AUD_FINE / IMG_QUERY / IQR /
OBJ_PRIOR / OGL / EXTRA_IQR_OGL`。任何数值不一致都会立即停止，之后不会解释 slot
结果。

Slot map 与其他 map 一样：先 bicubic resize 到 224×224、逐图 min-max normalize，
再按固定权重融合并再次 min-max normalize。固定结果为：

```text
AUD_SLOT_L3 = normalize(0.6 * AUD_FINE + 0.4 * SLOT_L3)
AUD_SLOT_L4 = normalize(0.6 * AUD_FINE + 0.4 * SLOT_L4)
OGL         = normalize(0.6 * AUD_FINE + 0.4 * OBJ_PRIOR)
```

透明诊断会完整输出 `alpha={0.5,0.6,0.7,0.8,0.9}`，不会隐藏其他组合。成功标准沿用
当前 evaluator：单样本 IoU >= 0.5。

Complementarity 在 resize、normalize 后的 224×224 map 上计算 Pearson、Spearman 和
JS divergence；同时保留真实 ImageNet ResNet18 `OBJ_PRIOR` 作为只读诊断参考。

## 运行

从仓库根目录用两张卡同时运行正式两个 144k checkpoint：

```bash
bash chuagnxindian/2.0_slot_objectness_probe/run_144k_two_gpus.sh
```

也可单独运行：

```bash
python chuagnxindian/2.0_slot_objectness_probe/probe.py \
  --experiment vggss_144k --gpu 0
python chuagnxindian/2.0_slot_objectness_probe/probe.py \
  --experiment flickr_144k --gpu 1
```

## 输出

默认结果位于 `results/`：

- `combined_metrics.csv`：两数据集正式 alpha=0.6 表；
- `combined_alpha_sweep.csv`：所有透明 alpha 组合；
- `combined_rescue_hurt.csv`：rescue、hurt、overlap、only-slot、only-OGL；
- `combined_map_complementarity.csv`：Pearson、Spearman、JS；
- `<dataset>/per_sample_metrics.csv`：每个样本全部 IoU；
- `<dataset>/slot_attention_normalization_audit.json`：两阶段 attention 归一化审计；
- `<dataset>/zero_training_audit.json`：checkpoint 哈希/时间戳和零训练声明；
- `<dataset>/qualitative/`：固定规则选择的成功、失败、rescue、hurt 样本面板；
- 数值图同时保存 PNG 和 PDF。

定性样本不是人工挑选：按测试集原始顺序，在预先定义的 success/rescue/hurt/failure
类别中 round-robin 取最早样本，再用测试集最早样本补足；选择规则和 ID 写入 manifest。
