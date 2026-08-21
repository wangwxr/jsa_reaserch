# Experiment 2.2：High-Resolution Internal Slot Ownership Probe

本实验完全零训练。它用 L4 visual Slot Attention 的最终 query `Q4`，让两个
L4 slots 在 1.3G 的 196 个 fine tokens `K34` 上重新竞争：

```text
Q4             [B, 2, 512]
F34            [B, 512, 14, 14]
K34            [B, 196, 512]
logits14       einsum(Q4, K34) * l4_branch.scale
ownership14    softmax(logits14, dim=slot)  -> [B, 2, 196]
SLOT_L4_HR14   ownership14[:, 0]            -> [B, 1, 14, 14]
```

这不是对 7×7 ownership 的插值。

## 强制审计

| Audit | VGG | Flickr |
|---|---:|---:|
| Q4 × K4 重构 7×7 ownership 最大误差 | 0.0 | 0.0 |
| local AUD_FINE vs 正式 1.3G 最大误差 | 0.0 | 0.0 |
| 7×7 ownership slot-sum 最大误差 | 1.19e-7 | 1.19e-7 |
| 14×14 ownership slot-sum 最大误差 | 1.19e-7 | 1.19e-7 |
| F4 token 路径误差 | 0.0 | 0.0 |

AUD、OBJ_PRIOR、OGL 与正式 1.3G 结果误差全部为 0。checkpoint 的 SHA256 与
mtime 在运行前后完全一致；没有 optimizer、backward、新参数或 trainable 参数。

## 完整主结果

正式融合固定为 `0.6 * AUD + 0.4 * Slot`。

| Method | VGG cIoU / AUC | Flickr cIoU / AUC |
|---|---:|---:|
| AUD_FINE | **0.4269 / 0.4230** | 0.8120 / 0.6356 |
| SLOT_L4_7 | 0.2144 / 0.3472 | 0.7320 / 0.5952 |
| SLOT_L4_HR14 | 0.2010 / 0.3449 | **0.8120 / 0.6212** |
| AUD + SLOT_L4_7 | 0.3858 / 0.4157 | 0.8360 / 0.6396 |
| AUD + SLOT_L4_HR14 | 0.3777 / 0.4175 | **0.8440 / 0.6502** |
| OBJ_PRIOR | 0.3478 / 0.3924 | 0.4480 / 0.4668 |
| OGL reference | 0.4570 / 0.4401 | **0.8680 / 0.6596** |

相对 AUD，HR14 固定融合在 VGG 为 `-0.0492 cIoU / -0.0055 AUC`，在 Flickr
为 `+0.0320 cIoU / +0.0146 AUC`。相对 7×7 slot fusion，HR14 在 Flickr 又增加
`+0.0080 cIoU / +0.0106 AUC`，但在 VGG cIoU 进一步下降 0.0081。

## Rescue / Hurt / Oracle

| Candidate | Rescue | Hurt | Net | Oracle cIoU / AUC |
|---|---:|---:|---:|---:|
| VGG 7×7 | 355 | 567 | -212 | 0.4957 / 0.4617 |
| VGG HR14 | **430** | 684 | -254 | **0.5103 / 0.4694** |
| Flickr 7×7 | 11 | 5 | +6 | 0.8560 / 0.6614 |
| Flickr HR14 | **12** | **4** | **+8** | **0.8600 / 0.6664** |

VGG 的 HR14 candidate capacity 明显上升：oracle cIoU 从 0.4957 提升到 0.5103，
且超过 OGL 的 0.4570；但新增 75 个 Rescue 的同时新增 117 个 Hurt，所以固定融合
更差。Flickr 则同时增加 Rescue、减少 Hurt，实际融合和 oracle 都改善；HR14 oracle
距 OGL cIoU 只差 0.008。

候选转移的关键计数：

- VGG：339 个 7×7 Rescue 被 HR 保留，16 个丢失，新增 91 个 HR-only Rescue；
  31 个 7×7 Hurt 被 HR 修复，但新增 148 个 HR Hurt。
- Flickr：9 个 7×7 Rescue 被保留，2 个丢失，新增 3 个 HR-only Rescue；1 个
  7×7 Hurt 被 HR 修复，且没有新增 Hurt。

## Reliability：7×7 与 HR14

AUROC 都使用各自 candidate 的 Rescue/Hurt 标签。

| Reliability | VGG 7 / HR | Flickr 7 / HR |
|---|---:|---:|
| ownership confidence | 0.4857 / 0.5073 | 0.7455 / 0.5833 |
| eval seed top10 | **0.5936 / 0.5835** | 0.6909 / **0.7292** |
| eval seed top20 | 0.5931 / 0.5683 | 0.7455 / 0.7292 |
| raw seed top10 | 0.5022 / 0.4819 | **0.8000 / 0.7292** |
| raw seed top20 | 0.5077 / 0.4860 | 0.7818 / 0.7292 |
| centroid distance | 0.5610 / 0.5636 | 0.6909 / 0.5208 |
| JS divergence | 0.5078 / 0.4973 | 0.5091 / 0.5625 |
| extent ratio | 0.4955 / 0.4930 | 0.4364 / 0.5417 |

HR14 没有让 reliability 更容易判断。VGG 最佳仍只有约 0.584；Flickr HR 最佳
约 0.729，但仅含 12 个 Rescue 和 4 个 Hurt，统计不稳定。

## 透明 alpha diagnostic

| AUD alpha | VGG 7 cIoU/AUC | VGG HR cIoU/AUC | Flickr 7 cIoU/AUC | Flickr HR cIoU/AUC |
|---:|---:|---:|---:|---:|
| 0.5 | 0.3548/0.4058 | 0.3519/0.4079 | 0.8280/0.6360 | 0.8440/0.6488 |
| 0.6 | 0.3858/0.4157 | 0.3777/0.4175 | 0.8360/0.6396 | 0.8440/0.6502 |
| 0.7 | 0.4064/0.4224 | 0.4046/0.4245 | 0.8320/0.6422 | 0.8400/0.6480 |
| 0.8 | 0.4184/0.4257 | 0.4188/0.4272 | 0.8200/0.6424 | 0.8240/0.6472 |
| 0.9 | 0.4246/0.4260 | 0.4236/0.4269 | 0.8120/0.6398 | 0.8120/0.6420 |

alpha sweep 仅用于展示趋势，没有据此选择或改写正式结果。VGG 的 cIoU 在所有
固定 alpha 下都没有超过 AUD，说明问题不是简单地把 slot 权重调小就能解决。

## 定性观察与结论

- HR14 的边界/峰值确实与 7×7 插值不同，并能产生 HR-only Rescue，例如 Flickr
  `10549911663`：AUD 0.469、AUD+7 0.478、AUD+HR 0.501，而 OGL 0.453。
- VGG `0RZRFj7zDnQ_000030` 中 HR 把 7×7 fusion 的 0.499 推到 0.511，说明细粒度
  重新竞争可在阈值附近修复 candidate。
- 但 VGG 上常见的是 Slot map对整块高层语义区域扩张，覆盖背景或非声源结构；
  HR14 提高候选多样性，同时也放大了错误 extent，导致 Hurt 增长更快。
- 因此 2.3 不应直接使用手工 reliability hard gate。大规模 VGG 上所有内部单特征
  仍接近随机，Flickr 的较高 AUROC 又只有 16 个有效样本。更稳妥的路线是进入
  **Semantic-Spatial Decoupled Slot Learning**：保留 HR14 ownership 作为有潜力的
  completion candidate，但显式改善“目标语义”和“对象 extent/ownership”的解耦，
  而不是继续依赖固定融合或基于当前特征的门控。本文仅给出路线判断，没有实现或
  训练 2.3。

完整输出：

- `results/combined_method_metrics.csv`
- `results/combined_rescue_hurt_oracle.csv`
- `results/combined_reliability_auroc.csv`
- `results/combined_alpha_sweep.csv`
- `results/*/per_sample_metrics.csv`
- `results/*/qualitative/`
- `results/*/tensor_reconstruction_audit.json`
- `results/*/zero_training_audit.json`

