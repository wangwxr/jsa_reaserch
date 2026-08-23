# Experiment 5.4 结果：Value-Aware Query Readout

## 结论摘要

本实验属于 **Case C**：直接使用 `ATT_FINE @ V34` 构造的 `Q_VALUE` 会严重
破坏 semantic slot identity；加入产生 Q4 的原 slot state 作为 residual 后，定位
几乎完全恢复为 5.3 的 `Q4 -> K34`，但没有产生稳定的新信息。因此不建议继续做
learnable query refinement。

## 完整指标

| Method | VGGSS-144k cIoU | VGGSS AUC | Flickr-144k cIoU | Flickr AUC |
|---|---:|---:|---:|---:|
| AUD_FINE | 0.426910 | 0.422955 | 0.812000 | 0.635600 |
| IMG_L4 | 0.406941 | 0.416615 | 0.804000 | 0.616600 |
| IMG_FINE | 0.423032 | 0.425989 | 0.816000 | 0.640800 |
| IMG_VALUE | 0.116518 | 0.219698 | 0.176000 | 0.271800 |
| IMG_VALUE_RES | 0.424583 | 0.425989 | 0.816000 | 0.640600 |
| IQR_FINE | 0.430399 | 0.426086 | 0.804000 | 0.639400 |
| IQR_VALUE | 0.427297 | 0.423149 | 0.808000 | 0.635600 |
| IQR_VALUE_RES | 0.430593 | 0.425688 | 0.808000 | 0.639600 |

## Delta

| Comparison | VGGSS ΔcIoU / ΔAUC | Flickr ΔcIoU / ΔAUC |
|---|---:|---:|
| IMG_VALUE − IMG_FINE | -0.306514 / -0.206291 | -0.640000 / -0.369000 |
| IMG_VALUE_RES − IMG_FINE | +0.001551 / 0.000000 | 0.000000 / -0.000200 |
| IQR_VALUE − IQR_FINE | -0.003102 / -0.002937 | +0.004000 / -0.003800 |
| IQR_VALUE_RES − IQR_FINE | +0.000194 / -0.000397 | +0.004000 / +0.000200 |
| IQR_VALUE − AUD_FINE | +0.000388 / +0.000194 | -0.004000 / 0.000000 |
| IQR_VALUE_RES − AUD_FINE | +0.003684 / +0.002734 | -0.004000 / +0.004000 |

## Redundancy / identity

| Statistic (mean ± std) | VGGSoundSS | Flickr |
|---|---:|---:|
| `||Q4||` | 14.8396 ± 1.1095 | 15.7031 ± 1.7270 |
| `||Q_VALUE||` | 12.8269 ± 0.2748 | 12.9553 ± 0.2873 |
| `cos(Q4,Q_VALUE)` | -0.0191 ± 0.0482 | -0.0310 ± 0.0618 |
| `cos(Q4,Q_VALUE_RES)` | 0.8434 ± 0.0305 | 0.8667 ± 0.0357 |
| `cos(U,S4_QUERY_INPUT)` | -0.0066 ± 0.0328 | -0.0006 ± 0.0407 |
| IMG_FINE/IMG_VALUE Pearson | 0.2335 ± 0.5063 | -0.0235 ± 0.4971 |
| IMG_FINE/IMG_VALUE cosine | 0.8684 ± 0.1019 | 0.8619 ± 0.0942 |
| IMG_FINE/IMG_VALUE MAE | 0.2467 ± 0.0968 | 0.2836 ± 0.1017 |
| IMG_FINE/IMG_VALUE_RES Pearson | 0.999811 ± 0.000490 | 0.999727 ± 0.000409 |
| IMG_FINE/IMG_VALUE_RES cosine | 0.999966 ± 0.000076 | 0.999967 ± 0.000037 |
| IMG_FINE/IMG_VALUE_RES MAE | 0.004098 ± 0.003298 | 0.004704 ± 0.002798 |

Q4 与 Q_VALUE 的范数处在同一数量级，但方向近乎正交；直接 readout 的失败不是简单
的 norm mismatch。Residual 使 query 和 map 回到原语义方向，并导致输出几乎与
IMG_FINE 重合。

`UPDATE_ATTENTION` 与 raw slot-competition 再做 token L1 normalization 的全测试集
最大绝对误差为 0，二者数学和数值上完全相同，因此没有伪造为两个不同实验结果。

## 逐样本变化

Improved/Hurt 使用 ±0.01；positive/negative mean 使用所有大于/小于 0 的 delta。

### VGGSoundSS-144k

| Comparison | Improved | Hurt | Unchanged | Mean + | Mean - | Total net IoU |
|---|---:|---:|---:|---:|---:|---:|
| IMG_VALUE vs IMG_FINE | 604 | 4272 | 282 | 0.068622 | -0.257918 | -1087.4284 |
| IMG_VALUE_RES vs IMG_FINE | 172 | 106 | 4880 | 0.003725 | -0.002672 | +0.0532 |
| IQR_VALUE vs IQR_FINE | 853 | 1423 | 2882 | 0.013577 | -0.013475 | -14.2649 |
| IQR_VALUE_RES vs IQR_FINE | 70 | 67 | 5021 | 0.002485 | -0.002202 | -1.7254 |
| IQR_VALUE vs AUD_FINE | 0 | 1 | 5157 | 0.000608 | -0.000657 | -0.0147 |
| IQR_VALUE_RES vs AUD_FINE | 1246 | 751 | 3161 | 0.011637 | -0.011448 | +12.5247 |

### Flickr-144k

| Comparison | Improved | Hurt | Unchanged | Mean + | Mean - | Total net IoU |
|---|---:|---:|---:|---:|---:|---:|
| IMG_VALUE vs IMG_FINE | 10 | 236 | 4 | 0.074157 | -0.394457 | -92.9910 |
| IMG_VALUE_RES vs IMG_FINE | 7 | 9 | 234 | 0.003299 | -0.003347 | -0.0691 |
| IQR_VALUE vs IQR_FINE | 49 | 73 | 128 | 0.011293 | -0.013954 | -0.9639 |
| IQR_VALUE_RES vs IQR_FINE | 0 | 3 | 247 | 0.001834 | -0.002408 | -0.1269 |
| IQR_VALUE vs AUD_FINE | 0 | 0 | 250 | 0.000882 | -0.000979 | -0.0791 |
| IQR_VALUE_RES vs AUD_FINE | 59 | 39 | 152 | 0.011870 | -0.009173 | +0.7579 |

## Reproduction 和只读审计

- 5.3 的 AUD_FINE、IMG_L4、IMG_FINE、IQR_OLD、IQR_FINE：汇总指标误差、
  逐样本 IoU 最大误差、raw/final map 最大误差全部为 0。
- `K34` 捕获值与冻结路径重算值误差为 0；Q4 从最后一次
  `S4_QUERY_INPUT` 重构误差为 0。
- optimizer 未创建，backward 未调用，新增可训练参数为 0；所有模型参数
  `requires_grad=False`，所有 `.grad=None`。
- K/V/query 路径的所有 LayerNorm/Linear 参数均确认冻结。
- NaN/Inf 为 0。
- 正式 G 与 Stage-1 checkpoint 的 SHA256、mtime、size 在运行前后完全一致。

## 判断

- 不是 Case A：IMG_VALUE 没有稳定提升，而是严重下降。
- 不是原定义的 Case B：`cos(Q4,Q_VALUE)` 不高；直接 ATT@V 不是简单复制，而是
  丢失了原 semantic identity。
- 明确属于 Case C：直接 value query 失败，semantic residual 恢复结果。
- Residual 恢复后又几乎完全等价于 IMG_FINE，因此没有足够证据支持学习式 query
  refinement。
- Flickr 的 residual IQR 仍低于 AUD_FINE，进一步支持 sample-dependent
  compatibility/fusion reliability 才是剩余瓶颈，而不是继续增强 visual readout。
