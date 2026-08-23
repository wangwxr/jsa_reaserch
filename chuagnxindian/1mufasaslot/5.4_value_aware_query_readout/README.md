# Experiment 5.4：Value-Aware Query Readout

这是一个严格 zero-training/read-only control。正式模型、G spatial student、
`img_norm_input/img_to_k/img_to_v/img_norm_slots/img_to_q` 全部冻结。

## Forward

```text
fine_norm = frozen_l4.img_norm_input(flatten(F34))
K34       = frozen_l4.img_to_k(fine_norm)
V34       = frozen_l4.img_to_v(fine_norm)

ATT_FINE  = teacher.slot_attn._attention(Q4, K34, infer_sharpening)
U         = ATT_FINE @ V34
Q_VALUE   = frozen_l4.img_to_q(frozen_l4.img_norm_slots(U))
IMG_VALUE = Attention(Q_VALUE, K34)[slot0]
```

最终 `Q4` 对应的 slot feature 通过 `img_norm_slots` 的最后一次 forward-pre-hook
无歧义取得，记作 `S4_QUERY_INPUT`。Residual control 为：

```text
Q_VALUE_RES   = img_to_q(img_norm_slots(S4_QUERY_INPUT + U))
IMG_VALUE_RES = Attention(Q_VALUE_RES, K34)[slot0]
```

IQR 固定使用 5.3 的原生 14×14、`0.6 AUD + 0.4 IMG` 定义，不调 alpha。

## 运行

从仓库根目录执行：

```bash
bash chuagnxindian/1mufasaslot/5.4_value_aware_query_readout/run_formal_two_gpus.sh
```

GPU 0 运行 VGGSoundSS-144k，GPU 1 运行 Flickr-144k。结果写入本目录
`results/`，不会写入 checkpoint。
