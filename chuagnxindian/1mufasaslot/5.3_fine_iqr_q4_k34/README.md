# Experiment 5.3：Fine IQR（Q4 → K34）

这是一个严格的 zero-training / read-only 诊断。它复用正式 Experiment G 的
`AUD_FINE = Qa → K34`，只将 IQR 的视觉分支由 `Q4 → K4` 改为
`IMG_FINE = Q4 → K34`。

## 数学定义

- `AUD_FINE = target(_attention(Qa, K34, 0.1))`，原生 14×14。
- `IMG_L4 = target(_attention(Q4, K4, 0.1))`，原生 7×7。
- `IMG_FINE = target(_attention(Q4, K34, 0.1))`，原生 14×14。
- `IQR_OLD`：各分支先 resize 到 224×224 并 min-max normalize，再以
  `0.6/0.4` 融合并 normalize；与根目录正式 evaluator 完全一致。
- `IQR_FINE`：在原生 14×14 概率图上以 `0.6/0.4` 融合、normalize，随后
  resize 并交给相同 evaluator normalization。
- `IQR_FINE_EVALSPACE`：AUD_FINE 与 IMG_FINE 分别 resize/normalize 到
  224×224 后再融合，是 evaluator-order control。

所有 attention 均调用冻结 teacher 的同一个 `slot_attn._attention()`。

## 运行

从仓库根目录执行：

```bash
bash chuagnxindian/1mufasaslot/5.3_fine_iqr_q4_k34/run_formal_two_gpus.sh
```

GPU 0 运行 VGGSoundSS-144k 正式 checkpoint，GPU 1 运行 Flickr-144k 正式
checkpoint。结果保存在本目录 `results/`，包括完整指标、逐样本 IoU、复现审计、
checkpoint SHA256/mtime 审计和日志。
