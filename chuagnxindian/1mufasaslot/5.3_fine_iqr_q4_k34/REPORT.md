# Experiment 5.3 结果：Fine IQR（Q4 → K34）

本实验完全零训练，仅使用正式 1.3G-144k best checkpoint 做只读评测。

## 完整指标

| Method | VGGSoundSS-144k cIoU | VGGSoundSS-144k AUC | Flickr-144k cIoU | Flickr-144k AUC |
|---|---:|---:|---:|---:|
| AUD_FINE | 0.426910 | 0.422955 | 0.812000 | 0.635600 |
| IMG_L4 | 0.406941 | 0.416615 | 0.804000 | 0.616600 |
| IMG_FINE | 0.423032 | 0.425989 | 0.816000 | 0.640800 |
| IQR_OLD | 0.423032 | 0.423420 | 0.804000 | 0.636600 |
| IQR_FINE | 0.430399 | 0.426086 | 0.804000 | 0.639400 |
| IQR_FINE_EVALSPACE | 0.431175 | 0.425931 | 0.800000 | 0.639000 |

主要 delta：

- VGGSoundSS：IMG_FINE - IMG_L4 = +0.016092 cIoU / +0.009374 AUC；
  IQR_FINE - IQR_OLD = +0.007367 / +0.002666；IQR_FINE - AUD_FINE =
  +0.003490 / +0.003131。
- Flickr：IMG_FINE - IMG_L4 = +0.012000 cIoU / +0.024200 AUC；
  IQR_FINE - IQR_OLD = +0.000000 / +0.002800；IQR_FINE - AUD_FINE =
  -0.008000 / +0.003800。

## 逐样本变化（IoU delta ±0.01）

| Comparison | Dataset | Improved | Hurt | Unchanged |
|---|---|---:|---:|---:|
| IQR_FINE vs IQR_OLD | VGGSoundSS | 1164 | 858 | 3136 |
| IQR_FINE vs AUD_FINE | VGGSoundSS | 1427 | 849 | 2882 |
| IQR_FINE vs IQR_OLD | Flickr | 91 | 81 | 78 |
| IQR_FINE vs AUD_FINE | Flickr | 72 | 49 | 129 |

## Reproduction / read-only audit

- AUD_FINE、IMG_L4、IQR_OLD 的正式指标误差均为 0。
- 所有 5158 个 VGGSoundSS 样本和 250 个 Flickr 样本的 AUD_FINE raw/final、
  IMG_L4 raw/final、IQR_OLD final reconstruction 最大绝对误差均为 0。
- 两次正式 forward 捕获的 Q4/K4 最大绝对误差为 0。
- optimizer 未创建，backward 未调用，新增可训练参数为 0；评测时模型可训练参数为
  0，所有参数 `.grad` 均为 `None`。
- NaN/Inf 数量为 0。
- 正式 G checkpoint 与 Stage-1 base checkpoint 的 SHA256、mtime、文件大小在
  评测前后完全一致。

## 结论

K34 明确改善了 Q4 视觉分支本身，说明 7×7 K4 确实限制了 IMG_QUERY。
但该改善只在 VGGSoundSS 上形成小幅 IQR 正增益，在 Flickr 上没有提高 IQR cIoU，
且仍低于 AUD_FINE。因此 IQR 的主要剩余瓶颈不是单纯空间分辨率，还包括视觉分支
与音频分支的逐样本互补性和固定融合的 hurt。

`IQR_FINE` 与 `IQR_FINE_EVALSPACE` 的差异很小：VGGSoundSS 为
+0.000775 cIoU / -0.000155 AUC（eval-space 相对 native），Flickr 为
-0.004000 / -0.000400，不能解释主要结论。

ATT@V → new query → K34 值得作为下一项独立的 zero-training query-readout
诊断，因为 K34 已证明有更好的视觉定位信息；但当前结果不足以支持直接训练或把它
升级为正式方法。下一步应检验 query 适配能否让视觉分支对 AUD 形成稳定互补，而不
只是继续提高单独 IMG 指标。
