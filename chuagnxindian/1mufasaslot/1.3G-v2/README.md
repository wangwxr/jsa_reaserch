# 1.3G-v2：Visual-Slot Semantic Preservation

本实验严格从正式 L3+L4 Stage-1 checkpoint 初始化，并复用 1.3G 的 frozen teacher、`proj3_spatial`、Top-down adapter、几何增强、AUD coarse anchor 与 AUD equivariance。已有实验目录和根目录 JSA 不会被修改。

## 唯一变化

冻结 teacher 对每个训练视图提供：

```text
Qa [B,2,512] + K4 [B,49,512]   -> AUD_L4  [B,1,7,7]
Q4 [B,2,512] + K4 [B,49,512]   -> IMG_L4  [B,1,7,7]

F34 [B,512,14,14] -> K34 [B,196,512]
Qa + K34                         -> AUD_FINE [B,1,14,14]
Q4 + K34                         -> IMG_FINE [B,1,14,14]
```

`Q4` 是冻结 L4 visual Slot Attention 最后一轮 query；没有新 query、Q/K/V、head 或其他可训练参数。新增损失只比较每个视图自己的视觉 coarse teacher 与 fine student：

```text
L_img_coarse = 1/2 * [
    KL(normalize(IMG_L4_A.detach), normalize(sum_pool_2x2(IMG_FINE_A))) +
    KL(normalize(IMG_L4_B.detach), normalize(sum_pool_2x2(IMG_FINE_B)))
]

L_total = L_aud_coarse + L_aud_equiv + L_img_coarse
```

`sum_pool_2x2` 完全复用 1.3G 的 probability-mass-preserving pooling。没有 IMG equivariance。正式 best 仍只按 `AUD_FINE cIoU` 保存。

## 参数与训练设置

- trainable：`proj3_spatial` + `TopDownL3Adapter`，共 1,443,072。
- frozen：完整 L3+L4 teacher、Qa/Q4/K4 及原始 q/k/v。
- AdamW，lr `5e-5`，weight decay `0.01`，batch size `256`，无 scheduler。
- 10k：100 epochs；144k：50 epochs。
- workers：VGGSoundSS 16；Flickr 12。
- seed、数据、augmentation 与 evaluator 均沿用正式 1.3G。

每次正式训练启动前都会执行完整 teacher reproduction、`lambda_img=0` 回归和梯度审计。每 20 iteration 记录 AUD/IMG 梯度范数、比例、cosine 和负 cosine 比例，但不会改变 `.grad`。

## 命令

只跑 10k 双卡：

```bash
bash chuagnxindian/1mufasaslot/1.3G-v2/run_10k_two_gpus.sh
```

只跑 144k 双卡：

```bash
bash chuagnxindian/1mufasaslot/1.3G-v2/run_144k_two_gpus.sh
```

按两条互不等待的流水线完整执行：GPU0 运行 `VGG10k → VGG门检 → VGG144k`，GPU1 运行 `Flickr10k → Flickr门检 → Flickr144k`；最后汇总：

```bash
bash chuagnxindian/1mufasaslot/1.3G-v2/run_all_staged_two_gpus.sh
```

门检只拦截 NaN/Inf、冻结违规或相对原 G 下降超过绝对 0.05 cIoU 的明显灾难；它不要求 10k 必须提升，也不以 IMG/IQR 选模。
