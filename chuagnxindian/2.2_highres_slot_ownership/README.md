# 2.2 High-Resolution Internal Slot Ownership Probe

本实验完全零训练。它不是对 7×7 Slot map 做插值，而是复用正式 1.3G 的：

```text
Q4  = L4 Visual Slot Attention final query       [B,2,512]
K34 = img_to_k(img_norm_input(F34 tokens))       [B,196,512]
ownership14 = softmax(Q4 @ K34^T * scale, dim=slot)
```

正式 evaluation 前会用同一个 `Q4` 和原始 `K4` 重构 7×7 ownership，并与 2.0 tensor
逐元素比较；不一致则停止。所有 alpha、Rescue/Hurt、oracle 和 reliability 都完整保存，
OGL/GT 只用于评测与离线分析。

运行：

```bash
bash chuagnxindian/2.2_highres_slot_ownership/run_144k_two_gpus.sh
```
