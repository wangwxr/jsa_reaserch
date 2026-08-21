# Experiment 2.3 — Semantic-Spatial Decoupled Slot Learning

本目录只新增一个可训练的 `SpatialSlotAttention`。正式 1.3G-144k best checkpoint
整体冻结，`F34`、L4 semantic slots、AUD_FINE 和旧 HR14 ownership 都来自冻结模型。

## 数据流

```text
Frozen 1.3G F34 [B,512,14,14] -> tokens [B,196,512]
Frozen L4 semantic S4 [B,2,512] -> stop-gradient initialization
                         |
                         v
NEW SpatialSlotAttention (independent norm/q/k/v/GRU/MLP, 5 iterations)
                         |
                         v
ownership = softmax(final logits, dim=slot) [B,2,196]
SPATIAL_SLOT0 [B,1,14,14]
```

`slot0` 固定为 target，不进行 dynamic selection 或 Hungarian matching。

## Loss

```text
L = 1.0 L_seed + 1.0 L_equiv + 0.1 L_visual + 0.1 L_mass
```

- seed：只在 frozen AUD_FINE 的 top10% token 上约束 slot0。
- equiv：crop/resize/flip 后完整两槽 categorical ownership 的 valid-region symmetric KL。
- visual：冻结 F34 上的 soft prototype reconstruction cosine loss。
- mass：只匹配旧 HR14 每个 slot 的总体质量，不匹配任何 pixel location。

## 固定训练配置

- VGGSoundSS-144k workers=16；Flickr-144k workers=12。
- batch size 沿用各自 1.3G/base config。
- AdamW，lr=5e-5，weight decay=0.01，无 scheduler。
- epochs=50，两张卡并行。
- best checkpoint：`AUD_SPATIAL` 在 alpha=0.6 下的 cIoU。

## 一条命令：smoke 后自动正式训练

```bash
bash chuagnxindian/2.3_semantic_spatial_decoupled_slot/run_144k_two_gpus.sh
```

只运行 smoke：

```bash
bash chuagnxindian/2.3_semantic_spatial_decoupled_slot/smoke_144k_two_gpus.sh
```

正式输出：

```text
checkpoints/2.3_semantic_spatial_decoupled_slot_vggss_144k/
checkpoints/2.3_semantic_spatial_decoupled_slot_flickr_144k_frame8_center5/
```

曲线同时保存 PNG/PDF；最终 results 包含全部 alpha、Rescue/Hurt、oracle、OGL 和
固定 test-order qualitative panels。GT/OGL/OBJ_PRIOR 从不进入训练或选模。

