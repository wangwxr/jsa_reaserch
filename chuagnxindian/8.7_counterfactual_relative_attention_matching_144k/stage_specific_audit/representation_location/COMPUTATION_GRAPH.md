# CRAM 作用位置与梯度通路

Stage-1 源码：`8.7/objective.py` 委托 `8.6/objective.py`。
Stage-2 源码：`stage2_with_cram/model.py` 继承原 `1.3G-multigeom_equivariant_l3_refine/model.py`。
实际 tensor shape 见 `../gradient_conflict/{dataset}/tensor_shapes.json`。

```text
Stage-1 (coarse; model parameters trainable)
image -> visual backbone / L3,L4 -> K4 [B,49,512] -----------+
image -> visual slot branch -> Qv [B,2,512] -> A(Qv,K4) -> detach target [B,49]
audio+ / audio- -> audio backbone + slot branch -> Qa [B,2,512]
                                         |               |
                                         +-> A(Qa,K4) <---+
                                               |
                                     [CRAM: slot0 [B,49]]
                                     Dpos vs mean(Dneg), softplus
                                               |
                              gradients to audio queries AND visual keys
Other simultaneous anchors: InfoNCE + recon + div + 100*(A2V + V2A match)
Inference: A(Qa,K4; sharpening=.1), slot0 -> 7x7 coarse localization
```

```text
Stage-2 (fine; teacher frozen)
frozen C3 image -> L4 projected feature -> upsample -----------+
frozen C3 image -> L3 native -> trainable proj3 -> adapter ----+-> F34 [B,512,14,14]
                                                                  |
                                   frozen LayerNorm + img_to_k <----+
                                               |
                                          K34 [B,196,512]
                                  /             |                \
frozen Qv ------------------> visual target      |                 |
                              detach            |                 |
frozen Qa+ ---------------------------------> A(Qa+,K34)      AUD_FINE [B,1,14,14]
frozen Qa- ---------------------------------> A(Qa-,K34)            |
                                              |                    +-> coarse KL + two-view equivariance
                                    [CRAM: slot0 [B,196]]
                                              |
                            gradients ONLY to existing proj3/adapter through K34
```

注意：上图中的 CRAM matched attention 与 AUD_FINE 是从共同 K34 分出的
两次计算，CRAM 没有通过 `AUD_FINE` 这个 Python tensor 反传。测 AUD_FINE 的
CRAM 梯度会返回 unused；这不能解释为 CRAM 不影响定位。正确共同祖先为 K34、F34。
base 同时使用 view A/B，CRAM 只使用 view A；component audit 单独检查了两路参数梯度。

| 属性 | Stage-1 C3 | 当前 Stage-2 + CRAM |
|---|---|---|
| query/key | Qa/Qv [B,2,512]; K4 [B,49,512] | 冻结 Qa/Qv [B,2,512]; K34 [B,196,512] |
| attention | [B,2,49]，slot0 [B,49] | [B,2,196]，slot0 [B,196] |
| attention 计算 | dot / sqrt(512)，先 slot-softmax 再空间归一化 | 同一形式，但 dot 乘 infer_sharpening=.1 |
| 训练倍率 | 1.0 | .1 |
| CRAM 权重 | 1.0 | 本次诊断 25,000 |
| target | stopgrad visual attention on current K4 | stopgrad visual attention on current K34 |
| 正锚定 | 全部原始 L_match，系数 100 | 原 coarse KL + equivariance；没有额外完整 L_match |
| CRAM 能调整 | 音频/视觉表示、slot 与 key 路径 | 仅 refinement student；音频与 Teacher 全冻结 |
| 下游 | coarse output，随后交给独立 Stage-2 | 直接产生最终 AUD_FINE |

`stopgrad` 只切断当次 target 的梯度，target 数值仍随下一次 K34 更新而变化。
因此此 target 不是跨 step 固定的语义锚点。

可以支持的解释是：S1 有多种可学习的跨模态适应路径与完整正锚定；S2 的追加
relative loss 将压力集中到有限的细化参数，并与 coarse 保真、几何一致性争夺
同一参数更新方向。不能把两种设置视为只改变“训练阶段”的完全受控消融：
冻结状态、分辨率、attention 倍率、原有监督、负样本采样方式均有差别。

所有已检查的可训练参数组均出现晚期冲突，没有找到一个已经证实无冲突的可训练层。
将 CRAM 简单移动到完全冻结的 coarse 输入会没有梯度；为它解冻 Teacher 又会改变
当前两阶段配方。因此本次没有据此实现新位置或新训练。
