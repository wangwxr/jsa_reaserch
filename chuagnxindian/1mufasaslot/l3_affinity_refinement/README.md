# L3 Affinity Spatial Refinement（纯 evaluation）

本目录只读取已经训练完成的 `mufasa_ablation2_l3_l4_ablation` 四个 best checkpoint，不训练、不微调、不修改 loss 或 checkpoint，也不加载 OBJ_PRIOR / OGL / 外部 objectness 模型。

## 特征与正式 AUD 路径

- 正式声音 seed 完整复用现有 L3+L4 `forward_eval`：`audio -> shared Audio Slot Attention -> Qa -> L4 keys -> AUD_L4`，输出 `[B,1,7,7]`。
- native L3 通过只读 forward hook 从 `model.imgnet.proj3` 输出端取得。该位置恰好位于 checkpoint 中原有 `Conv1x1(256,512)` 之后、训练所用 `AdaptiveAvgPool2d((7,7))` 之前，实际为 `[B,512,14,14]`。
- pooled control 直接对同一 `proj3` 输出调用模型训练路径使用的 `_pool_to_7x7`，得到 `[B,512,7,7]`。

hook 不改变模型 forward；脚本逐 tensor 检查加载后的 `proj3.weight/bias` 与 checkpoint 完全相等，并在每个输出目录的 `audit.json` 保存检查结果和权重 SHA-256。

## 精确公式

令正式 `AUD_L4` 为 $A_4$。先对其空间维归一化。native 路径使用双线性插值（`align_corners=False`）得到 $A_4^\uparrow$，再做一次空间归一化。对投影后的 L3 token 做 channel-wise L2 normalization：

$$
\hat F_i = \frac{F_i}{\lVert F_i\rVert_2},\qquad
p = \operatorname{norm}\left(\frac{\sum_i A_4^\uparrow(i)\hat F_i}{\sum_i A_4^\uparrow(i)+\epsilon}\right).
$$

$$
A_3(i)=\operatorname{softmax}_i\left(\frac{\hat F_i^T p}{\tau_{aff}}\right),\qquad
A_{ref}=\alpha A_4^\uparrow+(1-\alpha)A_3.
$$

pooled control 使用相同公式，只把 $F_3$ 与 seed 的空间大小都保持为 `7×7`。所有混合结果再次做单位空间质量归一化。

## Evaluation protocol

热图无论 `7×7` 还是 `14×14`，都直接用 bicubic 插值到 `224×224`，随后逐样本 min-max normalize、阈值 `0.6`、同一 GT 与同一 IoU/AUC 计算。没有把 `14×14` 下采样回 `7×7`。

为严格复现仓库现有输出，本地 wrapper 保留了一个历史命名事实：`test_model.py` 控制台标成 `cIoU` 的第一项实际来自 `Evaluator.finalize_AP50()`（样本 IoU ≥ 0.5 的比例）；CSV 的 `cIoU` 继续使用这一值以匹配已有四个参考数，同时额外记录真正的样本 IoU 均值 `mean_sample_cIoU` 便于审计。AUC 公式完全不变。

## 运行

```bash
bash chuagnxindian/1mufasaslot/l3_affinity_refinement/run_all_zero_shot.sh
```

脚本先独立复现四个 AUD baseline；任何一个在四位小数上不一致就终止，不发布 refinement 结果。baseline gate 与正式 sweep 都采用两个独立 GPU 队列：GPU 0 顺序运行两个 VGGSoundSS checkpoint，GPU 1 顺序运行两个 Flickr checkpoint；某张卡完成 10k 后会立即运行同卡 144k，不等待另一张卡。

完整 sweep 为 `tau_aff={0.05,0.1,0.2}` 与 `alpha={0.25,0.5,0.75}`。每个 checkpoint 输出 22 行，四个 checkpoint 合计 88 行，最终汇总到：

- `l3_affinity_refinement_results.csv`：所有参数组合；
- `summary.md`：默认参数表与 diagnostic best；
- `diagnostic_best.csv`：逐数据集、逐方法的 post-hoc best；
- `native_vs_pooled_comparison.csv`：默认与 diagnostic-best 下的 14×14 vs 7×7 差值；
- `ogl_reference_comparison.csv`：只含历史 OGL 数字与来源日志路径，不读取 OGL map；
- `outputs/*/audit.json`：shape、checkpoint 与 baseline 复现审计；
- `outputs/{vggss,flickr}_144k/qualitative/`：固定均匀索引的 10 个样本，PNG 与 PDF；
- `outputs/{vggss,flickr}_144k/qualitative_sample_ids.csv`：固定样本 ID、选择规则及成功/失败的 sample IoU。

定性图由 `figures/gen_fig_qualitative.py` 可复现生成，展示 Image、GT、AUD_L4、A4_up、native L3 affinity、native refined、pooled refined 和 native−pooled。GT 只在 map 已生成后用于指标与可视化，从不进入 prototype/refinement。
