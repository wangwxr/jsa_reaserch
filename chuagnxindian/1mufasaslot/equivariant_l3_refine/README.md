# Experiment F：Frozen Semantic Teacher + Equivariant Fine Spatial Refinement

本目录是独立实验，不修改 JSA 根目录、MUFASA-JSA v1/v1.1、L3+L4 ablation、zero-shot affinity、Frozen/Joint/Decoupled Top-down 实验。

## 固定教师与可训练学生

教师从现有 L3+L4 best checkpoint 严格加载，全部参数 `requires_grad=False`，并始终处于 `eval()`。正式 coarse map 不变：

```text
frozen audio SA target query Qa
        × frozen L4 keys K4
        → AUD_L4 [B,1,7,7]
```

Student 只有两个可训练模块：

1. `proj3_spatial`：`Conv1x1(256→512)`，初始化时逐参数复制 checkpoint 中 frozen teacher `proj3` 的权重和 bias，之后与 teacher 解耦。
2. `TopDownL3Adapter`：`Conv1x1(512→256) → GELU → Conv3x3(256→512)`；末层 3×3 convolution 保留上一版的 zero initialization。

Teacher forward 的 `proj3` pre-hook 读取其输入，即原生 layer3 `[B,256,14,14]`；student 产生：

```text
F3_spatial = proj3_spatial(layer3_native)       [B,512,14,14]
F34 = bilinear_up(F4_projected) + Adapter(F3_spatial)
```

`F34` flatten 后继续经过 frozen L4 visual branch 原有的 `img_norm_input → img_to_k`，没有新增 key/query projection。Frozen `Qa` 查询这些 fine keys，得到 `AUD_FINE [B,1,14,14]`。

## 双视图定义

DataLoader 仍使用原 L3+L4 配置和现有数据增强。对同一次 DataLoader 返回的 view A tensor，逐样本以 `p=0.5` 生成：

```text
view B = horizontal_flip(view A)  或  view A
```

audio 完全相同。因为 B 直接由 A tensor 变换，A/B 没有第二套随机 crop/color 参数，几何对应关系精确。B 的 14×14/7×7 heatmap 按同一个逐样本 mask 水平翻转回 A 坐标。

## Loss

所有 map 都是 spatial probability distribution。实现使用 `eps=1e-8`、float32 KL 和重新归一化：

```text
L_equiv = 0.5 * [KL(AUD_FINE_A || aligned AUD_FINE_B)
               + KL(aligned AUD_FINE_B || AUD_FINE_A)]

Pool(X) = avg_pool2d(X,2,2)*4，再将空间概率和归一化为 1

L_coarse = 0.5 * [KL(detach(AUD_L4_A) || Pool(AUD_FINE_A))
                + KL(detach(aligned AUD_L4_B) || Pool(aligned AUD_FINE_B))]

L_spatial = L_coarse + 1.0 * L_equiv
```

不计算 InfoNCE、reconstruction、divergence、IMG fine matching 或原 base attention loss；不使用 GT localization、OGL、OBJ_PRIOR 或外部模型。

## 训练与评测

Optimizer 严格只有 `proj3_spatial + adapter`：AdamW，lr `5e-5`，weight decay `0.01`，无 scheduler。batch size、epochs、seed、数据、augmentation 和 test protocol 继承对应 L3+L4 config。VGGSoundSS workers=16，Flickr workers=12。

测试只做 single view，不做 two-view averaging。每轮输出相同协议下的：

```text
AUD_L4 teacher cIoU / AUC
AUD_FINE      cIoU / AUC
```

best checkpoint 只按 `AUD_FINE cIoU` 选择。checkpoint 不复制 teacher 权重，只保存 base checkpoint 路径、`proj3_spatial`、adapter、optimizer、epoch 和 metrics。结果目录还包含 `epoch_metrics.csv`、训练曲线、config 和 sanity audit。

## Sanity checks

正式训练前自动：

- 在完整 test split 精确复现 teacher AUD_L4；
- 断言 teacher 全冻结、无 gradient；
- 断言 optimizer/trainable parameter 仅为 student；
- 断言 `proj3_spatial` 初始化与 teacher `proj3` 完全一致；
- 断言 zero-init 时 `F34 == Up(F4)`；
- 断言 F4 hook 与正式 L4 token 来源一致；
- 断言 coarse/fine map 和 sum-pool 后空间概率和为 1；
- 断言 KL finite 且 adapter 获得非零 gradient。

注意：末层 zero-init 导致第一次 backward 时 `proj3_spatial` 和 adapter 第一层的梯度为 0；末层经过第一次 optimizer update 后，梯度自然传播到这些上游参数。这是 zero-init residual 的预期行为。

## 从 repo root 运行

先并行执行完整 baseline/gradient sanity（不会写训练 checkpoint）：

```bash
bash chuagnxindian/1mufasaslot/equivariant_l3_refine/sanity_check_10k.sh
```

两个 10k 在 GPU 0/1 并行正式训练：

```bash
bash chuagnxindian/1mufasaslot/equivariant_l3_refine/run_10k_two_gpus.sh
```

分别运行：

```bash
bash chuagnxindian/1mufasaslot/equivariant_l3_refine/train_vggss_10k.sh 0
bash chuagnxindian/1mufasaslot/equivariant_l3_refine/train_flickr_10k.sh 1
```

可把第二个参数作为自定义 experiment name。若 shell 未激活项目环境，可在命令前设置 `JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python`。

本轮有意没有提供 144k launcher，避免 10k 尚无正信号时误启动 144k。
