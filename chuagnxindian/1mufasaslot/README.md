# MUFASA-inspired Multi-Layer JSA v1

本目录实现的实验只回答一个问题：**JSA 是否能从 ResNet 多层 visual slots 融合中受益？**

实现不修改根目录的 JSA baseline，不修改数据集和 evaluation protocol，也不增加 loss、数据增强或 backbone。训练和测试入口只是把模型构造替换为本目录的 `MUFASAJSA`，其余逻辑继续复用根目录的 `train_slot.py` 与 `test_model.py`。

## 1. 原 JSA 数据流

原 JSA 的视觉与音频数据流为：

```text
image -> ResNet18 layer4 + 1x1 projection -> [B,49,512]
audio -> audio ResNet18 + frequency pooling -> [B,T,512]
                     |
                     v
          Joint Slot Attention
                     |
          img_slots <-> aud_slots
```

Slot index 0 是参与跨模态 InfoNCE 和最终定位的 target slot；其他 slot 表示 off-target 内容。训练仍由 `info_loss + lam1*recon_loss + lam2*div_loss + lam3*att_loss` 构成。

## 2. 本版本修改

视觉 ResNet18 一次前向同时提取 layer2、layer3、layer4。每层使用独立的 visual Slot Attention，三层 visual slots 再经过 MUFASA 风格的 M-Fusion。音频编码器和 audio Slot Attention 仍然只运行一次。

```text
layer2 -> proj + pool -> visual SA2 -> S2 --\
layer3 -> proj + pool -> visual SA3 -> S3 ----> M-Fusion -> fused_img_slots
layer4 -> proj        -> visual SA4 -> S4 --/

audio -> original audio encoder -> one audio SA -> aud_slots
```

没有加入 dynamic fusion、native-resolution heatmap、Transformer fusion、新 loss、新增强或新 backbone。

## 3. 本地实际 shape

使用当前数据管线的 `224 x 224` 图像实测结果如下：

| 层 | ResNet 原始输出 | 投影和池化后 | flatten 后 |
|---|---|---|---|
| layer2 | `[B,128,28,28]` | `[B,512,7,7]` | `[B,49,512]` |
| layer3 | `[B,256,14,14]` | `[B,512,7,7]` | `[B,49,512]` |
| layer4 | `[B,512,7,7]` | `[B,512,7,7]` | `[B,49,512]` |

layer2 使用 `Conv1x1(128,512)`，layer3 使用 `Conv1x1(256,512)`，layer4 保留 JSA 的 `Conv1x1(512,512)` 投影。layer2/layer3 使用 `AdaptiveAvgPool2d((7,7))`；layer4 已经是 7x7 时不执行额外池化。

ImageNet ResNet18 权重继续通过 JSA 原来使用的 `resnet18-f37072fd.pth` 加载。`conv1`、BN 和 layer1-layer4 均匹配并加载；新增的三个 projection 按 PyTorch/Kaiming 方式随机初始化。

## 4. 为什么 v1 全部统一到 7x7

统一分辨率后，三个层级都具有 49 个 spatial token，slot 和 attention 可以直接做相邻层融合。这样本实验主要变量是“多层特征 + 多层 slots”，不会同时混入不同热图分辨率、插值策略和 native multi-scale localization 的影响。

## 5. MultiLayerJointSlotAttention

`MultiLayerJointSlotAttention` 包含三套互不共享的 visual 参数：

- `img_to_q / img_to_k / img_to_v`
- `img_gru / img_mlp`
- `img_norm_input / img_norm_slots / img_norm_pre_ff`

audio 参数只有一套：

- `aud_to_q / aud_to_k / aud_to_v`
- `aud_gru / aud_mlp`
- `aud_norm_input / aud_norm_slots / aud_norm_pre_ff`

三套 visual branch 和一套 audio branch 都从同一个 learnable `slots: [1,num_slots,512]` 展开初始化。这是为了维持 JSA 的 slot 0 target、slot 1 off-target 语义。

JSA 原有 masking 被保留：三个 visual level 分别随机采样 patch mask，共用一个 learnable `mask_token_img`；audio 只采样一次 mask 并使用一个 `mask_token_aud`。masking 只在训练模式生效。

## 6. Audio 为什么只运行一次

原 JSA 的 audio slot 更新只依赖 audio keys/values，visual feature 只参与 cross attention map。因此没有必要为三个视觉层复制三套完整 audio Slot Attention。本实现先得到一组 `aud_slots / aud_query / aud_keys`，再让同一个 audio query 分别查询 layer2、layer3、layer4 visual keys，随后融合三张 cross-modal map。

这既符合本实验目标，也避免把 audio 参数量和计算量扩大三倍。

## 7. M-Fusion

对三个 `[B,num_slots,512]` visual slots：

```text
P23 = (S2 + S3) / 2
P34 = (S3 + S4) / 2
X   = concat(P23, P34)                    # [B,num_slots,1024]
Sf  = Linear(1024,1536) -> GELU -> Linear(1536,512)
```

`Sf` 即 `fused_img_slots`。实现位于 `fusion.py::MFusion`，不依赖外部 MUFASA package。

## 8. Attention fusion

同一套 `LearnedAttentionFusion` 参数用于所有 visual-related attention：

- visual query -> visual key，token 数 49；
- audio query -> visual key，token 数 49；
- visual query -> audio key，三层的 audio token 数相同。

对于任意三张 attention `A2/A3/A4`：

```text
A23 = (A2 + A3) / 2
A34 = (A3 + A4) / 2
[w23,w34] = softmax(layer_weights)
Afused = w23*A23 + w34*A34
```

`layer_weights` 初始化为 `[0,0]`，所以训练开始时 `w23=w34=0.5`。audio query -> audio key 保持单分支，不做融合。evaluation 的 `img_attn` 和 `cross_attn` 分别来自 fused visual self attention 与 fused audio-to-visual cross attention。

## 9. 四个 loss 与原 JSA 的对应

没有新 loss，也没有新增 lambda。

1. `info_loss`：用归一化后的 `fused_img_slots <-> aud_slots`，仍只取 slot 0，保留 `tau`、reciprocal-k false-negative filtering 和双向 CE。
2. `recon_loss`：`fused_img_slots` 经原 JSA image MLP decoder 重建投影后的 layer4 `[B,49,512]`；audio slots 经原 audio decoder 重建原 audio tokens。target 的 `detach` 规则不变。
3. `div_loss`：分别在 `fused_img_slots` 和 `aud_slots` 上使用原 cosine/divergence loss。
4. `att_loss`：保持原 MSE 语义：
   - `fused_audq_imgk[:,0]` 对齐 `fused_imgq_imgk[:,0].detach()`；
   - `fused_imgq_audk[:,0]` 对齐 `audq_audk[:,0].detach()`。

训练入口仍由原 `train_slot.py` 组合：

```text
loss = info_loss + lam1*recon_loss + lam2*div_loss + lam3*att_loss
```

## 10. 与原 MUFASA 的区别

原 MUFASA 面向 permutation-invariant object slots，并从 ViT 多层 token 建立多层 Slot Attention。本版本面向 JSA 音视频定位：

- backbone 是当前 JSA 的 ResNet18，而不是 MUFASA 的 ViT encoder；
- 只扩展 visual branch，audio encoder 与 audio Slot Attention 保持单分支；
- 使用 JSA 的 learnable shared slot initialization；
- 保留 JSA 的四项 loss、target slot 规则和 evaluator；
- 只实现指定的 M-Fusion 与 learned attention fusion，不引入 MUFASA 其余训练设计。

## 11. 为什么没有 Hungarian matching

MUFASA 原 slots 是 permutation-invariant，所以跨层对齐可以用 Hungarian matching。JSA 的 slot 0 已被 InfoNCE 和 localization 显式当作 target slot；训练中交换 slot index 可能破坏 target/off-target 的固定语义。

因此 v1 固定 `slot_alignment=none`，三层通过同一组 learnable initial slots 保持语义一致。代码中保留了：

```text
TODO: add Hungarian cross-layer slot alignment ablation
```

## 12. 新增参数

本版本**没有新增命令行超参数**，训练脚本继续使用 baseline 的全部参数。

新增的可训练模型参数是：

- layer2、layer3、layer4 三个 visual projection；
- 三套独立 visual Slot Attention 参数；
- `Linear(1024,1536) -> GELU -> Linear(1536,512)` M-Fusion；
- learned attention fusion 的两个 logits。

`configs.json` 额外记录以下实验身份字段，但它们不是可调训练超参数：

```json
{
  "architecture": "mufasa_jsa_v1",
  "visual_levels": ["layer2", "layer3", "layer4"],
  "slot_alignment": "none"
}
```

## 13. 训练命令

所有命令从 JSA 根目录运行。参数位置与现有 baseline 脚本一致：`split GPU_ID EXPERIMENT_NAME`。

VGGSS 144k：

```bash
VGG_PREP_ROOT=/home/wxr/datasets/JSA/VGGSound_144k_npy \
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python \
bash chuagnxindian/1mufasaslot/train_vggss_mufasa.sh \
144k 0 mufasa_jsa_v1_vggss_144k
```

Flickr 144k，使用当前确定的 frame8-center5 数据：

```bash
FLICKR_PREP_ROOT=/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame8_center5_npy \
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python \
bash chuagnxindian/1mufasaslot/train_flickr_mufasa.sh \
144k 1 mufasa_jsa_v1_flickr_144k_frame8_center5
```

10k 可以复用相同的完整预计算目录，由现有 manifest 严格选出 10k：

```bash
VGG_PREP_ROOT=/home/wxr/datasets/JSA/VGGSound_144k_npy \
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python \
bash chuagnxindian/1mufasaslot/train_vggss_mufasa.sh \
10k 0 mufasa_jsa_v1_vggss_10k

FLICKR_PREP_ROOT=/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame8_center5_npy \
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python \
bash chuagnxindian/1mufasaslot/train_flickr_mufasa.sh \
10k 1 mufasa_jsa_v1_flickr_10k_frame8_center5
```

脚本继续执行当前 VGGSS split/preparation check；Flickr 继续使用当前 dataset loader 和 manifest。checkpoint、日志、`epoch_metrics.csv`、`training_curves.png` 仍保存在：

```text
checkpoints/<EXPERIMENT_NAME>/
checkpoints/<EXPERIMENT_NAME>/logs/
```

best checkpoint 的选择规则不变，仍是每轮测试集上的 IQR cIoU。

## 14. 测试命令

不显式给 checkpoint 时，脚本优先使用 `<testset>_best.pth`，否则使用 `final.pth`。

```bash
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python \
bash chuagnxindian/1mufasaslot/test_vggss_mufasa.sh \
mufasa_jsa_v1_vggss_144k 0

JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python \
bash chuagnxindian/1mufasaslot/test_flickr_mufasa.sh \
mufasa_jsa_v1_flickr_144k_frame8_center5 1
```

显式测试 final checkpoint：

```bash
bash chuagnxindian/1mufasaslot/test_vggss_mufasa.sh \
mufasa_jsa_v1_vggss_144k 0 0.6 final.pth
```

测试继续直接调用原 `test_model.validate_img_aud`，所以 cIoU/AUC、threshold scanning、GT、IQR、object prior、OGL 和额外组合指标均未改变。

### L4-only evaluation

L4-only 模式不修改训练、loss 或 checkpoint。`AUD` 使用 shared audio query 查询 L4 visual keys，`IMG_QUERY` 使用 L4 image query 查询 L4 visual keys；原 evaluator 继续计算 IQR、OBJ_PRIOR、OGL 和 EXTRA_IQR_OGL。

```bash
bash chuagnxindian/1mufasaslot/test_vggss_mufasa_l4.sh \
mufasa_jsa_v1_vggss_144k 0 0.6 vggss_best.pth

bash chuagnxindian/1mufasaslot/test_flickr_mufasa_l4.sh \
mufasa_jsa_v1_flickr_144k_frame8_center5 1 0.6 flickr_best.pth
```

L4-only 日志仍保存在实验自己的 `logs/`，逐样本指标单独保存在 `checkpoints/<EXPERIMENT_NAME>/eval_l4_only/`，不会覆盖原 fused evaluation 文件。

### Fused-query → L4 evaluation

该模式同样只改变 evaluation，不修改训练、loss 或 checkpoint。它先按训练路径得到
`Sf = MFusion(S2, S3, S4)`，再复用 L4 visual branch 的 query head：

```text
Qf = img_to_q_l4(img_norm_slots_l4(Sf))
IMG_FUSED_QUERY = attention(Qf, K4)
AUD = attention(Qaudio, K4)
IQR_NEW = normalize(0.6 * AUD + 0.4 * IMG_FUSED_QUERY)
```

其余 OBJ_PRIOR、OGL、EXTRA_IQR_OGL 与原 evaluator 完全一致。本模式只把原 evaluator
相应的控制台显示名改为 `IMG_FUSED_QUERY` 和 `IQR_NEW`，计算过程没有变化。

```bash
bash chuagnxindian/1mufasaslot/test_vggss_mufasa_fused_query_l4.sh \
mufasa_jsa_v1_vggss_144k 0 0.6 vggss_best.pth

bash chuagnxindian/1mufasaslot/test_flickr_mufasa_fused_query_l4.sh \
mufasa_jsa_v1_flickr_144k_frame8_center5 1 0.6 flickr_best.pth
```

逐样本指标保存在 `checkpoints/<EXPERIMENT_NAME>/eval_fused_query_l4/`，不会覆盖原 fused evaluation 或 L4-only 文件。

### MUFASA-JSA v1.1：L4-only spatial attention

v1.1 与 v1 并存，不修改已有 v1 代码、实验或 checkpoint。它完全保留三个 visual Slot
Attention branches、single audio branch 和 M-Fusion：

- `Sf = MFusion(S2, S3, S4)` 仍用于 InfoNCE、image reconstruction 和 visual divergence；
- audio slot 及 audio reconstruction/divergence 保持不变；
- masked reconstruction、四个 loss 的接口、lambda 和总 loss 组合保持不变；
- `att_loss` 的四张图固定为 `Q4→K4`、`Qa→K4`、`Q4→Ka`、`Qa→Ka`；
- 训练期验证和最终测试中的 `IMG_QUERY=Q4→K4`、`AUD=Qa→K4`；
- learned spatial attention fusion 不再实例化，也不出现在 v1.1 checkpoint 中；
- 每轮验证仍复用根目录 `train_slot.validate()`，因此 best checkpoint 自动按 L4-only IQR cIoU 保存。

两张卡依次训练四组实验：先并行训练两个 10k，二者结束后再并行训练两个 144k：

```bash
bash chuagnxindian/1mufasaslot/run_all_two_gpus_v1_1.sh
```

单独训练：

```bash
bash chuagnxindian/1mufasaslot/train_vggss_mufasa_v1_1.sh \
10k 0 mufasa_jsa_v1_1_vggss_10k

bash chuagnxindian/1mufasaslot/train_flickr_mufasa_v1_1.sh \
10k 1 mufasa_jsa_v1_1_flickr_10k_frame8_center5
```

测试 best checkpoint：

```bash
bash chuagnxindian/1mufasaslot/test_vggss_mufasa_v1_1.sh \
mufasa_jsa_v1_1_vggss_10k 0 0.6 vggss_best.pth

bash chuagnxindian/1mufasaslot/test_flickr_mufasa_v1_1.sh \
mufasa_jsa_v1_1_flickr_10k_frame8_center5 1 0.6 flickr_best.pth
```

### 双卡依次完成四组训练

下面的总控脚本先并行运行 VGGSS-10k（GPU 0）和 Flickr-10k（GPU 1）；两者都成功结束后，再并行运行 VGGSS-144k（GPU 0）和 Flickr-144k（GPU 1）：

```bash
bash chuagnxindian/1mufasaslot/run_all_two_gpus.sh
```

脚本默认使用本机现有的 VGGSound NPY 与 Flickr frame8-center5 NPY，也可以通过 `VGG_PREP_ROOT`、`FLICKR_PREP_ROOT` 和 `JSA_PYTHON` 覆盖。任一阶段有任务失败时，脚本会等待该阶段另一个任务结束并停止，不会继续启动下一阶段。

## 15. 与 baseline 公平比较

除模型结构外，至少保持以下条件完全一致：

- 同一个 train manifest、test manifest、GT 和 prepared root；
- Flickr 使用同一套 frame8-center5 图像/音频；
- `epochs`：10k 为 100、144k 为 50；
- `batch_size=256`、`init_lr=5e-5`、`weight_decay=0.01`；
- `alpha=0.6`、`infer_sharpening=0.1`；
- `lam1=0.1`、`lam2=0.1`、`lam3=100.0`；
- `tau=0.03`、`num_slots=2`、`iters=5`、`reciprocal_k=20`；
- `mask_ratio=0.1`、`hard_img=true`、`hard_aud=true`；
- 同一个 seed、evaluation protocol 和 best-checkpoint 选择规则。

当前代码实测原 JSA 为 31,818,498 个参数，本模型为 41,211,652 个参数，增加 9,393,154 个参数。参数量差异来自新增 visual branches、projection 和 fusion，是本实验结构变量的一部分；数据、loss 和训练超参数仍相同。

## 16. 后续 TODO

- Hungarian alignment ablation；
- native 28/14/7 multi-resolution heatmap；
- audio-adaptive fusion。

以上三项均未进入 v1。
