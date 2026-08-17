# Experiment A：No-Slot AV Baseline

## 定义依据

JSA 官方仓库没有单独发布 no-slot 模型文件；当前本地仓库已有的
`model_baseline.py` 实现了 conventional audio-visual MIL。JSA 论文将传统 SSL
描述为：聚合所有局部音频特征得到 global audio representation，并在视觉 feature map
中选择与之最相关的 local visual representation 做对比学习。

本目录在不修改根目录代码的前提下复用该最小定义。论文：
<https://arxiv.org/abs/2504.15118>。

## 模型与训练

```text
image -> ImageNet-pretrained ResNet18 -> L4 [B,512,7,7]
audio -> audio ResNet18 -> [B,512,T] -> global max pool -> [B,512]
```

所有 local visual feature 和 global audio feature 沿 channel 做 L2 normalization。
训练时计算 batch 内全部 image-audio pair 的 7x7 correspondence，并对每一对取 spatial
maximum，得到 `[B,B]` logits，执行双向 InfoNCE。

reciprocal-k 自然复用：每个样本用其配对 global audio 选出的 local visual feature 作为
image representation，以 global audio 作为 audio representation，再使用与 JSA 相同的
`k=20` reciprocal false-negative filtering。这里没有构造伪 slots；长度为 1 的维度仅用于
兼容现有工具函数的输入接口。

模型完全不包含 Slot Attention、learnable slots、GRU、decoder、divergence、masking、
attention matching 或 M-Fusion。训练接口返回的后三个零标量仅用于复用根目录训练循环，
配置中的 `lam1/lam2/lam3` 均为 0，这些 loss 在实验语义上为 N/A。

## Evaluation 与 checkpoint

复用相同的 7x7→224 bicubic interpolation、map normalization、threshold=0.6、GT、
`utils.Evaluator` 和 object-prior ResNet：

- `AUD`：global audio 与 L4 local feature 的 correspondence；
- `OBJ_PRIOR`：固定外部 ImageNet ResNet18；
- `OGL = 0.6 * AUD + 0.4 * OBJ_PRIOR`；
- `IMG_QUERY / IQR / EXTRA_IQR_OGL`：N/A。

由于没有 IQR，best checkpoint 按 `AUD cIoU` 保存。

## 命令

```bash
bash chuagnxindian/baseline/train_vggss.sh 10k 0 b0_baseline_vggss_10k
bash chuagnxindian/baseline/train_flickr.sh 10k 1 b0_baseline_flickr_10k_frame8_center5

bash chuagnxindian/baseline/test_vggss.sh b0_baseline_vggss_10k 0
bash chuagnxindian/baseline/test_flickr.sh b0_baseline_flickr_10k_frame8_center5 1
```
