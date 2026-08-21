# 1.1.1_14_14_L3

这是 MUFASA-JSA v1.1 的严格单变量消融。正式 v1.1 文件、checkpoint 与结果不修改。

```text
L2 layer2 -> proj2 -> pool 7x7 -> [B,49,512]  -> SA2 -> [B,2,512]
L3 layer3 -> proj3 -> NO POOL  -> [B,196,512] -> SA3 -> [B,2,512]
L4 layer4 -> proj4 -> 7x7      -> [B,49,512]  -> SA4 -> [B,2,512]
                                      |
                       unchanged MFusion of slot representations
```

Audio branch、masking、SA iterations/projections/GRU/MLP/norm/initialization、MFusion、L4-only spatial path、L4 reconstruction target、7×7 decoder、四项 loss、optimizer、数据与 evaluator 均不变。正式训练从与 v1.1 相同的 ImageNet/pretrained initialization 和随机初始化开始，不从 v1.1 best fine-tune。

Ownership probe 使用最后一次 SA iteration 的 query/key：

```text
logits = einsum(query,key) * 512**-0.5
ownership = softmax(logits, dim=1)
SLOT_L3_NATIVE = ownership_L3[:,0] -> [B,1,14,14]
SLOT_L4        = ownership_L4[:,0] -> [B,1,7,7]
```

它不使用随后为 slot update 构造的 token-normalized attention。`evaluate.py` 先调用根目录正式 evaluator，再保存 native/pooled ownership、cIoU/AUC、固定 0.6 融合的 rescue/hurt 与逐样本结果。

运行两个 10k：

```bash
bash chuagnxindian/1mufasaslot/1.1.1_14_14_L3/run_10k_two_gpus.sh
```

自动依次运行 10k 与 144k 的两阶段实验：

```bash
bash chuagnxindian/1mufasaslot/1.1.1_14_14_L3/run_all_two_gpus.sh
```

每个 dataset/GPU 的顺序为：zero-training sanity → 1.1.1 training → formal+ownership evaluation → 1.3G training/evaluation。任一 sanity 失败都会由 `set -e` 阻止后续正式训练。
