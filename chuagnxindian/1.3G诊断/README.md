# 1.3G 输出热力图诊断

这是一个严格只读的可视化脚本。它加载正式 1.3G-144k best checkpoint，不训练、不 backward、不创建 optimizer，也不写 checkpoint。

## 展示内容

每个样本固定显示八列：

1. 原图；
2. 真实 GT localization；
3. `AUD`：正式 1.3G `Qa -> K34`，原生 14×14；
4. `IMG_QUERY`：冻结视觉分支 `Q4 -> K4`，原生 7×7；
5. `IQR`：`0.6 AUD + 0.4 IMG_QUERY`；
6. `OBJ_PRIOR`：外部 ImageNet ResNet18，即 OGL 外部分支自身的输出；
7. `OGL`：正式 `0.6 AUD + 0.4 OBJ_PRIOR`；
8. `EXTRA_IQR_OGL`：`0.6 AUD + 0.2 IMG_QUERY + 0.2 OBJ_PRIOR`。

所有 map 都严格沿用正式 evaluator：先 bicubic resize 到 224×224，再逐 map min-max normalize。图中 IoU 使用正式阈值 0.6。

默认从 VGGSoundSS 和 Flickr 正式 test set 各取 10 个等距固定索引，不根据结果挑选。

## 运行

```bash
bash /data/wxr/audio_video/JSA/chuagnxindian/1.3G诊断/run.sh 0 10
```

参数依次为 GPU ID 和每个数据集的样本数。

输出位于：

```text
outputs/vggss/
outputs/flickr/
```

每套包含：

- 总览 contact sheet；
- 每个样本的 2×4 对比 panel；
- 每个分支的单独叠加图；
- `raw_maps.npz` 原始归一化 map；
- `sample_manifest.csv`，记录样本 ID、标签和六分支 IoU；
- `audit.json`，记录 checkpoint 哈希和只读审计。
