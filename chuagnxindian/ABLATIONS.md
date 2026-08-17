# MUFASA-JSA v1.1 严格消融实验

三个实验均不修改根目录 JSA baseline 或现有 v1.1，不修改数据、augmentation、optimizer、
LR、epoch、batch size、seed 或 evaluation protocol。

| 实验 | architecture | best checkpoint |
|---|---|---|
| A No-Slot | `b0_baseline` | AUD cIoU |
| B L4×3 | `mufasa_ablation1_l4x3_control` | L4-only IQR cIoU |
| C L3+L4 | `mufasa_ablation2_l3_l4_ablation` | L4-only IQR cIoU |

## 一条命令运行 12 组

```bash
cd /data/wxr/audio_video/JSA
bash chuagnxindian/run_all_ablations_two_gpus.sh
```

调度顺序如下。每一行的 VGGSS 使用 GPU 0、Flickr 使用 GPU 1 并行；一行成功结束后才进入
下一行：

```text
10k:  A VGGSS + A Flickr
10k:  B VGGSS + B Flickr
10k:  C VGGSS + C Flickr
144k: A VGGSS + A Flickr
144k: B VGGSS + B Flickr
144k: C VGGSS + C Flickr
```

总计 `3 experiments × 2 datasets × 2 scales = 12` 个实验。默认继续使用：

```text
JSA_PYTHON=/home/wxr/miniconda3/envs/wwww/bin/python
VGG_PREP_ROOT=/home/wxr/datasets/JSA/VGGSound_144k_npy
FLICKR_PREP_ROOT=/home/wxr/datasets/JSA/FlickrSoundNet_144k_frame8_center5_npy
```

每个直接训练脚本的参数仍是：

```text
{10k|144k} [GPU_ID] [EXPERIMENT_NAME]
```

每个直接测试脚本的参数仍是：

```text
EXPERIMENT_NAME [GPU_ID] [ALPHA] [CHECKPOINT]
```
