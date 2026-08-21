# Native 14×14 L3 + G 两阶段结果

所有数字均由脚本读取正式 epoch CSV 或 best-checkpoint evaluator JSON；缺失项标记为 N/A。

## VGGSS-10k

| Experiment | AUD cIoU/AUC | IMG_QUERY cIoU/AUC | IQR cIoU/AUC | OGL cIoU/AUC | EXTRA cIoU/AUC |
|---|---:|---:|---:|---:|---:|
| v1.1 baseline | 0.3823/0.3987 | 0.3881/0.4005 | 0.3885/0.4001 | 0.4387/0.4224 | 0.4143/0.4125 |
| 1.1.1_14_14_L3 | 0.3813/0.4002 | 0.3858/0.4006 | 0.3868/0.4013 | 0.4356/0.4254 | 0.4130/0.4151 |
| original 1.3G | 0.4112/0.4141 | 0.4064/0.4094 | 0.4131/0.4134 | 0.4527/0.4351 | 0.4378/0.4261 |
| 1.3G_14_14_L3 | 0.3870/0.4051 | 0.3858/0.4006 | 0.3903/0.4056 | 0.4447/0.4319 | 0.4236/0.4201 |

### Ownership

| Map | cIoU | AUC | Rescue | Hurt | Net |
|---|---:|---:|---:|---:|---:|
| SLOT_L3_POOLED_BASELINE | 0.0706 | 0.2437 | 97 | 653 | -556 |
| SLOT_L3_NATIVE | 0.0450 | 0.2399 | 123 | 458 | -335 |
| SLOT_L4 | 0.3088 | 0.3806 | 180 | 217 | -37 |

## Flickr-10k

| Experiment | AUD cIoU/AUC | IMG_QUERY cIoU/AUC | IQR cIoU/AUC | OGL cIoU/AUC | EXTRA cIoU/AUC |
|---|---:|---:|---:|---:|---:|
| v1.1 baseline | 0.7520/0.5950 | 0.7440/0.5928 | 0.7600/0.5956 | 0.8240/0.6208 | 0.8000/0.6134 |
| 1.1.1_14_14_L3 | 0.7200/0.5786 | 0.7000/0.5652 | 0.7240/0.5746 | 0.8000/0.6148 | 0.7640/0.6048 |
| original 1.3G | 0.8040/0.6164 | 0.7520/0.5890 | 0.7840/0.6074 | 0.8520/0.6340 | 0.8160/0.6260 |
| 1.3G_14_14_L3 | 0.7600/0.6012 | 0.7000/0.5652 | 0.7400/0.5890 | 0.8280/0.6330 | 0.7720/0.6174 |

### Ownership

| Map | cIoU | AUC | Rescue | Hurt | Net |
|---|---:|---:|---:|---:|---:|
| SLOT_L3_POOLED_BASELINE | 0.4120 | 0.4512 | 12 | 22 | -10 |
| SLOT_L3_NATIVE | 0.4080 | 0.4728 | 11 | 14 | -3 |
| SLOT_L4 | 0.5560 | 0.4986 | 4 | 19 | -15 |

## VGGSS-144k

| Experiment | AUD cIoU/AUC | IMG_QUERY cIoU/AUC | IQR cIoU/AUC | OGL cIoU/AUC | EXTRA cIoU/AUC |
|---|---:|---:|---:|---:|---:|
| v1.1 baseline | 0.4050/0.4079 | 0.4033/0.4099 | 0.4054/0.4103 | 0.4391/0.4294 | 0.4252/0.4218 |
| 1.1.1_14_14_L3 | 0.3860/0.4046 | 0.4033/0.4108 | 0.3967/0.4087 | 0.4372/0.4277 | 0.4195/0.4211 |
| original 1.3G | 0.4269/0.4230 | 0.4069/0.4166 | 0.4230/0.4234 | 0.4570/0.4401 | 0.4447/0.4343 |
| 1.3G_14_14_L3 | 0.4029/0.4118 | 0.4033/0.4108 | 0.4131/0.4155 | 0.4521/0.4374 | 0.4345/0.4282 |

### Ownership

| Map | cIoU | AUC | Rescue | Hurt | Net |
|---|---:|---:|---:|---:|---:|
| SLOT_L3_POOLED_BASELINE | 0.0834 | 0.2428 | 129 | 655 | -526 |
| SLOT_L3_NATIVE | 0.1099 | 0.2756 | 186 | 468 | -282 |
| SLOT_L4 | 0.3135 | 0.3859 | 170 | 222 | -52 |

## Flickr-144k

| Experiment | AUD cIoU/AUC | IMG_QUERY cIoU/AUC | IQR cIoU/AUC | OGL cIoU/AUC | EXTRA cIoU/AUC |
|---|---:|---:|---:|---:|---:|
| v1.1 baseline | 0.8280/0.6148 | 0.8000/0.6080 | 0.8200/0.6122 | 0.8760/0.6300 | 0.8640/0.6338 |
| 1.1.1_14_14_L3 | 0.8640/0.6492 | 0.8640/0.6434 | 0.8720/0.6510 | 0.8640/0.6502 | 0.8800/0.6596 |
| original 1.3G | 0.8120/0.6356 | 0.8040/0.6166 | 0.8040/0.6366 | 0.8680/0.6596 | 0.8440/0.6534 |
| 1.3G_14_14_L3 | 0.8560/0.6584 | 0.8640/0.6434 | 0.8520/0.6602 | 0.8920/0.6688 | 0.8840/0.6724 |

### Ownership

| Map | cIoU | AUC | Rescue | Hurt | Net |
|---|---:|---:|---:|---:|---:|
| SLOT_L3_POOLED_BASELINE | 0.0000 | 0.0750 | 5 | 81 | -76 |
| SLOT_L3_NATIVE | 0.0000 | 0.0288 | 7 | 77 | -70 |
| SLOT_L4 | 0.8480 | 0.6424 | 3 | 4 | -1 |

## 科研问题

### VGGSS-10k

Q1 AUD cIoU：下降 0.0010。 IQR cIoU：下降 0.0017。

Q2：SLOT_L3_NATIVE 相对 pooled L3 cIoU：下降 0.0256。

Q3 条件未满足或数据不足。

Q4：新 G 相对原 G 的 AUD cIoU：下降 0.0242。 Native fine slot teacher 使 G 的 AUD cIoU 下降，当前表现为冲突。

### Flickr-10k

Q1 AUD cIoU：下降 0.0320。 IQR cIoU：下降 0.0360。

Q2：SLOT_L3_NATIVE 相对 pooled L3 cIoU：下降 0.0040。

Q3 条件未满足或数据不足。

Q4：新 G 相对原 G 的 AUD cIoU：下降 0.0440。 Native fine slot teacher 使 G 的 AUD cIoU 下降，当前表现为冲突。

### VGGSS-144k

Q1 AUD cIoU：下降 0.0190。 IQR cIoU：下降 0.0087。

Q2：SLOT_L3_NATIVE 相对 pooled L3 cIoU：提升 0.0266。

Q3：fine object information 已经存在于 L3 Slot 内部，但当前 L4-only inference path 没有利用它。

Q4：新 G 相对原 G 的 AUD cIoU：下降 0.0240。 Native fine slot teacher 使 G 的 AUD cIoU 下降，当前表现为冲突。

### Flickr-144k

Q1 AUD cIoU：提升 0.0360。 IQR cIoU：提升 0.0520。

Q2：SLOT_L3_NATIVE 相对 pooled L3 cIoU：持平 0.0000。

Q3 条件未满足或数据不足。

Q4：新 G 相对原 G 的 AUD cIoU：提升 0.0440。 Native fine slot 与 G refinement 在 AUD cIoU 上表现为互补。

## 数据来源

| Dataset | Experiment | Source |
|---|---|---|
| VGGSS-10k | v1.1 baseline | `/data/wxr/audio_video/JSA/checkpoints/mufasa_jsa_v1_1_vggss_10k/epoch_metrics.csv (best IQR epoch 8)` |
| VGGSS-10k | 1.1.1_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.1.1_14_14_L3_vggss_10k/best_full_metrics.json` |
| VGGSS-10k | original 1.3G | `/data/wxr/audio_video/JSA/checkpoints/1.3G-multigeom_equivariant_l3_refine_vggss_10k/best_full_six_metrics.json` |
| VGGSS-10k | 1.3G_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.3G_14_14_L3_vggss_10k/best_full_six_metrics.json` |
| Flickr-10k | v1.1 baseline | `/data/wxr/audio_video/JSA/checkpoints/mufasa_jsa_v1_1_flickr_10k_frame8_center5/epoch_metrics.csv (best IQR epoch 77)` |
| Flickr-10k | 1.1.1_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.1.1_14_14_L3_flickr_10k/best_full_metrics.json` |
| Flickr-10k | original 1.3G | `/data/wxr/audio_video/JSA/checkpoints/1.3G-multigeom_equivariant_l3_refine_flickr_10k_frame8_center5/best_full_six_metrics.json` |
| Flickr-10k | 1.3G_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.3G_14_14_L3_flickr_10k/best_full_six_metrics.json` |
| VGGSS-144k | v1.1 baseline | `/data/wxr/audio_video/JSA/checkpoints/mufasa_jsa_v1_1_vggss_144k/epoch_metrics.csv (best IQR epoch 1)` |
| VGGSS-144k | 1.1.1_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.1.1_14_14_L3_vggss_144k/best_full_metrics.json` |
| VGGSS-144k | original 1.3G | `/data/wxr/audio_video/JSA/checkpoints/1.3G-multigeom_equivariant_l3_refine_vggss_144k/best_full_six_metrics.json` |
| VGGSS-144k | 1.3G_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.3G_14_14_L3_vggss_144k/best_full_six_metrics.json` |
| Flickr-144k | v1.1 baseline | `/data/wxr/audio_video/JSA/checkpoints/mufasa_jsa_v1_1_flickr_144k_frame8_center5/epoch_metrics.csv (best IQR epoch 9)` |
| Flickr-144k | 1.1.1_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.1.1_14_14_L3_flickr_144k/best_full_metrics.json` |
| Flickr-144k | original 1.3G | `/data/wxr/audio_video/JSA/checkpoints/1.3G-multigeom_equivariant_l3_refine_flickr_144k_frame8_center5/best_full_six_metrics.json` |
| Flickr-144k | 1.3G_14_14_L3 | `/data/wxr/audio_video/JSA/checkpoints/1.3G_14_14_L3_flickr_144k/best_full_six_metrics.json` |
