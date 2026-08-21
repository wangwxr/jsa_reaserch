# Experiment 2.5 - Dual-Path Decision Probe

## Zero-Training Audit

All four settings used `model.eval()` and `torch.inference_mode()`. No optimizer or backward call was created; all loaded parameters had `requires_grad=False`; checkpoint SHA256 and mtime values were unchanged.

| Setting | Q3 | K3_POOL | K3_NATIVE | Pooled reconstruction max error | Native slot-sum max error |
|---|---:|---:|---:|---:|---:|
| vggss_10k | [256, 2, 512] | [256, 49, 512] | [256, 196, 512] | 0.000e+00 | 1.192e-07 |
| flickr_10k | [32, 2, 512] | [32, 49, 512] | [32, 196, 512] | 0.000e+00 | 1.192e-07 |
| vggss_144k | [256, 2, 512] | [256, 49, 512] | [256, 196, 512] | 0.000e+00 | 1.192e-07 |
| flickr_144k | [32, 2, 512] | [32, 49, 512] | [32, 196, 512] | 0.000e+00 | 1.192e-07 |

## Part A - Ownership and Corresponding-Audio Fusion

`cIoU/AUC` below are standalone ownership metrics. Rescue/Hurt/Net and Oracle use the corresponding audio checkpoint and its alpha=0.6 fusion.

| Dataset | Map | Ownership cIoU/AUC | Fusion cIoU/AUC | Rescue | Hurt | Net | Oracle cIoU/AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| vggss_10k | SLOT_L3_POOLED | 0.1342/0.2692 | 0.3112/0.3715 | 101 | 567 | -466 | 0.4211/0.4203 |
| vggss_10k | SLOT_L3_NATIVE_UPDATE | 0.0450/0.2399 | 0.3164/0.3807 | 123 | 458 | -335 | 0.4052/0.4174 |
| vggss_10k | SLOT_L3_NATIVE_READOUT | 0.1380/0.2719 | 0.3261/0.3755 | 89 | 478 | -389 | 0.4188/0.4177 |
| vggss_10k | SLOT_L4 | 0.1060/0.2835 | 0.2916/0.3782 | 325 | 892 | -567 | 0.4645/0.4413 |
| flickr_10k | SLOT_L3_POOLED | 0.4760/0.4992 | 0.7560/0.5922 | 16 | 18 | -2 | 0.8280/0.6230 |
| flickr_10k | SLOT_L3_NATIVE_UPDATE | 0.4080/0.4728 | 0.7080/0.5756 | 11 | 14 | -3 | 0.7640/0.6030 |
| flickr_10k | SLOT_L3_NATIVE_READOUT | 0.4600/0.4876 | 0.7480/0.5898 | 12 | 16 | -4 | 0.8120/0.6180 |
| flickr_10k | SLOT_L4 | 0.3920/0.4424 | 0.6440/0.5360 | 5 | 35 | -30 | 0.7840/0.6020 |
| vggss_144k | SLOT_L3_POOLED | 0.1247/0.2560 | 0.2617/0.3604 | 104 | 818 | -714 | 0.4203/0.4239 |
| vggss_144k | SLOT_L3_NATIVE_UPDATE | 0.1099/0.2756 | 0.3313/0.3891 | 186 | 468 | -282 | 0.4221/0.4250 |
| vggss_144k | SLOT_L3_NATIVE_READOUT | 0.1214/0.2554 | 0.2770/0.3644 | 78 | 713 | -635 | 0.4153/0.4207 |
| vggss_144k | SLOT_L4 | 0.2144/0.3472 | 0.3466/0.4011 | 285 | 561 | -276 | 0.4554/0.4421 |
| flickr_144k | SLOT_L3_POOLED | 0.3520/0.4524 | 0.7240/0.5912 | 9 | 29 | -20 | 0.8400/0.6448 |
| flickr_144k | SLOT_L3_NATIVE_UPDATE | 0.0000/0.0288 | 0.5840/0.5236 | 7 | 77 | -70 | 0.8920/0.6618 |
| flickr_144k | SLOT_L3_NATIVE_READOUT | 0.3080/0.4256 | 0.7120/0.5758 | 7 | 30 | -23 | 0.8320/0.6360 |
| flickr_144k | SLOT_L4 | 0.7320/0.5952 | 0.8240/0.6238 | 7 | 2 | 5 | 0.8320/0.6400 |

## Part A - Original G Combinations (144k)

| Dataset | Method | cIoU/AUC | Rescue | Hurt | Net | Oracle cIoU/AUC |
|---|---|---:|---:|---:|---:|---:|
| vggss_144k | Original G AUD | 0.4269/0.4230 | 0 | 0 | 0 | 0.4269/0.4230 |
| vggss_144k | Original G + L3 pooled | 0.2780/0.3673 | 85 | 853 | -768 | 0.4434/0.4340 |
| vggss_144k | Original G + L3 native readout | 0.2891/0.3690 | 65 | 776 | -711 | 0.4395/0.4306 |
| vggss_144k | Original G + L4 | 0.3858/0.4157 | 355 | 567 | -212 | 0.4957/0.4617 |
| vggss_144k | OGL | 0.4570/0.4401 | 357 | 202 | 155 | 0.4961/0.4608 |
| flickr_144k | Original G AUD | 0.8120/0.6356 | 0 | 0 | 0 | 0.8120/0.6356 |
| flickr_144k | Original G + L3 pooled | 0.7240/0.5982 | 9 | 31 | -22 | 0.8480/0.6586 |
| flickr_144k | Original G + L3 native readout | 0.7040/0.5774 | 10 | 37 | -27 | 0.8520/0.6514 |
| flickr_144k | Original G + L4 | 0.8360/0.6396 | 11 | 5 | 6 | 0.8560/0.6614 |
| flickr_144k | OGL | 0.8680/0.6596 | 19 | 5 | 14 | 0.8880/0.6860 |

## Part B - Cross-Checkpoint Fusion

| Method | VGG cIoU/AUC | Flickr cIoU/AUC |
|---|---:|---:|
| Original G AUD | 0.4269/0.4230 | 0.8120/0.6356 |
| Original G + original HR14 | 0.3777/0.4175 | 0.8440/0.6502 |
| 2.4 AUD | 0.4153/0.4195 | 0.7880/0.6296 |
| 2.4 AUD + 2.4 OWN14 | 0.3971/0.4209 | 0.8480/0.6422 |
| Original G AUD + 2.4 OWN14 | 0.4002/0.4237 | 0.8480/0.6484 |
| OGL | 0.4570/0.4401 | 0.8680/0.6596 |

| Method | VGG Rescue/Hurt/Net, Oracle | Flickr Rescue/Hurt/Net, Oracle |
|---|---:|---:|
| Original HR14 | 430/684/-254, 0.5103/0.4694 | 12/4/8, 0.8600/0.6664 |
| 2.4 same-checkpoint | 397/491/-94, 0.4922/0.4620 | 17/2/15, 0.8560/0.6554 |
| Cross-checkpoint | 380/518/-138, 0.5006/0.4648 | 12/3/9, 0.8600/0.6632 |

## Map Complementarity

Statistics compare each normalized 224x224 candidate map against original G AUD.

| Dataset | Candidate | Pearson mean | Spearman mean | JS mean |
|---|---|---:|---:|---:|
| vggss_144k | ORIGINAL_HR14 | 0.9105 | 0.9750 | 0.0406 |
| vggss_144k | OWN14_24 | 0.9228 | 0.9702 | 0.0325 |
| vggss_144k | L3_NATIVE_READOUT | -0.1006 | -0.1108 | 0.0377 |
| flickr_144k | ORIGINAL_HR14 | 0.9560 | 0.9754 | 0.0167 |
| flickr_144k | OWN14_24 | 0.9521 | 0.9582 | 0.0125 |
| flickr_144k | L3_NATIVE_READOUT | 0.1259 | 0.1220 | 0.0433 |

## Alpha Diagnostic

Alpha=0.6 is the formal result; other values are diagnostics only.

| Dataset | Audio alpha | cIoU/AUC | Rescue | Hurt | Net |
|---|---:|---:|---:|---:|---:|
| vggss_144k | 0.5 | 0.3819/0.4171 | 444 | 676 | -232 |
| vggss_144k | 0.6 | 0.4002/0.4237 | 380 | 518 | -138 |
| vggss_144k | 0.7 | 0.4166/0.4280 | 307 | 360 | -53 |
| vggss_144k | 0.8 | 0.4228/0.4282 | 204 | 225 | -21 |
| vggss_144k | 0.9 | 0.4267/0.4272 | 93 | 94 | -1 |
| flickr_144k | 0.5 | 0.8520/0.6446 | 14 | 4 | 10 |
| flickr_144k | 0.6 | 0.8480/0.6484 | 12 | 3 | 9 |
| flickr_144k | 0.7 | 0.8240/0.6452 | 6 | 3 | 3 |
| flickr_144k | 0.8 | 0.8120/0.6442 | 3 | 3 | 0 |
| flickr_144k | 0.9 | 0.8120/0.6422 | 1 | 1 | 0 |

## Fixed-Selection Qualitative Findings

The panels use the unchanged deterministic Experiment 2.2 manifests; no 2.5 outcome was used for selection.

- L3 native-readout is finer than the pooled map, but its new detail is mostly fragmented high-frequency activation over foreground and background. It avoids the severe Flickr-144k native-update collapse, yet does not produce a coherent object boundary or a useful correction map.
- Original HR14 and 2.4 OWN14 are visually close to original G AUD. On Flickr, the shared broad region often overlaps the large/centered sounding object, so fusion helps. On VGG, the same behavior commonly over-expands object extent into surrounding scene context; occasional misplaced or undersized peaks occur, but over-expansion is the dominant fixed-sample failure.
- Cross fusion preserves the original G audio response, but 2.4 OWN14 usually changes intensity inside the same region instead of supplying an independent spatial correction. This matches its high Pearson/Spearman correlation with AUD.


## Decision

**Route C**

Do not continue the current object-ownership line: internal Slot ownership has oracle capacity, but the current self-supervised extent signal is not stable across datasets.

Decision checks:

- Route A `vgg_native_rescue_gt_pooled`: False
- Route A `vgg_native_hurt_lt_pooled`: True
- Route A `vgg_original_G_plus_native_gt_original_G`: False
- Route A `flickr_original_G_plus_native_no_material_drop`: False
- Route B `vggss_144k_cross_gt_same_checkpoint`: True
- Route B `vggss_144k_cross_near_or_above_original_G`: False
- Route B `flickr_144k_cross_gt_same_checkpoint`: False
- Route B `flickr_144k_cross_near_or_above_original_G`: True

No subsequent experiment was implemented or started.
