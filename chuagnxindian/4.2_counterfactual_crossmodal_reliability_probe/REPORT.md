# Experiment 4.2 - Counterfactual Cross-Modal Reliability Probe

## Zero-Training and Reproduction Audit

| Dataset | 4.1 metric error | Raw AUD error | Raw IMG error | Order mismatch | Trainable params | Checkpoints unchanged |
|---|---:|---:|---:|---:|---:|---|
| VGGSS-144k | 0.0 | 0.0 | 0.0 | 0 | 0 | True |
| Flickr-144k | 0.0 | 0.0 | 0.0 | 0 | 0 | True |

## Formal InfoNCE Metric Space

- Visual: `slot_fusion([L3 visual slots, L4 visual slots])[:,0]`.
- Audio: `audio_slots[:,0]`.
- Normalization: `torch.nn.functional.normalize(..., dim=2) in training; target vectors use dim=-1 equivalently`.
- Projection: `none after slot fusion/audio Slot Attention`.
- Similarity: `dot product of unit-normalized target slots, exactly cosine similarity`; training logit `S / tau` with `tau=0.03`.
- Counterfactual scores use pre-temperature cosine. `F34, K34, K4, raw visual tokens are not directly compared with audio slots`.

## Mask and Intervention Audit

Primary masks use the highest 40 of 196 positions independently for AUD and IMG. Nearest resizing maps every native cell to 16x16 input pixels.

| Dataset | A20 native/input | I20 native/input | Input area diff | Mean A20-I20 IoU | Mean AUD-extra/IMG-extra area |
|---|---:|---:|---:|---:|---:|
| VGGSS-144k | 40.0/10240.0 | 40.0/10240.0 | 0.0 | 0.7496 | 31.85/8.15 |
| Flickr-144k | 40.0/10240.0 | 40.0/10240.0 | 0.0 | 0.6377 | 35.16/4.84 |

- Gaussian Blur: kernel `[31, 31]`, sigma `[10.0, 10.0]`.
- Mean Fill: normalized value `[0.0, 0.0, 0.0]` because ImageNet channel means map to zero.
- Random equal-area control seed: `42020`.

Intervention strength (A/I pairs report mean absolute perturbation, L1, and L2):

| Dataset | Baseline | Keep A/I mean abs | Keep A/I L1 | Keep A/I L2 | Remove A/I mean abs | Remove A/I L1 | Remove A/I L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | BLUR | 0.240318/0.243314 | 36174.57/36625.51 | 156.857/158.653 | 0.088273/0.085277 | 13287.53/12836.59 | 101.668/99.083 |
| VGGSS-144k | MEAN | 0.838905/0.838005 | 126278.66/126143.21 | 417.218/417.044 | 0.200259/0.201159 | 30144.54/30279.99 | 198.011/198.542 |
| Flickr-144k | BLUR | 0.244723/0.249668 | 36837.62/37581.99 | 160.858/163.510 | 0.094949/0.090004 | 14292.45/13548.08 | 108.816/104.977 |
| Flickr-144k | MEAN | 0.808029/0.806124 | 121630.96/121344.16 | 408.332/407.738 | 0.188343/0.190249 | 28350.94/28637.74 | 188.968/190.501 |

## Semantic Score Stability

| Dataset | Score | Mean | Median | Std |
|---|---|---:|---:|---:|
| VGGSS-144k | S_ORIGINAL | 0.414988 | 0.416169 | 0.059658 |
| VGGSS-144k | S_KEEP_A_BLUR | 0.385829 | 0.383696 | 0.051770 |
| VGGSS-144k | S_KEEP_I_BLUR | 0.384431 | 0.382739 | 0.051941 |
| VGGSS-144k | S_REMOVE_A_BLUR | 0.388317 | 0.388219 | 0.050495 |
| VGGSS-144k | S_REMOVE_I_BLUR | 0.390120 | 0.389849 | 0.050574 |
| VGGSS-144k | S_KEEP_A_MEAN | 0.370629 | 0.370770 | 0.055911 |
| VGGSS-144k | S_KEEP_I_MEAN | 0.369227 | 0.368740 | 0.056276 |
| VGGSS-144k | S_REMOVE_A_MEAN | 0.376496 | 0.376014 | 0.050898 |
| VGGSS-144k | S_REMOVE_I_MEAN | 0.378709 | 0.378679 | 0.050983 |
| VGGSS-144k | CF_RANDOM_BLUR | -0.063118 | -0.057671 | 0.065902 |
| VGGSS-144k | CF_RANDOM_MEAN | -0.057169 | -0.050706 | 0.059601 |
| Flickr-144k | S_ORIGINAL | 0.426748 | 0.426640 | 0.051787 |
| Flickr-144k | S_KEEP_A_BLUR | 0.338155 | 0.337167 | 0.060392 |
| Flickr-144k | S_KEEP_I_BLUR | 0.338496 | 0.340356 | 0.063841 |
| Flickr-144k | S_REMOVE_A_BLUR | 0.396148 | 0.394342 | 0.053045 |
| Flickr-144k | S_REMOVE_I_BLUR | 0.399303 | 0.398250 | 0.053704 |
| Flickr-144k | S_KEEP_A_MEAN | 0.326221 | 0.329066 | 0.060250 |
| Flickr-144k | S_KEEP_I_MEAN | 0.323723 | 0.325575 | 0.062593 |
| Flickr-144k | S_REMOVE_A_MEAN | 0.396078 | 0.394622 | 0.052095 |
| Flickr-144k | S_REMOVE_I_MEAN | 0.397483 | 0.392980 | 0.054678 |
| Flickr-144k | CF_RANDOM_BLUR | -0.095561 | -0.089318 | 0.070984 |
| Flickr-144k | CF_RANDOM_MEAN | -0.104733 | -0.102907 | 0.065360 |

## Counterfactual Evidence

| Dataset | Evidence | AUROC IMG-better | AUPRC | AUROC/AUPRC IMG-only | BalAcc@0 | Delta>0 |
|---|---|---:|---:|---:|---:|---:|
| VGGSS-144k | DELTA_CF_BLUR | 0.5058 | 0.5087 | 0.5183/0.0335 | 0.5038 | 0.4428 |
| VGGSS-144k | DELTA_CF_MEAN | 0.4949 | 0.4970 | 0.4800/0.0311 | 0.4983 | 0.4595 |
| VGGSS-144k | DELTA_KEEP_BLUR | 0.5046 | 0.5107 | 0.5202/0.0335 | 0.5046 | 0.4771 |
| VGGSS-144k | DELTA_KEEP_MEAN | 0.4931 | 0.4997 | 0.4903/0.0323 | 0.4988 | 0.4775 |
| VGGSS-144k | DELTA_DROP_BLUR | 0.5016 | 0.5026 | 0.4990/0.0314 | 0.5018 | 0.4568 |
| VGGSS-144k | DELTA_DROP_MEAN | 0.4983 | 0.4972 | 0.4730/0.0301 | 0.4950 | 0.4604 |
| Flickr-144k | DELTA_CF_BLUR | 0.5241 | 0.4651 | 0.5664/0.0804 | 0.5321 | 0.4960 |
| Flickr-144k | DELTA_CF_MEAN | 0.5485 | 0.4908 | 0.4615/0.0593 | 0.5285 | 0.4440 |
| Flickr-144k | DELTA_KEEP_BLUR | 0.5159 | 0.4472 | 0.6125/0.0977 | 0.4947 | 0.4920 |
| Flickr-144k | DELTA_KEEP_MEAN | 0.5358 | 0.4586 | 0.4869/0.0583 | 0.5227 | 0.4600 |
| Flickr-144k | DELTA_DROP_BLUR | 0.5160 | 0.4694 | 0.4804/0.0686 | 0.4842 | 0.4480 |
| Flickr-144k | DELTA_DROP_MEAN | 0.5520 | 0.4915 | 0.5245/0.0683 | 0.5227 | 0.4320 |

Primary direction consistency:

- DELTA_CF_BLUR: VGG `positive` (0.5058), Flickr `positive` (0.5241); consistent=`True`.
- DELTA_CF_MEAN: VGG `negative` (0.4949), Flickr `positive` (0.5485); consistent=`False`.

## Official Zero-Threshold Selectors

| Dataset | Method | cIoU/AUC | Rescue/Hurt/Net | IMG rate | IMG-rescue retained | OGL-rescue captured | IMG-only wrong AUD | AUD-only wrong IMG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | AUD | 0.4269/0.4230 | 0/0/0 | 0 | 0/N/A | 0/N/A | N/A | N/A |
| VGGSS-144k | Fixed IQR | 0.4230/0.4234 | 83/103/-20 | N/A | N/A | N/A | N/A | N/A |
| VGGSS-144k | CF_BLUR | 0.4203/0.4208 | 83/117/-34 | 0.4428 | 56/104 (0.5385) | 56/357 (0.1569) | 85 | 117 |
| VGGSS-144k | CF_MEAN | 0.4184/0.4207 | 74/118/-44 | 0.4595 | 51/104 (0.4904) | 51/357 (0.1429) | 94 | 118 |
| VGGSS-144k | CF_CONSENSUS | 0.4221/0.4216 | 53/78/-25 | 0.2931 | 38/104 (0.3654) | 38/357 (0.1064) | 115 | 78 |
| VGGSS-144k | OGL | 0.4570/0.4401 | N/A | N/A | N/A | N/A | N/A | N/A |
| VGGSS-144k | Sample Oracle | 0.4595/0.4423 | N/A | N/A | N/A | N/A | N/A | N/A |
| Flickr-144k | AUD | 0.8120/0.6356 | 0/0/0 | 0 | 0/N/A | 0/N/A | N/A | N/A |
| Flickr-144k | Fixed IQR | 0.8040/0.6366 | 2/4/-2 | N/A | N/A | N/A | N/A | N/A |
| Flickr-144k | CF_BLUR | 0.8240/0.6274 | 8/5/3 | 0.4960 | 8/11 (0.7273) | 8/19 (0.4211) | 5 | 5 |
| Flickr-144k | CF_MEAN | 0.8120/0.6284 | 5/5/0 | 0.4440 | 4/11 (0.3636) | 4/19 (0.2105) | 8 | 5 |
| Flickr-144k | CF_CONSENSUS | 0.8120/0.6286 | 4/4/0 | 0.3320 | 4/11 (0.3636) | 4/19 (0.2105) | 9 | 4 |
| Flickr-144k | OGL | 0.8680/0.6596 | N/A | N/A | N/A | N/A | N/A | N/A |
| Flickr-144k | Sample Oracle | 0.8640/0.6598 | N/A | N/A | N/A | N/A | N/A | N/A |

## Mechanism by Group

| Dataset | Group | Count | Delta CF Blur mean/median | Delta CF Mean mean/median | Blur/Mean IMG fraction | A20-I20 IoU | Drop A/I Blur |
|---|---|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | IMG_ONLY | 168 | -0.002838/-0.000129 | -0.005613/-0.002904 | 0.4940/0.4405 | 0.7739 | 0.036066/0.033934 |
| VGGSS-144k | AUD_ONLY | 271 | -0.004609/-0.003515 | -0.004176/-0.002953 | 0.4317/0.4354 | 0.7258 | 0.023464/0.020049 |
| VGGSS-144k | BOTH_SUCCESS | 1931 | -0.003226/-0.002270 | -0.003519/-0.001981 | 0.4345/0.4609 | 0.7603 | 0.030600/0.028672 |
| VGGSS-144k | BOTH_FAIL | 2788 | -0.003070/-0.001919 | -0.003507/-0.002114 | 0.4466/0.4620 | 0.7431 | 0.023694/0.022154 |
| VGGSS-144k | OGL_RESCUE | 357 | -0.003293/0.000000 | -0.003747/-0.001042 | 0.4986/0.4902 | 0.7547 | 0.031400/0.029503 |
| Flickr-144k | IMG_ONLY | 13 | 0.006393/0.007049 | -0.004589/-0.014300 | 0.6154/0.3846 | 0.6448 | 0.036929/0.032658 |
| Flickr-144k | AUD_ONLY | 15 | -0.006064/-0.011581 | -0.010837/-0.017424 | 0.3333/0.3333 | 0.6540 | 0.053802/0.049950 |
| Flickr-144k | BOTH_SUCCESS | 188 | -0.005088/-0.000876 | -0.004050/-0.002160 | 0.4840/0.4628 | 0.6368 | 0.030001/0.025921 |
| Flickr-144k | BOTH_FAIL | 34 | 0.007677/0.008448 | 0.000238/-0.003573 | 0.5882/0.4118 | 0.6323 | 0.021256/0.023950 |
| Flickr-144k | OGL_RESCUE | 19 | 0.014179/0.011012 | -0.006237/-0.008197 | 0.7895/0.3158 | 0.6676 | 0.029183/0.030145 |

Signed disagreement and SHRINK diagnostics:

- VGGSS-144k IMG-better correction counts `{'SHRINK': 1425, 'MIXED': 1061, 'EXPAND': 107}`. SHRINK Delta CF Blur/Mean mean `-0.002369`/`-0.002974`, selected IMG fraction `0.4618`/`0.4596`.
- VGGSS-144k IMG_ONLY+SHRINK count `125`: AUD-extra exterior fraction `0.9436`; removal drop Blur/Mean `0.002121`/`0.010465`; density `0.00005358`/`0.00026481`.
- Flickr-144k IMG-better correction counts `{'SHRINK': 74, 'MIXED': 33}`. SHRINK Delta CF Blur/Mean mean `-0.000749`/`0.002950`, selected IMG fraction `0.5405`/`0.5270`.
- Flickr-144k IMG_ONLY+SHRINK count `10`: AUD-extra exterior fraction `0.8329`; removal drop Blur/Mean `0.009900`/`0.022572`; density `0.00027263`/`0.00062473`.

## Post-hoc Threshold Transfer

GT-optimal source thresholds are applied unchanged to the other dataset. These results are not official.

| Evidence | Source -> Target | Threshold | Target cIoU/AUC | Rescue/Hurt/Net | IMG rate |
|---|---|---:|---:|---:|---:|
| DELTA_CF_BLUR | VGGSS-144k -> Flickr-144k | 0.0884001 | 0.8120/0.6350 | 0/0/0 | 0.0240 |
| DELTA_CF_BLUR | Flickr-144k -> VGGSS-144k | 0.0049754 | 0.4207/0.4213 | 56/88/-32 | 0.3354 |
| DELTA_CF_MEAN | VGGSS-144k -> Flickr-144k | 0.1212659 | 0.8120/0.6358 | 0/0/0 | 0.0080 |
| DELTA_CF_MEAN | Flickr-144k -> VGGSS-144k | 0.0136924 | 0.4215/0.4215 | 36/64/-28 | 0.2237 |

## Qualitative Selection

- VGGSS-144k deterministic samples: `{'OGL_RESCUE_MISSED': '-4bPiXbovf0_000008', 'SHRINK_SUCCESS': '-5CGQGSFGyg_000060', 'AUD_ONLY_CF_CORRECT': '-GW1J75oAKU_000304', 'AUD_ONLY_CF_FAIL': '-JUhUI_KvUI_000026', 'IMG_ONLY_CF_CORRECT': '-Vo4CAMX26U_000030', 'OGL_RESCUE_CAPTURED': '-Vo4CAMX26U_000030', 'IMG_ONLY_CF_FAIL': '-hYRFCQdbLg_000030'}`.
- Flickr-144k deterministic samples: `{'SHRINK_SUCCESS': '10007936344', 'AUD_ONLY_CF_CORRECT': '10013411946', 'AUD_ONLY_CF_FAIL': '10106776154', 'IMG_ONLY_CF_CORRECT': '10548273474', 'OGL_RESCUE_CAPTURED': '10548273474', 'OGL_RESCUE_MISSED': '10939270325', 'IMG_ONLY_CF_FAIL': '12066557153'}`.
- Correct CF selections exist, but their score margins are often only 1e-3 to 1e-2. Visually similar keep/remove interventions can therefore reverse the branch choice.
- Failure examples show that equal-area masks still preserve almost the same object core. The semantic slot score reacts strongly to the intervention artifact but weakly to the subtle extent difference that determines localization IoU.
- Blur and Mean Fill can choose different branches for the same hard case, matching their inconsistent full-dataset direction/calibration.

## Decision

**Case D - Counterfactual Evidence Fails.**

- VGG is at chance: CF Blur AUROC 0.5058 and CF Mean 0.4949. Mean also has opposite direction between VGG and Flickr. Flickr's 0.5241/0.5485 does not approach the requested ~0.60 evidence level.
- Threshold-zero selectors all hurt VGG: CF Blur 0.4203 with Net -34, CF Mean 0.4184 with Net -44, and consensus 0.4221 with Net -25, versus AUD 0.4269.
- Flickr CF Blur gives a small 0.8120 -> 0.8240 gain with Net +3, but it is dataset-specific and is not supported by VGG. Mean and consensus do not improve Flickr cIoU.
- IMG-rescue retention is insufficient and unstable: Blur retains 56/104 on VGG and 8/11 on Flickr; Mean retains 51/104 and 4/11; consensus retains 38/104 and 4/11.
- IMG_ONLY+SHRINK has high AUD-extra exterior fraction, but deletion drop is not uniquely low across groups or interventions. The hypothesized context-leakage signal is not expressed reliably in the frozen semantic metric.
- Cross-dataset threshold transfer fails: VGG thresholds nearly suppress all Flickr IMG choices, while Flickr thresholds reduce VGG below AUD with negative Net.

**4.x next step: stop the hand-designed internal selector evidence line. Reconsider the supervision source or training objective; do not start 4.3 from this counterfactual score.**

No 4.3 experiment was implemented or started.
