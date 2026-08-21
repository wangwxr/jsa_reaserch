# Experiment 5.1 - Agreement-Seeded Visual Propagation Probe

## Protocol Audit

- Zero training: `model.eval()`, `torch.inference_mode()`, no optimizer, no backward, no new trainable parameters, and `parameters_with_grad=[]`.
- The only formal seed is frozen Stage1 `P=Top20(AUD_L4) intersect Top20(IMG_L4)` at `7x7`, nearest-resized to binary `P14`.
- `F34` propagation compares an F34 seed prototype only with F34 tokens. `K34` propagation compares a K34 seed prototype only with K34 tokens. No audio cosine or cross-space cosine is used.
- GT and OGL are used only for evaluation, oracle construction, and mechanism analysis.

## 5.0 Reproduction And Feature Spaces

### VGGSS-144k

- Raw tensor error: `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0}`; per-sample metric error: `{'Stage1_AUD': 0.0, 'Stage1_IMG': 0.0, 'Stage2_AUD': 0.0, 'Stage2_IMG': 0.0, 'OGL': 0.0}`; P error: `{'area': 0.0, 'fg': 0.0, 'bg': 0.0, 'fg_purity': 0.0, 'fg_recall': 0.0}`; aggregate P-purity error: `0.0`; sample mismatch: `0`.
- Shapes: `AUD_L4=[256, 1, 7, 7]`, `IMG_L4=[256, 1, 7, 7]`, `F3=[256, 512, 14, 14]`, `F4_up=[256, 512, 14, 14]`, `F34=[256, 512, 14, 14]`, `K34=[256, 196, 512]`, `AUD_FINE=[256, 1, 14, 14]`.
- Checkpoints unchanged: `True`; no NaN/Inf: `True`; trainable params: `0`; grad parameters: `[]`.

### Flickr-144k

- Raw tensor error: `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0}`; per-sample metric error: `{'Stage1_AUD': 0.0, 'Stage1_IMG': 0.0, 'Stage2_AUD': 0.0, 'Stage2_IMG': 0.0, 'OGL': 0.0}`; P error: `{'area': 0.0, 'fg': 0.0, 'bg': 0.0, 'fg_purity': 0.0, 'fg_recall': 0.0}`; aggregate P-purity error: `0.0`; sample mismatch: `0`.
- Shapes: `AUD_L4=[32, 1, 7, 7]`, `IMG_L4=[32, 1, 7, 7]`, `F3=[32, 512, 14, 14]`, `F4_up=[32, 512, 14, 14]`, `F34=[32, 512, 14, 14]`, `K34=[32, 196, 512]`, `AUD_FINE=[32, 1, 14, 14]`.
- Checkpoints unchanged: `True`; no NaN/Inf: `True`; trainable params: `0`; grad parameters: `[]`.

## Seed Audit

| Dataset | P14 pixels mean/median | Non-empty | P precision | P recall |
|---|---:|---:|---:|---:|
| VGGSS-144k | 34.48/36.0 | 1.0000 | 0.6231 | 0.4577 |
| Flickr-144k | 31.94/32.0 | 1.0000 | 0.8602 | 0.2960 |

## Main Table A - Standalone Localization

| Dataset | AUD_FINE | IMG | SEED_ONLY | PROP_F34 | PROP_K34 | OGL |
|---|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | 0.4269/0.4230 | 0.4069/0.4166 | 0.1159/0.3055 | 0.3660/0.3958 | 0.3377/0.3844 | 0.4570/0.4401 |
| Flickr-144k | 0.8120/0.6356 | 0.8040/0.6166 | 0.0400/0.2912 | 0.7360/0.6080 | 0.7400/0.6042 | 0.8680/0.6596 |

## Main Table B - Propagation Quality

| Dataset | Space | FG sim | BG sim | Margin | Seed P/R | Prop P/R | Recall gain | Precision loss | Expansion FG | Random expansion FG |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | F34 | 0.8884 | 0.7083 | 0.1802 | 0.6231/0.4577 | 0.4277/0.9244 | 0.4667 | 0.1954 | 0.3472 | 0.2448 |
| VGGSS-144k | K34 | 0.9074 | 0.7320 | 0.1754 | 0.6231/0.4577 | 0.4034/0.9566 | 0.4988 | 0.2197 | 0.3255 | 0.2448 |
| Flickr-144k | F34 | 0.8455 | 0.5900 | 0.2555 | 0.8602/0.2960 | 0.6518/0.9517 | 0.6557 | 0.2084 | 0.5914 | 0.4441 |
| Flickr-144k | K34 | 0.8408 | 0.5213 | 0.3195 | 0.8602/0.2960 | 0.6454/0.9546 | 0.6586 | 0.2147 | 0.5829 | 0.4442 |

## Main Table C - Complementarity With AUD

| Dataset | Prop | Pearson | Spearman | JS | Top20 | Mask IoU | PROP_ONLY | AUD_ONLY | Pair Oracle | Oracle gain | OGL rescue capture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | F34 | 0.8817 | 0.8896 | 0.0058 | 0.6905 | 0.6957 | 189 | 503 | 0.4636/0.4480 | 0.0366/0.0250 | 46/357 (0.1289) |
| VGGSS-144k | K34 | 0.8948 | 0.9072 | 0.0054 | 0.6981 | 0.6675 | 178 | 638 | 0.4614/0.4489 | 0.0345/0.0259 | 38/357 (0.1064) |
| Flickr-144k | F34 | 0.9228 | 0.9188 | 0.0033 | 0.6512 | 0.8169 | 4 | 23 | 0.8280/0.6522 | 0.0160/0.0166 | 3/19 (0.1579) |
| Flickr-144k | K34 | 0.9460 | 0.9396 | 0.0021 | 0.6726 | 0.8349 | 3 | 21 | 0.8240/0.6472 | 0.0120/0.0116 | 2/19 (0.1053) |

Reference AUD+IMG Sample Oracle from 4.0:

- VGGSS-144k: `0.4595/0.4423`.
- Flickr-144k: `0.8640/0.6598`.

## Main Table D - Seed Controls

| Dataset | Space | Seed | Prop cIoU/AUC | Pair-oracle gain cIoU/AUC |
|---|---|---|---:|---:|
| VGGSS-144k | F34 | RANDOM | 0.2307/0.3066 | 0.0275/0.0213 |
| VGGSS-144k | F34 | AUD | 0.3614/0.3935 | 0.0345/0.0244 |
| VGGSS-144k | F34 | IMG | 0.3616/0.3941 | 0.0363/0.0255 |
| VGGSS-144k | F34 | AGREEMENT | 0.3660/0.3958 | 0.0366/0.0250 |
| VGGSS-144k | K34 | RANDOM | 0.2286/0.3026 | 0.0266/0.0210 |
| VGGSS-144k | K34 | AUD | 0.3340/0.3821 | 0.0341/0.0258 |
| VGGSS-144k | K34 | IMG | 0.3325/0.3826 | 0.0345/0.0262 |
| VGGSS-144k | K34 | AGREEMENT | 0.3377/0.3844 | 0.0345/0.0259 |
| Flickr-144k | F34 | RANDOM | 0.5280/0.5074 | 0.0080/0.0068 |
| Flickr-144k | F34 | AUD | 0.7320/0.6060 | 0.0120/0.0156 |
| Flickr-144k | F34 | IMG | 0.7440/0.6062 | 0.0160/0.0162 |
| Flickr-144k | F34 | AGREEMENT | 0.7360/0.6080 | 0.0160/0.0166 |
| Flickr-144k | K34 | RANDOM | 0.5240/0.5014 | 0.0080/0.0060 |
| Flickr-144k | K34 | AUD | 0.7320/0.6012 | 0.0160/0.0112 |
| Flickr-144k | K34 | IMG | 0.7320/0.6030 | 0.0120/0.0122 |
| Flickr-144k | K34 | AGREEMENT | 0.7400/0.6042 | 0.0120/0.0116 |

## Top10/20/30 Diagnostic

| Dataset | Top-k | Seed purity/recall | F34 cIoU/AUC | F34 oracle gain | K34 cIoU/AUC | K34 oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | 10% | 0.6753/0.2429 | 0.3720/0.3985 | 0.0361 | 0.3540/0.3911 | 0.0345 |
| VGGSS-144k | 20% | 0.6231/0.4577 | 0.3660/0.3958 | 0.0366 | 0.3377/0.3844 | 0.0345 |
| VGGSS-144k | 30% | 0.5724/0.6138 | 0.3474/0.3878 | 0.0349 | 0.3207/0.3754 | 0.0333 |
| Flickr-144k | 10% | 0.8772/0.1374 | 0.7320/0.6114 | 0.0200 | 0.7440/0.6084 | 0.0080 |
| Flickr-144k | 20% | 0.8602/0.2960 | 0.7360/0.6080 | 0.0160 | 0.7400/0.6042 | 0.0120 |
| Flickr-144k | 30% | 0.8341/0.4575 | 0.7280/0.6008 | 0.0120 | 0.7280/0.5978 | 0.0120 |

## Error-Mechanism Groups

IMG_ONLY+SHRINK mean IoU / predicted-area / FP-area / FG-recall:

| Dataset | Method | IoU | Area | FP area | FG recall |
|---|---|---:|---:|---:|---:|
| VGGSS-144k | AUD | 0.4383 | 0.5250 | 0.2886 | 0.9505 |
| VGGSS-144k | IMG | 0.5599 | 0.3748 | 0.1511 | 0.9031 |
| VGGSS-144k | PROP_F34 | 0.3935 | 0.6192 | 0.3742 | 0.9808 |
| VGGSS-144k | PROP_K34 | 0.3497 | 0.7078 | 0.4594 | 0.9912 |
| Flickr-144k | AUD | 0.4320 | 0.6120 | 0.3180 | 0.9666 |
| Flickr-144k | IMG | 0.5439 | 0.4669 | 0.1874 | 0.9233 |
| Flickr-144k | PROP_F34 | 0.3720 | 0.7322 | 0.4312 | 0.9875 |
| Flickr-144k | PROP_K34 | 0.3644 | 0.7389 | 0.4387 | 0.9848 |

AUD over-expansion group:

- VGGSS-144k count `4111`: FP area AUD/F34/K34 `0.3095` / `0.4332` / `0.4990`; recall `0.9186` / `0.9465` / `0.9728`.
- Flickr-144k count `203`: FP area AUD/F34/K34 `0.2203` / `0.2906` / `0.2969`; recall `0.9249` / `0.9603` / `0.9640`.

AUD_ONLY risk:

- VGGSS-144k count `271`: F34 success/hurt `175/96`, mean IoU/recall delta `0.0029` / `0.1057`; K34 `168/103`, deltas `0.0116` / `0.1635`.
- Flickr-144k count `15`: F34 success/hurt `12/3`, mean IoU/recall delta `-0.0053` / `0.0784`; K34 `13/2`, deltas `0.0043` / `0.0959`.

## Agreement Confidence And Gain

| Dataset | Quartile | Seed purity | F34 cIoU | F34 margin | K34 cIoU | K34 margin |
|---|---|---:|---:|---:|---:|---:|
| VGGSS-144k | Q1 | 0.5428 | 0.2806 | 0.1580 | 0.2519 | 0.1516 |
| VGGSS-144k | Q2 | 0.6088 | 0.3527 | 0.1767 | 0.3233 | 0.1711 |
| VGGSS-144k | Q3 | 0.6464 | 0.3615 | 0.1883 | 0.3336 | 0.1839 |
| VGGSS-144k | Q4 | 0.6945 | 0.4694 | 0.1976 | 0.4422 | 0.1951 |
| Flickr-144k | Q1 | 0.8203 | 0.6825 | 0.2073 | 0.6984 | 0.2534 |
| Flickr-144k | Q2 | 0.8767 | 0.7937 | 0.2534 | 0.7619 | 0.3155 |
| Flickr-144k | Q3 | 0.8744 | 0.6935 | 0.2673 | 0.7258 | 0.3372 |
| Flickr-144k | Q4 | 0.8696 | 0.7742 | 0.2948 | 0.7742 | 0.3732 |

Seed purity vs propagation IoU gain correlation:

- VGGSS-144k F34 Pearson/Spearman `0.1741/0.1793`; K34 `0.1348/0.1191`.
- Flickr-144k F34 Pearson/Spearman `0.1311/0.1575`; K34 `-0.0090/0.0398`.

## Qualitative

- VGGSS-144k deterministic selections: `{'AUD_OVER_EXPANSION': '-0BIyqJj9ZU_000030', 'HIGH_CONFIDENCE': 'wy5edFMFcyM_000370', 'LOW_CONFIDENCE': '8mOksvfiImI_000130', 'PROP_HURT': '-0UuUoXQUoI_000107', 'OGL_RESCUE': '-4bPiXbovf0_000008', 'AUD_ONLY': '-GW1J75oAKU_000304', 'PROP_ONLY_SUCCESS': '-OAyRsvFGgc_000030', 'IMG_ONLY_SHRINK': '-hYRFCQdbLg_000030'}`.
- Flickr-144k deterministic selections: `{'HIGH_CONFIDENCE': '3790233127', 'LOW_CONFIDENCE': '9636000842', 'AUD_OVER_EXPANSION': '10007936344', 'AUD_ONLY': '10013411946', 'PROP_HURT': '10283938426', 'IMG_ONLY_SHRINK': '10548273474', 'OGL_RESCUE': '10548273474', 'PROP_ONLY_SUCCESS': '10939270325'}`.
- Propagation reliably fills most of the visually similar object and nearby context, converting the sparse core into a broad response.
- In successful PROP_ONLY cases this broad response covers object extent missed by AUD, explaining the sizable pair-oracle capacity.
- In IMG_ONLY+SHRINK and OGL-rescue cases, the same mechanism expands into exterior/context instead of preserving IMG's tighter correction.
- F34 is slightly less expansive than K34 on VGG, but neither space suppresses context leakage.

## Decision

**Case D - Propagation Introduces More Context Leakage.**

The frozen visual spaces do contain a real propagation signal: FG-BG margins are positive on both datasets, recall rises strongly over SEED_ONLY, expansion pixels beat matched random, and AUD+PROP pair oracles have non-trivial gains.

However, the propagation does not solve sounding-object extent. It drives recall toward 0.92-0.96 by producing much larger masks. False-positive area increases in both over-expansion groups, IMG_ONLY+SHRINK performance falls below both AUD and IMG, and OGL-rescue capture remains only about 10-16%.

Agreement specificity is also weak: agreement prototypes clearly beat random seeds, but improve only marginally and inconsistently over AUD/IMG seeds. Seed-purity versus propagation-gain correlations are weak, especially on Flickr.

**Close the current prototype-propagation line. Do not start 5.2 Agreement-Seeded Spatial Refinement automatically.**
