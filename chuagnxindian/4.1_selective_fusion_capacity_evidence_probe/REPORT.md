# Experiment 4.1 - Selective Fusion Capacity & Evidence Probe

## Protocol Audit

- Zero training: `model.eval()`, `torch.inference_mode()`, no optimizer, no backward, no new trainable parameters.
- Formal selector inputs use only original-model internal maps/representations. GT and OGL are used only for oracle construction, diagnostic labels, and evaluation.
- Official selectors use the fixed rule `Delta > 0 -> IMG`, otherwise AUD. No test threshold is used by an official result.

### VGGSS-144k

- 4.0 per-sample max error: `0.0`; sample-order mismatches: `0`.
- Shapes: `Qa=[256, 2, 512]`, `Qv=[256, 2, 512]`, `K4=[256, 49, 512]`, `K34=[256, 196, 512]`, `AUD=[256, 1, 14, 14]`.
- Tensor reconstruction max errors: `{'AUD_FINE': 0.0, 'IMG_QUERY': 0.0, 'f4_tokens': 0.0}`; ownership slot-sum error: `5.960e-08`.
- Checkpoints unchanged: `True`; no NaN/Inf: `True`.

### Flickr-144k

- 4.0 per-sample max error: `0.0`; sample-order mismatches: `0`.
- Shapes: `Qa=[32, 2, 512]`, `Qv=[32, 2, 512]`, `K4=[32, 49, 512]`, `K34=[32, 196, 512]`, `AUD=[32, 1, 14, 14]`.
- Tensor reconstruction max errors: `{'AUD_FINE': 0.0, 'IMG_QUERY': 0.0, 'f4_tokens': 0.0}`; ownership slot-sum error: `5.960e-08`.
- Checkpoints unchanged: `True`; no NaN/Inf: `True`.

## Capacity

| Dataset | AUD | IMG | Fixed IQR | OGL | Sample Oracle | Region Oracle | Pixel Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | 0.4269/0.4230 | 0.4069/0.4166 | 0.4230/0.4234 | 0.4570/0.4401 | 0.4595/0.4423 | 0.4668/N/A | 0.4965/0.4693 |
| Flickr-144k | 0.8120/0.6356 | 0.8040/0.6166 | 0.8040/0.6366 | 0.8680/0.6596 | 0.8640/0.6598 | 0.8800/N/A | 0.8880/0.6872 |

Capacity gaps (cIoU/success-rate scale):

- VGGSS-144k: Sample-AUD `+0.0326`, Region-Sample `+0.0074`, Pixel-Region `+0.0297`; OGL-Sample `-0.0025`, OGL-Region `-0.0099`, OGL-Pixel `-0.0396`.
- VGGSS-144k binary-mask Pixel Oracle: success/cIoU `0.4965`, mean sample IoU `0.4692`, AUC `N/A`.
- Flickr-144k: Sample-AUD `+0.0520`, Region-Sample `+0.0160`, Pixel-Region `+0.0080`; OGL-Sample `+0.0040`, OGL-Region `-0.0120`, OGL-Pixel `-0.0200`.
- Flickr-144k binary-mask Pixel Oracle: success/cIoU `0.9080`, mean sample IoU `0.7091`, AUC `N/A`.

## Complementarity Location

| Dataset | Group | Count | Mean disagreement | Top20 mass | Pearson | Mask IoU | IMG/AUD area | Centroid px |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | IMG_ONLY | 168 | 0.0967 | 0.4098 | 0.9531 | 0.7414 | 0.8046 | 13.69 |
| VGGSS-144k | AUD_ONLY | 271 | 0.0990 | 0.4054 | 0.9392 | 0.7148 | 0.7919 | 14.90 |
| VGGSS-144k | BOTH_SUCCESS | 1931 | 0.0698 | 0.4319 | 0.9684 | 0.8537 | 0.9241 | 6.78 |
| VGGSS-144k | BOTH_FAIL | 2788 | 0.0787 | 0.4213 | 0.9590 | 0.8082 | 0.8876 | 9.28 |
| VGGSS-144k | OGL_RESCUE | 357 | 0.0781 | 0.4205 | 0.9620 | 0.8111 | 0.8952 | 9.03 |
| Flickr-144k | IMG_ONLY | 13 | 0.0914 | 0.4095 | 0.9630 | 0.7949 | 0.8015 | 9.65 |
| Flickr-144k | AUD_ONLY | 15 | 0.1133 | 0.4065 | 0.9224 | 0.6697 | 0.7021 | 16.63 |
| Flickr-144k | BOTH_SUCCESS | 188 | 0.0849 | 0.4165 | 0.9575 | 0.8220 | 0.8547 | 8.85 |
| Flickr-144k | BOTH_FAIL | 34 | 0.0908 | 0.4235 | 0.9367 | 0.8043 | 0.8460 | 8.59 |
| Flickr-144k | OGL_RESCUE | 19 | 0.0852 | 0.4058 | 0.9661 | 0.8322 | 0.8514 | 8.55 |

Disagreement energy by GT-relative region (one-pixel boundary band):

- VGGSS-144k ALL: interior `0.2512`, boundary `0.0175`, exterior `0.7312`.
- VGGSS-144k IMG_ONLY: interior `0.1918`, boundary `0.0151`, exterior `0.7930`.
- VGGSS-144k OGL_RESCUE: interior `0.2004`, boundary `0.0159`, exterior `0.7838`.
- VGGSS-144k IMG_ONLY correction types: `{'MIXED': 30, 'SHRINK': 125, 'EXPAND': 13}`; OGL_RESCUE types: `{'SHRINK': 187, 'MIXED': 145, 'EXPAND': 25}`.
- Flickr-144k ALL: interior `0.4418`, boundary `0.0227`, exterior `0.5356`.
- Flickr-144k IMG_ONLY: interior `0.2285`, boundary `0.0174`, exterior `0.7541`.
- Flickr-144k OGL_RESCUE: interior `0.2511`, boundary `0.0180`, exterior `0.7309`.
- Flickr-144k IMG_ONLY correction types: `{'SHRINK': 10, 'MIXED': 3}`; OGL_RESCUE types: `{'SHRINK': 15, 'MIXED': 4}`.

## Metric-Space Audit

Only fused visual slots and audio slots are explicitly aligned by the training InfoNCE objective. `F34`, `K34`, `K4`, raw visual tokens, and `img_to_v` outputs are attention/key/value spaces and are not used for direct audio cosine. A direct token-pooled semantic verifier is therefore N/A.

`SEMANTIC_SLOT`: `H_sem(x)=sum_s OWN_s(x)*cos(Zv_s, Za_0)`, then `E(M)=fg_mean(H_sem)-bg_mean(H_sem)`. It uses aligned global slots and existing final L4 ownership, with no new projection.

`RECIPROCAL_L4`: `C_av=Qa_0->K4`; `r_s=1-JS(Qv_s->Ka, Qa_0->Ka)/log(2)`; `H_va=sum_s OWN_s*r_s`; `H_recip=.5*(C_av+H_va)`; then the same foreground-minus-background score. The `C_av` half is explicitly marked partially circular because it is the coarse precursor of AUD; the `Qv->Ka` reciprocal half is separate.

## Evidence Prediction

Evidence definitions use `Delta = E(IMG) - E(AUD)`: `CTRL_RAW_PEAK` is raw-map peak; `CTRL_NEG_ENTROPY` is negative normalized spatial entropy; `CTRL_NEG_AREA` is negative thresholded foreground area; `CTRL_TOP20_CONCENTRATION` is the mass in the highest 20% pixels; `CTRL_NEG_COMPONENTS` is negative foreground connected-component count. The semantic and reciprocal definitions are given above.

| Evidence | VGG AUROC/AUPRC | VGG IMG-only AUROC | VGG BalAcc@0 | Flickr AUROC/AUPRC | Flickr IMG-only AUROC | Flickr BalAcc@0 | Direction consistent |
|---|---:|---:|---:|---:|---:|---:|---|
| CTRL_RAW_PEAK | 0.5183/0.5062 | 0.5505 | 0.5000 | 0.5719/0.4641 | 0.6339 | 0.5000 | yes: +Delta=>IMG |
| CTRL_NEG_ENTROPY | 0.5508/0.5378 | 0.6024 | 0.5081 | 0.6075/0.4876 | 0.5855 | 0.5163 | yes: +Delta=>IMG |
| CTRL_NEG_AREA | 0.5858/0.5388 | 0.6584 | 0.5918 | 0.5070/0.4285 | 0.5826 | 0.5536 | yes: +Delta=>IMG |
| CTRL_TOP20_CONCENTRATION | 0.5688/0.5492 | 0.6799 | 0.5241 | 0.5292/0.4461 | 0.5677 | 0.5209 | yes: +Delta=>IMG |
| CTRL_NEG_COMPONENTS | 0.5056/0.5057 | 0.5222 | 0.4977 | 0.5182/0.4373 | 0.5977 | 0.5024 | yes: +Delta=>IMG |
| SEMANTIC_SLOT | 0.5070/0.5075 | 0.5332 | 0.4982 | 0.5777/0.4883 | 0.4995 | 0.5593 | yes: +Delta=>IMG |
| RECIPROCAL_L4 | 0.5049/0.5036 | 0.4797 | 0.5084 | 0.5873/0.4657 | 0.5576 | 0.5824 | yes: +Delta=>IMG |

## Label-Free Fusion at Fixed Zero Threshold

| Dataset | Evidence | Mode | cIoU/AUC | Rescue | Hurt | Net | IMG sample rate | Pixel switch rate | IMG-rescue retention | OGL-rescue capture |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | CTRL_RAW_PEAK | SAMPLE | 0.4069/0.4166 | 168 | 271 | -103 | 1.0000 | N/A | 104/104 (1.0000) | 104/357 (0.2913) |
| VGGSS-144k | CTRL_RAW_PEAK | DISAGREE20_SCALAR | 0.4153/0.4182 | 64 | 124 | -60 | 1.0000 | 0.2000 | 40/104 (0.3846) | 43/357 (0.1204) |
| VGGSS-144k | CTRL_NEG_ENTROPY | SAMPLE | 0.4083/0.4170 | 163 | 259 | -96 | 0.9715 | N/A | 103/104 (0.9904) | 103/357 (0.2885) |
| VGGSS-144k | CTRL_NEG_ENTROPY | DISAGREE20_SCALAR | 0.4166/0.4185 | 62 | 115 | -53 | 0.9715 | 0.1943 | 40/104 (0.3846) | 43/357 (0.1204) |
| VGGSS-144k | CTRL_NEG_AREA | SAMPLE | 0.4110/0.4185 | 146 | 228 | -82 | 0.7675 | N/A | 96/104 (0.9231) | 96/357 (0.2689) |
| VGGSS-144k | CTRL_NEG_AREA | DISAGREE20_SCALAR | 0.4172/0.4194 | 51 | 101 | -50 | 0.7675 | 0.1535 | 34/104 (0.3269) | 37/357 (0.1036) |
| VGGSS-144k | CTRL_TOP20_CONCENTRATION | SAMPLE | 0.4087/0.4175 | 159 | 253 | -94 | 0.9244 | N/A | 101/104 (0.9712) | 101/357 (0.2829) |
| VGGSS-144k | CTRL_TOP20_CONCENTRATION | DISAGREE20_SCALAR | 0.4168/0.4190 | 58 | 110 | -52 | 0.9244 | 0.1849 | 38/104 (0.3654) | 41/357 (0.1148) |
| VGGSS-144k | CTRL_NEG_COMPONENTS | SAMPLE | 0.4263/0.4230 | 11 | 14 | -3 | 0.0304 | N/A | 2/104 (0.0192) | 2/357 (0.0056) |
| VGGSS-144k | CTRL_NEG_COMPONENTS | DISAGREE20_SCALAR | 0.4256/0.4228 | 3 | 10 | -7 | 0.0304 | 0.0061 | 0/104 (0.0000) | 1/357 (0.0028) |
| VGGSS-144k | SEMANTIC_SLOT | SAMPLE | 0.4081/0.4169 | 165 | 262 | -97 | 0.9740 | N/A | 103/104 (0.9904) | 103/357 (0.2885) |
| VGGSS-144k | SEMANTIC_SLOT | DISAGREE20_SCALAR | 0.4164/0.4186 | 62 | 116 | -54 | 0.9740 | 0.1948 | 39/104 (0.3750) | 42/357 (0.1176) |
| VGGSS-144k | SEMANTIC_SLOT | DISAGREE20_LOCAL | 0.4153/0.4182 | 64 | 124 | -60 | N/A | 0.2000 | 40/104 (0.3846) | 43/357 (0.1204) |
| VGGSS-144k | RECIPROCAL_L4 | SAMPLE | 0.4137/0.4194 | 153 | 221 | -68 | 0.8813 | N/A | 94/104 (0.9038) | 94/357 (0.2633) |
| VGGSS-144k | RECIPROCAL_L4 | DISAGREE20_SCALAR | 0.4205/0.4202 | 53 | 86 | -33 | 0.8813 | 0.1763 | 35/104 (0.3365) | 36/357 (0.1008) |
| VGGSS-144k | RECIPROCAL_L4 | DISAGREE20_LOCAL | 0.4201/0.4193 | 21 | 56 | -35 | N/A | 0.1208 | 16/104 (0.1538) | 16/357 (0.0448) |
| VGGSS-144k | FIXED_IQR | SAMPLE_AVERAGE | 0.4230/0.4234 | 83 | 103 | -20 | N/A | N/A | N/A | N/A |
| Flickr-144k | CTRL_RAW_PEAK | SAMPLE | 0.8040/0.6166 | 13 | 15 | -2 | 1.0000 | N/A | 11/11 (1.0000) | 11/19 (0.5789) |
| Flickr-144k | CTRL_RAW_PEAK | DISAGREE20_SCALAR | 0.7920/0.6228 | 4 | 9 | -5 | 1.0000 | 0.2000 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | CTRL_NEG_ENTROPY | SAMPLE | 0.8040/0.6188 | 13 | 15 | -2 | 0.9720 | N/A | 11/11 (1.0000) | 11/19 (0.5789) |
| Flickr-144k | CTRL_NEG_ENTROPY | DISAGREE20_SCALAR | 0.7920/0.6240 | 4 | 9 | -5 | 0.9720 | 0.1944 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | CTRL_NEG_AREA | SAMPLE | 0.8040/0.6200 | 13 | 15 | -2 | 0.9200 | N/A | 11/11 (1.0000) | 11/19 (0.5789) |
| Flickr-144k | CTRL_NEG_AREA | DISAGREE20_SCALAR | 0.7920/0.6252 | 4 | 9 | -5 | 0.9200 | 0.1840 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | CTRL_TOP20_CONCENTRATION | SAMPLE | 0.8040/0.6192 | 13 | 15 | -2 | 0.9480 | N/A | 11/11 (1.0000) | 11/19 (0.5789) |
| Flickr-144k | CTRL_TOP20_CONCENTRATION | DISAGREE20_SCALAR | 0.7920/0.6246 | 4 | 9 | -5 | 0.9480 | 0.1896 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | CTRL_NEG_COMPONENTS | SAMPLE | 0.8200/0.6358 | 2 | 0 | 2 | 0.0160 | N/A | 2/11 (0.1818) | 2/19 (0.1053) |
| Flickr-144k | CTRL_NEG_COMPONENTS | DISAGREE20_SCALAR | 0.8160/0.6358 | 1 | 0 | 1 | 0.0160 | 0.0032 | 1/11 (0.0909) | 1/19 (0.0526) |
| Flickr-144k | SEMANTIC_SLOT | SAMPLE | 0.8040/0.6238 | 12 | 14 | -2 | 0.8480 | N/A | 10/11 (0.9091) | 10/19 (0.5263) |
| Flickr-144k | SEMANTIC_SLOT | DISAGREE20_SCALAR | 0.7920/0.6276 | 4 | 9 | -5 | 0.8480 | 0.1696 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | SEMANTIC_SLOT | DISAGREE20_LOCAL | 0.7920/0.6228 | 4 | 9 | -5 | N/A | 0.2000 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | RECIPROCAL_L4 | SAMPLE | 0.8160/0.6324 | 11 | 10 | 1 | 0.6440 | N/A | 10/11 (0.9091) | 10/19 (0.5263) |
| Flickr-144k | RECIPROCAL_L4 | DISAGREE20_SCALAR | 0.8000/0.6336 | 4 | 7 | -3 | 0.6440 | 0.1288 | 4/11 (0.3636) | 4/19 (0.2105) |
| Flickr-144k | RECIPROCAL_L4 | DISAGREE20_LOCAL | 0.7920/0.6256 | 2 | 7 | -5 | N/A | 0.1157 | 2/11 (0.1818) | 2/19 (0.1053) |
| Flickr-144k | FIXED_IQR | SAMPLE_AVERAGE | 0.8040/0.6366 | 2 | 4 | -2 | N/A | N/A | N/A | N/A |

Sample-selector failure decomposition:

- VGGSS-144k CTRL_NEG_COMPONENTS: `{'true_AUD_choice': 2481, 'true_IMG_choice': 73, 'false_AUD_choice': 2520, 'false_IMG_choice': 84}`.
- VGGSS-144k SEMANTIC_SLOT: `{'true_AUD_choice': 62, 'true_IMG_choice': 2521, 'false_AUD_choice': 72, 'false_IMG_choice': 2503}`.
- VGGSS-144k RECIPROCAL_L4: `{'true_AUD_choice': 326, 'true_IMG_choice': 2307, 'false_AUD_choice': 286, 'false_IMG_choice': 2239}`.
- Flickr-144k CTRL_NEG_COMPONENTS: `{'true_AUD_choice': 141, 'true_IMG_choice': 2, 'false_AUD_choice': 105, 'false_IMG_choice': 2}`.
- Flickr-144k SEMANTIC_SLOT: `{'true_AUD_choice': 29, 'true_IMG_choice': 98, 'false_AUD_choice': 9, 'false_IMG_choice': 114}`.
- Flickr-144k RECIPROCAL_L4: `{'true_AUD_choice': 61, 'true_IMG_choice': 79, 'false_AUD_choice': 28, 'false_IMG_choice': 82}`.

## Post-hoc Threshold Transfer

These are non-official diagnostics. The threshold was optimized on the source dataset using GT, then applied unchanged to the other dataset.

| Evidence | Source -> Target | Threshold | Target cIoU/AUC | Rescue/Hurt/Net | IMG rate |
|---|---|---:|---:|---:|---:|
| CTRL_RAW_PEAK | VGGSS-144k -> Flickr-144k | 0.020824 | 0.8120/0.6356 | 0/0/0 | 0.0000 |
| CTRL_RAW_PEAK | Flickr-144k -> VGGSS-144k | 0.017695 | 0.4238/0.4228 | 49/65/-16 | 0.2625 |
| CTRL_NEG_ENTROPY | VGGSS-144k -> Flickr-144k | 0.014996 | 0.8120/0.6356 | 0/0/0 | 0.0000 |
| CTRL_NEG_ENTROPY | Flickr-144k -> VGGSS-144k | 0.005238 | 0.4188/0.4222 | 88/130/-42 | 0.3895 |
| CTRL_NEG_AREA | VGGSS-144k -> Flickr-144k | 0.555923 | 0.8120/0.6356 | 0/0/0 | 0.0000 |
| CTRL_NEG_AREA | Flickr-144k -> VGGSS-144k | 0.463588 | 0.4261/0.4226 | 0/4/-4 | 0.0014 |
| CTRL_TOP20_CONCENTRATION | VGGSS-144k -> Flickr-144k | 0.079671 | 0.8080/0.6356 | 0/1/-1 | 0.0120 |
| CTRL_TOP20_CONCENTRATION | Flickr-144k -> VGGSS-144k | 0.048068 | 0.4219/0.4223 | 58/84/-26 | 0.1565 |
| CTRL_NEG_COMPONENTS | VGGSS-144k -> Flickr-144k | 1.000000 | 0.8120/0.6356 | 0/0/0 | 0.0000 |
| CTRL_NEG_COMPONENTS | Flickr-144k -> VGGSS-144k | 0.000000 | 0.4263/0.4230 | 11/14/-3 | 0.0304 |
| SEMANTIC_SLOT | VGGSS-144k -> Flickr-144k | 0.078253 | 0.8120/0.6360 | 0/0/0 | 0.0120 |
| SEMANTIC_SLOT | Flickr-144k -> VGGSS-144k | 0.025541 | 0.4159/0.4195 | 121/178/-57 | 0.6902 |
| RECIPROCAL_L4 | VGGSS-144k -> Flickr-144k | 0.063707 | 0.8120/0.6354 | 0/0/0 | 0.0040 |
| RECIPROCAL_L4 | Flickr-144k -> VGGSS-144k | 0.017446 | 0.4164/0.4211 | 100/154/-54 | 0.6470 |

## Qualitative Selection

- VGGSS-144k deterministic categories: `{'SELECTOR_WRONG_IMG': '-0BIyqJj9ZU_000030', 'SELECTOR_CORRECT_IMG': '-3Kv4fdm7Uk_000030', 'OGL_RESCUE_MISSED': '-4bPiXbovf0_000008', 'SELECTOR_CORRECT_AUD': '-D64b_8YJK4_000046', 'AUD_ONLY': '-GW1J75oAKU_000304', 'FIXED_IQR_HURT': '-GW1J75oAKU_000304', 'IMG_ONLY': '-Vo4CAMX26U_000030', 'OGL_RESCUE_CAPTURED': '-Vo4CAMX26U_000030', 'FIXED_IQR_RESCUE': '-hYRFCQdbLg_000030'}`.
- Flickr-144k deterministic categories: `{'SELECTOR_WRONG_IMG': '10000130166', 'SELECTOR_CORRECT_IMG': '10007936344', 'AUD_ONLY': '10013411946', 'FIXED_IQR_HURT': '10013411946', 'SELECTOR_CORRECT_AUD': '10035917404', 'IMG_ONLY': '10548273474', 'OGL_RESCUE_CAPTURED': '10548273474', 'FIXED_IQR_RESCUE': '10701841844', 'OGL_RESCUE_MISSED': '10939270325'}`.
- IMG-only examples mostly show a tighter response that removes AUD exterior/context activation. The disagreement image is broad and low-amplitude rather than a clean object-boundary signal.
- Selector failures are not caused by missing AUD/IMG differences: evidence often prefers IMG on samples where IMG merely shifts or shrinks an already incorrect response.
- OGL-rescue misses show that internal slot/reciprocal support can rank AUD and IMG almost equally even when OGL makes the task-relevant extent correction.

## Decision

**Case C - Capacity Exists - No Reliable Self-Supervised Selector Evidence.**

- Sample routing capacity is sufficient to close the observed OGL gap: Sample Oracle is 0.0025 above OGL on VGG and only 0.0040 below OGL on Flickr. Region routing adds 0.0074/0.0160 over Sample Oracle; Pixel Oracle adds larger idealized headroom, but this does not make spatial routing the immediate bottleneck.
- No evidence satisfies the fixed rule. The strongest semantic/reciprocal AUROC on VGG is effectively random, and every semantic/reciprocal zero-threshold method lowers VGG. Reciprocal sample selection improves Flickr from 0.8120 to 0.8160 but lowers VGG to 0.4137.
- The sparse component-count control reaches 0.8200 on Flickr while nearly preserving VGG at 0.4263, but its AUROC and balanced accuracy are approximately random on both datasets. This is not reliable selector evidence.
- Disagreement20 scalar/local correction does not solve the problem: all variants remain below AUD and retain only a minority of IMG's known OGL-rescue capacity.
- Post-hoc threshold transfer is unstable: VGG-derived thresholds mostly collapse to selecting no Flickr samples, while Flickr-derived thresholds generally hurt VGG.

**Next action: stop the current internal-evidence selector line and reconsider the supervision source. Do not train an MLP gate and do not start 4.2 from these signals.**

Test-optimal and transferred thresholds remain diagnostics only and are not counted as official inference methods.
