# Experiment 5.2 - Expansion-Shrink Error Diagnosis

## 1. Audit

- Formal Stage1 and original 1.3G checkpoints, loaders, preprocessing, evaluator, sample order, and cached 5.1 propagation maps are reused unchanged.
- Frozen model inference uses `model.eval()` and `torch.inference_mode()`; no optimizer, backward pass, localization training, or new torch parameter is created.
- The scikit-learn logistic regressions are analysis-only probes evaluated out of fold. They never modify or feed back into AUD/IMG/PROP predictions.

### VGGSS-144k

- Raw reconstruction errors: `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0}`; metric errors: `{'AUD': 0.0, 'IMG': 0.0, 'PROP_F34': 0.0, 'PROP_K34': 0.0, 'OGL': 0.0}`; sample mismatch: `0`.
- Shapes: `AUD/IMG L3=[256, 2, 49]/[256, 2, 49]`, `AUD/IMG L4=[256, 2, 49]/[256, 2, 49]`, `AUD_FINE=[256, 1, 14, 14]`, `F34=[256, 512, 14, 14]`, `K34=[256, 196, 512]`.
- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`, `parameters_with_grad=[]`.
- Checkpoint SHA256/mtime unchanged: `True`; NaN/Inf: `False`.

### Flickr-144k

- Raw reconstruction errors: `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0}`; metric errors: `{'AUD': 0.0, 'IMG': 0.0, 'PROP_F34': 0.0, 'PROP_K34': 0.0, 'OGL': 0.0}`; sample mismatch: `0`.
- Shapes: `AUD/IMG L3=[32, 2, 49]/[32, 2, 49]`, `AUD/IMG L4=[32, 2, 49]/[32, 2, 49]`, `AUD_FINE=[32, 1, 14, 14]`, `F34=[32, 512, 14, 14]`, `K34=[32, 196, 512]`.
- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`, `parameters_with_grad=[]`.
- Checkpoint SHA256/mtime unchanged: `True`; NaN/Inf: `False`.

## 2. Error-Type Definition

- EXPAND candidates: `per-pixel max(AUD_norm, PROP_F34_norm); cannot delete an AUD-positive pixel` and `per-pixel max(AUD_norm, PROP_K34_norm); cannot delete an AUD-positive pixel`; GT selects the better fixed candidate only for oracle labeling.
- SHRINK candidate: `binary(AUD_norm>=0.6) intersection binary(IMG_norm>=0.6); cannot add outside AUD`.
- Beneficial threshold: absolute IoU gain `>=0.01`.
- Strict three-class rule: `EXPAND/SHRINK requires gain>=0.01 and >= opposite gain+0.01; all other samples are KEEP_AMBIGUOUS`.
- The threshold and dominance margin were fixed before the formal run and were not tuned per dataset.

## 3. Dataset Distribution

| Dataset | Expand only | Shrink only | Both beneficial | Neither | Strict Expand | Strict Shrink | Keep/Ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | 1094 (21.2%) | 1986 (38.5%) | 97 (1.9%) | 1981 (38.4%) | 1130 (21.9%) | 2012 (39.0%) | 2016 (39.1%) |
| Flickr-144k | 55 (22.0%) | 96 (38.4%) | 5 (2.0%) | 94 (37.6%) | 57 (22.8%) | 99 (39.6%) | 94 (37.6%) |

The strict distributions are almost identical across datasets: roughly 22% EXPAND, 39% SHRINK, and 38-39% KEEP/AMBIGUOUS. This is strong evidence that the bidirectional error structure is not a VGG-only artifact.

## 4. Spatial Disagreement Statistics

| Dataset | Type | IMG/AUD area | AUD-only/AUD | IMG-only/IMG | AUD-IMG mask IoU |
|---|---|---:|---:|---:|---:|
| VGGSS-144k | EXPAND | 0.9051 | 0.1444 | 0.0458 | 0.8171 |
| VGGSS-144k | SHRINK | 0.8504 | 0.1754 | 0.0271 | 0.8036 |
| VGGSS-144k | KEEP_AMBIGUOUS | 0.9301 | 0.1231 | 0.0493 | 0.8332 |
| Flickr-144k | EXPAND | 0.8264 | 0.1971 | 0.0253 | 0.7840 |
| Flickr-144k | SHRINK | 0.8396 | 0.1714 | 0.0119 | 0.8194 |
| Flickr-144k | KEEP_AMBIGUOUS | 0.8529 | 0.1686 | 0.0215 | 0.8134 |

- VGG SHRINK samples show the expected geometry: smaller IMG/AUD area (`0.8504`) and larger AUD-only fraction (`0.1754`) than EXPAND (`0.9051`, `0.1444`).
- Flickr does not preserve that ordering: EXPAND has a smaller IMG/AUD ratio and larger AUD-only fraction than SHRINK. Area disagreement alone therefore cannot identify which removed pixels are context rather than true extent.
- Across both datasets, SHRINK tends to have very little IMG-only addition. The most stable geometry signal is low `IMG_only_area`, but its EXPAND-vs-SHRINK AUROC is only about `0.60`.

## 5. Internal Signals

| Dataset | Type | AUD-only F34 support | AUD-only K34 support | Common-minus-extra F34 | Common-minus-extra K34 |
|---|---|---:|---:|---:|---:|
| VGGSS-144k | EXPAND | 0.7079 | 0.7590 | 0.1598 | 0.1348 |
| VGGSS-144k | SHRINK | 0.6936 | 0.7491 | 0.1798 | 0.1507 |
| VGGSS-144k | KEEP_AMBIGUOUS | 0.7153 | 0.7653 | 0.1543 | 0.1309 |
| Flickr-144k | EXPAND | 0.7333 | 0.7606 | 0.1339 | 0.1296 |
| Flickr-144k | SHRINK | 0.7109 | 0.7409 | 0.1645 | 0.1559 |
| Flickr-144k | KEEP_AMBIGUOUS | 0.7262 | 0.7549 | 0.1384 | 0.1339 |

AUD-only regions in SHRINK samples receive slightly weaker agreement-prototype support in both F34 and K34. The direction transfers, but the separation is modest on VGG and does not form a standalone reliable router.

## 6. Single-Variable Diagnostics

Best same-direction signals that remain near the top on both datasets:

| Task | Signal | Positive direction | VGG AUROC | Flickr AUROC |
|---|---|---|---:|---:|
| EXPAND_vs_SHRINK | IMG_only_area | lower | 0.6035 | 0.6026 |
| EXPAND_vs_SHRINK | IMG_L3_L4_Spearman | higher | 0.6147 | 0.5966 |
| EXPAND_vs_SHRINK | IMG_only_ratio | lower | 0.5964 | 0.6009 |
| SHRINK_vs_OTHERS | IMG_only_area | lower | 0.6283 | 0.5904 |
| SHRINK_vs_OTHERS | IMG_only_ratio | lower | 0.6224 | 0.5851 |
| SHRINK_vs_OTHERS | REGION_DELTA_PROP_F34_similarity | higher | 0.5830 | 0.6483 |
| EXPAND_vs_OTHERS | IMG_L3_L4_Spearman | lower | 0.6080 | 0.5754 |
| EXPAND_vs_OTHERS | L3_ownership_entropy | higher | 0.5656 | 0.5785 |
| EXPAND_vs_OTHERS | IMG_L3_L4_mask_IoU | lower | 0.5638 | 0.5850 |

No single signal is strong on both datasets. The best stable EXPAND-vs-SHRINK diagnostics remain around `0.60 AUROC`; dataset-specific peaks reach about `0.65-0.66` but use different signals.

## 7. Lightweight Probe

All results are out-of-fold. VGG uses video-id grouped folds; Flickr uses sample-id groups. Imputation and standardization are fitted only inside each training fold.

| Dataset | Features | Task | AUROC | Balanced acc | Macro F1 |
|---|---|---|---:|---:|---:|
| VGGSS-144k | SPATIAL | EXPAND_vs_SHRINK | 0.6389 | 0.6113 | 0.6042 |
| VGGSS-144k | SPATIAL | SHRINK_vs_OTHERS | 0.6574 | 0.6176 | 0.6022 |
| VGGSS-144k | SPATIAL | EXPAND_vs_OTHERS | 0.5662 | 0.5527 | 0.4990 |
| VGGSS-144k | INTERNAL | EXPAND_vs_SHRINK | 0.7593 | 0.6946 | 0.6853 |
| VGGSS-144k | INTERNAL | SHRINK_vs_OTHERS | 0.7026 | 0.6446 | 0.6344 |
| VGGSS-144k | INTERNAL | EXPAND_vs_OTHERS | 0.7135 | 0.6538 | 0.6060 |
| VGGSS-144k | ALL | EXPAND_vs_SHRINK | 0.7664 | 0.6996 | 0.6918 |
| VGGSS-144k | ALL | SHRINK_vs_OTHERS | 0.7145 | 0.6546 | 0.6423 |
| VGGSS-144k | ALL | EXPAND_vs_OTHERS | 0.7160 | 0.6629 | 0.6157 |
| VGGSS-144k | ALL | THREE_CLASS | N/A | 0.5179 | 0.5057 |
| Flickr-144k | SPATIAL | EXPAND_vs_SHRINK | 0.5609 | 0.5789 | 0.5775 |
| Flickr-144k | SPATIAL | SHRINK_vs_OTHERS | 0.5499 | 0.5197 | 0.5067 |
| Flickr-144k | SPATIAL | EXPAND_vs_OTHERS | 0.4850 | 0.4694 | 0.4669 |
| Flickr-144k | INTERNAL | EXPAND_vs_SHRINK | 0.6211 | 0.6063 | 0.5969 |
| Flickr-144k | INTERNAL | SHRINK_vs_OTHERS | 0.6168 | 0.5676 | 0.5617 |
| Flickr-144k | INTERNAL | EXPAND_vs_OTHERS | 0.5719 | 0.5344 | 0.5172 |
| Flickr-144k | ALL | EXPAND_vs_SHRINK | 0.6055 | 0.5598 | 0.5516 |
| Flickr-144k | ALL | SHRINK_vs_OTHERS | 0.6052 | 0.5676 | 0.5617 |
| Flickr-144k | ALL | EXPAND_vs_OTHERS | 0.5491 | 0.5178 | 0.5019 |
| Flickr-144k | ALL | THREE_CLASS | N/A | 0.3811 | 0.3827 |

- VGG has a real combined signal: ALL EXPAND-vs-SHRINK AUROC `0.7664`, balanced accuracy `0.6996`.
- Flickr is only moderate: its best EXPAND-vs-SHRINK result is INTERNAL AUROC `0.6211`, balanced accuracy `0.6063`; adding all spatial features lowers it.
- Three-class routing is weak: balanced accuracy `0.5179` on VGG and `0.3811` on Flickr.
- The signal combination is therefore diagnostic, not reliable enough for a general adaptive router.

## 8. Oracle Routing Upper Bound

| Dataset | AUD | Expand-only oracle | Shrink-only oracle | Combined oracle | Combined gain |
|---|---:|---:|---:|---:|---:|
| VGGSS-144k | 0.4269/0.4230 | 0.4604/0.4498 | 0.4591/0.4425 | 0.4913/0.4689 | +0.0644/+0.0460 |
| Flickr-144k | 0.8120/0.6356 | 0.8240/0.6480 | 0.8640/0.6606 | 0.8760/0.6724 | +0.0640/+0.0368 |

- VGG: expansion and shrink independently contribute about `+.0335` and `+.0322 cIoU`; combined capacity reaches `.4913/.4689`.
- Flickr: shrink is dominant (`+.0520 cIoU`) but expansion still adds `+.0120`; combined capacity reaches `.8760/.6724`.
- Bidirectional correction capacity is substantial and consistently larger than either one-direction oracle.

## 9. Relationship With Experiment 5.1

| Dataset | 5.1 group | Count | Expand | Shrink | Keep/Ambiguous |
|---|---|---:|---:|---:|---:|
| VGGSS-144k | PROP_ONLY | 210 | 177 | 26 | 7 |
| VGGSS-144k | IMG_ONLY_SHRINK | 125 | 1 | 123 | 1 |
| VGGSS-144k | AUD_OVER_EXPANSION | 4111 | 271 | 2003 | 1837 |
| VGGSS-144k | PROP_HURT | 480 | 4 | 228 | 248 |
| VGGSS-144k | OGL_RESCUE | 357 | 44 | 235 | 78 |
| Flickr-144k | PROP_ONLY | 4 | 3 | 1 | 0 |
| Flickr-144k | IMG_ONLY_SHRINK | 10 | 0 | 10 | 0 |
| Flickr-144k | AUD_OVER_EXPANSION | 203 | 28 | 97 | 78 |
| Flickr-144k | PROP_HURT | 19 | 0 | 13 | 6 |
| Flickr-144k | OGL_RESCUE | 19 | 3 | 16 | 0 |

- PROP_ONLY maps primarily to EXPAND: `177/210` VGG and `3/4` Flickr.
- IMG_ONLY+SHRINK maps almost perfectly to SHRINK: `123/125` VGG and `10/10` Flickr.
- OGL rescues are mostly SHRINK: `235/357` VGG and `16/19` Flickr.
- PROP hurt is overwhelmingly SHRINK or KEEP: only `4/480` VGG and `0/19` Flickr are EXPAND.
- This directly explains 5.1: unconditional propagation helps the EXPAND minority but damages samples that require deletion or no change.

## 10. Failure-Case Analysis

### VGGSS-144k

- `MAX_EXPAND_GAIN` `MPe6ztPtF0Y_000030`: true `EXPAND`, probe `KEEP_AMBIGUOUS`, AUD `0.1628`, expand gain `0.6606`, shrink gain `0.0000`, IMG/AUD area `1.0740`.
- `MAX_SHRINK_GAIN` `Bpw53tN6h8E_000030`: true `SHRINK`, probe `EXPAND`, AUD `0.3887`, expand gain `0.0000`, shrink gain `0.3750`, IMG/AUD area `0.5031`.
- `EXPAND_MISROUTED_AS_SHRINK` `VAfO711tnQA_000030`: true `EXPAND`, probe `SHRINK`, AUD `0.4687`, expand gain `0.4008`, shrink gain `0.0000`, IMG/AUD area `0.8906`.
- `SHRINK_MISROUTED_AS_EXPAND` `Bpw53tN6h8E_000030`: true `SHRINK`, probe `EXPAND`, AUD `0.3887`, expand gain `0.0000`, shrink gain `0.3750`, IMG/AUD area `0.5031`.

### Flickr-144k

- `MAX_EXPAND_GAIN` `4407899725`: true `EXPAND`, probe `SHRINK`, AUD `0.6523`, expand gain `0.1338`, shrink gain `0.0000`, IMG/AUD area `0.6962`.
- `MAX_SHRINK_GAIN` `4180455681`: true `SHRINK`, probe `SHRINK`, AUD `0.4260`, expand gain `0.0000`, shrink gain `0.2271`, IMG/AUD area `0.6049`.
- `EXPAND_MISROUTED_AS_SHRINK` `4407899725`: true `EXPAND`, probe `SHRINK`, AUD `0.6523`, expand gain `0.1338`, shrink gain `0.0000`, IMG/AUD area `0.6962`.
- `SHRINK_MISROUTED_AS_EXPAND` `2897653916`: true `SHRINK`, probe `EXPAND`, AUD `0.3170`, expand gain `0.0000`, shrink gain `0.1915`, IMG/AUD area `0.6335`.

The misrouted examples show the core ambiguity: a large AUD-only region can be missing extent in one sample and removable context in another. IMG suppression is a valid operation, but the frozen model does not consistently encode the semantic status of the suppressed pixels.

## 11. Final Decision

**Case B - Error Structure Exists but Routing Signal Is Weak.**

EXPAND and SHRINK are stable, complementary failure modes with a large combined oracle upper bound. IMG is a highly effective SHRINK candidate in the known IMG_ONLY+SHRINK and OGL-rescue groups. However, IMG/AUD area disagreement is not directionally stable across datasets, the strongest transferable single signals are only around 0.60 AUROC, and the three-class probe is weak, especially on Flickr.

The evidence does not support implementing a complex adaptive expand/shrink localization module yet.

## 12. Recommended Next Experiment

Continue only with a narrowly scoped routing-cue study that targets the semantic status of AUD-only pixels: true object extent versus context leakage. Do not resume unconditional prototype propagation, and do not treat IMG area or disagreement magnitude alone as a SHRINK gate.

No Experiment 5.3 was started.
