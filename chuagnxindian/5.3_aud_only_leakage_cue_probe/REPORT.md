# Experiment 5.3 - AUD-Only Leakage Cue Probe

## 1. Audit

- Frozen Stage1/original 1.3G inference only; no optimizer, backward pass, trainable localization parameter, checkpoint modification, or correction module.
- Official evaluator reproduction uses the original GT values. Candidate-independent TP/FP/FN diagnosis and pixel labels use fixed binary `GT >= 0.5`.
- Pixel folds are `sha256(seed, sample_id) mod 5`; every pixel from one sample stays in one fold.

### VGGSS-144k

- Raw errors `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0, 'PROP_F34': 0.0, 'PROP_K34': 0.0}`; metric errors `{'AUD': 0.0, 'IMG': 0.0, 'PROP_F34': 0.0, 'PROP_K34': 0.0}`; sample mismatch `0`.
- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`, `parameters_with_grad=[]`.
- Checkpoint unchanged `True`; NaN/Inf `False`.
- Saved pixels `105425`; balanced probe pixels `22309/22309` (TRUE_EXTENT/LEAKAGE).

### Flickr-144k

- Raw errors `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0, 'PROP_F34': 0.0, 'PROP_K34': 0.0}`; metric errors `{'AUD': 0.0, 'IMG': 0.0, 'PROP_F34': 0.0, 'PROP_K34': 0.0}`; sample mismatch `0`.
- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`, `parameters_with_grad=[]`.
- Checkpoint unchanged `True`; NaN/Inf `False`.
- Saved pixels `5679`; balanced probe pixels `1355/1355` (TRUE_EXTENT/LEAKAGE).

## 2. Candidate-Independent Intrinsic Diagnosis

`IoU_expand*=|GT|/(|GT|+FP)` and `IoU_shrink*=TP/(TP+FN)` use only AUD and binary GT. Gain threshold and dominance margin are both fixed at `.01`.

| Dataset | Intrinsic Expand | Intrinsic Shrink | Mixed | Keep |
|---|---:|---:|---:|---:|
| VGGSS-144k | 1152 (22.3%) | 3928 (76.2%) | 73 (1.4%) | 5 (0.1%) |
| Flickr-144k | 55 (22.0%) | 191 (76.4%) | 4 (1.6%) | 0 (0.0%) |

## 3. Agreement With Experiment 5.2

| Dataset | Agreement | Candidate EXPAND P/R | Candidate SHRINK P/R |
|---|---:|---:|---:|
| VGGSS-144k | 0.5733 | 0.8133/0.7977 | 0.9901/0.5071 |
| Flickr-144k | 0.5320 | 0.5789/0.6000 | 0.9798/0.5079 |

The bidirectional structure survives: both intrinsic EXPAND and SHRINK exist at nearly identical rates across datasets. The 5.2 SHRINK candidate is very high precision (`.99/.98`) but only about `.51` recall, so many candidate KEEP cases are intrinsically shrink-beneficial rather than truly KEEP.

## 4. AUD-Only Pixel Composition

| Dataset / intrinsic type | AUD-only pixels | True extent | Context leakage | Macro leakage ratio |
|---|---:|---:|---:|---:|
| VGGSS-144k / ALL | 20467569 | 0.3099 | 0.6901 | 0.6843 |
| VGGSS-144k / INTRINSIC_EXPAND | 3748011 | 0.7749 | 0.2251 | 0.2419 |
| VGGSS-144k / INTRINSIC_SHRINK | 16413086 | 0.1996 | 0.8004 | 0.8176 |
| VGGSS-144k / MIXED_AMBIGUOUS | 276460 | 0.5917 | 0.4083 | 0.4396 |
| VGGSS-144k / KEEP | 30012 | 0.0000 | 1.0000 | 1.0000 |
| Flickr-144k / ALL | 1415864 | 0.4931 | 0.5069 | 0.5173 |
| Flickr-144k / INTRINSIC_EXPAND | 305420 | 0.7609 | 0.2391 | 0.2249 |
| Flickr-144k / INTRINSIC_SHRINK | 1058104 | 0.3953 | 0.6047 | 0.6103 |
| Flickr-144k / MIXED_AMBIGUOUS | 52340 | 0.9075 | 0.0925 | 0.1416 |
| Flickr-144k / KEEP | 0 | N/A | N/A | N/A |

The target distinction is real and strong in GT space: intrinsic SHRINK AUD-only pixels are `80.0%/60.5%` leakage on VGG/Flickr, while intrinsic EXPAND pixels are only `22.5%/23.9%` leakage. The remaining question is whether frozen features expose this distinction without GT.

## 5. Single-Variable Pixel Signals

Top same-direction signals across both datasets:

| Signal | Direction | VGG AUROC | Flickr AUROC |
|---|---|---:|---:|
| K34_IMG_core_similarity | lower=leakage | 0.5942 | 0.6335 |
| K34_agreement_core_similarity | lower=leakage | 0.5891 | 0.6348 |
| F34_IMG_core_similarity | lower=leakage | 0.5889 | 0.6358 |
| K34_AUD_core_similarity | lower=leakage | 0.5851 | 0.6395 |
| F34_agreement_core_similarity | lower=leakage | 0.5830 | 0.6380 |
| F34_AUD_core_similarity | lower=leakage | 0.5787 | 0.6476 |
| AUD_norm_score | lower=leakage | 0.5686 | 0.6152 |
| AUD_threshold_distance | lower=leakage | 0.5686 | 0.6152 |
| F34_local_similarity | lower=leakage | 0.5615 | 0.5978 |
| K34_local_similarity | lower=leakage | 0.5590 | 0.5864 |

The strongest transferable single cue is lower K34 similarity to the IMG-supported core (`.5942/.6335` AUROC). All single cues remain weak-to-moderate; no non-prototype or prototype signal is independently reliable.

## 6. Pixel Linear Probe

All OOF metrics use sample-disjoint folds. Probe-selected pixels are exactly balanced within every mixed-label sample.

| Dataset | Features | AUROC | AUPRC | Balanced acc | F1 |
|---|---|---:|---:|---:|---:|
| VGGSS-144k | PREDICTION | 0.5945 | 0.5785 | 0.5647 | 0.5727 |
| VGGSS-144k | WITHOUT_PROTOTYPE | 0.6584 | 0.6484 | 0.6123 | 0.6099 |
| VGGSS-144k | WITH_PROTOTYPE | 0.6615 | 0.6548 | 0.6126 | 0.6095 |
| VGGSS-144k | PROTOTYPE_ONLY | 0.5954 | 0.5996 | 0.5660 | 0.5374 |
| Flickr-144k | PREDICTION | 0.6150 | 0.5833 | 0.5871 | 0.6203 |
| Flickr-144k | WITHOUT_PROTOTYPE | 0.6599 | 0.6362 | 0.6181 | 0.6218 |
| Flickr-144k | WITH_PROTOTYPE | 0.6786 | 0.6698 | 0.6321 | 0.6287 |
| Flickr-144k | PROTOTYPE_ONLY | 0.6480 | 0.6544 | 0.6159 | 0.5954 |

Adding prototype cues changes AUROC only from `.6584` to `.6615` on VGG and `.6599` to `.6786` on Flickr. The signal is not a disguised replay of prototype propagation, but its absolute reliability remains moderate.

## 7. Cross-Dataset Transfer

Target normalization and thresholds are never fitted or tuned. Threshold remains `.5`.

| Direction | Features | AUROC | AUPRC | Balanced acc | F1 |
|---|---|---:|---:|---:|---:|
| vggss_144k_to_flickr_144k | PREDICTION | 0.6045 | 0.5715 | 0.5786 | 0.5871 |
| vggss_144k_to_flickr_144k | WITHOUT_PROTOTYPE | 0.6380 | 0.6180 | 0.5952 | 0.6357 |
| vggss_144k_to_flickr_144k | WITH_PROTOTYPE | 0.6544 | 0.6375 | 0.6052 | 0.6575 |
| vggss_144k_to_flickr_144k | PROTOTYPE_ONLY | 0.6293 | 0.6204 | 0.5166 | 0.6622 |
| flickr_144k_to_vggss_144k | PREDICTION | 0.5763 | 0.5607 | 0.5545 | 0.6232 |
| flickr_144k_to_vggss_144k | WITHOUT_PROTOTYPE | 0.5984 | 0.5866 | 0.5683 | 0.5720 |
| flickr_144k_to_vggss_144k | WITH_PROTOTYPE | 0.6017 | 0.5997 | 0.5700 | 0.5409 |
| flickr_144k_to_vggss_144k | PROTOTYPE_ONLY | 0.5748 | 0.5835 | 0.5401 | 0.2916 |

Pixel AUROC transfers partially rather than collapsing: WITH_PROTOTYPE is `.6544` VGG-to-Flickr and `.6017` Flickr-to-VGG. Calibration is weaker, especially in the Flickr-to-VGG direction, but the main failure appears after pixel scores are aggregated into a routing decision.

## 8. Aggregated Sample-Level Routing

The score is the fraction of deterministic AUD-only routing pixels predicted as leakage at threshold `.5`.

| Dataset | Pixel score source | Features | Shrink-vs-non AUROC | Expand-vs-shrink AUROC |
|---|---|---|---:|---:|
| VGGSS-144k | OOF | PREDICTION | 0.5011 | 0.5038 |
| VGGSS-144k | OOF | WITHOUT_PROTOTYPE | 0.5722 | 0.5744 |
| VGGSS-144k | OOF | WITH_PROTOTYPE | 0.5649 | 0.5679 |
| VGGSS-144k | OOF | PROTOTYPE_ONLY | 0.5167 | 0.5220 |
| VGGSS-144k | cross-transfer | PREDICTION | 0.4605 | 0.4608 |
| VGGSS-144k | cross-transfer | WITHOUT_PROTOTYPE | 0.5065 | 0.5075 |
| VGGSS-144k | cross-transfer | WITH_PROTOTYPE | 0.4942 | 0.4972 |
| VGGSS-144k | cross-transfer | PROTOTYPE_ONLY | 0.5333 | 0.5351 |
| Flickr-144k | OOF | PREDICTION | 0.5237 | 0.5208 |
| Flickr-144k | OOF | WITHOUT_PROTOTYPE | 0.5896 | 0.5944 |
| Flickr-144k | OOF | WITH_PROTOTYPE | 0.6027 | 0.6067 |
| Flickr-144k | OOF | PROTOTYPE_ONLY | 0.6874 | 0.7011 |
| Flickr-144k | cross-transfer | PREDICTION | 0.5777 | 0.5650 |
| Flickr-144k | cross-transfer | WITHOUT_PROTOTYPE | 0.6965 | 0.6945 |
| Flickr-144k | cross-transfer | WITH_PROTOTYPE | 0.6732 | 0.6733 |
| Flickr-144k | cross-transfer | PROTOTYPE_ONLY | 0.5960 | 0.5929 |

The pixel cue does not solve the 5.2 routing problem. WITH_PROTOTYPE OOF shrink-routing AUROC is only `.5649` on VGG and `.6027` on Flickr; direct transfer to VGG falls to `.4942`.

## 9. Mapping To Experiments 5.1/5.2

### VGGSS-144k

| Group | Count | Intrinsic Expand | Intrinsic Shrink | Mixed | Keep | True leakage | Predicted leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| PROP_ONLY | 210 | 152 | 58 | 0 | 0 | 0.3499 | 0.4901 |
| PROP_HURT | 480 | 38 | 437 | 5 | 0 | 0.7058 | 0.5264 |
| IMG_ONLY_SHRINK | 125 | 2 | 123 | 0 | 0 | 0.9165 | 0.6233 |
| OGL_RESCUE | 357 | 38 | 315 | 4 | 0 | 0.8232 | 0.5592 |
| AUD_OVER_EXPANSION | 4111 | 105 | 3928 | 73 | 5 | 0.8033 | 0.5422 |
| CANDIDATE_EXPAND | 1130 | 919 | 183 | 28 | 0 | 0.2721 | 0.4769 |
| CANDIDATE_SHRINK | 2012 | 16 | 1992 | 4 | 0 | 0.9317 | 0.5774 |

### Flickr-144k

| Group | Count | Intrinsic Expand | Intrinsic Shrink | Mixed | Keep | True leakage | Predicted leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| PROP_ONLY | 4 | 1 | 3 | 0 | 0 | 0.6208 | 0.1667 |
| PROP_HURT | 19 | 1 | 18 | 0 | 0 | 0.7347 | 0.4035 |
| IMG_ONLY_SHRINK | 10 | 0 | 10 | 0 | 0 | 0.9090 | 0.4833 |
| OGL_RESCUE | 19 | 0 | 19 | 0 | 0 | 0.8773 | 0.5789 |
| AUD_OVER_EXPANSION | 203 | 8 | 191 | 4 | 0 | 0.5921 | 0.5004 |
| CANDIDATE_EXPAND | 57 | 33 | 23 | 1 | 0 | 0.2483 | 0.4108 |
| CANDIDATE_SHRINK | 99 | 2 | 97 | 0 | 0 | 0.8084 | 0.5463 |

## 10. Failure Analysis

### VGGSS-144k

- `SHRINK_LOWEST_PREDICTED_LEAKAGE` `-2sOH8XovEE_000484`: intrinsic `INTRINSIC_SHRINK`, 5.2 `KEEP_AMBIGUOUS`, true leakage `1.0000`, predicted `0.0000`.
- `EXPAND_HIGHEST_PREDICTED_LEAKAGE` `z_WasI8m3iM_000012`: intrinsic `INTRINSIC_EXPAND`, 5.2 `EXPAND`, true leakage `0.7191`, predicted `1.0000`.

### Flickr-144k

- `SHRINK_LOWEST_PREDICTED_LEAKAGE` `10045181004`: intrinsic `INTRINSIC_SHRINK`, 5.2 `KEEP_AMBIGUOUS`, true leakage `0.5161`, predicted `0.0000`.
- `EXPAND_HIGHEST_PREDICTED_LEAKAGE` `7740990330`: intrinsic `INTRINSIC_EXPAND`, 5.2 `SHRINK`, true leakage `0.6346`, predicted `1.0000`.

## 11. Final Decision

**Case C - Weak Pixel Signal.**

Even direct AUD-only pixel supervision does not produce a consistently strong frozen linear signal across VGG and Flickr, and aggregation does not resolve sample-level SHRINK routing.

## 12. Research-Line Decision

Stop the hand-designed adaptive expand/shrink routing line; do not start 5.4 from the current frozen cues.

No Experiment 5.4 was started.
