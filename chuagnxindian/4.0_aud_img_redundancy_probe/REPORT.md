# Experiment 4.0 - AUD-IMG Redundancy & Attention-Loss Mechanism Probe

## Protocol And Att-Loss Audit

This is a zero-training diagnostic on formal L3+L4 Stage1 and original 1.3G. All maps use the unchanged bicubic resize, per-sample min-max normalization, threshold 0.6, cIoU, and AUC evaluator.

The formal Stage1 code computes:

```python
att_loss = MSE(audq_imgk_attn[:, 0], imgq_imgk_attn[:, 0].detach())
att_loss += MSE(imgq_audk_attn[:, 0], audq_audk_attn[:, 0].detach())
total = info + lam1 * recon + lam2 * div + lam3 * att_loss
```

- `lam3 = 100.0` in both formal 144k configs.
- Spatial term: `AUD query -> L4 image keys` is optimized toward detached `IMG query -> L4 image keys`.
- Reciprocal audio-token term: `IMG query -> audio keys` is optimized toward detached `AUD query -> audio keys`.
- The spatial term directly contains the same L4 AUD/IMG attention tensors used by formal Stage1 localization, with scale multiplier 1.0 during training versus infer sharpening 0.1 during evaluation.
- Source: `model_mufasa_jsa.py:113-120`, `l3_l4_slot_attention.py:102-120`, `train_slot.py:339-343`.

## Tensor And Reproduction Audit

| Setting | Qa | Qv | K4 | AUD_L4 | IMG_L4 | K34 | AUD_FINE | max tensor error | max evaluator error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | [256, 2, 512] | [256, 2, 512] | [256, 49, 512] | [256, 1, 7, 7] | [256, 1, 7, 7] | [256, 196, 512] | [256, 1, 14, 14] | 0.000e+00 | 4.719e-05 |
| flickr_144k | [32, 2, 512] | [32, 2, 512] | [32, 49, 512] | [32, 1, 7, 7] | [32, 1, 7, 7] | [32, 196, 512] | [32, 1, 14, 14] | 0.000e+00 | 2.220e-16 |

`optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`; all models remained in eval/inference mode. Every used checkpoint SHA256 and mtime was identical before and after.

Direct best-checkpoint discrepancy using the exact training MSE tensors:

| Setting | Spatial MSE | Reciprocal audio-token MSE | Total att MSE |
|---|---:|---:|---:|
| vggss_144k | 2.623e-05 | 3.838e-05 | 6.461e-05 |
| flickr_144k | 9.978e-06 | 7.367e-05 | 8.365e-05 |

Historical audit: VGG best-checkpoint evaluation exactly agrees with epoch 3. For Flickr, epoch 7 training CSV records IQR `0.8120/0.6236`, while the independent formal best-checkpoint test log and this exact reconstruction both give `0.8080/0.6234`. AUD, IMG, OBJ, and OGL agree; this pre-existing one-sample IQR difference is retained rather than hidden.

## Final Redundancy

| Setting | Stage | AUD cIoU/AUC | IMG cIoU/AUC | IQR cIoU/AUC | OBJ cIoU/AUC | OGL cIoU/AUC | Pearson | Spearman | JS | Top20 | Mask IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | Stage1 | 0.4002/0.4127 | 0.4069/0.4166 | 0.4069/0.4160 | 0.3478/0.3924 | 0.4343/0.4307 | 0.9725 | 0.9680 | 0.0025 | 0.7809 | 0.8580 |
| vggss_144k | Stage2 | 0.4269/0.4230 | 0.4069/0.4166 | 0.4230/0.4234 | 0.3478/0.3924 | 0.4570/0.4401 | 0.9613 | 0.9580 | 0.0049 | 0.7494 | 0.8181 |
| flickr_144k | Stage1 | 0.8040/0.6228 | 0.8040/0.6166 | 0.8080/0.6234 | 0.4480/0.4668 | 0.8440/0.6392 | 0.9709 | 0.9605 | 0.0020 | 0.7024 | 0.8678 |
| flickr_144k | Stage2 | 0.8120/0.6356 | 0.8040/0.6166 | 0.8040/0.6366 | 0.4480/0.4668 | 0.8680/0.6596 | 0.9528 | 0.9387 | 0.0042 | 0.6331 | 0.8091 |

Raw-space Pearson/Spearman equal evaluator-normalized values because independent min-max normalization is a positive affine transform. Raw and normalized JS are both retained in `stage_summaries.json`.

## Task Complementarity

| Setting | Stage | Pair | AUD only | AUX only | Both success | Both fail | Oracle cIoU/AUC | Oracle gain | Fixed gain | Rescue | Hurt | Net |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | Stage1 | AUD+IMG | 147 | 182 | 1917 | 2912 | 0.4354/0.4307 | +0.0353 | +0.0068 | 95 | 60 | 35 |
| vggss_144k | Stage1 | AUD+OBJ | 921 | 651 | 1143 | 2443 | 0.5264/0.4695 | +0.1262 | +0.0341 | 367 | 191 | 176 |
| vggss_144k | Stage2 | AUD+IMG | 271 | 168 | 1931 | 2788 | 0.4595/0.4423 | +0.0326 | -0.0039 | 83 | 103 | -20 |
| vggss_144k | Stage2 | AUD+OBJ | 1035 | 627 | 1167 | 2329 | 0.5485/0.4807 | +0.1216 | +0.0301 | 357 | 202 | 155 |
| flickr_144k | Stage1 | AUD+IMG | 8 | 8 | 193 | 41 | 0.8360/0.6400 | +0.0320 | +0.0040 | 1 | 0 | 1 |
| flickr_144k | Stage1 | AUD+OBJ | 106 | 17 | 95 | 32 | 0.8720/0.6546 | +0.0680 | +0.0400 | 17 | 7 | 10 |
| flickr_144k | Stage2 | AUD+IMG | 15 | 13 | 188 | 34 | 0.8640/0.6598 | +0.0520 | -0.0080 | 2 | 4 | -2 |
| flickr_144k | Stage2 | AUD+OBJ | 112 | 21 | 91 | 26 | 0.8960/0.6706 | +0.0840 | +0.0560 | 19 | 5 | 14 |

## OGL Rescue Decomposition

| Setting | Stage | OGL rescue pool | IMG captured | IMG rate | IQR captured | IQR rate | IMG IoU > AUD |
|---|---|---:|---:|---:|---:|---:|---:|
| vggss_144k | Stage1 | 367 | 88 | 0.240 | 52 | 0.142 | 0.559 |
| vggss_144k | Stage2 | 357 | 104 | 0.291 | 56 | 0.157 | 0.681 |
| flickr_144k | Stage1 | 17 | 8 | 0.471 | 1 | 0.059 | 0.882 |
| flickr_144k | Stage2 | 19 | 11 | 0.579 | 1 | 0.053 | 0.895 |

## AUD-IMG Versus AUD-OBJ

| Setting | Stage | Pair | Pearson | Spearman | JS | Top10 | Top20 | Top30 | Mask IoU | AUX-only frac | Oracle gain | Fusion gain |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | Stage1 | AUD+IMG | 0.9725 | 0.9680 | 0.0025 | 0.6891 | 0.7809 | 0.8302 | 0.8580 | 0.0353 | +0.0353 | +0.0068 |
| vggss_144k | Stage1 | AUD+OBJ | 0.7801 | 0.7870 | 0.0225 | 0.3408 | 0.4896 | 0.5813 | 0.5074 | 0.1262 | +0.1262 | +0.0341 |
| vggss_144k | Stage2 | AUD+IMG | 0.9613 | 0.9580 | 0.0049 | 0.6373 | 0.7494 | 0.8059 | 0.8181 | 0.0326 | +0.0326 | -0.0039 |
| vggss_144k | Stage2 | AUD+OBJ | 0.8052 | 0.8143 | 0.0231 | 0.3853 | 0.5199 | 0.6049 | 0.4964 | 0.1216 | +0.1216 | +0.0301 |
| flickr_144k | Stage1 | AUD+IMG | 0.9709 | 0.9605 | 0.0020 | 0.5987 | 0.7024 | 0.7722 | 0.8678 | 0.0320 | +0.0320 | +0.0040 |
| flickr_144k | Stage1 | AUD+OBJ | 0.7120 | 0.7352 | 0.0269 | 0.2046 | 0.3736 | 0.4958 | 0.4268 | 0.0680 | +0.0680 | +0.0400 |
| flickr_144k | Stage2 | AUD+IMG | 0.9528 | 0.9387 | 0.0042 | 0.4936 | 0.6331 | 0.7209 | 0.8091 | 0.0520 | +0.0520 | -0.0080 |
| flickr_144k | Stage2 | AUD+OBJ | 0.7345 | 0.7638 | 0.0279 | 0.2528 | 0.4150 | 0.5274 | 0.4059 | 0.0840 | +0.0840 | +0.0560 |

## Alpha Diagnostic

Formal IQR remains alpha AUD = 0.6. The sweep is diagnostic only.

| alpha AUD | VGG Stage1 | VGG Stage2 | Flickr Stage1 | Flickr Stage2 |
|---:|---:|---:|---:|---:|
| 0.0 | 0.4069/0.4166 | 0.4069/0.4166 | 0.8040/0.6166 | 0.8040/0.6166 |
| 0.1 | 0.4077/0.4167 | 0.4116/0.4180 | 0.8000/0.6170 | 0.8120/0.6224 |
| 0.2 | 0.4077/0.4166 | 0.4147/0.4200 | 0.8040/0.6198 | 0.8160/0.6258 |
| 0.3 | 0.4071/0.4167 | 0.4168/0.4212 | 0.8080/0.6220 | 0.8040/0.6304 |
| 0.4 | 0.4075/0.4165 | 0.4211/0.4224 | 0.8120/0.6224 | 0.8040/0.6330 |
| 0.5 | 0.4079/0.4163 | 0.4221/0.4230 | 0.8120/0.6236 | 0.8000/0.6350 |
| 0.6 | 0.4069/0.4160 | 0.4230/0.4234 | 0.8080/0.6234 | 0.8040/0.6366 |
| 0.7 | 0.4050/0.4151 | 0.4257/0.4236 | 0.8080/0.6232 | 0.8040/0.6372 |
| 0.8 | 0.4048/0.4144 | 0.4281/0.4239 | 0.8040/0.6224 | 0.8120/0.6374 |
| 0.9 | 0.4023/0.4135 | 0.4283/0.4236 | 0.8040/0.6232 | 0.8120/0.6386 |
| 1.0 | 0.4002/0.4127 | 0.4269/0.4230 | 0.8040/0.6228 | 0.8120/0.6356 |

## Epoch Trajectory

Only two distinct model states exist per formal Stage1 directory. `latest.pth` duplicates `final.pth`; no epoch 1/5/10/25 weights exist. The requested correlations are therefore reported but are not statistically interpretable.

| Setting | Epoch | att loss | direct eval MSE | AUD | IMG | IQR | Pearson | Top20 | IMG-only | Oracle gain | IQR gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | 3 | 7.756e-05 | 6.461e-05 | 0.4002 | 0.4069 | 0.4069 | 0.9725 | 0.7809 | 0.0353 | +0.0353 | +0.0068 |
| vggss_144k | 50 | 4.773e-04 | 4.969e-04 | 0.3416 | 0.3290 | 0.3373 | 0.9817 | 0.8459 | 0.0312 | +0.0312 | -0.0043 |
| flickr_144k | 7 | 1.745e-04 | 8.365e-05 | 0.8040 | 0.8040 | 0.8080 | 0.9709 | 0.7024 | 0.0320 | +0.0320 | +0.0040 |
| flickr_144k | 50 | 3.495e-04 | 3.152e-04 | 0.7360 | 0.7400 | 0.7440 | 0.9798 | 0.7829 | 0.0360 | +0.0360 | +0.0080 |

Two-point observational correlations with logged `train_attention_match_loss`:

| Setting | Target | Pearson | Spearman |
|---|---|---:|---:|
| vggss_144k | AUD_IMG_Pearson | 1.0000 | 1.0000 |
| vggss_144k | AUD_IMG_Spearman | 1.0000 | 1.0000 |
| vggss_144k | Top20Overlap | 1.0000 | 1.0000 |
| vggss_144k | IMG_ONLY_fraction | -1.0000 | -1.0000 |
| vggss_144k | OracleGain | -1.0000 | -1.0000 |
| vggss_144k | IQRGain | -1.0000 | -1.0000 |
| flickr_144k | AUD_IMG_Pearson | 1.0000 | 1.0000 |
| flickr_144k | AUD_IMG_Spearman | 1.0000 | 1.0000 |
| flickr_144k | Top20Overlap | 1.0000 | 1.0000 |
| flickr_144k | IMG_ONLY_fraction | 1.0000 | 1.0000 |
| flickr_144k | OracleGain | 1.0000 | 1.0000 |
| flickr_144k | IQRGain | 1.0000 | 1.0000 |

## Sample-Level Disagreement

Disagreement is `1 - Pearson(AUD_FINE, IMG_QUERY)` in evaluator space.

| Setting | Group | Mean | Median | Std | N |
|---|---|---:|---:|---:|---:|
| vggss_144k | IMG_ONLY | 0.0469 | 0.0369 | 0.0359 | 168 |
| vggss_144k | AUD_ONLY | 0.0608 | 0.0416 | 0.0571 | 271 |
| vggss_144k | BOTH_SUCCESS | 0.0316 | 0.0235 | 0.0261 | 1931 |
| vggss_144k | BOTH_FAIL | 0.0410 | 0.0301 | 0.0361 | 2788 |
| vggss_144k | OGL_RESCUE | 0.0380 | 0.0273 | 0.0327 | 357 |
| flickr_144k | IMG_ONLY | 0.0370 | 0.0368 | 0.0157 | 13 |
| flickr_144k | AUD_ONLY | 0.0776 | 0.0683 | 0.0550 | 15 |
| flickr_144k | BOTH_SUCCESS | 0.0425 | 0.0347 | 0.0268 | 188 |
| flickr_144k | BOTH_FAIL | 0.0633 | 0.0457 | 0.0549 | 34 |
| flickr_144k | OGL_RESCUE | 0.0339 | 0.0284 | 0.0208 | 19 |

## Qualitative Audit

Selection is deterministic: the lexicographically first available sample for each fixed category. Panels contain Image, GT, AUD, IMG_QUERY, IQR, OBJ_PRIOR, OGL, |AUD-IMG|, and |AUD-OBJ|.

- `vggss_144k`: `{"AUD_ONLY": "-GW1J75oAKU_000304", "BOTH_FAIL": "-0BIyqJj9ZU_000030", "IMG_ONLY": "-Vo4CAMX26U_000030", "IQR_HURT": "-GW1J75oAKU_000304", "IQR_RESCUE": "-hYRFCQdbLg_000030", "OGL_RESCUE_CAPTURED_BY_IMG": "-Vo4CAMX26U_000030", "OGL_RESCUE_NOT_CAPTURED_BY_IMG": "-4bPiXbovf0_000008"}`
- `flickr_144k`: `{"AUD_ONLY": "10013411946", "BOTH_FAIL": "10008553263", "IMG_ONLY": "10548273474", "IQR_HURT": "10013411946", "IQR_RESCUE": "10701841844", "OGL_RESCUE_CAPTURED_BY_IMG": "10548273474", "OGL_RESCUE_NOT_CAPTURED_BY_IMG": "10939270325"}`

Observed fixed-sample phenomena:

- AUD and IMG usually share the same dominant broad lobe; their useful differences are concentrated at boundaries or secondary peaks rather than forming a consistently independent object map.
- IMG-only cases show real but modest recentering/contraction. In VGG `-Vo4CAMX26U_000030`, AUD/IMG/IQR IoU is `0.477/0.509/0.495`; in Flickr `10548273474` it is `0.448/0.516/0.477`. The fixed mixture can erase an IMG success.
- IQR-hurt cases show the opposite boundary shift: VGG violin `0.565/0.441/0.495` and Flickr cyclists `0.523/0.481/0.497` for AUD/IMG/IQR.
- OGL-only corrections are visibly larger extent changes. The VGG air-conditioner sample has AUD/IMG/IQR `0.368/0.491/0.409`, while OBJ/OGL reach `0.740/0.511`.
- IQR rescues exist, but selected examples are threshold-boundary improvements; they do not offset the larger hurt count at Stage2.

## Decision

**Case B - Complementarity Exists - Fusion Bottleneck**

Study fusion, not att-loss. Do not start a follow-up automatically.

The cross-sectional redundancy and task evidence is kept separate from the att-loss causality claim. With only two saved states, epoch trajectory cannot establish that att-loss caused the redundancy even when final maps are highly similar.

Decision evidence:

- `high_similarity_both`: `False`
- `IMG_ONLY_fraction_at_most_0.05_both`: `False`
- `IMG_oracle_gain_less_than_75pct_OBJ_both`: `True`
- `IMG_OGL_rescue_capture_at_most_0.25_both`: `False`
- `trajectory_has_at_least_3_distinct_checkpoints_both`: `False`
- `trajectory_expected_direction`: `False`
- `clear_complementarity_both`: `True`
- `fixed_fusion_failed_both`: `True`
- `threshold_note`: `For Case B, noticeable means at least 3% AUX-only, +0.03 pair-oracle cIoU, and 25% OGL-rescue capture in both datasets. The report exposes every count.`

## Git Status

```text
?? chuagnxindian/1mufasaslot/1.1.1_14_14_L3/
?? chuagnxindian/1mufasaslot/1.3G_14_14_L3/
?? chuagnxindian/1mufasaslot/14_14_L3_COMPARISON.md
?? chuagnxindian/2.1R_fixed_slot_reliability/
?? chuagnxindian/2.2_highres_slot_ownership/
?? chuagnxindian/2.3_semantic_spatial_decoupled_slot/
?? chuagnxindian/2.4_object_aware_multigeom_spatial_specialization/
?? chuagnxindian/2.5_dual_path_decision_probe/
?? chuagnxindian/3.0_temporal_audio_grounding_probe/
?? chuagnxindian/3.1_hierarchical_audio_representation/
?? chuagnxindian/3.2_a4_temporal_grounding_probe/
?? chuagnxindian/4.0_aud_img_redundancy_probe/
```
