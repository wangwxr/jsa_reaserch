# Experiment 3.2 - A4 Temporal Grounding Probe

## Protocol And Audit

Original formal 1.3G checkpoints are frozen. Every temporal chunk reuses the same learned initial slots, AudioSlotBranch, K34, infer sharpening, slot-softmax, spatial normalization, and target slot0 as official AUD_FINE.

| Setting | raw A4 | A4 tokens | F34 | K34 | Full tensor error | Evaluator error |
|---|---:|---:|---:|---:|---:|---:|
| vggss_144k | [256, 512, 9, 16] | [256, 16, 512] | [256, 512, 14, 14] | [256, 196, 512] | 0.000e+00 | 0.000e+00 |
| flickr_144k | [32, 512, 9, 16] | [32, 16, 512] | [32, 512, 14, 14] | [32, 196, 512] | 0.000e+00 | 0.000e+00 |

- T2 boundaries: `[0:8], [8:16]`.
- T4 boundaries: `[0:4], [4:8], [8:12], [12:16]`.
- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`.
- All checkpoint SHA256 and mtimes are unchanged; all models remained in eval/inference mode.

## Localization Metrics

| Method | VGG cIoU/AUC | Flickr cIoU/AUC |
|---|---:|---:|
| ORIGINAL_AUD | 0.4269/0.4230 | 0.8120/0.6356 |
| T2_CHUNK1 | 0.4257/0.4229 | 0.8040/0.6360 |
| T2_CHUNK2 | 0.4259/0.4228 | 0.8080/0.6338 |
| T2_RAW_MEAN | 0.4277/0.4229 | 0.8080/0.6356 |
| T2_RAW_GEO | 0.4277/0.4229 | 0.8080/0.6356 |
| T2_NORM_MEAN | 0.4275/0.4228 | 0.8080/0.6356 |
| T2_NORM_GEO | 0.4279/0.4228 | 0.8080/0.6356 |
| T4_CHUNK1 | 0.4242/0.4224 | 0.7960/0.6348 |
| T4_CHUNK2 | 0.4257/0.4225 | 0.8040/0.6366 |
| T4_CHUNK3 | 0.4256/0.4227 | 0.8040/0.6352 |
| T4_CHUNK4 | 0.4244/0.4223 | 0.8040/0.6348 |
| T4_RAW_MEAN | 0.4271/0.4227 | 0.8080/0.6364 |
| T4_RAW_GEO | 0.4271/0.4227 | 0.8080/0.6364 |
| T4_NORM_MEAN | 0.4265/0.4226 | 0.8080/0.6364 |
| T4_NORM_GEO | 0.4269/0.4226 | 0.8080/0.6362 |
| OGL_REFERENCE | 0.4570/0.4401 | 0.8680/0.6596 |

## Temporal Query Semantics

Visual target is the final frozen L4 visual query slot0. Negative pairs are batch-shuffled visual queries.

| Setting | Scale | Positive | Negative | Margin | Query pairwise cosine | Query variance |
|---|---|---:|---:|---:|---:|---:|
| vggss_144k | T2 | 0.0103 | -0.0276 | 0.0380 | 0.8864 | 0.020574 |
| vggss_144k | T4 | 0.0085 | -0.0281 | 0.0366 | 0.8522 | 0.040147 |
| flickr_144k | T2 | -0.0142 | -0.0508 | 0.0366 | 0.8669 | 0.021920 |
| flickr_144k | T4 | -0.0182 | -0.0529 | 0.0347 | 0.8242 | 0.043654 |

## Temporal Map Similarity

| Setting | Scale | Pearson | Spearman | JS divergence |
|---|---|---:|---:|---:|
| vggss_144k | T2 | 0.9968 | 0.9957 | 3.610e-06 |
| vggss_144k | T4 | 0.9959 | 0.9947 | 4.538e-06 |
| flickr_144k | T2 | 0.9981 | 0.9969 | 4.696e-06 |
| flickr_144k | T4 | 0.9974 | 0.9961 | 6.131e-06 |

## Rescue, Hurt, Oracle, And OGL Capture

| Setting | Method | Rescue | Hurt | Net | Oracle cIoU/AUC | OGL pool | Captured | Rate | Rescue intersect OGL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | T2_RAW_MEAN | 10 | 6 | 4 | 0.4288/0.4240 | 357 | 6 | 0.017 | 6 |
| vggss_144k | T2_RAW_GEO | 10 | 6 | 4 | 0.4288/0.4240 | 357 | 6 | 0.017 | 6 |
| vggss_144k | T4_RAW_MEAN | 16 | 15 | 1 | 0.4300/0.4247 | 357 | 9 | 0.025 | 9 |
| vggss_144k | T4_RAW_GEO | 16 | 15 | 1 | 0.4300/0.4247 | 357 | 9 | 0.025 | 9 |
| flickr_144k | T2_RAW_MEAN | 0 | 1 | -1 | 0.8120/0.6364 | 19 | 0 | 0.000 | 0 |
| flickr_144k | T2_RAW_GEO | 0 | 1 | -1 | 0.8120/0.6364 | 19 | 0 | 0.000 | 0 |
| flickr_144k | T4_RAW_MEAN | 0 | 1 | -1 | 0.8120/0.6380 | 19 | 0 | 0.000 | 0 |
| flickr_144k | T4_RAW_GEO | 0 | 1 | -1 | 0.8120/0.6380 | 19 | 0 | 0.000 | 0 |

## Temporal Stability

T4 stability uses independently evaluator-normalized 224x224 chunk maps.

| Setting | Region | STD mean/median/std | CV mean/median/std | Samples |
|---|---|---:|---:|---:|
| vggss_144k | GT_REGION | 0.010078/0.008351/0.006965 | 0.014531/0.011618/0.011305 | 5156 |
| vggss_144k | AUD_FP_REGION | 0.012237/0.010507/0.007252 | 0.017172/0.014689/0.010346 | 5104 |
| vggss_144k | FP > GT fraction | 0.7181 | 0.7025 | 5102 |
| flickr_144k | GT_REGION | 0.008810/0.008049/0.004211 | 0.012182/0.010731/0.006718 | 250 |
| flickr_144k | AUD_FP_REGION | 0.010774/0.010137/0.005480 | 0.015237/0.014024/0.007954 | 249 |
| flickr_144k | FP > GT fraction | 0.7269 | 0.7430 | 249 |

## VGG Over-Expansion

- GT_REGION: STD mean/median/std=0.009813/0.008024/0.006998; CV mean/median/std=0.013238/0.010397/0.010561; n=4109.
- OVER_EXPANSION_REGION: STD mean/median/std=0.012227/0.010579/0.007006; CV mean/median/std=0.016973/0.014664/0.009801; n=4109.
- Fraction over-expansion STD > GT STD: 0.7501; CV fraction: 0.7676.

## Temporal Delta Versus OGL Delta

| Setting | Method | Pearson | Spearman |
|---|---|---:|---:|
| vggss_144k | T2_RAW_MEAN | -0.0368 | -0.0373 |
| vggss_144k | T2_RAW_GEO | -0.0390 | -0.0402 |
| vggss_144k | T4_RAW_MEAN | -0.0391 | -0.0550 |
| vggss_144k | T4_RAW_GEO | -0.0416 | -0.0587 |
| flickr_144k | T2_RAW_MEAN | -0.1433 | -0.1248 |
| flickr_144k | T2_RAW_GEO | -0.1448 | -0.1274 |
| flickr_144k | T4_RAW_MEAN | -0.1537 | -0.1334 |
| flickr_144k | T4_RAW_GEO | -0.1566 | -0.1393 |

## Qualitative Selection

Across the fixed panels, T4 M1-M4 preserve nearly the same hotspot, extent, and context response. VGG temporal rescues and hurts are predominantly small threshold-boundary changes rather than consistent removal of background support. OGL-rescue/temporal-fail panels show OGL changing spatial extent enough to cross the success threshold while raw and normalized temporal consensus remain close to ORIGINAL_AUD. TEMP_STD often highlights context, borders, and object edges; although this produces the aggregate FP/over-expansion variance signal, arithmetic and geometric averaging do not convert it into a reliable correction. Flickr has no temporal rescue and one temporal hurt under the four formal raw consensus maps.

- vggss_144k: `{"BASELINE_SUCCESS_STABLE": 2187, "OGL_RESCUE_TEMPORAL_FAIL": 348, "OGL_RESCUE_TEMPORAL_RESCUE": 9, "TEMPORAL_HURT": 15, "TEMPORAL_RESCUE": 17, "VGG_OVER_EXPANSION": 4111}`
- flickr_144k: `{"BASELINE_SUCCESS_STABLE": 202, "OGL_RESCUE_TEMPORAL_FAIL": 19, "OGL_RESCUE_TEMPORAL_RESCUE": 0, "TEMPORAL_HURT": 1, "TEMPORAL_RESCUE": 0, "VGG_OVER_EXPANSION": 203}`

## Mechanism Interpretation

T2/T4 chunk queries are not identical: T4 pairwise query cosine is about 0.85 on VGG and 0.82 on Flickr. However, their spatial maps have Pearson above 0.995 and JS divergence near zero. The temporal-specific query components therefore have little effect after projection onto the shared K34 spatial keys.

Evaluator-space temporal variance is higher in VGG false-positive and over-expansion regions, so the probe finds a limited mechanism signal. It is not the signal needed by fixed mean/geometric consensus: OGL-rescue capture is at most 2.5% on VGG and zero on Flickr, temporal and OGL IoU deltas are uncorrelated or negatively correlated, and oracle gains remain small.

## Fixed-Rule Decision

**Negative**

Audio auxiliary representation line closed. Next candidate: AUD-IMG_QUERY redundancy / att-loss. Do not start it automatically.

```json
{
  "label": "Negative",
  "evidence_A_localization": {
    "T2_RAW_MEAN": {
      "gains": {
        "vggss_144k": 0.0007754943776657752,
        "flickr_144k": -0.0040000000000000036
      },
      "both_non_decrease": false,
      "one_clear_gain_at_least_0.01": false,
      "both_positive": false,
      "rescue_at_least_hurt_both": false,
      "passed": false
    },
    "T2_RAW_GEO": {
      "gains": {
        "vggss_144k": 0.0007754943776657752,
        "flickr_144k": -0.0040000000000000036
      },
      "both_non_decrease": false,
      "one_clear_gain_at_least_0.01": false,
      "both_positive": false,
      "rescue_at_least_hurt_both": false,
      "passed": false
    },
    "T4_RAW_MEAN": {
      "gains": {
        "vggss_144k": 0.00019387359441647156,
        "flickr_144k": -0.0040000000000000036
      },
      "both_non_decrease": false,
      "one_clear_gain_at_least_0.01": false,
      "both_positive": false,
      "rescue_at_least_hurt_both": false,
      "passed": false
    },
    "T4_RAW_GEO": {
      "gains": {
        "vggss_144k": 0.00019387359441647156,
        "flickr_144k": -0.0040000000000000036
      },
      "both_non_decrease": false,
      "one_clear_gain_at_least_0.01": false,
      "both_positive": false,
      "rescue_at_least_hurt_both": false,
      "passed": false
    }
  },
  "evidence_B_mechanism": {
    "VGG_FP_to_GT_STD_ratio": 1.2141959708849392,
    "VGG_FP_to_GT_CV_ratio": 1.1817263444701103,
    "VGG_fraction_FP_STD_gt_GT_STD": 0.7181497451979616,
    "region_signal": true,
    "max_OGL_rescue_capture_rate": {
      "vggss_144k": 0.025210084033613446,
      "flickr_144k": 0.0
    },
    "capture_signal_at_least_0.10": false,
    "passed": true
  },
  "recommendation": "Audio auxiliary representation line closed. Next candidate: AUD-IMG_QUERY redundancy / att-loss. Do not start it automatically."
}
```

No training or follow-up experiment was started.
