# Experiment 3.0 - Temporal Audio Grounding Probe

## Protocol And Zero-Training Audit

This experiment uses the formal original 1.3G checkpoints only. All temporal maps use the unchanged frozen Audio Slot Branch and the same G K34/readout. OGL is evaluation-only and is never used to construct a temporal map.

- `model.eval()` and `torch.inference_mode()` were used throughout.
- `optimizer_created=false`, `backward_called=false`, `new_trainable_params=0`.
- All Stage1, original G, and evaluation-only object-prior checkpoint SHA256/mtime values are unchanged.
- No 3.1 experiment was implemented or started.

| Setting | Audio feature | Audio tokens | T | 4-chunk boundaries | FULL tensor error | Evaluator error |
|---|---:|---:|---:|---|---:|---:|
| vggss_144k | [256, 512, 16] | [256, 16, 512] | 16 | [[0, 4], [4, 8], [8, 12], [12, 16]] | 0.000e+00 | 0.000e+00 |
| flickr_144k | [32, 512, 16] | [32, 16, 512] | 16 | [[0, 4], [4, 8], [8, 12], [12, 16]] | 0.000e+00 | 0.000e+00 |

## Main Localization Results

| Method | VGG cIoU/AUC | Flickr cIoU/AUC |
|---|---:|---:|
| FULL_AUD | 0.4269/0.4230 | 0.8120/0.6356 |
| TEMP_MEAN_4 | 0.4271/0.4227 | 0.8080/0.6364 |
| TEMP_GEO_4 | 0.4271/0.4227 | 0.8080/0.6364 |
| FULL_TEMP_MEAN_4 | 0.4269/0.4228 | 0.8120/0.6356 |
| FULL_TEMP_GEO_4 | 0.4269/0.4228 | 0.8120/0.6356 |
| TEMP_MEAN_2 | 0.4277/0.4229 | 0.8080/0.6356 |
| TEMP_GEO_2 | 0.4277/0.4229 | 0.8080/0.6356 |
| OGL | 0.4570/0.4401 | 0.8680/0.6596 |

## Four-Chunk Standalone Results

| Chunk | VGG cIoU/AUC | Flickr cIoU/AUC |
|---|---:|---:|
| CHUNK_1 | 0.4242/0.4224 | 0.7960/0.6348 |
| CHUNK_2 | 0.4257/0.4225 | 0.8040/0.6366 |
| CHUNK_3 | 0.4256/0.4227 | 0.8040/0.6352 |
| CHUNK_4 | 0.4244/0.4223 | 0.8040/0.6348 |

## Primary Rescue, Hurt, Oracle, And OGL Capture

| Dataset | Method | Rescue | Hurt | Net | Oracle cIoU/AUC | OGL pool | Captured | Rate | Capture-Hurt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | FULL_TEMP_MEAN_4 | 7 | 7 | 0 | 0.4283/0.4236 | 357 | 4 | 0.011 | -3 |
| vggss_144k | FULL_TEMP_GEO_4 | 7 | 7 | 0 | 0.4283/0.4236 | 357 | 4 | 0.011 | -3 |
| flickr_144k | FULL_TEMP_MEAN_4 | 0 | 0 | 0 | 0.8120/0.6360 | 19 | 0 | 0.000 | 0 |
| flickr_144k | FULL_TEMP_GEO_4 | 0 | 0 | 0 | 0.8120/0.6360 | 19 | 0 | 0.000 | 0 |

## OGL Marginal Gap

| Dataset | Method | Original gap | New gap | Reduction | Reduction % |
|---|---|---:|---:|---:|---:|
| vggss_144k | TEMP_MEAN_4 | 0.0301 | 0.0299 | 0.0002 | 0.6% |
| vggss_144k | TEMP_GEO_4 | 0.0301 | 0.0299 | 0.0002 | 0.6% |
| vggss_144k | FULL_TEMP_MEAN_4 | 0.0301 | 0.0301 | 0.0000 | 0.0% |
| vggss_144k | FULL_TEMP_GEO_4 | 0.0301 | 0.0301 | 0.0000 | 0.0% |
| vggss_144k | TEMP_MEAN_2 | 0.0301 | 0.0293 | 0.0008 | 2.6% |
| vggss_144k | TEMP_GEO_2 | 0.0301 | 0.0293 | 0.0008 | 2.6% |
| flickr_144k | TEMP_MEAN_4 | 0.0560 | 0.0600 | -0.0040 | -7.1% |
| flickr_144k | TEMP_GEO_4 | 0.0560 | 0.0600 | -0.0040 | -7.1% |
| flickr_144k | FULL_TEMP_MEAN_4 | 0.0560 | 0.0560 | 0.0000 | 0.0% |
| flickr_144k | FULL_TEMP_GEO_4 | 0.0560 | 0.0560 | 0.0000 | 0.0% |
| flickr_144k | TEMP_MEAN_2 | 0.0560 | 0.0600 | -0.0040 | -7.1% |
| flickr_144k | TEMP_GEO_2 | 0.0560 | 0.0600 | -0.0040 | -7.1% |

## Chunk Slot Identity Stability

| Dataset | Chunk | q0->full q0 rate | q0->full q1 rate | cos(q0,f0) | cos(q0,f1) | cos(q1,f0) | cos(q1,f1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| vggss_144k | 1 | 1.000 | 0.000 | 0.9164 | -0.0186 | -0.0030 | 0.9918 |
| vggss_144k | 2 | 1.000 | 0.000 | 0.9392 | 0.0129 | 0.0130 | 0.9942 |
| vggss_144k | 3 | 1.000 | 0.000 | 0.9443 | 0.0136 | 0.0130 | 0.9943 |
| vggss_144k | 4 | 1.000 | 0.000 | 0.9266 | 0.0001 | 0.0052 | 0.9916 |
| vggss_144k | OVERALL | 1.000 | 0.000 | - | - | - | - |
| flickr_144k | 1 | 1.000 | 0.000 | 0.9012 | 0.1558 | 0.1389 | 0.9876 |
| flickr_144k | 2 | 1.000 | 0.000 | 0.9235 | 0.1490 | 0.1746 | 0.9903 |
| flickr_144k | 3 | 1.000 | 0.000 | 0.9146 | 0.1481 | 0.1731 | 0.9905 |
| flickr_144k | 4 | 1.000 | 0.000 | 0.9153 | 0.1411 | 0.1635 | 0.9884 |
| flickr_144k | OVERALL | 1.000 | 0.000 | - | - | - | - |

## Temporal Region Stability

Per-sample region means are aggregated below. Stability is computed from raw 14x14 chunk attention before evaluator normalization.

| Dataset | Region | Temporal mean | Temporal STD | Temporal CV | Samples |
|---|---|---:|---:|---:|---:|
| vggss_144k | GT_FOREGROUND | 0.005369 | 0.000017 | 0.0031 | 5156 |
| vggss_144k | CORRECT_AUD_FOREGROUND | 0.005423 | 0.000017 | 0.0031 | 5150 |
| vggss_144k | AUD_FALSE_POSITIVE_CONTEXT | 0.005291 | 0.000015 | 0.0027 | 5104 |
| flickr_144k | GT_FOREGROUND | 0.005331 | 0.000017 | 0.0032 | 250 |
| flickr_144k | CORRECT_AUD_FOREGROUND | 0.005380 | 0.000017 | 0.0031 | 250 |
| flickr_144k | AUD_FALSE_POSITIVE_CONTEXT | 0.005197 | 0.000014 | 0.0028 | 249 |

## Sample-Level Temporal Agreement

Each sample value is the mean over all six pairs among the four raw chunk maps.

| Dataset | Group | Pearson | Spearman | JS divergence | Samples |
|---|---|---:|---:|---:|---:|
| vggss_144k | ALL | 0.9959 | 0.9947 | 4.54e-06 | 5158 |
| vggss_144k | FULL_AUD_SUCCESS | 0.9962 | 0.9951 | 4.77e-06 | 2202 |
| vggss_144k | FULL_AUD_FAILURE | 0.9957 | 0.9945 | 4.36e-06 | 2956 |
| vggss_144k | OGL_RESCUE | 0.9960 | 0.9951 | 4.30e-06 | 357 |
| flickr_144k | ALL | 0.9974 | 0.9961 | 6.13e-06 | 250 |
| flickr_144k | FULL_AUD_SUCCESS | 0.9975 | 0.9962 | 6.17e-06 | 203 |
| flickr_144k | FULL_AUD_FAILURE | 0.9969 | 0.9955 | 5.96e-06 | 47 |
| flickr_144k | OGL_RESCUE | 0.9979 | 0.9968 | 7.61e-06 | 19 |

## Chunk Quality

| Dataset | Chunk | Token mean norm | Temporal variance | q0 norm | q0/full-q0 cosine |
|---|---:|---:|---:|---:|---:|
| vggss_144k | 1 | 61.7918 | 0.76353 | 13.6292 | 0.9164 |
| vggss_144k | 2 | 68.8762 | 0.54980 | 13.6551 | 0.9392 |
| vggss_144k | 3 | 68.9815 | 0.56556 | 13.6628 | 0.9443 |
| vggss_144k | 4 | 64.9610 | 0.65253 | 13.6375 | 0.9266 |
| flickr_144k | 1 | 62.3999 | 0.72618 | 13.0542 | 0.9012 |
| flickr_144k | 2 | 67.9786 | 0.57831 | 13.0168 | 0.9235 |
| flickr_144k | 3 | 68.0481 | 0.57394 | 12.9672 | 0.9146 |
| flickr_144k | 4 | 64.5371 | 0.65305 | 12.9235 | 0.9153 |

## Alpha Diagnostic

Alpha=0.6 is formal; other rows are diagnostics only and are not used for model selection.

| Dataset | Family | Full alpha | cIoU/AUC | Rescue | Hurt | Net |
|---|---|---:|---:|---:|---:|---:|
| vggss_144k | MEAN | 0.5 | 0.4271/0.4228 | 8 | 7 | 1 |
| vggss_144k | MEAN | 0.6 | 0.4269/0.4228 | 7 | 7 | 0 |
| vggss_144k | MEAN | 0.7 | 0.4269/0.4229 | 5 | 5 | 0 |
| vggss_144k | MEAN | 0.8 | 0.4267/0.4230 | 4 | 5 | -1 |
| vggss_144k | MEAN | 0.9 | 0.4267/0.4230 | 2 | 3 | -1 |
| vggss_144k | GEO | 0.5 | 0.4269/0.4228 | 8 | 8 | 0 |
| vggss_144k | GEO | 0.6 | 0.4269/0.4228 | 7 | 7 | 0 |
| vggss_144k | GEO | 0.7 | 0.4269/0.4229 | 5 | 5 | 0 |
| vggss_144k | GEO | 0.8 | 0.4267/0.4230 | 4 | 5 | -1 |
| vggss_144k | GEO | 0.9 | 0.4267/0.4230 | 2 | 3 | -1 |
| flickr_144k | MEAN | 0.5 | 0.8080/0.6356 | 0 | 1 | -1 |
| flickr_144k | MEAN | 0.6 | 0.8120/0.6356 | 0 | 0 | 0 |
| flickr_144k | MEAN | 0.7 | 0.8120/0.6354 | 0 | 0 | 0 |
| flickr_144k | MEAN | 0.8 | 0.8120/0.6354 | 0 | 0 | 0 |
| flickr_144k | MEAN | 0.9 | 0.8120/0.6356 | 0 | 0 | 0 |
| flickr_144k | GEO | 0.5 | 0.8080/0.6356 | 0 | 1 | -1 |
| flickr_144k | GEO | 0.6 | 0.8120/0.6356 | 0 | 0 | 0 |
| flickr_144k | GEO | 0.7 | 0.8120/0.6354 | 0 | 0 | 0 |
| flickr_144k | GEO | 0.8 | 0.8120/0.6354 | 0 | 0 | 0 |
| flickr_144k | GEO | 0.9 | 0.8120/0.6356 | 0 | 0 | 0 |

## Qualitative

The manifests use a fixed first-in-test-order round-robin across AUD success, OGL rescue, temporal rescue, temporal hurt, and all-fail categories; twelve panels are saved per dataset without cherry-picking.

- Across both datasets, CHUNK_1..4 preserve almost the same hotspot shape, extent, and background response as FULL_AUD. TEMP_MEAN_4 and TEMP_GEO_4 are therefore visually indistinguishable from each other and nearly identical to FULL_AUD.
- The fixed VGG temporal-rescue examples are threshold-boundary changes rather than a consistent removal of context. Matching temporal-hurt examples show the same small boundary movement in the opposite direction, consistent with the aggregate 7 Rescue / 7 Hurt result.
- In fixed OGL-rescue examples, OGL changes the spatial support enough to cross the success threshold, while temporal fusion remains close to FULL_AUD. This matches the 4/357 VGG and 0/19 Flickr OGL-rescue capture counts.
- TEMP_STD frequently highlights object edges, image borders, or non-target regions. It does not consistently separate correct grounding from context false positives; the quantitative false-positive STD/CV is in fact lower than the correct-region STD/CV on both datasets.
- Flickr contains no primary temporal Rescue or Hurt cases. Its fixed panels show the temporal maps preserving the original response, including its misses, rather than introducing complementary localization evidence.

## Fixed-Rule Decision

**Route C**

Temporal consensus fails. Stop this direction and use Hierarchical Audio Representation as the next candidate innovation direction.

### Route A checks

```json
{
  "FULL_TEMP_MEAN_4": {
    "gains": {
      "vggss_144k": 0.0,
      "flickr_144k": 0.0
    },
    "nets": {
      "vggss_144k": 0,
      "flickr_144k": 0
    },
    "non_decrease_both": true,
    "gain_at_least_0_01_one_dataset": false,
    "positive_net_both": false
  },
  "FULL_TEMP_GEO_4": {
    "gains": {
      "vggss_144k": 0.0,
      "flickr_144k": 0.0
    },
    "nets": {
      "vggss_144k": 0,
      "flickr_144k": 0
    },
    "non_decrease_both": true,
    "gain_at_least_0_01_one_dataset": false,
    "positive_net_both": false
  }
}
```

### Route B checks

```json
{
  "FULL_TEMP_MEAN_4": {
    "vggss_144k": {
      "false_positive_to_correct_std_ratio": 0.8613686092637177,
      "false_positive_to_correct_cv_ratio": 0.8866643541258121,
      "OGL_capture_lift_vs_other_failures": 0.01005019168113571,
      "oracle_cIoU_gain": 0.0013571151609150789,
      "region_signal": false,
      "capture_or_oracle_signal": false,
      "signal_positive": false
    },
    "flickr_144k": {
      "false_positive_to_correct_std_ratio": 0.8570331767997972,
      "false_positive_to_correct_cv_ratio": 0.8914111993147288,
      "OGL_capture_lift_vs_other_failures": 0.0,
      "oracle_cIoU_gain": 0.0,
      "region_signal": false,
      "capture_or_oracle_signal": false,
      "signal_positive": false
    }
  },
  "FULL_TEMP_GEO_4": {
    "vggss_144k": {
      "false_positive_to_correct_std_ratio": 0.8613686092637177,
      "false_positive_to_correct_cv_ratio": 0.8866643541258121,
      "OGL_capture_lift_vs_other_failures": 0.01005019168113571,
      "oracle_cIoU_gain": 0.0013571151609150789,
      "region_signal": false,
      "capture_or_oracle_signal": false,
      "signal_positive": false
    },
    "flickr_144k": {
      "false_positive_to_correct_std_ratio": 0.8570331767997972,
      "false_positive_to_correct_cv_ratio": 0.8914111993147288,
      "OGL_capture_lift_vs_other_failures": 0.0,
      "oracle_cIoU_gain": 0.0,
      "region_signal": false,
      "capture_or_oracle_signal": false,
      "signal_positive": false
    }
  }
}
```

No 3.1 implementation or run was started.
