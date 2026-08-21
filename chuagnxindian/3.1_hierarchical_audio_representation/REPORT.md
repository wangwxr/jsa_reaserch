# Experiment 3.1 - Hierarchical Audio Representation

## Protocol

Visual L3/L4, visual fusion, A4 localization, A4 attention loss, A4 reconstruction, loss weights, optimizer, scheduler, data configuration, and IQR-cIoU checkpoint selection are inherited from the formal L3+L4 Stage1.

The only added trainable modules are aud_proj3, AudioSlotBranch_A3, and the zero-initialized A4-residual AudioHierarchicalFusion. No G or Stage2 training is part of this experiment.

## Best Epochs

| Setting | Baseline epoch | 3.1 epoch | Selection |
|---|---:|---:|---|
| vggss_10k | 37 | 25 | IQR_cIoU=0.4069 |
| vggss_144k | 3 | 2 | IQR_cIoU=0.3903 |
| flickr_10k | 81 | 100 | IQR_cIoU=0.7400 |
| flickr_144k | 7 | 6 | IQR_cIoU=0.7880 |

## Six Metrics

| Setting | Method | Baseline cIoU/AUC | 3.1 cIoU/AUC | Delta |
|---|---|---:|---:|---:|
| vggss_10k | AUD | 0.4015/0.4074 | 0.4011/0.4071 | -0.0004/-0.0003 |
| vggss_10k | IMG_QUERY | 0.4064/0.4094 | 0.4050/0.4087 | -0.0014/-0.0007 |
| vggss_10k | IQR | 0.4073/0.4092 | 0.4069/0.4085 | -0.0004/-0.0007 |
| vggss_10k | OBJ_PRIOR | 0.3478/0.3924 | 0.3478/0.3924 | +0.0000/+0.0000 |
| vggss_10k | OGL | 0.4432/0.4292 | 0.4420/0.4264 | -0.0012/-0.0028 |
| vggss_10k | EXTRA_IQR_OGL | 0.4314/0.4215 | 0.4321/0.4197 | +0.0008/-0.0017 |
| vggss_144k | AUD | 0.4002/0.4127 | 0.3841/0.4069 | -0.0161/-0.0058 |
| vggss_144k | IMG_QUERY | 0.4069/0.4166 | 0.3891/0.4092 | -0.0178/-0.0074 |
| vggss_144k | IQR | 0.4069/0.4160 | 0.3903/0.4090 | -0.0167/-0.0069 |
| vggss_144k | OBJ_PRIOR | 0.3478/0.3924 | 0.3478/0.3924 | +0.0000/+0.0000 |
| vggss_144k | OGL | 0.4343/0.4307 | 0.4244/0.4253 | -0.0099/-0.0053 |
| vggss_144k | EXTRA_IQR_OGL | 0.4273/0.4256 | 0.4174/0.4193 | -0.0099/-0.0063 |
| flickr_10k | AUD | 0.7640/0.5916 | 0.7280/0.5794 | -0.0360/-0.0122 |
| flickr_10k | IMG_QUERY | 0.7520/0.5890 | 0.7400/0.5738 | -0.0120/-0.0152 |
| flickr_10k | IQR | 0.7720/0.5922 | 0.7400/0.5772 | -0.0320/-0.0150 |
| flickr_10k | OBJ_PRIOR | 0.4480/0.4668 | 0.4480/0.4668 | +0.0000/+0.0000 |
| flickr_10k | OGL | 0.8400/0.6154 | 0.8000/0.6138 | -0.0400/-0.0016 |
| flickr_10k | EXTRA_IQR_OGL | 0.8120/0.6124 | 0.7720/0.6042 | -0.0400/-0.0082 |
| flickr_144k | AUD | 0.8040/0.6228 | 0.7880/0.6126 | -0.0160/-0.0102 |
| flickr_144k | IMG_QUERY | 0.8040/0.6166 | 0.7600/0.6062 | -0.0440/-0.0104 |
| flickr_144k | IQR | 0.8120/0.6236 | 0.7880/0.6112 | -0.0240/-0.0124 |
| flickr_144k | OBJ_PRIOR | 0.4480/0.4668 | 0.4480/0.4668 | +0.0000/+0.0000 |
| flickr_144k | OGL | 0.8440/0.6392 | 0.8520/0.6410 | +0.0080/+0.0018 |
| flickr_144k | EXTRA_IQR_OGL | 0.8280/0.6380 | 0.8240/0.6364 | -0.0040/-0.0016 |

## OGL Gaps

| Setting | Baseline OGL-AUD | 3.1 OGL-AUD | Reduction | Baseline OGL-IQR | 3.1 OGL-IQR | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| vggss_10k | 0.0417 | 0.0409 | +0.0008 | 0.0359 | 0.0351 | +0.0008 |
| vggss_144k | 0.0341 | 0.0403 | -0.0062 | 0.0273 | 0.0341 | -0.0068 |
| flickr_10k | 0.0760 | 0.0720 | +0.0040 | 0.0680 | 0.0600 | +0.0080 |
| flickr_144k | 0.0400 | 0.0640 | -0.0240 | 0.0320 | 0.0640 | -0.0320 |

## A3 Query Diagnostic

| Setting | A3_QUERY_AUD cIoU/AUC |
|---|---:|
| vggss_10k | 0.0355/0.1650 |
| vggss_144k | 0.0735/0.2026 |
| flickr_10k | 0.0240/0.1470 |
| flickr_144k | 0.1880/0.3128 |

## Semantic Alignment

| Setting | Audio representation | Positive | Shuffled negative | Margin |
|---|---|---:|---:|---:|
| vggss_10k | A3 | -0.0024 | -0.0057 | 0.0033 |
| vggss_10k | A4 | 0.2323 | 0.0603 | 0.1720 |
| vggss_10k | FUSED | 0.2447 | 0.0621 | 0.1827 |
| vggss_144k | A3 | 0.0053 | 0.0032 | 0.0021 |
| vggss_144k | A4 | 0.3906 | 0.2694 | 0.1212 |
| vggss_144k | FUSED | 0.4034 | 0.2824 | 0.1210 |
| flickr_10k | A3 | -0.0040 | -0.0031 | -0.0009 |
| flickr_10k | A4 | 0.1945 | 0.0762 | 0.1183 |
| flickr_10k | FUSED | 0.2090 | 0.0841 | 0.1249 |
| flickr_144k | A3 | 0.0022 | 0.0048 | -0.0025 |
| flickr_144k | A4 | 0.4242 | 0.3063 | 0.1180 |
| flickr_144k | FUSED | 0.4447 | 0.3337 | 0.1111 |

## Temporal Diversity At Best Checkpoint

| Setting | Level | Adjacent cosine | Pairwise cosine | Temporal variance |
|---|---|---:|---:|---:|
| vggss_10k | A3 | 0.9177 | 0.9093 | 0.34195 |
| vggss_10k | A4 | 0.8793 | 0.8692 | 0.82790 |
| vggss_144k | A3 | 0.9261 | 0.9015 | 0.43262 |
| vggss_144k | A4 | 0.9322 | 0.8924 | 0.97364 |
| flickr_10k | A3 | 0.9142 | 0.9098 | 0.33067 |
| flickr_10k | A4 | 0.8769 | 0.8708 | 0.84403 |
| flickr_144k | A3 | 0.9226 | 0.9112 | 0.38513 |
| flickr_144k | A4 | 0.9200 | 0.8950 | 0.97579 |

## Fusion Utilization

| Setting | Best delta/A4 | cos(fused,A4) slot0/slot1 | Degenerated to A4 |
|---|---:|---:|---:|
| vggss_10k | 0.6677 | 0.9747/0.6815 | False |
| vggss_144k | 0.1072 | 0.9986/0.9882 | False |
| flickr_10k | 2.9090 | 0.9678/0.0715 | False |
| flickr_144k | 0.7058 | 0.9948/0.5739 | False |

## Qualitative

Twelve deterministic panels per setting compare the formal baseline and 3.1 AUD/IMG/IQR maps, A3 query diagnostic, OGL, and absolute map changes.

On isolated improvements, 3.1 sometimes suppresses broad background response and moves the A4 peak onto the sounding person or object. The failure cases are more common in both 144k evaluations: response is displaced toward roads, room structure, or another contextual region, or becomes too narrow to cover the annotated source. A3_QUERY_AUD is usually diffuse or context-focused and does not provide a stable localization cue. AUD and IMG changes frequently move together, consistent with the semantic objective perturbing the shared cross-modal solution rather than adding a reliably complementary acoustic representation.

| Setting | AUD improve | AUD hurt | IQR improve | IQR hurt |
|---|---:|---:|---:|---:|
| vggss_10k | 256 | 258 | 39 | 41 |
| vggss_144k | 287 | 370 | 50 | 63 |
| flickr_10k | 11 | 20 | 1 | 3 |
| flickr_144k | 16 | 20 | 0 | 2 |

## Fixed-Rule Decision

**Negative**

Stop the A3+A4 hierarchical-audio mainline; do not expand to A2/A3/A4 or automatically train G.

```json
{
  "label": "Negative",
  "recommendation": "Stop the A3+A4 hierarchical-audio mainline; do not expand to A2/A3/A4 or automatically train G.",
  "formal_144k_best_AUD_or_IQR_gains": {
    "vggss_144k": -0.01609150833656453,
    "flickr_144k": -0.016000000000000014
  },
  "formal_144k_best_gap_reductions": {
    "vggss_144k": -0.006203955021326035,
    "flickr_144k": -0.02400000000000002
  },
  "checks": {
    "one_dataset_gain_at_least_0.01": false,
    "other_dataset_at_least_baseline_minus_0.004": false,
    "mixed_condition": false,
    "both_datasets_improve": false,
    "both_gaps_shrink_at_least_0.005": false
  }
}
```

No G, Stage2, or follow-up experiment was trained automatically.
