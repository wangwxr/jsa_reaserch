# Stage-1 training-speed comparison

Snapshot: 2026-09-15.  All entries use the 144k split, batch size 256,
seed 12345, and 50-epoch recipe unless noted.  `epoch_seconds` includes the
in-training AUD/IQR validation and checkpoint bookkeeping, so the comparison
matches wall-clock progress observed by the user.

| Dataset | Run | Steady-state epoch seconds (median) | Source |
|---|---|---:|---|
| VGG-SS | Original 1.3G Stage-1 (`jsa_vggss_144k`) | 312.91 | `checkpoints/jsa_vggss_144k/epoch_metrics.csv` |
| VGG-SS | Mean-CRAM C3, pre-reuse | 575.25 | `8.7_counterfactual_relative_attention_matching/training_curves/vggss/C3/seed12345.csv` |
| VGG-SS | Default Hard-CRAM, pre-reuse | 590.65 | `8.7_hard_negative_cram/training_curves/vggss/C3/seed12345.csv` |
| VGG-SS | Hard-CRAM, exact-reuse, tau variant | 311.31 | `tau_variants/vggss_tau_stronger_neff2/training_curves/vggss/C3/seed12345.csv` (epochs 1--6 snapshot) |
| Flickr | Original 1.3G Stage-1 (`jsa_flickr_144k_frame8_center5`) | 177.03 | `checkpoints/jsa_flickr_144k_frame8_center5/epoch_metrics.csv` |
| Flickr | Mean-CRAM C3, pre-reuse | 568.50 | `8.7_counterfactual_relative_attention_matching/training_curves/flickr/C3/seed12345.csv` |
| Flickr | Default Hard-CRAM, pre-reuse | 566.36 | `8.7_hard_negative_cram/training_curves/flickr/C3/seed12345.csv` |
| Flickr | Hard-CRAM, exact-reuse, tau variant | 266.86 | `tau_variants/flickr_tau_weaker_neff3_08/training_curves/flickr/C3/seed12345.csv` (epochs 1--7 snapshot) |

The formal B=256 exactness benchmark measured a 2.269x VGG-SS and 2.249x
Flickr speedup for the replaced CRAM path, with zero differences in D_pos,
D_neg, loss, and gradients.  It is recorded in
`acceleration/{vggss,flickr}/acceleration_validation.json`.

## Interpretation

The current exact-reuse runs are materially faster than the prior CRAM runs:
about 1.85x on VGG-SS and 2.12x on Flickr based on the steady-state medians.
VGG-SS is now effectively equal to the historical original 1.3G epoch time.
Flickr remains about 1.51x slower than the historical original 1.3G run.
That residual is expected compute from the CRAM objective: each batch still
needs the matched path plus four counterfactual audio-conditioned attention
evaluations.  Reuse removes redundant audio encodes, but cannot remove those
four required attention/gradient paths without changing the objective.

The residual is not explained by the validation pass.  In particular, the
historical Flickr run evaluates only 250 test examples in one B=256 batch;
the current replay uses B=32 (eight batches) but both are tiny compared with
562 training batches.  The historical original recipe itself evaluated more
readouts (AUD, IMG_QUERY, IQR, OBJ_PRIOR, OGL, EXTRA_IQR_OGL) per epoch than
the current training-time AUD/IQR validation.  Therefore the dominant extra
wall-clock time is training-side CRAM computation, not test evaluation.
