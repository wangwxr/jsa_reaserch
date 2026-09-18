# Experiment 8.7 Report — CRAM-v0

## Result

**CRAM-NO-GO** on the registered Stage-1 mechanism gate (originally
`stage2_allowed = false`). On 2026-09-12 the user explicitly authorized a C3
Stage-2 performance follow-up on both GPUs. Execution is now allowed under
that amendment; see `stage2/PROTOCOL.md`. This does not change the mechanism
decision reported below.

All results use the preregistered first-pass seed (`12345`) and selected
Stage-1 checkpoint. C0 is unchanged 8.6 M0; C1/C2/C3 add CRAM at 0.1/0.5/1.0.
C3 was permitted before test evaluation by the predeclared gradient rule.

CRAM created the intended training ordering: at epoch 50, `D_neg - D_pos` is
`1.18e-4`, `1.33e-4`, `1.32e-4` on VGG-SS and `5.50e-5`, `6.94e-5`,
`6.04e-5` on Flickr for C1/C2/C3. But the unscaled softplus stays near 0.6931,
so this did not become a strong calibrated margin.

## Central comparison

`swap rho` is matched-vs-wrong mean Spearman (lower is more audio-sensitive).
`S_CF` and `z_specific` are GT-vs-NearFP macro AUROC; the second value is
AUD-matched. M1 is the exact 8.6 No-L_match control.

| Data | Group | swap rho | S_CF raw / matched | z_spec raw / matched | full/common rho | cIoU | Precision / Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG | C0 | .9718 | .5375 / .5006 | .5473 / .5013 | .9891 | .3990 | .5428 / .7645 |
| VGG | M1 no match | .7907 | .5050 / .4646 | .5079 / .4611 | .9157 | .2662 | .4472 / .7254 |
| VGG | C1 (.1) | .9630 | .5358 / .5047 | .5422 / .5022 | .9856 | .3755 | .5425 / .7426 |
| VGG | C2 (.5) | .9681 | .5355 / .5029 | .5420 / .5004 | .9877 | .3850 | .5445 / .7500 |
| VGG | C3 (1) | .9734 | .5351 / .5008 | .5412 / .4986 | .9895 | .3912 | .5442 / .7577 |
| Flickr | C0 | .9862 | .4964 / .4612 | .4958 / .4488 | .9937 | .7920 | .7057 / .8008 |
| Flickr | M1 no match | .5820 | .5167 / .4636 | .5201 / .4597 | .7942 | .3240 | .5896 / .5191 |
| Flickr | C1 (.1) | .9876 | .5357 / .4885 | .5288 / .4762 | .9941 | .8280 | .7212 / .7904 |
| Flickr | C2 (.5) | .9835 | .5033 / .4619 | .4986 / .4528 | .9926 | .8400 | .7161 / .7913 |
| Flickr | C3 (1) | .9860 | .5393 / .4789 | .5405 / .4944 | .9937 | .8640 | .7492 / .7655 |

NearFP conditional 95% CIs are C1/C2/C3=`[.4989,.5105]`, `[.4974,.5088]`,
`[.4951,.5067]` on VGG and `[.4578,.5190]`, `[.4331,.4928]`,
`[.4499,.5081]` on Flickr. No lower bound exceeds 0.5.

## Gate evidence and interpretation

- **Anchor:** retained overall. Fixed-train-batch C0 `D_pos` is `4.77e-5` VGG
  and `1.50e-5` Flickr; C1 is `3.75e-5`/`1.59e-5`. C2/C3 worsen Flickr to
  `2.89e-5`/`2.40e-5`. Full/common geometry remains close to C0, unlike M1.
- **Sensitivity:** C2 lowers swap rho on both datasets only trivially
  (`-.0037`, `-.0027`) and gains no grounding. C1 helps VGG but raises Flickr
  rho; C3 raises VGG rho.
- **Correctness:** Flickr C1/C3 raw NearFP separation improves, but VGG raw
  separation falls at every weight. AUD-matched evidence is inconclusive.
- **Natural localization:** CRAM avoids M1's collapse, but VGG cIoU/coverage
  decline for every setting. VGG matched-coverage precision deltas are C1
  `-.00843` (95% CI `[-.01020,-.00667]`), C2 `-.00312`, C3 `-.00170`, which
  violates the no-decrease gate despite positive Flickr values for C1/C3.

Thus CRAM-v0 is safer than deleting L_match but does not turn preserved spatial
organization into reproducible semantic grounding. Stage 2 was not started
at the time of this Stage-1 report; the later user-authorized C3 follow-up is
recorded above.

## Artifacts

- exact tables: `natural_localization/`, `experiment_8_1_replay/`,
  `experiment_8_3_geometry/`;
- training/gradient records: `training_curves/`, `gradient_audit/`;
- fixed replay figures: `figures/`.
