# 8.7 Experiment Index

Last audited: 2026-09-16.  This is an inventory, not a new result.

## Naming rule

- **144k** is the training corpus/recipe: the original 144k NPY pipeline,
  batch size 256, 50 Stage-1 epochs (and, where run, 50 vanilla Stage-2
  epochs).
- `lambda025k` and `lambda050k` in `stage2_with_cram/` mean loss weights
  **25,000** and **50,000**.  They do **not** mean 25k/50k training samples.
- There is **no completed or running 8.7 10k training experiment**.  Any
  `10k` checkpoints elsewhere in the repository belong to earlier baseline
  work, not to this experiment directory.

## Main Mean-CRAM study (all use 144k NPY, seed 12345)

| ID | Stage-1 objective | Stage-1 status | Vanilla Stage-2 | VGG AUD_FINE cIoU / AUC | Flickr AUD_FINE cIoU / AUC |
|---|---|---|---|---:|---:|
| C0 | Original loss; reusable 8.6 M0 control | reused | not a new 8.7 run | — | — |
| C1 | Mean-CRAM, lambda=.1, K=4 | complete, 50e | not run | .3755 / (see Stage-1 curve) | .8280 / (see Stage-1 curve) |
| C2 | Mean-CRAM, lambda=.5, K=4 | complete, 50e | not run | .3850 / (see Stage-1 curve) | .8400 / (see Stage-1 curve) |
| **C3** | **Mean-CRAM, lambda=1.0, K=4** | **complete, 50e** | **complete, 50e** | **.4343 / .4296** | **.8960 / .6532** |

C3 is the established best 8.7 main formulation.  Its Stage-1 values are
VGG `.3912 / .4132` and Flickr `.8640 / .6184`; the table's bold values are
after its vanilla Stage-2 refinement.

## Direct Stage-2 CRAM ablations (all use 144k NPY, seed 12345)

| Variant | Datasets | Status | Interpretation |
|---|---|---|---|
| `lambda050k` | VGG, Flickr | stopped: instability/`inf` gradients | coefficient 50,000; not 50k data |
| `lambda025k` | VGG, Flickr | stopped: instability/`inf` gradients | coefficient 25,000; not 25k data |

Both begin from the C3 Stage-1 teacher and retain original Stage-2 otherwise.
They are failed stage-specific ablations, not final competitors.

## Stage-specific and mechanism analyses (no full new training)

| Directory | Scope | Outcome |
|---|---|---|
| `mechanism_audit/` | C3 S1/S2 vs original 1.3G | Object-level GT-vs-NearFP discrimination improves; final audio-swap ranking does not materially change. |
| `residual_ogl_gap_audit/` | VGG C3 Stage-2 AUD vs OGL | Residual OGL gain is concentrated in over-activation/context false positives, principally RemoveFP. |
| `stage_specific_audit/` | short 25k-coefficient diagnostic | Stage-2 CRAM conflicts with/refines poorly against the fine-stage objective; not a 25k-sample run. |

## Hard-negative CRAM ablations (all use 144k NPY, seed 12345)

| Variant | Negative aggregation | Stage-1 -> vanilla Stage-2 | VGG AUD_FINE cIoU / AUC | Flickr AUD_FINE cIoU / AUC | Status |
|---|---|---|---:|---:|---|
| Hard default | soft-min, calibrated default tau | complete, 50e -> 50e | .4275 / .4283 | .8720 / .6430 | below Mean-C3 |
| VGG stronger tau | soft-min, `N_eff≈2` | complete, 50e -> 50e | .4254 / .4280 | — | below Mean-C3 |
| Flickr smoother tau | soft-min, `N_eff≈3.08` | complete, 50e -> 50e | — | .8480 / .6558 | below Mean-C3 on cIoU |

These are formulation ablations.  They do not replace C3 Mean-CRAM.

## Authoritative locations

- C1/C2/C3 Stage-1 checkpoints/configs: `checkpoints/C*/<dataset>/seed12345/`
- Main C3 Stage-2 result: `stage2/C3/<dataset>/seed12345/best_test_metrics.json`
- Direct Stage-2 CRAM: `stage2_with_cram/checkpoints/`
- Hard-negative variants: `8.7_hard_negative_cram/`
