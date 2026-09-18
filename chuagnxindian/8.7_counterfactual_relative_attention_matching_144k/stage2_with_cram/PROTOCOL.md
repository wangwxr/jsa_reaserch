# 8.7 C3 Stage-2 with CRAM: Minimal Protocol

## Scope

This independent experiment starts from the selected 8.7 C3 Stage-1 teacher and
uses the original 1.3G Stage-2 architecture, 144k NPY training data,
augmentation, optimizer, schedule, and all original Stage-2 losses unchanged.
It never overwrites `8.7_counterfactual_relative_attention_matching/stage2/C3`.

The only additive term is:

`L_stage2 = L_coarse + L_equiv + lambda_stage2_cram * L_CRAM`.

## Direct Stage-1 CRAM reuse at the Stage-2 decision channel

For the current Stage-2 image representation `K34` and frozen C3 queries:

`T34 = stopgrad(A(Qv, K34)[slot0])`

`D_pos = mean_pixel((A(Qa_matched, K34)[slot0] - T34)^2)`

`D_neg = mean_k mean_pixel((A(Qa_wrong_k, K34)[slot0] - T34)^2)`

`L_CRAM = mean_batch softplus(D_pos - D_neg)`.

`Qa`, wrong `Qa`, and `Qv` all come from the frozen existing C3 Teacher. `K34`
is the current trainable 14x14 Stage-2 key. The visual anchor is detached, so
the new loss updates only the existing Stage-2 spatial student through its
decision keys. No architecture or auxiliary network is introduced.

## Negatives

Each batch uses four deterministic uniform other-video audios from its existing
original Stage-2 batch. Sampling is seeded by `(12345, epoch, batch)`, checks
different video IDs, uses no labels, and records the realized indices/IDs when
training is run. This retains the original data pipeline while matching the
Stage-1 K=4 self-supervised CRAM definition.

## Pre-registered variants

The initial `0.25/0.5/1.0` Stage-1-relative suggestion was rejected before
training by the required Stage-2 gradient audit: a direct 14x14 CRAM derivative
is about `1.4e-5` of the original Stage-2 derivative at lambda 1.0 and rounds
to zero at the AMP fp16 boundary. The only change below is loss-weight scale;
the CRAM definition is unchanged.

- `lambda025k`: 25,000 (about 0.35x original Stage-2 gradient on audit batch)
- `lambda050k`: 50,000 (about 0.71x)
- `lambda100k`: 100,000 (about 1.41x)

Run the supplied `--sanity-only` path before any training. It verifies frozen
teacher gradients, different-video negatives, probability normalization,
finite loss, and nonzero effective CRAM gradient. No variant has been launched by
creation of this directory.
