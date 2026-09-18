# Experiment 8.7 — Counterfactual Relative Attention Matching

## Frozen scope

CRAM-v0 is an additive Stage-1-only regularizer. It preserves the complete
original `L_match = L_A2V + L_V2A`, InfoNCE, reconstruction, diversity,
architecture, optimizer, schedule, augmentations, initialization, and fixed
epoch orders from Experiment 8.6 M0. Stage 2 is forbidden unless the frozen
Stage-1 decision is `CRAM-GO`.

No OGL, SAM, detector, MLLM, GT mask, external supervision, object network,
third stage, replacement loss, synthetic provenance, or Teacher ownership
target is used.

## Loss

For the matched pair, `D_pos = D_match(ca_av(I_i,a_i), stopgrad(ia_vv(I_i)))`.
For four uniformly sampled, different-video audios from the same fixed training
batch, `D_neg` is the mean of the analogous four distances. The only new term is

`L_CRAM = softplus(D_pos - D_neg)`.

`L_stage1 = L_original_stage1 + lambda_cram * L_CRAM`.

`D_match` is the verified original A2V MSE, and the V2V target is detached at
the same stop-gradient location as the original loss. Wrong maps are never used
as masks or pixel targets.

## Controls and execution

| Group | Stage-1 loss | Status |
|---|---|---|
| C0 | exact reused 8.6 M0 | immutable reference |
| C1 | original + `0.1 L_CRAM` | primary |
| C2 | original + `0.5 L_CRAM` | fixed strength control |
| C3 | original + `1.0 L_CRAM` | only if pre-training audit permits |

Primary seed is `12345`. C1 and C2 run regardless of test metrics. C3 is allowed
only when the frozen training-batch audit finds `0.1||grad L_CRAM||` below 1% of
`100||grad L_match||`; it is decided before evaluation. Every sample uses four
different-video negatives. The index offsets, sample-ID manifest, reused order,
and initialization hashes are persisted.

## Required evaluation and gate

Use the exact 8.1 replay, full VGG-SS/Flickr swaps, raw and AUD-matched
GT/NearFP counterfactual AUROC with 2,000 image bootstrap repeats, the 8.3
geometry replay, and natural AUD/IQR localization. Directly compare C0, C1,
C2 and 8.6 M1.

`CRAM-GO` requires all three: preserved natural localization/positive anchor,
lower swap Spearman, and improved semantic grounding (including an AUD-matched
GT/NearFP CI above 0.5 on at least one dataset and same direction on the other),
with no matched-coverage precision or large coverage loss. `CRAM-WEAK-GO` is a
positive but insufficient conditional-grounding trend and leaves Stage 2 false.
Any M1-like collapse, mere randomization of wrong maps, anchor degradation,
FarFP-only benefit, or conflicting datasets is `CRAM-NO-GO`.
