# 8.7 Negative Quality / Counterfactual Pool Audit

## Scope

This is a read-only audit of the frozen Mean-CRAM Stage-1 checkpoints.  It
does not train, alter a checkpoint, modify `L_CRAM`, change `lambda`, change
`K`, or invoke Stage 2.

## Exact current training negative construction

`prepare_negatives.py` uses `K=4`.  At each frozen epoch/batch, for anchor
`i`, it selects four samples uniformly without replacement from the *other
members of the same training batch* whose parsed video ID differs from the
anchor's.  For VGG-SS the parser strips frame suffixes; for Flickr the sample
ID is the video ID.  This excludes self and same video/source.  There is no
class, audio embedding, semantic similarity, provenance, GT, OGL, or object
filter.  The globally advancing generator is
`np.random.default_rng(seed + 87_000_001)` (seed `12345`), and the frozen
offsets are saved per epoch in `configs/*negative_offsets.npy`.

The exact loss distance is

`D_pos = mean_t MSE(a2v_prob[i,0,t], stopgrad(v2v_prob[i,0,t]))`

and `D_neg,j` replaces the matched audio query by wrong audio `j`, retaining
the anchor image's visual keys and detached `v2v_prob` target.  Hence lower
`D_neg,j` means a wrong audio can better reproduce the image-derived spatial
target and is more counterfactually confusing.

## M=32 evaluation-pool protocol

For each frozen test image, the first four candidates are the existing
Experiment 8.1 `set1` uniform, different-video draw.  They form a fixed,
test-side *random-K=4 surrogate*.  A deterministic SHA-256 seeded draw adds 28
unique, different-video candidates, producing a nested M=32 pool.  Candidate
selection only sees IDs for video exclusion and the frozen D-distance; it does
not see label, GT, OGL, Group-A, or any localization metric.

This protocol measures whether random counterfactual coverage is associated
with the known residual failure.  It is deliberately not a claim that the
test-side four candidates equal a particular training batch's saved offsets.

## Models and controls

Frozen Mean-CRAM Stage-1 checkpoints are audited at 144k (VGG-SS/Flickr) and
10k (VGG-SS/Flickr).  The main test is VGG-SS 144k.  Fixed VGG residual groups
and Hard-help/Hurt labels are loaded *after* distance statistics are written.
Flickr has no usable class metadata in this pipeline, so class-based
false-negative rate is marked unavailable there.

## Statistics

Group comparisons report mean, median, IQR, 4,000-resample bootstrap 95% CI
for the mean difference, Mann-Whitney p, and rank-biserial effect.  All seeds
for analysis bootstrap and candidate-tail draws are encoded in the scripts.
