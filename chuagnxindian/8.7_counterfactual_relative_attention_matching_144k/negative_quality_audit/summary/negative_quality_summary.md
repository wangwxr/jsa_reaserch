# Negative Quality / Counterfactual Pool Audit — Final

## Decision

**Case C — the negative-construction hypothesis is not established.**

The M=32 pool demonstrates a generic fact: uniform random K=4 often fails to
include the very lowest-D candidates.  It does **not** demonstrate the needed
causal association: the fixed VGG residual Group-A (428 OGL-help/OVER
examples) is not enriched for that miss, and the internal signals do not
predict its OGL gain.  Therefore this audit does not authorize confusion-aware
top-K mining.

## Main VGG-SS 144k result

| Metric | Fixed Group-A (n=428) | non-Group-A (n=4,730) | A − non-A, bootstrap 95% CI | MW p |
|---|---:|---:|---:|---:|
| opportunity_gap_mean | 5.58e-5 | 4.99e-5 | [-0.93e-6, 1.30e-5] | .0609 |
| opportunity_gap_min | 2.14e-5 | 1.87e-5 | [-1.18e-6, 7.00e-6] | .7927 |
| pool violation fraction | .3042 | .3018 | [-.0245, .0289] | .6318 |
| random-4 violation fraction | .2979 | .3033 | [-.0365, .0256] | .9233 |
| missed confusing-negative rate | .2827 | .2947 | [-.0572, .0334] | .6016 |
| pool minimum D | 1.80e-5 | 1.82e-5 | [-1.75e-6, 1.36e-6] | .9507 |
| top-4 mean D | 2.44e-5 | 2.42e-5 | [-1.92e-6, 3.00e-6] | .7434 |

The one weak pattern is Group-A's slightly greater *mean* opportunity.  Its
CI crosses zero and it is not accompanied by more violations, a lower pool
minimum, or a higher missed rate.  It is insufficient evidence for mining.

Pool-wide VGG-SS 144k values explain why a naive global top-K change remains
tempting but unsafe: random-4 mean D=7.50e-5 versus top-4 mean D=2.42e-5,
mean opportunity=5.08e-5, and 29.4% of individual samples have a pool
violation missed by random-4.  Yet this occurs broadly, not specifically in
the residual OVER failure.

## OGL relation and Hard controls

OGL-help versus AUD-better has a small opportunity-gap difference (9.68e-6,
bootstrap CI [0.23e-6, 19.0e-6], MW p=.021), but neither pool violation
(p=.703) nor missed-negative rate (p=.492) differs.  Across all 5,158 VGG
examples, opportunity-gap versus `OGL-AUD IoU` has Spearman rho=.0325;
missed-negative rho=.0046; pool-violation rho=.0033.  These effects are not
useful predictors of the residual spatial failure.

VGG Hard-help versus Hard-hurt has a nominally larger opportunity gap
(2.55e-5, CI [0.01e-5,5.10e-5]) but its Mann-Whitney p=.071 and missed rate,
pool violation, minimum D and top-4 mean are all non-significant.  Thus it
does not rescue a reliable hard-negative criterion.  Flickr help/hurt subsets
are very small (8/22) and all corresponding inference is inconclusive.

## Cross-scale / dataset controls

| scale, dataset | mean opportunity gap | pool violation | random-4 violation | missed rate |
|---|---:|---:|---:|---:|
| 144k VGG-SS | 5.08e-5 | .3020 | .3028 | .2937 |
| 144k Flickr | 1.65e-5 | .3481 | .3480 | .3040 |
| 10k VGG-SS | 2.90e-5 | .3444 | .3461 | .2660 |
| 10k Flickr | 0.80e-6 | .5465 | .5360 | .1480 |

Distances themselves are scale-dependent, so they are not compared as a
single absolute threshold.  The relevant result is that a nontrivial generic
miss rate exists in all controls, while it fails to align with VGG Group-A.
This rules out the claim that VGG's residual OGL gap is uniquely a random-K=4
coverage deficiency.

## False-negative and diversity audit

For VGG, top-4 confusing candidates have a same-class rate of 1.28%, versus
.51% for random-4.  They also have higher anchor-audio cosine (.661 vs .605)
and higher within-top4 cosine (.675 vs .603), although their mean class count
remains high (3.93 of four).  Same-video/self count is exactly zero.  Thus a
pure top-K implementation carries a modest but real semantic false-negative
and redundancy shift.  Flickr class labels are unavailable in this evaluation
pipeline; its top-4 similarly raises embedding cosine (.649 vs .611) and
within-set cosine (.677 vs .613), while preserving zero same-video/self.

## One main direction, one backup

**Main direction: do not mine negatives; retain Mean-CRAM and diagnose its
sample-wise violation/margin distribution before changing any training.**
The observed failure is not identifiable by the proposed candidate-pool
signals.  The minimal next *diagnostic* should test whether the Mean-CRAM
constraint is dominated by already-satisfied samples and whether the frozen
Group-A is enriched among constraint violations.  Candidate construction,
K=4, mean aggregation, lambda=1, Stage-1 location, seed, and vanilla Stage-2
must remain frozen.  A later modification would be justified only if a
label-free sample-level violation signal predicts Group-A/OGL gain with a
predeclared effect and CI.

**Backup: if that violation diagnostic is also negative, stop CRAM-internal
mining/weighting routes rather than adding a second innovation.**  The
remaining VGG gap would then require a separately evidenced diagnosis.

## What this does and does not imply

This does not weaken the established Mean-CRAM result.  It says specifically
that selecting top-D negatives from a larger pool is not currently supported
as the way to close the remaining VGG OGL gap.  No training was launched by
this audit.
