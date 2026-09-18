# 8.7 Margin and Positive-Side Spatial Audit

## Scope

This is a read-only audit.  It starts no training and changes no checkpoint,
loss, lambda, K, negative sampling, architecture, or Stage-2 recipe.

## Phase A

The frozen Mean-CRAM test-distance cache contains the existing fixed
Experiment-8.1 set1 different-video random K=4 draw.  For each sample:

`m = mean_k(D_neg,k) - D_pos` and `violation = (m <= 0)`.

`D_pos`, all four `D_neg,k`, and their min/mean/max were generated from the
same detached V2V target and A2V attention distance as Mean-CRAM.  The
predeclared A-positive rule required all of: Group-A margin CI wholly below
zero, Group-A violation CI wholly above zero, and
rho(margin, OGL-AUD IoU) < -0.1 with CI wholly below zero.  Otherwise Phase B
is mandatory.

## Phase B

The full-set AUD/OGL response and transition statistics are reused from the
existing frozen residual and mechanism audits.  Fixed Group-A remains exactly
the original 428 VGG OGL-help examples.  AUD/IMG_QUERY/IQR map readout
comparison is replayed from immutable cached maps at their native 14x14/7x7
grid, solely for descriptive group-level comparison.  The C3 7x7 visual-key
cache is pooled over GT and the existing fixed NearFP regions; it is never fed
back into a loss.

All CIs use a deterministic 300-resample percentile bootstrap; rank tests and
point estimates use all samples.  The Spearman CI bootstraps correlation of
fixed full-sample ranks, while the point estimate is standard Spearman.
