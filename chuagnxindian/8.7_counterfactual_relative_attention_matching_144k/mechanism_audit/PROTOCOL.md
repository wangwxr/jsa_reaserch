# Experiment 8.7 Mechanism Audit Protocol

## Scope

This is a frozen, post-hoc comparison of exactly three model decisions:

1. the formal original 1.3G final `AUD_FINE`;
2. 8.7 C3 Stage-1 `AUD`;
3. 8.7 C3 Stage-2 `AUD_FINE`.

No checkpoint is trained or modified. The test splits, cached evaluation audio,
sample order, and same-image counterfactual assignments are inherited from
Experiments 8.0, 8.1, and 8.6.

## Fixed inputs

- Seed: `12345` for model construction and `87012345` for bootstrap resampling.
- Counterfactuals: the first eight `set1_indices` from the immutable Experiment
  8.1 files. Every counterfactual must have a different video ID.
- Regions: the immutable Experiment 8.0 `GT`, `FP`, `NearFP`, and `FarFP`
  pixel indices.
- Natural-map processing: bicubic resize to 224x224 (`align_corners=False`),
  per-map min-max normalization, threshold 0.6.
- Binary GT for precision/coverage and transition counts: `GT >= 0.5`.
- Formal Flickr IoU retains its soft consensus GT.

## Comparable counterfactual decision map

Stage-1 is natively 7x7. Each 14x14 Stage-2 probability map is sum-pooled in
non-overlapping 2x2 cells and spatially renormalized to 7x7. Primary audio-swap
statistics therefore compare 49 spatial decisions for every model. Native
14x14 final maps remain the source for natural localization.

Per image, metrics are averaged over the same eight wrong audios:

- Spearman and Pearson correlation;
- cosine similarity;
- mean absolute difference, exactly `|Hm-Hc|_1 / 49`;
- native Top-10 Jaccard overlap;
- matched-minus-median-wrong GT, NearFP, and GT-minus-NearFP response changes.

## Matched-audio object metrics

For each image and model:

- GT, NearFP, FarFP, and all-background mean response;
- `GT - NearFP` response gap;
- within-image GT-vs-NearFP AUROC;
- precision, coverage, predicted area, official IoU;
- cIoU and formal AUC aggregated exactly from sample IoUs.

`RemoveFP`, `AddTP`, `RemoveTP`, and `AddFP` retain the paired D2 definitions,
with original 1.3G final fixed as the reference map. Consequently the reference
column is zero by definition; rates and all four directions are reported so the
comparison cannot conceal a precision/coverage trade-off.

## Uncertainty and interpretation

All distribution summaries contain mean, median, Q1, Q3, and a deterministic
2,000-resample bootstrap 95% confidence interval for the mean. Candidate-minus-
baseline paired confidence intervals are also saved.

The final mechanism label is descriptive, not a new efficacy gate:

- Case 1: audio sensitivity improves while object discrimination is unchanged;
- Case 2: object discrimination improves while audio sensitivity is unchanged;
- Case 3: both improve;
- Case 4: neither improves.

An improvement must be directionally consistent on both datasets for the main
cross-dataset label. Dataset-specific effects and Stage-1-to-Stage-2 conversion
are reported even when the global label is mixed.

