# Hard-negative CRAM: frozen final audit

This directory compares the completed default soft-min Hard-CRAM experiment to
the completed Mean-CRAM `C3 Stage-1 -> vanilla Stage-2` experiment.  It does
not train a model or modify prior checkpoints.

## Protocol

- Test splits, threshold (`.6`), six readout definitions, fixed Experiment-8.1
  eight-audio swaps, and the Experiment-8.0 GT/NearFP regions are reused
  exactly.
- The Hard teacher's own audio queries are recomputed from the cached test
  NPY spectrograms.  Mean teacher queries are never substituted.
- `mean_cram` AUD per-image IoU reproduces the established formal C3 result
  to <= `1.12e-16` (VGG-SS) and <= `5.56e-17` (Flickr).
- `Hard_vs_Mean_RemoveFP/RemoveTP/AddTP/AddFP` are direct transition counts
  from the Mean mask to the Hard mask.  They are not legacy comparisons to
  the original 1.3G map.

## Default-tau conclusion

The default soft-min does not outperform Mean-CRAM globally.  On VGG Group A
(the frozen residual OGL-help group), it raises precision and GT-NearFP gap
while preserving coverage within its CI, but its aggregate AUD remains below
Mean-CRAM and OGL.  On Flickr it lowers GT-NearFP gap, AUROC, precision and
AUD.  Thus it is evidence that the hard-negative direction is not uniformly
beneficial at the default temperature, not evidence for a second module.

## Files

For each dataset:

- `hard_maps.npz`: matched/wrong Hard maps and own IMG_QUERY map.
- `six_readout_summary.csv`: AUD, IMG_QUERY, IQR, OBJ_PRIOR, OGL,
  EXTRA_IQR_OGL cIoU/AUC.
- `object_per_sample.csv`, `audio_swap_per_sample.csv`: sample-level metrics.
- `paired_hard_minus_mean.csv`: paired bootstrap CIs for continuous metrics.
- `hard_vs_mean_transition_summary.csv`: direct mask transition counts.
- `group_deltas.csv`: VGG Group-A/B/C/transition results (Flickr `all`).
