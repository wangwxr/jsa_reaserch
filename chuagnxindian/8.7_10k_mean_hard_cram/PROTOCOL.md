# 8.7 Mean-CRAM vs default Hard-CRAM on the original 10k manifests

This is an isolated, two-stage replication study.  It compares the established
8.7 C3 Mean-CRAM objective with the established default Hard-CRAM objective on
the pre-existing VGG-SS and Flickr 10k manifests.

## Fixed factors

- VGG manifest: `metadata_vggss/vggss_10k.csv` (10,000 rows).
- Flickr manifest: `metadata_flickr/flickr_10k.csv` (10,000 rows).
- Feature roots remain the original NPY roots named in the corresponding 1.3G
  10k configs; no data are copied or resampled.
- Seed 12345, batch size 256, K=4, same-video exclusion, and lambda CRAM=1.0.
- Stage 1: the original 1.3G 10k budget of 100 epochs.
- Stage 2: the original 1.3G 10k budget of 100 epochs, vanilla refinement
  only.  CRAM is not added to Stage 2.
- Mean and Hard use one shared newly-recorded initialization, frozen 100-epoch
  sample order, and frozen negative mapping per dataset.

## Only variable

Mean-CRAM uses `D_neg = mean_k D_neg,k`.  Default Hard-CRAM instead uses the
previously selected smooth soft-min aggregation.  All individual distances,
positive target detachment, Stage-1 architecture, and original Stage-1 losses
are unchanged.

Default soft-min tau values are the established 144k defaults:

- VGG-SS: `2.6128406021282223e-05`.
- Flickr: `8.700235267679427e-06`.

Outputs are isolated below `mean/` and `hard_default/`; existing 8.7 and 1.3G
artifacts are never overwritten.
