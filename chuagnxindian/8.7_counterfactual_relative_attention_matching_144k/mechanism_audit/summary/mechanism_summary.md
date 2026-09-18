# 8.7 Mechanism Audit Summary

## Classification

**Case 2 — object-level spatial discrimination is the primary confirmed repair.** C3 Stage-1 produces a small but consistent audio-counterfactual perturbation; Stage-2 converts it into better object-level localization, but does not retain lower matched/wrong map Spearman.

## Core mechanism table

All swap values use the shared Experiment-8.1 eight-audio counterfactual set. `RemoveFP` and `AddTP` are paired pixels versus original 1.3G final at the existing threshold 0.6.

| Dataset | Model | Swap Spearman | Swap MAE | GT | NearFP | GT−NearFP | GT/Near AUROC | RemoveFP | AddTP | cIoU | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vggss | 1.3G | 0.9755 | 0.000203 | 0.7979 | 0.7862 | 0.0150 | 0.5920 | 0.0 | 0.0 | 0.4269 | 0.4230 |
| vggss | 8.7-S1 | 0.9734 | 0.000333 | 0.7309 | 0.7034 | 0.0304 | 0.5901 | 4845.7 | 351.7 | 0.3912 | 0.4132 |
| vggss | 8.7-S2 | 0.9779 | 0.000291 | 0.7631 | 0.7203 | 0.0455 | 0.6256 | 3887.7 | 441.6 | 0.4343 | 0.4296 |
| flickr | 1.3G | 0.9876 | 0.000171 | 0.8060 | 0.7640 | 0.0427 | 0.6600 | 0.0 | 0.0 | 0.8120 | 0.6356 |
| flickr | 8.7-S1 | 0.9860 | 0.000303 | 0.7132 | 0.6254 | 0.0889 | 0.6732 | 5175.4 | 222.4 | 0.8640 | 0.6184 |
| flickr | 8.7-S2 | 0.9893 | 0.000260 | 0.7563 | 0.6590 | 0.0981 | 0.7078 | 4183.7 | 367.0 | 0.8960 | 0.6532 |

## Paired evidence

### Stage-1 counterfactual change

- VGG-SS: Spearman Δ = -0.0021 [-0.0026, -0.0015], MAE Δ = +0.000130.
- Flickr: Spearman Δ = -0.0016 [-0.0029, -0.0002], MAE Δ = +0.000132.

### Stage-2 object discrimination

- VGG-SS: GT−NearFP Δ = +0.0305 [+0.0291, +0.0322]; GT/Near AUROC Δ = +0.0336.
- Flickr: GT−NearFP Δ = +0.0554 [+0.0482, +0.0625]; GT/Near AUROC Δ = +0.0478.

### Important qualification

Stage-2's raw matched/wrong Spearman rises rather than falls (VGG-SS +0.0025, Flickr +0.0016), although MAE remains above original and cosine remains below original. Thus the evidence does not support the stronger claim that Stage-2 retains a more audio-sensitive *spatial ranking*. The robust final effect is better target-versus-context/object discrimination.

The AUD-matched counterfactual GT/NearFP AUROC improves significantly on VGG-SS but its Flickr CI crosses zero. It is recorded as partial supporting evidence, not a cross-dataset complementary information claim.

## Reproducibility

- All full extraction manifests record checkpoint paths, hashes, wrong-audio mappings, and exact replay errors.
- Stage-2 matched forward/offline attention replay error: 0 for both models and both datasets.
- Original final maps agree with the established 4.1 artifacts to <= 4.1e-7 native-map absolute error.
- Stage-2 cIoU/AUC and per-sample IoU reproduce the existing formal outputs exactly.
