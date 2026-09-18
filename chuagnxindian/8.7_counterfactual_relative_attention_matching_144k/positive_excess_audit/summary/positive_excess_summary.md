# Result: S2 — excess contains both FP and useful TP

On VGG Group-A, AUD–IMG_QUERY positive excess has a significantly higher FP mass fraction than non-Group-A (0.793 vs 0.732; difference 0.060, 95% bootstrap CI [0.041, 0.080], Mann–Whitney p=0.0018). This is driven by background excess (0.790 vs 0.729), not NearFP pixels at the formal grid. Group-A excess occupies OGL RemoveFP much more selectively: the fraction of all excess pixels that are RemoveFP is 0.111 versus 0.040 outside Group-A; by contrast its RemoveTP occupancy is only 0.0065 versus 0.0206.

That is useful mechanism evidence, but it is not a safe control signal. At the pixel level `PositiveExcess` has AUROC 0.488 for distinguishing RemoveFP from RemoveTP in VGG’s Group-A/AUD-better audit subset—chance-level. It also has no meaningful monotonic association with the magnitude of OGL’s IoU gain inside Group-A (Spearman −0.049). In AUD-better samples, excess is much more GT-aligned (GT ratio 0.494; FP ratio 0.506) and overlaps RemoveTP more (0.683), so simple suppression would remove useful target coverage.

Flickr is the required safety control: its excess is more GT-aligned globally (GT ratio 0.464 vs VGG Group-A 0.207) and global excess magnitude does not predict OGL gain (rho 0.048). This fits the already observed AUD>OGL result and argues strongly against a VGG-only global penalty.

Therefore the audit selects **S2**, not S1: AUD–IMG_QUERY disagreement identifies a mixture of context spread and useful audio-added target response. It cannot be directly used as a suppressive positive-side CRAM term.
