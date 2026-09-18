# C3 Stage-2 performance follow-up

On 2026-09-12 the user explicitly authorized Stage 2 on two GPUs following
review of C3's Flickr cIoU improvement and the four-metric performance goal.
This overrides the execution restriction; the original Stage-1 mechanism
decision remains CRAM-NO-GO and is not retroactively relabeled CRAM-GO.

Use C3 lambda=1.0 selected Stage-1 checkpoints on both datasets, seed 12345.
Run the existing 1.3G Stage-2 architecture and loss unchanged for 50 epochs,
batch 256, AdamW lr 5e-5, weight decay .01, no scheduler. Frozen Teacher,
copied proj3_spatial, zero-initialized existing adapter, coarse anchoring and
geometric equivariance remain the original recipe. No CRAM term in Stage 2.

The final method consists of two training stages. Train data remains the
original 144k NPY datasets. Evaluation uses the byte-verified 8.6 audio cache
with the original transforms, IDs, annotations, and cIoU/AUC protocol.
The original full-split Teacher reproduction and geometry/gradient sanity
checks must pass before the first formal training epoch.

Use the original AUD_FINE cIoU checkpoint-selection rule. Report both cIoU
and AUC at that one checkpoint (and preserve epoch-50 metrics). Do not combine
separately selected maxima. This is an exploratory single-seed follow-up with
test-based epoch selection inherited from the baseline, not independent
confirmation or a pre-test choice of C3.

The performance objective is to exceed the original 1.3G final AUD cIoU/AUC
on both VGG-SS and Flickr simultaneously. Historical targets are VGG-SS
0.42690965 / 0.42295463 and Flickr 0.812 / 0.6356. Preserve all results even
when only some metrics improve. A positive result needs a second seed before
a final method claim. No subsequent experiment is launched automatically.
