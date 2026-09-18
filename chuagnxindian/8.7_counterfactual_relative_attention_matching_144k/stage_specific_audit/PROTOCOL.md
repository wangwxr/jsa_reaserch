# 8.7 stage-specific diagnostic audit

Fixed seed 12345. No new method or lambda search. Short diagnostic uses the existing
25,000 CRAM definition, C3 Stage-1 initialization, original 144k NPY loader,
batch 256, AdamW 5e-5 / weight decay .01 and original two-view spatial losses.
Budget: 64 batches per dataset, not a full epoch. Save step 0/16/64 students.
This is a newly seeded diagnostic trajectory, not a bit-identical replay of the old run.

Measure scaled backward gradients of base, weighted CRAM and total separately
on the same graph; divide by the measured AMP scale before statistics. Do not
multiply an already underflowed raw fp16 CRAM gradient by lambda. Use FP32
read-only checks at steps 0/16/63 and old best/latest states. No GT enters training.
Report gradient cosines only for nonzero, finite pairs; zero-gradient groups are
undefined, never evidence of alignment. Parameter groups overlap deliberately:
all student, L3 projection, adapter, adapter last convolution. Activation groups:
K34 and F34 (common ancestors of both objectives). AUD_FINE itself is a sibling
readout to CRAM's recomputed attention, not a shared ancestor. Teacher audio,
visual query and key projection parameters are frozen (N/A, not zero cosine).

Read-only loss-scale sweep 2^16, 2^24, 2^28, 2^32 compares old late states against
FP32. This tests AMP overflow susceptibility; no saved historical scaler state
exists, so it cannot establish the exact failing historical batch mechanism.

Matched-audio evaluation uses all fixed VGG-SS/Flickr evaluation samples and
the original mechanism_audit GT/NearFP indices, interpolation, normalization,
threshold .6 and macro AUROC. RemoveFP/AddTP/RemoveTP/AddFP use original 1.3G
as reference; also report paired metric deltas vs vanilla C3 Stage-2. Reuse
available old epoch 1 / late checkpoints; epoch 5/10/20 checkpoints do not exist.
Entropy is measured on native probability maps (also divided by log(pixel count));
activated area is the original normalized map threshold .6. Audio swap metrics
remain auxiliary. OGL uses original ImageNet prior and original fusion weights
as test-only readout, never training supervision.

Hypotheses are tested, not assumed: gradient conflict, GT/NearFP loss, coverage
collapse, moving visual anchor, and numerical AMP overflow are distinct claims.
No further training variant is authorized by this diagnostic protocol.
