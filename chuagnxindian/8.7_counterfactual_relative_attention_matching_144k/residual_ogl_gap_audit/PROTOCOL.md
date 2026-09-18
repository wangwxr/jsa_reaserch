# Protocol

Scope: VGG-SS only, frozen 8.7 C3 Stage-1 -> vanilla Stage-2 AUD_FINE.

AUD is the saved fine attention map, bicubic-resized to 224 and independently
min-max normalized. OGL is `minmax(0.6 * AUD + 0.4 * ImageNet ResNet18 object
prior)`, with the same prior resize/normalization and threshold 0.6 used by the
existing evaluation. Per-sample AUD/OGL IoUs must reproduce the saved cache.

Groups are preregistered here: A `delta_iou > .10`, B `abs(delta_iou) <= .05`,
C `delta_iou < -.10`; values between these regions are reported as transition,
not reassigned.  AUD-to-OGL Add/Remove counts are paired pixel changes; they
are not the older comparisons against the 1.3G baseline.
