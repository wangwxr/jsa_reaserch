# Six predeclared label-free candidates

All cosine signals are higher-is-more-supported and are computed from L2-normalized existing feature vectors.

1. `f34_core_cos`: F34 token cosine to the mean F34 vector in the internal core.
2. `k34_core_cos`: K34/fine-key cosine to its internal-core prototype.
3. `l4_core_cos`: L4 visual-key cosine to a coarse internal-core prototype.
4. `cross_level_cos`: cosine between F34 and the upsampled existing F4/L4 representation after their existing channel-compatible projection.
5. `local_query_support`: 3×3 mean of normalized IMG_QUERY around each pixel.
6. `combined`: mean of within-image z-scored `f34_core_cos` and `local_query_support`.

No learned combination, classifier, GT, OGL, object prior, or slot relabeling is permitted.
