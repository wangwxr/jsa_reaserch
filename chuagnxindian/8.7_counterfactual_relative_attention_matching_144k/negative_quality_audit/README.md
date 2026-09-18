# Negative Quality Audit

This directory contains the completed, training-free audit requested for 8.7.
The central result is **Case C: negative-construction hypothesis not
established for the VGG residual Group-A.**  Random K=4 often misses a lower
distance candidate in M=32, but that opportunity is not enriched in the fixed
OGL-help/OVER group and does not predict OGL residual gain.

See `PROTOCOL.md` for the exact frozen, label-free candidate construction and
`summary/negative_quality_summary.md` for the evidence and recommendation.

Key reproducible artifacts:

- `candidate_pool/per_sample_pool_stats_*`: all sample-level D statistics.
- `groupA/`: fixed Group-A comparisons; no group was redefined.
- `ogl_relation/`: OGL relation and continuous correlations.
- `false_negative/` and `diversity/`: audit-only semantic risk and embedding
  diversity checks.
- `cross_scale/`: 10k/144k and VGG/Flickr controls.
- `figures/`: 300 dpi PNG and vector PDF figures plus their source script.
