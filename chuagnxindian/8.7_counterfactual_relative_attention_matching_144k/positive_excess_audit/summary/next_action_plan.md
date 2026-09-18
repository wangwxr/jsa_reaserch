# Decision

Do not train a direct `AUD≈IMG_QUERY` or `penalize(AUD-IMG_QUERY)` objective. The next permitted positive-side direction, if pursued, is to audit a finer **internal spatial confidence** signal that can distinguish audio-only FP from audio-added TP without GT/OGL. Keep Stage-1 only, Mean negative aggregation, K=4, lambda=1, and vanilla Stage-2 frozen. This is still a CRAM-internal refinement, not a second module.
