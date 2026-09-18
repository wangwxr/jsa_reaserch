# Next action (not launched)

Do **not** train confusion-aware negative mining.  The mandatory Group-A
enrichment condition failed.  The one evidence-based next step is a read-only
Mean-CRAM margin/violation audit: quantify `D_pos - mean(D_neg)` for frozen
training-side samples, then test whether violation state predicts the fixed
VGG residual Group-A without using GT/OGL for its definition.  No changes to
K, lambda, aggregation, Stage 2, architecture, or data recipe are authorized.
