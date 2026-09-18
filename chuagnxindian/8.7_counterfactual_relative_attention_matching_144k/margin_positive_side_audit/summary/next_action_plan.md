# Next action — not launched

The next target is **positive spatial side**, not the negative side.  Before
training anything, run a read-only validation of a label-free positive-map
confidence/compactness signal.  It must predict broad matched activation on
the fixed residual protocol without accessing GT, OGL, Group-A, or class
metadata in its definition.

Only if that validation succeeds may a minimal Stage-1-only positive-anchor
weighting variant be proposed.  Mean negative aggregation, K=4, lambda=1,
the architecture, seed, data recipe, and vanilla Stage-2 must remain frozen.
